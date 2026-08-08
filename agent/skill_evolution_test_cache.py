"""skill_evolution_test_cache.py

Cache de tests ejecutables generados por el reflection agent del Skill Evolution
Engine (SEE). Los tests son codigo Python caro de generar (modelo mas costoso,
p.ej. Claude Opus); cachearlos entre evoluciones del mismo skill evita volver a
llamar al LLM.

API publica: los cinco metodos de TestCache replican la firma pedida
(get_cached_test / store_test / invalidate_stale_tests / prune_expired_tests /
archive_refuted_tests). La cache es por-skill, thread-safe y best-effort: cualquier
fallo de I/O o de locking degrada a "miss" sin bloquear la evolucion.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_TTL_DAYS = 30
MANIFEST_VERSION = 2
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


@dataclass
class TestCase:
    """Un test ejecutable: check(output, files) -> {pass, reason, category}."""

    test_id: str                       # p.ej. "T1_check_exit_code"
    hypothesis_id: str                 # p.ej. "H1"
    hypothesis_description: str
    observable_behavior: str
    code: str                          # cuerpo del .py cacheado
    category: str = "hard"             # "hard" | "semantic"
    validated: bool = False
    skill_version_hash: str = ""
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    stale: bool = False

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    def cache_key(self, skill_hash: str) -> str:
        """Clave de cache = hash(hypothesis_description + skill_version_hash + observable_behavior)."""
        payload = f"{self.hypothesis_description}|{skill_hash}|{self.observable_behavior}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(hypothesis_desc: str, skill_hash: str, observable_behavior: str) -> str:
        payload = f"{hypothesis_desc}|{skill_hash}|{observable_behavior}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TestCacheError(Exception):
    """Fallo best-effort del cache; nunca debe abortar una evolucion."""


class TestCache:
    """Cache de tests de un skill conexionado por advisory lock (fcntl.flock)."""

    def __init__(self, category: str, skill_name: str, base_dir: Optional[Path] = None):
        root = (base_dir or HERMES_HOME) / "skills" / category / skill_name / ".evolution_cache"
        self.tests_dir = root / "tests"
        self.lock_path = root / ".test_cache.lock"
        self.manifest_path = self.tests_dir / "manifest.json"
        self.archive_path = self.tests_dir / "tests_archive.json"
        self._hits = 0
        self._misses = 0

    # ---- telemetria ----
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    # ---- locking advisory ----
    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # ---- persistencia manifest/archive ----
    def _load_manifest(self) -> dict:
        try:
            with open(self.manifest_path) as f:
                data = json.load(f)
            data.setdefault("tests", {})
            data["version"] = MANIFEST_VERSION
            return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": MANIFEST_VERSION, "tests": {}}

    def _save_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def _load_archive(self) -> dict:
        try:
            with open(self.archive_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_archive(self, archive: dict) -> None:
        tmp = self.archive_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(archive, f, indent=2)
        os.replace(tmp, self.archive_path)

    def _load_test_code(self, test_id: str, meta: dict) -> Optional[TestCase]:
        try:
            code = (self.tests_dir / f"{test_id}.py").read_text()
        except OSError:
            return None
        return TestCase(
            test_id=test_id,
            hypothesis_id=meta.get("hypothesis_id", ""),
            hypothesis_description=meta.get("hypothesis_description", ""),
            observable_behavior=meta.get("observable_behavior", ""),
            code=code,
            category=meta.get("category", "hard"),
            validated=meta.get("validated", False),
            skill_version_hash=meta.get("skill_version_hash", ""),
            created_at=meta.get("created_at", 0.0),
            last_used_at=meta.get("last_used_at", 0.0),
            use_count=meta.get("use_count", 0),
            stale=meta.get("stale", False),
        )

    # ---- API publica ----
    def get_cached_test(self, hypothesis_desc: str, skill_hash: str,
                        observable_behavior: str) -> Optional[TestCase]:
        """Reusa un test valido y no-stale; incrementa hit-rate. Best-effort."""
        target = TestCase._key(hypothesis_desc, skill_hash, observable_behavior)
        try:
            with self._locked():
                manifest = self._load_manifest()
                for tid, meta in manifest["tests"].items():
                    if meta.get("cache_key") != target or meta.get("stale"):
                        continue
                    test = self._load_test_code(tid, meta)
                    if test is None:
                        break
                    meta["last_used_at"] = time.time()
                    meta["use_count"] = meta.get("use_count", 0) + 1
                    self._save_manifest(manifest)
                    self._hits += 1
                    return test
        except OSError as exc:  # lock / I/O: degrade a miss
            raise TestCacheError(str(exc)) from exc
        self._misses += 1
        return None

    def store_test(self, test: TestCase, skill_hash: str) -> None:
        """Persiste el test (manifest + T<n>_<slug>.py). Best-effort."""
        test.skill_version_hash = skill_hash
        key = test.cache_key(skill_hash)
        try:
            with self._locked():
                self.tests_dir.mkdir(parents=True, exist_ok=True)
                manifest = self._load_manifest()
                manifest["tests"][test.test_id] = {
                    "hypothesis_id": test.hypothesis_id,
                    "hypothesis_description": test.hypothesis_description,
                    "skill_version_hash": skill_hash,
                    "observable_behavior": test.observable_behavior,
                    "code_hash": test.code_hash,
                    "category": test.category,
                    "validated": test.validated,
                    "created_at": test.created_at,
                    "last_used_at": test.last_used_at,
                    "use_count": test.use_count,
                    "stale": False,
                    "cache_key": key,
                }
                self._save_manifest(manifest)
                code_path = self.tests_dir / f"{test.test_id}.py"
                tmp = code_path.with_suffix(".py.tmp")
                with open(tmp, "w") as f:
                    f.write(test.code)
                os.replace(tmp, code_path)
        except OSError:
            pass  # best-effort: la evolucion continua sin cache

    def invalidate_stale_tests(self, current_skill_hash: str) -> list[str]:
        """Marca stale (no borra) los tests de versiones anteriores del skill."""
        invalidated: list[str] = []
        try:
            with self._locked():
                manifest = self._load_manifest()
                for tid, meta in manifest["tests"].items():
                    if meta.get("skill_version_hash") != current_skill_hash and not meta.get("stale"):
                        meta["stale"] = True
                        invalidated.append(tid)
                if invalidated:
                    self._save_manifest(manifest)
        except OSError:
            pass
        return invalidated

    def prune_expired_tests(self, ttl_days: int = DEFAULT_TTL_DAYS) -> int:
        """Elimina tests con created_at mas viejo que ttl_days. Retorna count."""
        cutoff = time.time() - ttl_days * 86400
        removed = 0
        try:
            with self._locked():
                manifest = self._load_manifest()
                to_drop = [tid for tid, m in manifest["tests"].items()
                           if m.get("created_at", 0) < cutoff]
                for tid in to_drop:
                    manifest["tests"].pop(tid, None)
                    try:
                        (self.tests_dir / f"{tid}.py").unlink(missing_ok=True)
                    except OSError:
                        pass
                    removed += 1
                if to_drop:
                    self._save_manifest(manifest)
        except OSError:
            pass
        return removed

    def archive_refuted_tests(self, hypothesis_id: str) -> None:
        """Mueve al archive de auditoria los tests de una hipotesis refutada."""
        try:
            with self._locked():
                manifest = self._load_manifest()
                archive = self._load_archive()
                changed = False
                for tid, meta in list(manifest["tests"].items()):
                    if meta.get("hypothesis_id") != hypothesis_id:
                        continue
                    code_path = self.tests_dir / f"{tid}.py"
                    if code_path.exists():
                        meta["code"] = code_path.read_text(errors="replace")
                        try:
                            code_path.unlink()
                        except OSError:
                            pass
                    archive[tid] = meta
                    manifest["tests"].pop(tid, None)
                    changed = True
                if changed:
                    self._save_manifest(manifest)
                    self._save_archive(archive)
        except OSError:
            pass
