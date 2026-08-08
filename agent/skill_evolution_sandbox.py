"""
SEE Prototype — Skill Evolution Engine: Test Sandbox

Sandbox restringido para ejecutar tests generados por el rol "test" del
reflection agent. Los tests son funciones Python `def check(output, files)
-> dict` que se ejecutan contra outputs cacheados del agente.

El sandbox:
1. Compila el código (syntax check) — ImportError/SyntaxError capturados.
2. Ejecuta en namespace restringido con solo json/re/os.path/yaml.
3. Detecta tests triviales (always-true) que contaminan la matriz.
4. Prohíbe imports peligrosos: subprocess, socket, urllib, requests, open(),
   __import__, eval(), exec(), compile(), globals(), locals(), getattr con
   string dinámico.

Uso principal:
    result = run_test_sandboxed(code, output, files)
    if result["pass"]:
        # M[v, test_id] = 1
    else:
        # M[v, test_id] = 0

El sandbox NO usa subprocess para mantenerse ligero. Ejecuta en el mismo
proceso con globals restringidos — los imports prohibidos se detectan por
inspección del código fuente antes de exec(). No es seguro contra
adversarios dedicados, pero los tests los genera un LLM configurado por
el usuario, no código hostil.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Imports permitidos en el sandbox ──────────────────────────────────
_ALLOWED_MODULES = {"json", "re", "os", "os.path", "yaml", "io"}

# ── Tokens prohibidos (inspección del código fuente) ─────────────────
_FORBIDDEN_TOKENS = [
    "subprocess", "socket", "urllib", "requests", "http.client",
    "ftplib", "smtplib", "telnetlib", "webbrowser",
    "__import__", "eval(", "exec(", "compile(",
    "globals(", "locals(", "vars(", "dir(",
    "getattr((",  # getattr con string dinámico — se permite getattr(obj, "x")
    "ctypes", "cffi", "pickle", "marshal",
    "open(",  # sin acceso al filesystem — los archivos vienen vía `files`
    "input(", "breakpoint(", "exit(", "quit(",
    "__builtins__", "__import__",
]

# Estimación conservadora: un test no necesita más de 4KB de código.
MAX_TEST_CODE_CHARS = 4000


@dataclass
class TestResult:
    """Resultado de ejecutar un test en el sandbox."""
    passed: bool
    reason: str = ""
    category: str = "semantic"
    error: Optional[str] = None
    runtime_s: float = 0.0


@dataclass
class ValidationResult:
    """Resultado de validar un test antes de ejecutarlo."""
    valid: bool
    reason: str = ""
    code_hash: str = ""


# ──────────────────────────────────────────────────────────────────────
# 1. Validación estática (antes de exec)
# ──────────────────────────────────────────────────────────────────────

def _code_hash(code: str) -> str:
    import hashlib
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _forbidden_token_found(code: str) -> Optional[str]:
    """Inspecciona el código fuente para detectar tokens prohibidos.
    Retorna el primer token prohibido encontrado, o None si el código está limpio.
    """
    # Normalizar: remover comentarios y strings para falsos positivos
    # checamos el código crudo — los modelos caros no deben generar payloads
    # maliciosos, pero esto es cinturón de seguridad.
    for token in _FORBIDDEN_TOKENS:
        if token in code:
            return token
    return None


def _has_required_signature(code: str) -> bool:
    """Verifica que el código define `def check(output, files) -> dict` o similar."""
    # Aceptar: def check(output, files) -> dict:  |  def check(output: str, files: dict) -> dict:
    pattern = r"def\s+check\s*\(\s*output[^)]*files[^)]*\)\s*(?:->\s*\S+)?\s*:"
    return bool(re.search(pattern, code))


def _returns_dict_hint(code: str) -> bool:
    """Detecta si el código retorna un dict con 'pass' bool.
    Heurística: busca `return {"pass": ...}` o `return {\"pass\": ...` en cualquier forma.
    """
    pattern = r'return\s*\{[^}]*["\']pass["\']'
    return bool(re.search(pattern, code, re.DOTALL))


def validate_test(code: str) -> ValidationResult:
    """Valida un test antes de incorporarlo a la matriz de evidencia.
    Retorna ValidationResult(valid, reason, code_hash).
    """
    if not code or not code.strip():
        return ValidationResult(False, "empty code")

    if len(code) > MAX_TEST_CODE_CHARS:
        return ValidationResult(
            False, f"code exceeds {MAX_TEST_CODE_CHARS} chars"
        )

    # 1. Syntax check
    try:
        compile(code, "<test>", "exec")
    except SyntaxError as e:
        return ValidationResult(False, f"syntax error: {e}")

    # 2. Tokens prohibidos
    bad_token = _forbidden_token_found(code)
    if bad_token:
        return ValidationResult(False, f"forbidden token: {bad_token}")

    # 3. Debe definir check()
    if not _has_required_signature(code):
        return ValidationResult(
            False, "no `def check(output, files) -> dict` found"
        )

    # 4. Debe retornar dict con 'pass' (heurística)
    if not _returns_dict_hint(code):
        return ValidationResult(
            False, "check() must return dict with 'pass' key"
        )

    return ValidationResult(True, "valid", _code_hash(code))


# ──────────────────────────────────────────────────────────────────────
# 2. Validación reforzada: detectar tests triviales (always-true)
# ──────────────────────────────────────────────────────────────────────

def _exec_test_sandboxed(code: str, stdout: str, files: dict) -> dict:
    """Ejecuta el test en namespace restringido. Retorna el dict de check()."""
    import json as _json
    import re as _re
    import os as _os
    import os.path as _ospath

    # __import__ controlado: solo permite módulos del whitelist.
    # El código del test puede hacer `import json` o `import re`, pero no
    # `import subprocess` o `import socket`.
    _ALLOWED = {"json": _json, "re": _re, "os": _os, "os.path": _ospath,
                "yaml": None}
    try:
        import yaml as _yaml
        _ALLOWED["yaml"] = _yaml
    except ImportError:
        pass

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("relative imports not allowed")
        if name not in _ALLOWED:
            raise ImportError(f"module '{name}' not allowed in sandbox")
        return _ALLOWED[name]

    safe_builtins = {
        "__import__": _safe_import,
        "json": _json,
        "re": _re,
        "os": _os,
        "os.path": _ospath,
        "len": len, "str": str, "int": int, "float": float,
        "bool": bool, "list": list, "dict": dict, "set": set,
        "tuple": tuple, "range": range, "enumerate": enumerate,
        "isinstance": isinstance, "hasattr": hasattr,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "sorted": sorted, "reversed": reversed,
        "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
        "True": True, "False": False, "None": None,
        "print": lambda *a, **k: None,  # silenciar prints
    }
    # yaml opcional
    try:
        import yaml as _yaml
        safe_builtins["yaml"] = _yaml
    except ImportError:
        pass

    safe_globals = {"__builtins__": safe_builtins}
    local_ns: dict[str, Any] = {}

    exec(compile(code, "<test>", "exec"), safe_globals, local_ns)
    check_fn = local_ns.get("check")
    if not check_fn or not callable(check_fn):
        return {"pass": False, "reason": "check() not defined or not callable"}

    # Normalizar files: solo strings
    files_dict = {
        str(k): (v if isinstance(v, str) else str(v))
        for k, v in (files or {}).items()
    }
    result = check_fn(output=stdout or "", files=files_dict)
    if not isinstance(result, dict):
        return {"pass": False, "reason": "check() did not return dict"}
    if "pass" not in result:
        return {"pass": False, "reason": "check() returned dict without 'pass'"}
    return result


def validate_test_strict(code: str) -> tuple[bool, str]:
    """Validación reforzada: valida el test Y detecta patrones triviales.
    Esto es lo que atrapa tests que pasan el syntax check pero son
    semánticamente inútiles (always-true, demasiado permisivos).
    """
    # 1. Validación básica
    v = validate_test(code)
    if not v.valid:
        return False, v.reason

    # 2. Dry-run en tres escenarios contrastantes
    scenarios = [
        ("empty", "", {}),
        ("garbage", "GARBAGE_TEST_INPUT_12345_xqz", {}),
        ("plausible", "PR #42 created successfully\nfiles: 3", {"output.txt": "sample"}),
    ]
    results = []
    for name, stdout, files in scenarios:
        try:
            r = _exec_test_sandboxed(code, stdout, files)
            results.append((name, r.get("pass", False)))
        except Exception as e:
            return False, f"dry-run failed on {name}: {e}"

    # 3. Si los tres escenarios pasan, el test es trivially permissive
    if all(p for _, p in results):
        return False, "test passes on empty, garbage, and plausible input — trivially permissive"

    # 4. No marcamos "trivially strict" con solo 3 escenarios genéricos.
    # Un test que busca un patrón específico ("gh: command not found") va a
    # fallar en los 3 escenarios arbitrarios sin ser trivial. Si el test
    # ejecuta sin excepción y retorna dict válido, es válido.
    # Los 3 escenarios solo se usan para detectar always-true (regla 3).

    # 5. Si empty y garbage pasan pero plausible falla, sospechoso
    if results[0][1] and results[1][1] and not results[2][1]:
        logger.warning("validate_test_strict: test passes on junk but fails on plausible — review needed")

    return True, "valid (strict)"


# ──────────────────────────────────────────────────────────────────────
# 3. Ejecución con timeout y manejo de errores
# ──────────────────────────────────────────────────────────────────────

import time
import signal


class _TestTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TestTimeout("test exceeded time limit")


def run_test_sandboxed(
    code: str,
    stdout: str,
    files: dict,
    timeout_s: float = 5.0,
) -> TestResult:
    """Ejecuta un test validado contra el output cacheado.
    Retorna TestResult con passed, reason, category, error, runtime_s.
    """
    t0 = time.time()
    
    # Re-validar antes de ejecutar (defensa en profundidad)
    ok, reason = validate_test_strict(code)
    if not ok:
        return TestResult(
            passed=False, reason=reason, error="validation_failed", runtime_s=0.0
        )

    # Timeout via SIGALRM (Unix only)
    old_handler = None
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)

    try:
        result = _exec_test_sandboxed(code, stdout, files)
        elapsed = time.time() - t0
        return TestResult(
            passed=bool(result.get("pass", False)),
            reason=str(result.get("reason", "")),
            category=str(result.get("category", "semantic")),
            error=None,
            runtime_s=elapsed,
        )
    except _TestTimeout:
        return TestResult(
            passed=False, reason=f"timeout after {timeout_s}s",
            error="timeout", runtime_s=timeout_s,
        )
    except Exception as e:
        return TestResult(
            passed=False, reason=str(e)[:200], error="runtime_error",
            runtime_s=time.time() - t0,
        )
    finally:
        if has_alarm and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
