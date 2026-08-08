"""
SEE Prototype — Skill Evolution Engine: Test Sandbox

Sandbox restringido para ejecutar tests generados por el rol "test" del
reflection agent. Los tests son funciones Python `def check(output, files)
-> dict` que se ejecutan contra outputs cacheados del agente.

El sandbox:
1. Compile the code (syntax check) — ImportError/SyntaxError caught.
2. Ejecuta en namespace restringido con solo json/re/os.path/yaml.
3. Detecta tests triviales (always-true) que contaminan la matriz.
4. Forbid dangerous imports: subprocess, socket, urllib, requests, open(),
   __import__, eval(), exec(), compile(), globals(), locals(), getattr con
   dynamic string.

Uso principal:
    result = run_test_sandboxed(code, output, files)
    if result["pass"]:
        # M[v, test_id] = 1
    else:
        # M[v, test_id] = 0

El sandbox NO usa subprocess para mantenerse ligero. Ejecuta en el mismo
proceso con globals restringidos — los imports prohibidos se detectan por
source code inspection before exec(). It is not secure against
adversarios dedicados, pero los tests los genera un LLM configurado por
the user, not hostile code.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Imports allowed in the sandbox ──────────────────────────────────
_ALLOWED_MODULES = {"json", "re", "os", "os.path", "yaml", "io"}

# ── Forbidden tokens (source code inspection) ─────────────────
_FORBIDDEN_TOKENS = [
    "subprocess", "socket", "urllib", "requests", "http.client",
    "ftplib", "smtplib", "telnetlib", "webbrowser",
    "__import__", "eval(", "exec(", "compile(",
    "globals(", "locals(", "vars(", "dir(",
    "getattr((",  # getattr with dynamic string — getattr(obj, "x") is allowed
    "ctypes", "cffi", "pickle", "marshal",
    "open(",  # no filesystem access — files come via `files`
    "input(", "breakpoint(", "exit(", "quit(",
    "__builtins__", "__import__",
]

# Conservative estimate: a test needs no more than 4KB of code.
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
# 1. Static validation (before exec)
# ──────────────────────────────────────────────────────────────────────

def _code_hash(code: str) -> str:
    import hashlib
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _forbidden_token_found(code: str) -> Optional[str]:
    """Inspect source code to detect forbidden tokens.
    Returns the first forbidden token found, or None if the code is clean.
    """
    # Normalize: remove comments and strings to avoid false positives
    # we check raw code — expensive models shouldn't generate malicious payloads,
    # but this is a safety belt.
    for token in _FORBIDDEN_TOKENS:
        if token in code:
            return token
    return None


def _has_required_signature(code: str) -> bool:
    """Verify that the code defines `def check(output, files) -> dict` or similar."""
    # Aceptar: def check(output, files) -> dict:  |  def check(output: str, files: dict) -> dict:
    pattern = r"def\s+check\s*\(\s*output[^)]*files[^)]*\)\s*(?:->\s*\S+)?\s*:"
    return bool(re.search(pattern, code))


def _returns_dict_hint(code: str) -> bool:
    """Detect whether the code returns a dict with a bool 'pass'.
    Heuristic: looks for `return {"pass": ...}` or `return {\"pass\": ...` in any form.
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

    # 4. Must return dict with 'pass' (heuristic)
    if not _returns_dict_hint(code):
        return ValidationResult(
            False, "check() must return dict with 'pass' key"
        )

    return ValidationResult(True, "valid", _code_hash(code))


# ──────────────────────────────────────────────────────────────────────
# 2. Reinforced validation: detect trivial (always-true) tests
# ──────────────────────────────────────────────────────────────────────

def _exec_test_sandboxed(code: str, stdout: str, files: dict) -> dict:
    """Ejecuta el test en namespace restringido. Retorna el dict de check()."""
    import json as _json
    import re as _re
    import os as _os
    import os.path as _ospath

    # Controlled __import__: only whitelist modules allowed.
    # Test code may `import json` or `import re`, but not
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
    """Reinforced validation: validates the test AND detects trivial patterns.
    Esto es lo que atrapa tests que pasan el syntax check pero son
    semantically useless (always-true, too permissive).
    """
    # 1. Basic validation
    v = validate_test(code)
    if not v.valid:
        return False, v.reason

    # 2. Dry-run on three contrasting scenarios
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

    # 3. If all three scenarios pass, the test is trivially permissive
    if all(p for _, p in results):
        return False, "test passes on empty, garbage, and plausible input — trivially permissive"

    # 4. We don't mark "trivially strict" with only 3 generic scenarios.
    # A test seeking a specific pattern ("gh: command not found") will
    # fail on all 3 arbitrary scenarios without being trivial. If the test
    # executes without exception and returns a valid dict, it's valid.
    # The 3 scenarios are only used to detect always-true (rule 3).

    # 5. If empty and garbage pass but plausible fails, suspicious
    if results[0][1] and results[1][1] and not results[2][1]:
        logger.warning("validate_test_strict: test passes on junk but fails on plausible — review needed")

    return True, "valid (strict)"


# ──────────────────────────────────────────────────────────────────────
# 3. Execution with timeout and error handling
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
    
    # Re-validate before execution (defense in depth)
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
