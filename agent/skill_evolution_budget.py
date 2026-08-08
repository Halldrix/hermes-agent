"""Budget tracking activo para el Skill Evolution Engine (SEE).

Uso mínimo:

    tracker = BudgetTracker.from_config(cfg.get("evolution", {}))
    ...
    tracker.check_budget(role, model)          # antes del spawn -> puede raise
    text = child.run(prompt)
    u = getattr(child, "last_usage", None)
    tracker.track(role, model, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))

Hermes Agent v0.20.0 / Python 3.13.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("hermes.evolution.budget")

# --------------------------------------------------------------------------- #
# Pricing defaults (USD por 1M tokens, estimaciones 2026)
# Clave: "provider:model" en minúsculas. Valor: (input, output).
# --------------------------------------------------------------------------- #
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-4-7": (15.0, 75.0),
    "anthropic:claude-sonnet-4-5": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (0.80, 4.0),
    "openai:gpt-5": (1.25, 10.0),
    "openai:gpt-5-mini": (0.25, 2.0),
    "openrouter:meta-llama/llama-3.3-70b-instruct": (0.05, 0.05),
    "openrouter:qwen/qwen3-235b-a22b": (0.13, 0.60),
    "openrouter:deepseek/deepseek-v3": (0.25, 0.85),
    "google:gemini-2.5-pro": (1.25, 10.0),
    "google:gemini-2.5-flash": (0.30, 2.50),
    "groq:llama-3.3-70b-versatile": (0.59, 0.79),
    "groq:moonshotai/kimi-k2-instruct": (1.0, 3.0),
    "zai:glm-4.6": (0.60, 2.20),
    "zai:glm-4.5-air": (0.20, 1.10),
    "nous:hermes-4-405b": (1.0, 3.0),
    "nous:hermes-4-70b": (0.30, 0.80),
}

# Defaults conservadores por rol para la primera predicción (in, out).
ROLE_TOKEN_DEFAULTS: dict[str, tuple[int, int]] = {
    "hypothesis": (3000, 500),
    "test": (2000, 800),
    "patch": (4000, 1000),
    "execute": (6000, 2000),
    "decide": (2000, 200),
}
_FALLBACK_TOKENS = (3000, 800)


class BudgetExceededError(RuntimeError):
    """El costo proyectado supera evolution.max_cost_usd."""

    def __init__(self, role: str, model: str, spent: float, projected: float, limit: float):
        self.role, self.model = role, model
        self.spent, self.projected, self.limit = spent, projected, limit
        super().__init__(
            f"Budget exceeded before role={role} model={model}: "
            f"spent=${spent:.4f} + estimated=${projected - spent:.4f} "
            f"= ${projected:.4f} > limit=${limit:.4f}"
        )


@dataclass
class _RoleStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BudgetTracker:
    max_cost_usd: float | None = None
    warning_threshold: float = 0.8
    # {role: {"provider":..,"model":..,"cost_per_1m_input":..,"cost_per_1m_output":..}}
    model_config: dict[str, dict[str, Any]] = field(default_factory=dict)

    cost_accumulator: dict[str, _RoleStats] = field(default_factory=dict)
    _warned_missing: set[str] = field(default_factory=set)
    _warned_threshold: bool = False

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, evolution_cfg: dict[str, Any]) -> "BudgetTracker":
        return cls(
            max_cost_usd=evolution_cfg.get("max_cost_usd"),
            warning_threshold=float(evolution_cfg.get("cost_warning_threshold", 0.8)),
            model_config=dict(evolution_cfg.get("models") or {}),
        )

    # ---------------------------- pricing ----------------------------- #
    def _pricing(self, role: str, model: str) -> tuple[float, float]:
        """(usd_per_1m_input, usd_per_1m_output) para role/model."""
        rc = self.model_config.get(role, {}) or {}
        cin, cout = rc.get("cost_per_1m_input"), rc.get("cost_per_1m_output")
        if cin is not None and cout is not None:
            return float(cin), float(cout)

        provider = (rc.get("provider") or "").lower()
        model_l = (model or rc.get("model") or "").lower()
        for key in (f"{provider}:{model_l}", model_l):
            if key in DEFAULT_PRICING:
                return DEFAULT_PRICING[key]
        # match por sufijo (model sin prefijo de provider)
        for key, val in DEFAULT_PRICING.items():
            if model_l and key.endswith(":" + model_l):
                return val

        tag = f"{provider}:{model_l}"
        if tag not in self._warned_missing:
            self._warned_missing.add(tag)
            log.warning("[evolve] no pricing for %s — assuming $0.00 (free)", tag)
        return 0.0, 0.0

    @staticmethod
    def _cost(inp: int, out: int, cin: float, cout: float) -> float:
        return (inp / 1_000_000.0) * cin + (out / 1_000_000.0) * cout

    # ---------------------------- API --------------------------------- #
    @property
    def total_usd(self) -> float:
        return sum(s.cost_usd for s in self.cost_accumulator.values())

    def track(self, role: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Registra el uso real de una llamada. Devuelve el costo de esa llamada."""
        input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
        cin, cout = self._pricing(role, model)
        cost = self._cost(input_tokens, output_tokens, cin, cout)

        st = self.cost_accumulator.setdefault(role, _RoleStats())
        st.calls += 1
        st.input_tokens += input_tokens
        st.output_tokens += output_tokens
        st.cost_usd += cost

        total = self.total_usd
        if self.max_cost_usd:
            pct = total / self.max_cost_usd * 100
            budget_s = f"${total:.2f}/${self.max_cost_usd:.2f} ({pct:.0f}%)"
        else:
            budget_s = f"${total:.2f}/unlimited"
        log.info(
            "[evolve] role=%s model=%s in=%d out=%d cost=$%.3f cumulative=%s",
            role, model, input_tokens, output_tokens, cost, budget_s,
        )

        if (
            self.max_cost_usd
            and not self._warned_threshold
            and total >= self.max_cost_usd * self.warning_threshold
        ):
            self._warned_threshold = True
            log.warning(
                "[evolve] cost warning: $%.2f of $%.2f budget consumed (>=%.0f%%)",
                total, self.max_cost_usd, self.warning_threshold * 100,
            )
        return cost

    def predict(self, role: str, model: str) -> float:
        """Costo estimado de la próxima llamada del rol (promedios de sesión)."""
        st = self.cost_accumulator.get(role)
        if st and st.calls:
            inp = st.input_tokens / st.calls
            out = st.output_tokens / st.calls
        else:
            inp, out = ROLE_TOKEN_DEFAULTS.get(role, _FALLBACK_TOKENS)
        cin, cout = self._pricing(role, model)
        return self._cost(int(inp), int(out), cin, cout)

    def check_budget(self, role: str = "?", model: str = "?") -> None:
        """Raise BudgetExceededError si spent + predict(role) supera el tope."""
        if not self.max_cost_usd:
            return
        spent = self.total_usd
        projected = spent + self.predict(role, model)
        if projected > self.max_cost_usd:
            raise BudgetExceededError(role, model, spent, projected, float(self.max_cost_usd))

    def summary(self) -> dict[str, Any]:
        return {
            "cost_total_usd": round(self.total_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "cost_breakdown": {
                role: {
                    "calls": s.calls,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for role, s in self.cost_accumulator.items()
            },
        }


# --------------------------------------------------------------------------- #
# INTEGRACIÓN
# --------------------------------------------------------------------------- #
# 1) En _delegate_role(self, role, prompt, ...):
#
#       model = self._model_for_role(role)          # de evolution.models[role]
#       self.budget.check_budget(role, model)       # <-- ANTES del spawn (raise)
#       child = AIAgent(model=model, ...)
#       text = child.run(prompt)
#       u = getattr(child, "last_usage", None)      # o callback on_usage
#       self.budget.track(role, model,
#                         getattr(u, "input_tokens", 0) if u else 0,
#                         getattr(u, "output_tokens", 0) if u else 0)
#       return text
#
# 2) En evolve_skill(...):
#
#       self.budget = BudgetTracker.from_config(cfg.get("evolution", {}))
#       budget_exceeded = False
#       try:
#           ... bucle PUCT ...
#       except BudgetExceededError as e:
#           log.warning("[evolve] aborting: %s", e)
#           budget_exceeded = True
#       best = self._best_node()                    # mejor nodo hasta ahora
#       return {**self._result(best),
#               "budget_exceeded": budget_exceeded,
#               **self.budget.summary()}            # cost_total_usd + cost_breakdown
#
# Nota: si AIAgent no expone last_usage, registrar un callback
# `child.on_usage = lambda u: self.budget.track(role, model, u.input_tokens, u.output_tokens)`
# y omitir el track manual para no contar doble.
