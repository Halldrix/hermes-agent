"""Active budget tracking for the Skill Evolution Engine (SEE).

Minimal usage:

    tracker = BudgetTracker.from_config(cfg.get("evolution", {}))
    ...
    tracker.check_budget(role, model)          # before spawn -> may raise
    text = child.run(prompt)
    u = getattr(child, "last_usage", None)
    tracker.track(role, model, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))

Hermes Agent v0.20.0 / Python 3.13.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("hermes.evolution.budget")

# --------------------------------------------------------------------------- #
# Pricing is resolved via the central agent.usage_pricing module, which
# looks up provider catalogues, OpenRouter /models, and official-docs
# snapshots — the same source Hermes uses for session cost tracking.
# Subscription-included routes (Nous-managed, etc.) resolve to $0.00
# automatically.  Unknown models also fall back to $0.00 so evolution
# never aborts on a model the pricing service hasn't indexed yet.
# --------------------------------------------------------------------------- #

# Conservative per-role defaults for the first prediction (in, out).
ROLE_TOKEN_DEFAULTS: dict[str, tuple[int, int]] = {
    "hypothesis": (3000, 500),
    "test": (2000, 800),
    "patch": (4000, 1000),
    "execute": (6000, 2000),
    "decide": (2000, 200),
}
_FALLBACK_TOKENS = (3000, 800)


class BudgetExceededError(RuntimeError):
    """Projected cost exceeds evolution.max_cost_usd."""

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
    def _pricing(
        self,
        role: str,
        model: str,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> tuple[float, float]:
        """Resolve (usd_per_1m_input, usd_per_1m_output) for role/model.

        Delegates to agent.usage_pricing.get_pricing_entry(), which consults
        provider catalogues, OpenRouter /models, and official-docs snapshots
        — the same source Hermes uses for session cost tracking.

        Per-role ``cost_per_1m_input`` / ``cost_per_1m_output`` overrides in
        ``evolution.models.<role>`` take priority, then the pricing service.
        Unknown models fall back to $0.00 so evolution never aborts on a
        model the pricing service hasn't indexed yet.
        """
        rc = self.model_config.get(role, {}) or {}
        cin, cout = rc.get("cost_per_1m_input"), rc.get("cost_per_1m_output")
        if cin is not None and cout is not None:
            return float(cin), float(cout)

        provider = provider or rc.get("provider")
        base_url = base_url or rc.get("base_url")
        model_name = model or rc.get("model") or ""

        try:
            from agent.usage_pricing import get_pricing_entry

            entry = get_pricing_entry(
                model_name,
                provider=provider,
                base_url=base_url,
            )
            if entry:
                return (
                    float(entry.input_cost_per_million or 0),
                    float(entry.output_cost_per_million or 0),
                )
        except Exception:
            pass  # pricing service unavailable — fall through to $0.00

        tag = f"{provider or '?'}:{model_name}"
        if tag not in self._warned_missing:
            self._warned_missing.add(tag)
            log.warning(
                "[evolve] no pricing for %s via usage_pricing — assuming $0.00", tag
            )
        return 0.0, 0.0

    @staticmethod
    def _cost(inp: int, out: int, cin: float, cout: float) -> float:
        return (inp / 1_000_000.0) * cin + (out / 1_000_000.0) * cout

    # ---------------------------- API --------------------------------- #
    @property
    def total_usd(self) -> float:
        return sum(s.cost_usd for s in self.cost_accumulator.values())

    def track(
        self,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> float:
        """Record actual usage of a call. Returns the cost of that call."""
        input_tokens, output_tokens = int(input_tokens or 0), int(output_tokens or 0)
        cin, cout = self._pricing(role, model, provider=provider, base_url=base_url)
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

    def predict(
        self,
        role: str,
        model: str,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> float:
        """Estimated cost of the next role call (session averages)."""
        st = self.cost_accumulator.get(role)
        if st and st.calls:
            inp = st.input_tokens / st.calls
            out = st.output_tokens / st.calls
        else:
            inp, out = ROLE_TOKEN_DEFAULTS.get(role, _FALLBACK_TOKENS)
        cin, cout = self._pricing(role, model, provider=provider, base_url=base_url)
        return self._cost(int(inp), int(out), cin, cout)

    def check_budget(
        self,
        role: str = "?",
        model: str = "?",
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Raise BudgetExceededError if spent + predict(role) exceeds the limit."""
        if not self.max_cost_usd:
            return
        spent = self.total_usd
        projected = spent + self.predict(role, model, provider=provider, base_url=base_url)
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
# INTEGRATION
# --------------------------------------------------------------------------- #
# 1) In _delegate_role(self, role, prompt, ...):
#
#       model = self._model_for_role(role)          # from evolution.models[role]
#       self.budget.check_budget(role, model)       # <-- BEFORE spawn (raise)
#       child = AIAgent(model=model, ...)
#       text = child.run(prompt)
#       u = getattr(child, "last_usage", None)      # or callback on_usage
#       self.budget.track(role, model,
#                         getattr(u, "input_tokens", 0) if u else 0,
#                         getattr(u, "output_tokens", 0) if u else 0)
#       return text
#
# 2) In evolve_skill(...):
#
#       self.budget = BudgetTracker.from_config(cfg.get("evolution", {}))
#       budget_exceeded = False
#       try:
#           ... PUCT loop ...
#       except BudgetExceededError as e:
#           log.warning("[evolve] aborting: %s", e)
#           budget_exceeded = True
#       best = self._best_node()                    # best node so far
#       return {**self._result(best),
#               "budget_exceeded": budget_exceeded,
#               **self.budget.summary()}            # cost_total_usd + cost_breakdown
#
# Note: if AIAgent does not expose last_usage, register a callback
# `child.on_usage = lambda u: self.budget.track(role, model, u.input_tokens, u.output_tokens)`
# and omit manual tracking to avoid double counting.
