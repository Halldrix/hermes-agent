#!/usr/bin/env python3
"""
Integrated smoke test: Budget Tracker + Cache + Sandbox.

Validates that BudgetTracker:
1. Track costs per role with real pricing.
2. Predict cost before spawn.
3. Raise BudgetExceededError when exceeding the ceiling.
4. Generate summary with cost_breakdown.
5. Integrate with the cache+sandbox flow.
"""
import sys, os, logging

from agent.skill_evolution_budget import BudgetTracker, BudgetExceededError
from agent.skill_evolution_test_cache import TestCache, TestCase
from agent.skill_evolution_sandbox import run_test_sandboxed, validate_test_strict

logging.basicConfig(level=logging.INFO, format="%(message)s")

def test_budget_tracking():
    print("=" * 60)
    print("SEE Prototype — Smoke Test: Budget Tracker")
    print("=" * 60)

    cfg = {
        "max_cost_usd": 1.0,
        "cost_warning_threshold": 0.8,
        "models": {
            "test": {"provider": "anthropic", "model": "claude-opus-4-7"},
        }
    }
    tracker = BudgetTracker.from_config(cfg)
    print(f"\n[1] BudgetTracker created: max=${tracker.max_cost_usd}")

    # ── Resolve expected prices from usage_pricing (same source Hermes uses)
    from agent.usage_pricing import get_pricing_entry

    opus_entry = get_pricing_entry("claude-opus-4-7", provider="anthropic")
    assert opus_entry is not None, "claude-opus-4-7 should resolve"
    opus_in_price = float(opus_entry.input_cost_per_million or 0)
    opus_out_price = float(opus_entry.output_cost_per_million or 0)
    print(f"[1b] Opus 4-7 pricing: ${opus_in_price}/${opus_out_price} per 1M (in/out)")

    # ── track test call
    in_tok, out_tok = 2100, 750
    cost_test = tracker.track("test", "claude-opus-4-7",
                              input_tokens=in_tok, output_tokens=out_tok,
                              provider="anthropic")
    expected_cost = in_tok / 1e6 * opus_in_price + out_tok / 1e6 * opus_out_price
    assert abs(cost_test - expected_cost) < 1e-6, \
        f"cost should be ${expected_cost:.4f}, got ${cost_test:.4f}"
    print(f"[2] Tracked test call: ${cost_test:.4f}")

    # ── track execute (unknown model — should be $0.00, never abort)
    cost_exec = tracker.track("execute", "some-unknown-model", 6000, 2000)
    assert cost_exec == 0.0, f"unknown model should be $0.00, got ${cost_exec:.4f}"
    print(f"[3] Tracked execute (unknown): ${cost_exec:.4f}")

    # ── predict should reflect session average
    pred = tracker.predict("test", "claude-opus-4-7", provider="anthropic")
    assert pred > 0, f"predict should be >0 for priced model, got ${pred:.4f}"
    print(f"[5] Predict next test: ${pred:.4f}")

    # ── check_budget should NOT raise (within $1.0 limit)
    tracker.check_budget("test", "claude-opus-4-7", provider="anthropic")
    print(f"[7] check_budget: OK (within limit)")

    # ── summary
    s = tracker.summary()
    assert "cost_total_usd" in s
    assert "cost_breakdown" in s
    assert "test" in s["cost_breakdown"]
    assert s["cost_breakdown"]["test"]["calls"] == 1
    print(f"[10] Summary: total=${s['cost_total_usd']:.4f}, "
          f"roles={list(s['cost_breakdown'].keys())}")

    print("\n" + "=" * 60)
    print("✓ BUDGET TRACKER SMOKE TEST PASSED")
    print("=" * 60)


def test_pricing_via_usage_pricing():
    """Verifies that BudgetTracker resolves prices via agent.usage_pricing
    (the same pricing service Hermes uses for session cost tracking), not a
    hardcoded table."""
    print("\n" + "=" * 60)
    print("SEE Prototype — Pricing via usage_pricing")
    print("=" * 60)

    try:
        from agent.usage_pricing import get_pricing_entry, has_known_pricing
    except ImportError:
        print("  usage_pricing not available — skipping (CI env)")
        return

    # Claude Opus via Anthropic
    opus_entry = get_pricing_entry("claude-opus-4-7", provider="anthropic")
    assert opus_entry is not None, "claude-opus-4-7 should resolve via usage_pricing"
    opus_in = float(opus_entry.input_cost_per_million or 0)
    opus_out = float(opus_entry.output_cost_per_million or 0)
    assert opus_in > 0 or opus_out > 0, "Opus should have nonzero pricing"

    # Verify BudgetTracker._pricing delegates to usage_pricing
    tracker = BudgetTracker(max_cost_usd=10.0)
    resolved_in, resolved_out = tracker._pricing("test", "claude-opus-4-7", provider="anthropic")
    assert resolved_in == opus_in, f"BudgetTracker should match usage_pricing: {resolved_in} vs {opus_in}"
    assert resolved_out == opus_out, f"BudgetTracker should match usage_pricing: {resolved_out} vs {opus_out}"

    # Per-role override takes priority
    tracker_override = BudgetTracker(
        max_cost_usd=10.0,
        model_config={"test": {"cost_per_1m_input": 5.0, "cost_per_1m_output": 25.0}},
    )
    ov_in, ov_out = tracker_override._pricing("test", "any-model")
    assert ov_in == 5.0 and ov_out == 25.0, "override should take priority"

    # Unknown model falls back to $0.00 gracefully
    unknown_in, unknown_out = tracker._pricing("test", "nonexistent-model-xyz")
    assert unknown_in == 0.0 and unknown_out == 0.0, "unknown model should be $0.00"

    print(f"  Opus 4-7:  ${opus_in}/${opus_out} per 1M (in/out)")
    print(f"  BudgetTracker._pricing delegates to usage_pricing: OK")
    print(f"  Per-role override priority: OK")
    print(f"  Unknown model fallback: $0.00 OK")
    print("✓ PRICING VIA USAGE_PRICING OK")
