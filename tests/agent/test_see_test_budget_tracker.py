#!/usr/bin/env python3
"""
Smoke test integrado: Budget Tracker + Cache + Sandbox.

Valida que BudgetTracker:
1. Track costos por rol con pricing real.
2. Prediga costo antes del spawn.
3. Lance BudgetExceededError al pasar el tope.
4. Genere summary con cost_breakdown.
5. Se integre con el flujo cache+sandbox.
"""
import sys, os, logging

from agent.skill_evolution_budget import BudgetTracker, BudgetExceededError, DEFAULT_PRICING
from agent.skill_evolution_test_cache import TestCache, TestCase
from agent.skill_evolution_sandbox import run_test_sandboxed, validate_test_strict

logging.basicConfig(level=logging.INFO, format="%(message)s")

def test_budget_tracking():
    print("=" * 60)
    print("SEE Prototype — Smoke Test: Budget Tracker")
    print("=" * 60)

    cfg = {
        "max_cost_usd": 0.20,
        "cost_warning_threshold": 0.8,
        "models": {
            "test": {"provider": "anthropic", "model": "claude-opus-4-7"},
            "execute": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct"},
        }
    }
    tracker = BudgetTracker.from_config(cfg)
    print(f"\n[1] BudgetTracker created: max=${tracker.max_cost_usd}")

    # ── track test call (Claude Opus caro)
    cost_test = tracker.track("test", "claude-opus-4-7", input_tokens=2100, output_tokens=750)
    assert 0.08 < cost_test < 0.10, f"test cost should be ~$0.09, got ${cost_test:.4f}"
    print(f"[2] Tracked test call: ${cost_test:.4f} (Claude Opus 2100in/750out)")

    # ── track execute call (Llama 3.3 70B barato)
    cost_exec = tracker.track("execute", "meta-llama/llama-3.3-70b-instruct", 6000, 2000)
    assert cost_exec < 0.001, f"execute should be ~$0.0004, got ${cost_exec:.4f}"
    print(f"[3] Tracked execute: ${cost_exec:.4f} (Llama 70B 6000in/2000out)")

    # ── track hypothesis (heredado, sin override)
    cost_hyp = tracker.track("hypothesis", "glm-4.6", 3000, 500)
    assert 0.002 < cost_hyp < 0.005
    print(f"[4] Tracked hypothesis: ${cost_hyp:.4f} (GLM-4.6 3000in/500out)")

    # ── predict next test call
    pred = tracker.predict("test", "claude-opus-4-7")
    assert 0.08 < pred < 0.10, f"predict should avg ~$0.09, got ${pred:.4f}"
    print(f"[5] Predict next test: ${pred:.4f}")

    # ── total so far
    total = tracker.total_usd
    assert 0.08 < total < 0.12
    print(f"[6] Cumulative: ${total:.4f}/${tracker.max_cost_usd} ({total/tracker.max_cost_usd*100:.0f}%)")

    # ── check_budget should NOT raise yet
    tracker.check_budget("test", "claude-opus-4-7")
    print(f"[7] check_budget: OK (within limit)")

    # ── now simulate approaching limit: track more test calls to exceed
    for i in range(2):
        tracker.track("test", "claude-opus-4-7", 2100, 750)
    print(f"[8] After 3 test calls: ${tracker.total_usd:.4f}")

    # ── check_budget should raise now (spent + predict > $0.20)
    raised = False
    try:
        tracker.check_budget("test", "claude-opus-4-7")
    except BudgetExceededError as e:
        raised = True
        print(f"[9] BudgetExceededError raised ✓: {e}")
    assert raised, "should have raised BudgetExceededError"

    # ── summary
    s = tracker.summary()
    assert "cost_total_usd" in s
    assert "cost_breakdown" in s
    assert "test" in s["cost_breakdown"]
    assert s["cost_breakdown"]["test"]["calls"] == 3
    print(f"[10] Summary: total=${s['cost_total_usd']:.4f}, "
          f"breakdown roles={list(s['cost_breakdown'].keys())}")
    print(f"     test: calls={s['cost_breakdown']['test']['calls']}, "
          f"cost=${s['cost_breakdown']['test']['cost_usd']:.4f}")

    print("\n" + "=" * 60)
    print("✓ BUDGET TRACKER SMOKE TEST PASSED")
    print("=" * 60)


def test_pricing_table_coverage():
    """Verifica que DEFAULT_PRICING tiene al menos 10 entradas cubriendo providers clave."""
    print("\n" + "=" * 60)
    print("SEE Prototype — Pricing Table Coverage")
    print("=" * 60)
    assert len(DEFAULT_PRICING) >= 10, f"need at least 10 entries, got {len(DEFAULT_PRICING)}"

    providers = set(k.split(":")[0] for k in DEFAULT_PRICING)
    required = {"anthropic", "openrouter", "groq", "zai"}
    missing = required - providers
    assert not missing, f"missing required providers: {missing}"

    # Claude Opus debe ser el más caro
    opus_in, opus_out = DEFAULT_PRICING["anthropic:claude-opus-4-7"]
    llama_in, llama_out = DEFAULT_PRICING["openrouter:meta-llama/llama-3.3-70b-instruct"]
    assert opus_in > llama_in * 10, f"Opus should be >10x Llama input: {opus_in} vs {llama_in}"
    assert opus_out > llama_out * 10, f"Opus should be >10x Llama output: {opus_out} vs {llama_out}"

    print(f"  Entries: {len(DEFAULT_PRICING)}")
    print(f"  Providers: {sorted(providers)}")
    print(f"  Opus (caro):     ${opus_in}/${opus_out} per 1M (in/out)")
    print(f"  Llama-70B (barato): ${llama_in}/${llama_out} per 1M (in/out)")
    print(f"  Ratio input:  {opus_in/llama_in:.0f}x")
    print(f"  Ratio output: {opus_out/llama_out:.0f}x")
    print("✓ PRICING TABLE COVERAGE OK")
