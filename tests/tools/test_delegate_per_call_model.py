"""Tests for per-call model/provider overrides in delegate_task.

Covers the resolution precedence (per-task > top-level > delegation.model >
parent inheritance), both accepted model forms (short alias and fully qualified
id / inference URI), the operator allowlist, and the fail-fast contract: an
invalid model must raise rather than silently fall back to the parent model.
"""
from __future__ import annotations

import pytest

from tools.delegate_tool import _resolve_per_call_model


class _FakeParent:
    """Minimal stand-in for the parent agent's attribute surface."""

    def __init__(self, provider="bedrock", model="parent-model"):
        self.provider = provider
        self.model = model


def test_qualified_model_passes_through_verbatim():
    """A Bedrock inference-profile ARN must not go through alias resolution."""
    arn = (
        "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/"
        "eu.anthropic.claude-opus-5"
    )
    creds = _resolve_per_call_model(arn, None, {}, _FakeParent())
    assert creds["model"] == arn
    assert creds["_requested_model"] == arn
    assert creds["_alias_used"] is None


def test_slash_qualified_model_passes_through_verbatim():
    """An aggregator-style 'vendor/model' id is treated as qualified."""
    creds = _resolve_per_call_model(
        "anthropic/claude-opus-4", None, {}, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-opus-4"
    assert creds["_alias_used"] is None


def test_empty_model_raises():
    with pytest.raises(ValueError, match="empty"):
        _resolve_per_call_model("   ", None, {}, _FakeParent())


def test_unknown_bare_word_raises_and_lists_aliases():
    """Unknown model must fail loudly, never fall back to the parent model."""
    with pytest.raises(ValueError) as exc:
        _resolve_per_call_model("definitely-not-a-model", None, {}, _FakeParent())
    msg = str(exc.value)
    assert "unknown model" in msg.lower()
    # The error is only actionable if it names valid options.
    assert "alias" in msg.lower()


def test_alias_resolution_uses_provider_catalog(monkeypatch):
    """A known alias resolves against the effective provider's catalog."""
    import hermes_cli.model_switch as ms

    def _fake_resolve_alias(raw, provider):
        assert raw == "opus"
        assert provider == "bedrock"
        return ("bedrock", "eu.anthropic.claude-opus-5", "opus")

    monkeypatch.setattr(ms, "resolve_alias", _fake_resolve_alias)
    creds = _resolve_per_call_model("opus", None, {}, _FakeParent())
    assert creds["model"] == "eu.anthropic.claude-opus-5"
    assert creds["_alias_used"] == "opus"


def test_known_alias_unavailable_on_provider_raises(monkeypatch):
    """A known alias with no catalog match must not degrade silently."""
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(ms, "resolve_alias", lambda raw, provider: None)
    with pytest.raises(ValueError) as exc:
        _resolve_per_call_model("opus", None, {}, _FakeParent())
    assert "no matching" in str(exc.value).lower()


def test_explicit_provider_wins_over_config(monkeypatch):
    """Per-call provider takes precedence for catalog lookup."""
    import hermes_cli.model_switch as ms
    import tools.delegate_tool as dt

    seen = {}

    def _fake_resolve_alias(raw, provider):
        seen["provider"] = provider
        return (provider, "some-model", raw)

    monkeypatch.setattr(ms, "resolve_alias", _fake_resolve_alias)
    # Stub the credential resolver: this test asserts which provider the alias
    # catalog is searched against, not that real provider credentials exist.
    monkeypatch.setattr(
        dt,
        "_resolve_delegation_credentials",
        lambda cfg, parent: {"model": cfg.get("model"), "provider": cfg.get("provider")},
    )
    _resolve_per_call_model(
        "opus", "openrouter", {"provider": "bedrock"}, _FakeParent()
    )
    assert seen["provider"] == "openrouter"


def test_allowlist_permits_listed_model():
    arn = "arn:aws:bedrock:eu-west-1:1:inference-profile/eu.anthropic.claude-opus-5"
    cfg = {"allowed_models": [arn]}
    creds = _resolve_per_call_model(arn, None, cfg, _FakeParent())
    assert creds["model"] == arn


def test_allowlist_rejects_unlisted_model():
    cfg = {"allowed_models": ["anthropic/claude-sonnet-4"]}
    with pytest.raises(ValueError) as exc:
        _resolve_per_call_model(
            "anthropic/claude-opus-4", None, cfg, _FakeParent()
        )
    msg = str(exc.value)
    assert "not permitted" in msg
    # Must tell the caller what IS allowed.
    assert "anthropic/claude-sonnet-4" in msg


def test_allowlist_accepts_string_scalar():
    """A single string (not a list) is a valid allowlist form."""
    cfg = {"allowed_models": "anthropic/claude-opus-4"}
    creds = _resolve_per_call_model(
        "anthropic/claude-opus-4", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-opus-4"


def test_allowlist_matches_on_alias_name(monkeypatch):
    """Allowlisting by alias works even though the resolved id differs."""
    import hermes_cli.model_switch as ms

    monkeypatch.setattr(
        ms,
        "resolve_alias",
        lambda raw, provider: ("bedrock", "eu.anthropic.claude-opus-5", "opus"),
    )
    cfg = {"allowed_models": ["opus"]}
    creds = _resolve_per_call_model("opus", None, cfg, _FakeParent())
    assert creds["model"] == "eu.anthropic.claude-opus-5"


def test_empty_allowlist_means_no_restriction():
    """An unset/empty allowlist keeps existing configs working unchanged."""
    for cfg in ({}, {"allowed_models": []}, {"allowed_models": None}):
        creds = _resolve_per_call_model(
            "anthropic/claude-opus-4", None, cfg, _FakeParent()
        )
        assert creds["model"] == "anthropic/claude-opus-4"


def test_override_preserves_other_delegation_fields():
    """Only model/provider are swapped; base_url and friends keep resolving."""
    cfg = {"base_url": "https://example.invalid/v1", "api_key": "k-test"}
    creds = _resolve_per_call_model(
        "anthropic/claude-opus-4", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-opus-4"
    assert creds["base_url"] == "https://example.invalid/v1"
    assert creds["api_key"] == "k-test"


def test_cfg_is_not_mutated():
    """The shared delegation config dict must not be modified in place."""
    cfg = {"model": "configured-model", "provider": "bedrock"}
    snapshot = dict(cfg)
    _resolve_per_call_model("anthropic/claude-opus-4", None, cfg, _FakeParent())
    assert cfg == snapshot


# --- Escalation gate: downgrade free, upgrade gated -------------------------


_TIERS = {"model_tiers": ["haiku", "sonnet", "opus"]}


def test_downgrade_is_always_permitted():
    """sonnet -> haiku needs no permission: cannot blow a budget."""
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4")
    creds = _resolve_per_call_model(
        "anthropic/claude-haiku-3", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-haiku-3"


def test_upgrade_is_blocked_without_allow_escalation():
    """sonnet -> opus must fail while allow_escalation is unset."""
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4")
    with pytest.raises(ValueError) as exc:
        _resolve_per_call_model(
            "anthropic/claude-opus-4", None, cfg, _FakeParent()
        )
    msg = str(exc.value)
    assert "escalation" in msg.lower()
    assert "sonnet -> opus" in msg
    # The error must name the exact opt-in switch.
    assert "allow_escalation" in msg


def test_upgrade_permitted_when_allow_escalation_true():
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4", allow_escalation=True)
    creds = _resolve_per_call_model(
        "anthropic/claude-opus-4", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-opus-4"


def test_same_tier_is_not_an_escalation():
    """Switching within a tier is neither up nor down."""
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4")
    creds = _resolve_per_call_model(
        "anthropic/claude-sonnet-4-5", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-sonnet-4-5"


def test_escalation_baseline_falls_back_to_parent_model():
    """With no delegation.model, the parent's model is the baseline."""
    cfg = dict(_TIERS)
    parent = _FakeParent(model="anthropic/claude-sonnet-4")
    with pytest.raises(ValueError, match="escalation"):
        _resolve_per_call_model("anthropic/claude-opus-4", None, cfg, parent)


def test_escalation_gate_matches_tier_inside_inference_uri():
    """A Bedrock ARN must be ranked by the tier substring it contains."""
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4")
    arn = (
        "arn:aws:bedrock:eu-west-1:1:inference-profile/"
        "eu.anthropic.claude-opus-5"
    )
    with pytest.raises(ValueError, match="escalation"):
        _resolve_per_call_model(arn, None, cfg, _FakeParent())


def test_no_tiers_configured_means_no_gate():
    """Without model_tiers there is no ladder — full backward compatibility."""
    cfg = {"model": "anthropic/claude-sonnet-4"}
    creds = _resolve_per_call_model(
        "anthropic/claude-opus-4", None, cfg, _FakeParent()
    )
    assert creds["model"] == "anthropic/claude-opus-4"


def test_unranked_model_is_not_gated():
    """A model outside the ladder cannot be compared, so it passes."""
    cfg = dict(_TIERS, model="anthropic/claude-sonnet-4")
    creds = _resolve_per_call_model(
        "mistral/mistral-large", None, cfg, _FakeParent()
    )
    assert creds["model"] == "mistral/mistral-large"


def test_allowlist_and_escalation_gate_compose():
    """Allowlist is checked first; escalation gate applies to what survives."""
    cfg = dict(
        _TIERS,
        model="anthropic/claude-sonnet-4",
        allowed_models=["anthropic/claude-opus-4"],
    )
    # Rejected by the allowlist, not the ladder.
    with pytest.raises(ValueError, match="not permitted"):
        _resolve_per_call_model(
            "anthropic/claude-haiku-3", None, cfg, _FakeParent()
        )
    # Allowlisted but still an escalation.
    with pytest.raises(ValueError, match="escalation"):
        _resolve_per_call_model(
            "anthropic/claude-opus-4", None, cfg, _FakeParent()
        )
