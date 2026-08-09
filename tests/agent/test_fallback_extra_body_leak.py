"""Regression test: extra_body leak across providers on fallback.

When a primary provider has a per-provider ``extra_body`` (e.g.
``custom:nvidia`` with ``thinking.type=adaptive``) and a fallback is
activated to a different provider whose ``custom_providers`` entry has
NO ``extra_body`` (e.g. ``custom:modal`` hosting Kimi-K3 on SGLang),
the primary's ``extra_body`` must NOT leak into the fallback request.

SGLang rejects ``thinking.type=adaptive`` (only ``enabled``/``disabled``
are accepted) with HTTP 400 — ``BadRequestError``, which is non-retryable
and kills the session.

Root cause: ``_try_activate_fallback`` changed ``agent.model`` /
``agent.provider`` / ``agent.base_url`` but never cleared or re-merged
``agent.request_overrides['extra_body']``, so the transport
(``chat_completions.py:584``) injected the stale primary ``extra_body``
into every fallback request.

See: agent/agent_init.py::_merge_custom_provider_extra_body (re-entrant
merge with key tracking) and agent/chat_completion_helpers.py::
try_activate_fallback (clear + re-merge on swap).
"""

from types import SimpleNamespace

from agent.agent_init import _merge_custom_provider_extra_body


# ── Unit: _merge_custom_provider_extra_body is re-entrant ─────────────────


def test_re_merge_drops_previous_provider_keys():
    """Re-calling merge with a provider that has NO extra_body must purge
    keys injected by the *previous* provider's extra_body."""
    agent = SimpleNamespace(
        provider="custom:nvidia",
        model="z-ai/glm-5.2",
        base_url="https://integrate.api.nvidia.com/v1",
        request_overrides={},
    )
    custom_providers = [
        {
            "name": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "key_env": "NVIDIA_API_KEY",
            "extra_body": {"thinking": {"type": "adaptive"}},
            "models": {"z-ai/glm-5.2": {"context_length": 1048576}},
        },
        {
            "name": "modal",
            "base_url": "https://modal.example/v1",
            "key_env": "MODAL_API_KEY",
            # NOTE: no extra_body
            "models": {"moonshotai/Kimi-K3": None},
        },
    ]

    # Primary provider merge — injects thinking.type=adaptive
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides["extra_body"] == {"thinking": {"type": "adaptive"}}
    assert agent._custom_provider_extra_body_keys == ["thinking"]

    # Simulate fallback: agent switches to modal (no extra_body)
    agent.provider = "custom:modal"
    agent.model = "moonshotai/Kimi-K3"
    agent.base_url = "https://modal.example/v1"

    _merge_custom_provider_extra_body(agent, custom_providers)

    # The stale thinking key from nvidia must be GONE
    assert "extra_body" not in agent.request_overrides or \
        "thinking" not in agent.request_overrides.get("extra_body", {}), \
        "extra_body leaked from primary provider to fallback!"


def test_re_merge_preserves_user_override_keys():
    """User-set keys in extra_body that were NOT injected by the provider
    merge must survive a re-merge."""
    agent = SimpleNamespace(
        provider="custom:nvidia",
        model="z-ai/glm-5.2",
        base_url="https://integrate.api.nvidia.com/v1",
        request_overrides={
            "extra_body": {"user_custom_key": "keep_me"},
        },
    )
    custom_providers = [
        {
            "name": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "extra_body": {"thinking": {"type": "adaptive"}},
            "models": {"z-ai/glm-5.2": {"context_length": 1048576}},
        },
    ]

    # First merge: provider key + user key coexist
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides["extra_body"]["thinking"] == {"type": "adaptive"}
    assert agent.request_overrides["extra_body"]["user_custom_key"] == "keep_me"

    # Simulate fallback to a provider with no extra_body
    agent.provider = "custom:modal"
    agent.model = "moonshotai/Kimi-K3"
    agent.base_url = "https://modal.example/v1"
    custom_providers.append({
        "name": "modal",
        "base_url": "https://modal.example/v1",
        "models": {"moonshotai/Kimi-K3": None},
    })

    _merge_custom_provider_extra_body(agent, custom_providers)

    # Provider-injected key purged, user key preserved
    extra = agent.request_overrides.get("extra_body", {})
    assert "thinking" not in extra, "stale provider key leaked!"
    assert extra.get("user_custom_key") == "keep_me", "user key was purged!"


def test_re_merge_new_provider_extra_body_replaces_old():
    """When the new provider ALSO has an extra_body, it must replace the
    old provider's keys (not merge on top of the stale ones)."""
    agent = SimpleNamespace(
        provider="custom:nvidia",
        model="z-ai/glm-5.2",
        base_url="https://integrate.api.nvidia.com/v1",
        request_overrides={},
    )
    custom_providers = [
        {
            "name": "nvidia",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "extra_body": {"thinking": {"type": "adaptive"}},
            "models": {"z-ai/glm-5.2": {"context_length": 1048576}},
        },
        {
            "name": "other-provider",
            "base_url": "https://other.example/v1",
            "extra_body": {"separate_reasoning": True},
            "models": {"some-model": None},
        },
    ]

    # Primary merge
    _merge_custom_provider_extra_body(agent, custom_providers)
    assert agent.request_overrides["extra_body"] == {"thinking": {"type": "adaptive"}}

    # Fallback to other-provider
    agent.provider = "custom:other-provider"
    agent.model = "some-model"
    agent.base_url = "https://other.example/v1"

    _merge_custom_provider_extra_body(agent, custom_providers)

    # Old key gone, new key present
    extra = agent.request_overrides["extra_body"]
    assert "thinking" not in extra, "old provider key leaked!"
    assert extra == {"separate_reasoning": True}


def test_no_extra_body_no_override_no_keys():
    """When neither the primary nor any fallback has extra_body, calling
    merge must be a no-op (no keys tracked, no extra_body set)."""
    agent = SimpleNamespace(
        provider="custom:modal",
        model="moonshotai/Kimi-K3",
        base_url="https://modal.example/v1",
        request_overrides={},
    )
    custom_providers = [
        {
            "name": "modal",
            "base_url": "https://modal.example/v1",
            "models": {"moonshotai/Kimi-K3": None},
        },
    ]

    _merge_custom_provider_extra_body(agent, custom_providers)

    assert "extra_body" not in agent.request_overrides
    # When there's no extra_body and no previous keys, the function returns
    # early; _custom_provider_extra_body_keys may not be set on a fresh agent.
    assert getattr(agent, "_custom_provider_extra_body_keys", []) == []
