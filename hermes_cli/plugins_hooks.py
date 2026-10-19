"""Mutating-hook metadata: which hook classes replace a payload the host already produced.

A plugin hook can REPLACE the payload a surface already showed (a ``transform_*``
hook returning a replacement string) or only observe it. A surface that streamed
that payload must suppress it when a mutator is live, or the bytes on screen and
the final text disagree (#102203).

The declaration lives on ``register_hook(..., mutates=)``; the query lives on
``PluginManager.has_mutating_hook``. Both live here so the rule that decides a
hook class's default has one home, and so ``hermes_cli/plugins.py`` — already
over its line cap — does not grow for it.
"""

from __future__ import annotations

# Hook classes whose callback return value REPLACES a payload the host already
# produced, so a surface that streamed that payload must suppress it (#102203).
# A plugin that registers one of these without saying so would garble the
# terminal, so these classes default to ``mutates=True`` and the plugin must
# opt out explicitly with ``mutates=False``.
#
# Only the classes whose replacement is visible in a STREAMED surface belong
# here: ``transform_tool_result`` and ``transform_terminal_output`` also
# replace a payload, but they do so before the host renders it, so a
# mutating flag on them has no consumer to suppress.
MUTATING_HOOKS: set[str] = {"transform_llm_output"}


def hook_mutates_by_default(hook_name: str) -> bool:
    """True when ``hook_name`` is a payload-replacing hook class."""
    return hook_name in MUTATING_HOOKS


def has_mutating_hook(hook_name: str) -> bool:
    """True when a loaded plugin handles a hook and marked it ``mutates=True`` at registration.

    Lazy-discovers first — same gate-before-invoke rationale as ``has_hook`` (tracking #64178).
    """
    from hermes_cli.plugins import _delivery_manager

    return _delivery_manager().has_mutating_hook(hook_name)
