"""Plugin-triggered machine notices for a running turn (``PluginContext.deliver_to_running_turn``).

A machine notice is appended to the running turn's newest tool result under a declared machine
provenance header — never a user row, never the out-of-band user marker — so the model reads the
gateway's words, not an instruction from the user. Nothing touches the session when no turn is
running on the key; nothing is durable, so a caller that needs exactly-once must keep its own
settlement (the background-process completion keeps it via the admission transfer).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Optional

logger = logging.getLogger("gateway.run")


class GatewayRunningTurnMixin:
    def _install_running_turn_injector(self) -> None:
        """Publish this live gateway's running-turn injector process-wide."""
        from hermes_cli.plugins import publish_running_turn_host

        publish_running_turn_host(self, self._deliver_to_running_turn)

    def _clear_running_turn_injector(self) -> None:
        """Remove this runner's injector without clobbering a newer owner."""
        from hermes_cli.plugins import clear_published_running_turn_host

        clear_published_running_turn_host(self)

    def _deliver_to_running_turn(self, *, content: str, plugin_id: str, session_key: Optional[str] = None,
                                 plugin_home: Optional[object] = None) -> bool:
        """Deliver a machine notice into the turn running on *session_key*, or nothing.

        Plugins that run inside one session pass the session's key (or pass none and let the
        plugin layer resolve the calling session); there is no keyless resolution across chats —
        a notice is bound to the turn it landed on, not broadcast.
        """
        if not session_key or not isinstance(content, str) or not content.strip():
            return False
        state = self._peek_session_state(session_key)
        agent = getattr(getattr(state, "turn", None), "agent", None)
        from gateway.run import _AGENT_PENDING_SENTINEL
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return False
        notice = getattr(agent, "notice", None)
        if not callable(notice):
            # Cron/managed-worker turns have no notice seam — same boundary as the steer seam.
            return False
        return bool(notice(content.strip()))
