"""Deliver an admitted process completion into the running turn instead of the next one.

A background process that finishes while a turn is still running has two audiences: the chat,
which already gets a concise receipt (#112033), and the agent, whose running turn is the one that
can act on the result. Queueing the completion teaches the agent about it only after that turn
ends — minutes later in a long review, and the turn may be waiting on that very process.

This hands the admitted row to the running claim instead, through the same durable transfer the
client-facing ``prompt.steer_queued`` uses (``session_queued_controls.transfer_into_running_claim``,
``kind="notice"``): the row moves ``queued`` -> terminal ``steering`` bound to the running
generation, so the queue drain can never claim it too, and a delivery that reached nobody puts the
row back in its FIFO place instead of stranding it. The text rides the claim's machine-notice
drain — an appendix inside the newest tool result, under a declared machine-provenance header —
never the out-of-band user marker, so the turn reads it as the gateway speaking rather than as an
instruction from the user. Nothing is lost and nothing is delivered twice.

Off by default (``display.background_process_delivery: queue``): delivering an automated notice
mid-turn is a policy decision, not just plumbing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("hermes")

DELIVERY_MODES = ("queue", "notice")


def delivery_mode(mode: str | None = None) -> str:
    """How a process completion reaches the agent: ``queue`` (default) or ``notice``.

    ``mode`` is the resolved knob. A missing or unknown value keeps today's behaviour — the
    completion waits for the next turn. ``notice`` delivers it into the turn that is running now.
    """
    raw = str(mode or "").strip().lower()
    return raw if raw in DELIVERY_MODES else "queue"


async def notice_completion_into_running_turn(authority, receipt, text: str) -> bool:
    """Hand an admitted completion to the turn that is running right now.

    True when the running claim took the text: it is delivered into that turn and the queue drain
    can no longer claim the row. False when nothing live can take it — no running claim, no notice
    seam, or the drain claimed the row first — and the row stays queued and runs as the next turn,
    exactly as before.
    """
    from gateway.session_contract import SessionRef
    from gateway.session_queued_controls import transfer_into_running_claim
    from hermes_state_runtime import RuntimeStoreError

    session_id = receipt.ref.session_id
    live = authority.sessions.get(session_id)
    if live is None:
        return False
    ref = SessionRef(authority.profile_id, session_id)
    generation = authority.executing.get(session_id)
    if generation is None:
        return False
    try:
        committed, _fresh, delivered = await transfer_into_running_claim(
            authority, ref, generation, receipt.admission_id, text, kind="notice")
    except RuntimeStoreError:
        # The drain claimed the row first, or the epoch moved under us: it runs as the next turn.
        return False
    if committed is None or not delivered:
        return False
    logger.info("Process completion steered into the running turn for session %s", session_id)
    return True
