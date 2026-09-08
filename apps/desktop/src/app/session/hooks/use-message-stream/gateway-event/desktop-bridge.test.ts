import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { $toursEnabled } from '@/store/tours'

import { handleDesktopBridgeEvent } from './desktop-bridge'
import type { GatewayEventContext } from './types'

const requestMock = vi.fn()

function bridgeCtx(eventType: string, isActiveEvent: boolean): GatewayEventContext {
  const sessionId = 'rt-owner-001'

  return {
    deps: {},
    event: { session_id: sessionId, type: eventType },
    explicitSid: sessionId,
    fromActiveSource: () => true,
    isActiveEvent,
    occurredAt: Date.now() / 1000,
    payload: { action: 'elements', request_id: 'req-1' },
    scheduleConfigRefresh: vi.fn(),
    sessionId
  } as unknown as GatewayEventContext
}

describe('handleDesktopBridgeEvent scoped fan-out (#104435)', () => {
  beforeEach(() => {
    requestMock.mockClear()
    $gateway.set({ request: requestMock } as unknown as NonNullable<ReturnType<typeof $gateway.get>>)
    $toursEnabled.set(true)
  })

  it('stays silent on preview.act.request for a background session (no refusal)', async () => {
    expect(handleDesktopBridgeEvent(bridgeCtx('preview.act.request', false))).toBe(true)

    // The refusal used to be synchronous; the owner's success needs a dynamic
    // import first, so any answer here would win the gateway's first-answer
    // race and veto the owner. Flush and assert zero respond calls.
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(requestMock).not.toHaveBeenCalled()
  })

  it('answers preview.act.request for the active session exactly once', async () => {
    expect(handleDesktopBridgeEvent(bridgeCtx('preview.act.request', true))).toBe(true)

    await vi.waitFor(() => {
      expect(requestMock).toHaveBeenCalledTimes(1)
    })
    expect(requestMock.mock.calls[0][0]).toBe('preview.act.respond')
  })

  it('stays silent on tour.request for a background session (no refusal)', async () => {
    expect(handleDesktopBridgeEvent(bridgeCtx('tour.request', false))).toBe(true)

    await new Promise(resolve => setTimeout(resolve, 50))
    expect(requestMock).not.toHaveBeenCalled()
  })
})
