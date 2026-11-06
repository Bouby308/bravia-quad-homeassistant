"""Tests for gRPC auto-reconnect when the notify stream drops."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.bravia_quad.bravia_grpc_client import BraviaGrpcClientAsync
from custom_components.bravia_quad.const import NOTIFY_STALL_TIMEOUT_S

if TYPE_CHECKING:
    from collections.abc import Callable

    from custom_components.bravia_quad.notify_state import NotifyStateUpdate


@pytest.fixture(autouse=True)
def _mock_ensure_external_control() -> None:
    with patch(
        "custom_components.bravia_quad.bravia_grpc_client.async_ensure_external_control_enabled",
        new=AsyncMock(),
    ):
        yield


@pytest.fixture
def grpc_async() -> BraviaGrpcClientAsync:
    client = BraviaGrpcClientAsync(
        "192.168.1.50",
        device_id="dev",
        key_id="kid",
        session_key="s" * 64,
        hmac_key="h" * 64,
    )
    client._connected = True
    client._client._notify_state = {"main.power": True}
    return client


def _fake_start_notify_factory(
    *,
    lose_after: int = 1,
    stop_client: BraviaGrpcClientAsync | None = None,
) -> Any:
    """Return a start_notify that signals connection_lost after *lose_after* starts."""
    calls = {"n": 0}

    def fake_start_notify(
        on_delta: Callable[[str, Any], None],
        on_connection_lost: Callable[[], None] | None = None,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        del on_delta, on_reconnect
        calls["n"] += 1
        if calls["n"] >= lose_after and on_connection_lost is not None:
            on_connection_lost()
            if stop_client is not None and calls["n"] >= 2:
                stop_client._notify_stop.set()
                stop_client._notify_restore_event.set()

    fake_start_notify.calls = calls  # type: ignore[attr-defined]
    return fake_start_notify


async def test_reconnects_after_notify_stream_ends(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """When notify connection_lost fires, the manager should restore the session."""
    fake_start = _fake_start_notify_factory(lose_after=1, stop_client=grpc_async)
    grpc_async._client.start_notify = fake_start  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async.async_connect = AsyncMock(return_value=True)
    grpc_async.async_fetch_capabilities = AsyncMock(return_value=frozenset({"power"}))
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=3)
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    grpc_async.async_connect.assert_awaited()
    grpc_async.async_seed_notify_from_snapshot.assert_awaited()
    assert fake_start.calls["n"] >= 2


async def test_reconnect_callback_and_snapshot_callbacks(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Successful reconnect should refresh HA callbacks from cached notify_state."""
    updates: list[NotifyStateUpdate] = []

    def capture(update: NotifyStateUpdate) -> None:
        updates.append(update)

    grpc_async.add_state_callback(capture)
    reconnect_cb = AsyncMock()
    grpc_async.set_reconnect_callback(reconnect_cb)

    fake_start = _fake_start_notify_factory(lose_after=1, stop_client=grpc_async)
    grpc_async._client.start_notify = fake_start  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async.async_connect = AsyncMock(return_value=True)
    grpc_async.async_fetch_capabilities = AsyncMock(return_value=frozenset({"power"}))
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=1)
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    reconnect_cb.assert_awaited_once()
    assert any(u.path == "main.power" for u in updates)


async def test_disconnect_stops_connection_manager(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Intentional shutdown should not keep retrying reconnect."""

    def fake_start_notify(
        on_delta: Callable[[str, Any], None],
        on_connection_lost: Callable[[], None] | None = None,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        del on_delta, on_connection_lost, on_reconnect

    grpc_async._client.start_notify = fake_start_notify  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await grpc_async.async_disconnect()

    assert grpc_async._notify_task is None
    assert not grpc_async.is_connected


async def test_stall_probe_ok_requests_stream_only_renew(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Silent stream + healthy GetCapabilities probe must renew only the stream."""
    grpc_async._last_notify_at = time.monotonic() - (NOTIFY_STALL_TIMEOUT_S + 1)
    grpc_async._client.probe = MagicMock(return_value=True)

    assert await grpc_async._async_check_stream_health() is True
    assert grpc_async._stream_renew_pending is True


async def test_stall_probe_failure_requests_session_renew(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Silent stream + failing probe must escalate to a full session renew."""
    grpc_async._last_notify_at = time.monotonic() - (NOTIFY_STALL_TIMEOUT_S + 1)
    grpc_async._client.probe = MagicMock(return_value=False)

    assert await grpc_async._async_check_stream_health() is True
    assert grpc_async._stream_renew_pending is False


async def test_recent_delta_skips_stall_probe(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """A fresh delta keeps the watchdog quiet — idle-but-healthy streams are fine."""
    grpc_async._last_notify_at = time.monotonic()
    probe = MagicMock(return_value=False)
    grpc_async._client.probe = probe

    assert await grpc_async._async_check_stream_health() is False
    probe.assert_not_called()


async def test_stream_renew_restarts_stream_without_session_restore(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Watchdog stream renew must not re-auth or mark entities unavailable."""
    availability: list[bool] = []
    grpc_async.register_availability_callback(availability.append)
    calls = {"n": 0}

    async def fake_bridge() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            grpc_async._stream_renew_pending = True
        else:
            grpc_async._notify_stop.set()

    grpc_async._run_notify_bridge = fake_bridge  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async._async_restore_session = AsyncMock(return_value=True)
    grpc_async._async_light_reconnect = AsyncMock()

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    assert calls["n"] == 2
    grpc_async._async_restore_session.assert_not_awaited()
    grpc_async._async_light_reconnect.assert_awaited_once()
    assert False not in availability
    assert grpc_async.is_connected


async def test_session_lost_renews_keeping_last_state_available(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Stream loss renews the session before ever publishing unavailable."""
    availability: list[bool] = []
    grpc_async.register_availability_callback(availability.append)
    calls = {"n": 0}

    async def fake_bridge() -> None:
        calls["n"] += 1
        if calls["n"] >= 2:
            grpc_async._notify_stop.set()

    grpc_async._run_notify_bridge = fake_bridge  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async._async_restore_session = AsyncMock(return_value=True)

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    grpc_async._async_restore_session.assert_awaited_once()
    assert False not in availability
    assert grpc_async.is_connected


async def test_renew_failure_publishes_unavailable_after_attempt(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Unavailable is shown only after the renew attempt fails, not before."""
    events: list[str] = []
    grpc_async.register_availability_callback(
        lambda available: events.append(f"avail:{available}")
    )

    calls = {"n": 0}

    async def fake_bridge() -> None:
        # First return is the session loss; the stop below ends the manager.
        calls["n"] += 1

    async def failing_restore() -> bool:
        events.append("restore")
        # async_connect flips the raw flag on failure inside a real restore.
        grpc_async._connected = False
        grpc_async._notify_stop.set()
        return False

    grpc_async._run_notify_bridge = fake_bridge  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async._async_restore_session = AsyncMock(side_effect=failing_restore)

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    assert events == ["restore", "avail:False"]


async def test_refresh_callback_exception_does_not_kill_manager(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """ConfigEntryAuthFailed from the refresh callback must not escape restore."""
    from homeassistant.exceptions import ConfigEntryAuthFailed

    async def boom() -> bool:
        msg = "refresh failed"
        raise ConfigEntryAuthFailed(msg)

    grpc_async.set_refresh_keys_callback(boom)
    grpc_async.async_connect = AsyncMock(return_value=False)

    assert await grpc_async._async_restore_session() is False


async def test_light_reconnect_seed_failure_schedules_retry(
    grpc_async: BraviaGrpcClientAsync, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed light-reconnect re-seed must schedule a background retry."""
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=0)
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))
    grpc_async._client.last_rpc_error = "GetStates failed: StatusCode.UNAVAILABLE"
    schedule = MagicMock()
    grpc_async.schedule_seed_retry = schedule

    await grpc_async._async_light_reconnect()

    schedule.assert_called_once()
    assert "GetStates re-seed failed" in caplog.text


async def test_seed_retry_loop_recovers_and_dispatches(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Transient failure then success: re-seed, backfill, and refresh entities."""
    updates: list[NotifyStateUpdate] = []
    grpc_async.add_state_callback(updates.append)
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(side_effect=[0, 4])
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await asyncio.wait_for(grpc_async._async_seed_retry_loop(), timeout=2.0)

    assert grpc_async.async_seed_notify_from_snapshot.await_count == 2
    grpc_async.async_backfill_entity_paths.assert_awaited_once()
    assert any(u.path == "main.power" for u in updates)


async def test_seed_retry_loop_escalates_invalid_argument_to_renew(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Stale rolling-auth INVALID_ARGUMENT must renew the session, not re-sign."""
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=0)
    grpc_async._client.last_rpc_error = "GetStates failed: StatusCode.INVALID_ARGUMENT"

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await asyncio.wait_for(grpc_async._async_seed_retry_loop(), timeout=2.0)

    assert grpc_async._notify_restore_event.is_set()
    assert grpc_async._seed_retry_escalations == 1
    grpc_async.async_seed_notify_from_snapshot.assert_awaited_once()


async def test_seed_retry_loop_respects_escalation_cap(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Past the renew cap: quiet backoff retries, no more renew churn."""
    grpc_async._seed_retry_escalations = 3
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=0)
    grpc_async._client.last_rpc_error = "GetStates failed: StatusCode.INVALID_ARGUMENT"
    waits = {"n": 0}

    async def fake_wait(_seconds: float) -> None:
        waits["n"] += 1
        if waits["n"] >= 2:
            grpc_async._notify_stop.set()

    with patch.object(grpc_async, "_async_wait", new=fake_wait):
        await asyncio.wait_for(grpc_async._async_seed_retry_loop(), timeout=2.0)

    assert not grpc_async._notify_restore_event.is_set()
    assert grpc_async._seed_retry_escalations == 3
    assert waits["n"] == 2


async def test_seed_retry_loop_exits_when_disconnected(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """The connection manager owns recovery once the session is down."""
    grpc_async._connected = False
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=0)

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await asyncio.wait_for(grpc_async._async_seed_retry_loop(), timeout=2.0)

    grpc_async.async_seed_notify_from_snapshot.assert_not_awaited()


async def test_schedule_seed_retry_idempotent_and_guards(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """One loop at a time; no loop while stopping or disconnected."""

    async def stub_loop() -> None:
        await asyncio.sleep(30)

    grpc_async._async_seed_retry_loop = stub_loop  # type: ignore[method-assign]
    grpc_async.schedule_seed_retry()
    first = grpc_async._seed_retry_task
    assert first is not None
    grpc_async.schedule_seed_retry()
    assert grpc_async._seed_retry_task is first

    grpc_async._cancel_seed_retry()
    with contextlib.suppress(asyncio.CancelledError):
        await first
    assert first.cancelled()

    grpc_async._notify_stop.set()
    grpc_async.schedule_seed_retry()
    assert grpc_async._seed_retry_task is None
    grpc_async._notify_stop.clear()

    grpc_async._connected = False
    grpc_async.schedule_seed_retry()
    assert grpc_async._seed_retry_task is None


async def test_disconnect_cancels_seed_retry(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Config-entry unload must not leave a retry loop running."""

    async def stub_loop() -> None:
        await asyncio.sleep(30)

    grpc_async._async_seed_retry_loop = stub_loop  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()
    grpc_async._client.close = MagicMock()
    grpc_async.schedule_seed_retry()
    task = grpc_async._seed_retry_task
    assert task is not None

    await grpc_async.async_disconnect()

    assert grpc_async._seed_retry_task is None
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_restore_session_reschedules_retry_on_seed_failure(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Restore seeds itself; a failed restore seed restarts the retry fresh."""
    grpc_async.async_connect = AsyncMock(return_value=True)
    grpc_async.async_fetch_capabilities = AsyncMock(return_value=None)
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=0)
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))
    stale = MagicMock()
    stale.done.return_value = False
    grpc_async._seed_retry_task = stale

    async def stub_loop() -> None:
        await asyncio.sleep(30)

    grpc_async._async_seed_retry_loop = stub_loop  # type: ignore[method-assign]
    try:
        assert await grpc_async._async_restore_session() is True
        stale.cancel.assert_called_once()
        new_task = grpc_async._seed_retry_task
        assert new_task is not None
        assert new_task is not stale
    finally:
        grpc_async._cancel_seed_retry()


async def test_restore_session_no_retry_on_seed_success(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """A successful restore seed clears the retry slot with nothing pending."""
    grpc_async.async_connect = AsyncMock(return_value=True)
    grpc_async.async_fetch_capabilities = AsyncMock(return_value=None)
    grpc_async.async_seed_notify_from_snapshot = AsyncMock(return_value=5)
    grpc_async.async_backfill_entity_paths = AsyncMock(return_value=(0, 0, 0))
    stale = MagicMock()
    stale.done.return_value = False
    grpc_async._seed_retry_task = stale

    assert await grpc_async._async_restore_session() is True

    stale.cancel.assert_called_once()
    assert grpc_async._seed_retry_task is None


async def test_notify_delta_clears_seeds_write_guard(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """The device's own delta releases the write guard early (Seeds can resume)."""
    path = "sound_setting.dsee_ultimate"
    grpc_async._seeds_write_guard[path] = time.monotonic() + 30
    loop = asyncio.get_running_loop()

    def fake_start_notify(
        on_delta: Callable[[str, Any], None],
        on_connection_lost: Callable[[], None] | None = None,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        del on_connection_lost, on_reconnect
        on_delta(path, False)
        # Queue the stop after the delta dispatch (same loop FIFO order) so
        # _dispatch runs before the stop guard inside it takes effect.
        loop.call_soon_threadsafe(grpc_async._notify_stop.set)

    grpc_async._client.start_notify = fake_start_notify  # type: ignore[method-assign]
    grpc_async._client.stop_notify = MagicMock()

    with patch.object(grpc_async, "_async_wait", new=AsyncMock()):
        await grpc_async.async_start_notify()
        await asyncio.wait_for(grpc_async._notify_task, timeout=2.0)

    assert path not in grpc_async._seeds_write_guard
