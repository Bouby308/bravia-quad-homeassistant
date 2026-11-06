"""Tests for BraviaGrpcClientAsync exec failure recovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.bravia_quad.bravia_grpc_client import BraviaGrpcClientAsync


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
    client._client.exec_command = MagicMock()
    return client


@pytest.mark.asyncio
async def test_async_exec_command_retries_after_session_restore(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    grpc_async._client.exec_command.side_effect = [False, True]

    with patch.object(
        grpc_async, "_async_restore_session", new=AsyncMock(return_value=True)
    ) as restore:
        ok = await grpc_async.async_exec_command("power", bool_value=True)

    assert ok is True
    assert grpc_async._client.exec_command.call_count == 2
    restore.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_exec_command_no_retry_when_restore_fails(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    grpc_async._client.exec_command.return_value = False

    with patch.object(
        grpc_async, "_async_restore_session", new=AsyncMock(return_value=False)
    ):
        ok = await grpc_async.async_exec_command("power", bool_value=True)

    assert ok is False
    grpc_async._client.exec_command.assert_called_once()


@pytest.mark.asyncio
async def test_async_exec_command_skips_restore_on_success(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    grpc_async._client.exec_command.return_value = True

    with patch.object(grpc_async, "_async_restore_session", new=AsyncMock()) as restore:
        ok = await grpc_async.async_exec_command("volume", int_value=10)

    assert ok is True
    restore.assert_not_awaited()


DSEE_PATH = "sound_setting.dsee_ultimate"


@pytest.mark.asyncio
async def test_exec_success_merges_written_value_into_notify_cache(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Exec success must update the notify cache or the pre-write value bounces back."""
    grpc_async._client.exec_command.return_value = True
    grpc_async._client.update_notify_cache({DSEE_PATH: True})

    with (
        patch.object(grpc_async, "_async_confirm_exec_readback", new=AsyncMock()),
        patch.object(grpc_async, "_async_restore_session", new=AsyncMock()),
    ):
        ok = await grpc_async.async_exec_command(DSEE_PATH, bool_value=False)

    assert ok is True
    assert grpc_async.notify_state[DSEE_PATH] is False
    assert DSEE_PATH in grpc_async._seeds_write_guard


@pytest.mark.asyncio
async def test_exec_retry_success_also_merges_written_value(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    grpc_async._client.exec_command.side_effect = [False, True]

    with (
        patch.object(
            grpc_async, "_async_restore_session", new=AsyncMock(return_value=True)
        ),
        patch.object(grpc_async, "_async_confirm_exec_readback", new=AsyncMock()),
    ):
        ok = await grpc_async.async_exec_command(DSEE_PATH, bool_value=False)

    assert ok is True
    assert grpc_async.notify_state[DSEE_PATH] is False


@pytest.mark.asyncio
async def test_exec_readback_merges_device_value_for_readable_path(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """Readable paths are re-read right after exec; the device value wins."""
    grpc_async._client.exec_command.return_value = True

    readback = AsyncMock(return_value=12)
    with (
        patch.object(grpc_async, "async_get_states_single_path", new=readback),
        patch.object(grpc_async, "_async_restore_session", new=AsyncMock()),
    ):
        ok = await grpc_async.async_exec_command("volume", int_value=10)

    assert ok is True
    readback.assert_awaited_once_with("volume")
    assert grpc_async.notify_state["volume"] == 12


@pytest.mark.asyncio
async def test_exec_readback_skipped_for_notify_only_path(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """The device rejects GetStates for notify-only paths — never probe them."""
    grpc_async._client.exec_command.return_value = True

    readback = AsyncMock(return_value=None)
    with (
        patch.object(grpc_async, "async_get_states_single_path", new=readback),
        patch.object(grpc_async, "_async_restore_session", new=AsyncMock()),
    ):
        ok = await grpc_async.async_exec_command(DSEE_PATH, bool_value=False)

    assert ok is True
    readback.assert_not_awaited()
    # Optimistic value stands untouched.
    assert grpc_async.notify_state[DSEE_PATH] is False


@pytest.mark.asyncio
async def test_exec_readback_failure_keeps_optimistic_value(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    """A failed read-back is soft: the optimistic merge must survive."""
    grpc_async._client.exec_command.return_value = True

    with (
        patch.object(
            grpc_async, "async_get_states_single_path", new=AsyncMock(return_value=None)
        ),
        patch.object(grpc_async, "_async_restore_session", new=AsyncMock()),
    ):
        ok = await grpc_async.async_exec_command("volume", int_value=10)

    assert ok is True
    assert grpc_async.notify_state["volume"] == 10


@pytest.mark.asyncio
async def test_exec_failure_does_not_merge(
    grpc_async: BraviaGrpcClientAsync,
) -> None:
    grpc_async._client.exec_command.return_value = False
    grpc_async._client.update_notify_cache({DSEE_PATH: True})

    with patch.object(
        grpc_async, "_async_restore_session", new=AsyncMock(return_value=False)
    ):
        ok = await grpc_async.async_exec_command(DSEE_PATH, bool_value=False)

    assert ok is False
    assert grpc_async.notify_state[DSEE_PATH] is True
    assert DSEE_PATH not in grpc_async._seeds_write_guard
