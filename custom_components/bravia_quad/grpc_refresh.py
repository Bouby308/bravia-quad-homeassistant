"""Refresh Sony Seeds gRPC credentials and persist them to the config entry."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import CONF_NAME
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from pybravia_connect import (
    CredentialsError,
    CredentialsRefreshError,
    async_refresh_credentials,
    credentials_to_json,
    keys_need_refresh,
    parse_credentials_json,
)

from .bravia_grpc_client import BraviaGrpcClientAsync
from .const import (
    CONF_FEATURE_UNAVAILABLE_REASONS,
    CONF_GRPC_DEBUG,
    CONF_GRPC_KEYS,
    CONF_GRPC_SEEDS_POLL,
    DEFAULT_NAME,
    DOMAIN,
)
from .external_control import async_ensure_external_control_enabled
from .transport import grpc_keys_json

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

_INVALID_GRPC_KEYS_MSG = (
    "Invalid Sony Seeds credentials. Re-authenticate the integration in Settings."
)
_REFRESH_FAILED_MSG = (
    "Sony Seeds gRPC key refresh failed. Re-authenticate the integration."
)


async def async_refresh_grpc_keys(
    hass: HomeAssistant,
    entry: ConfigEntry,
    credentials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh Sony Seeds credentials and write them back to the config entry."""
    if credentials is None:
        keys_json = grpc_keys_json(entry)
        if not keys_json:
            msg = "No Sony Seeds keys configured"
            raise CredentialsError(msg)
        credentials = parse_credentials_json(keys_json)

    session = async_get_clientsession(hass)
    refreshed = await async_refresh_credentials(session, credentials)
    new_json = credentials_to_json(refreshed)
    hass.config_entries.async_update_entry(
        entry,
        data={**entry.data, CONF_GRPC_KEYS: new_json},
    )
    _LOGGER.info(
        "Refreshed Sony Seeds gRPC keys for %s (key_id=%s, expires_at=%s)",
        entry.data.get("host", entry.entry_id),
        refreshed.get("key_id"),
        refreshed.get("session_keys_expires_at"),
    )
    return refreshed


def should_refresh_grpc_keys(credentials: dict[str, Any]) -> bool:
    """Return True when proactive refresh is recommended before connect."""
    return keys_need_refresh(credentials)


async def async_try_refresh_grpc_keys(
    hass: HomeAssistant,
    entry: ConfigEntry,
    credentials: dict[str, Any],
) -> dict[str, Any] | None:
    """Refresh credentials; return None when refresh is not possible or fails."""
    if not credentials.get("refresh_token"):
        return None
    try:
        return await async_refresh_grpc_keys(hass, entry, credentials)
    except CredentialsError:
        _LOGGER.warning(
            "Sony Seeds gRPC keys for %s cannot be refreshed (missing refresh_token)",
            entry.data.get("host", entry.entry_id),
        )
        return None
    except CredentialsRefreshError:
        _LOGGER.exception(
            "Sony Seeds gRPC key refresh failed for %s",
            entry.data.get("host", entry.entry_id),
        )
        return None


async def _maybe_refresh_credentials(
    hass: HomeAssistant,
    entry: ConfigEntry,
    credentials: dict[str, Any],
    *,
    proactive: bool,
) -> tuple[dict[str, Any], bool]:
    """Refresh credentials when proactive or when caller retries after auth failure."""
    if proactive and not should_refresh_grpc_keys(credentials):
        return credentials, False
    if not proactive and not credentials.get("refresh_token"):
        msg = (
            "No refresh_token in credentials; re-authenticate with Sony Seeds "
            "(pybravia-connect tools/get_session_keys.py --login, or the HA "
            "scripts/grpc/get_session_keys.py wrapper)"
        )
        raise CredentialsError(msg)
    try:
        refreshed = await async_refresh_grpc_keys(hass, entry, credentials)
    except CredentialsError:
        raise
    except CredentialsRefreshError as err:
        if proactive:
            # Transient failure (Sony endpoint / network): keep the existing
            # keys and let the connect attempt decide. Failing setup here
            # would drop entities to unavailable even though the session may
            # still be perfectly usable.
            _LOGGER.warning(
                "Proactive Sony Seeds gRPC key refresh failed for %s; "
                "continuing with existing keys: %s",
                entry.data.get("host", entry.entry_id),
                err,
            )
            return credentials, False
        raise ConfigEntryAuthFailed(_REFRESH_FAILED_MSG) from err
    else:
        return refreshed, True


async def _seed_snapshot_logged(grpc_client: BraviaGrpcClientAsync, host: str) -> bool:
    """Bulk-seed the notify cache from GetStates; log the outcome."""
    seeded = await grpc_client.async_seed_notify_from_snapshot()
    if seeded:
        _LOGGER.info("gRPC GetStates snapshot seeded %d fields for %s", seeded, host)
    else:
        _LOGGER.warning(
            "gRPC GetStates snapshot failed for %s (%s); retrying in background",
            host,
            grpc_client.last_rpc_error or "no RPC error recorded",
        )
    return bool(seeded)


async def async_setup_grpc_client(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> BraviaGrpcClientAsync | None:
    """Connect gRPC control plane and start the notify stream."""
    keys_json = grpc_keys_json(entry)
    if not keys_json:
        _LOGGER.warning("gRPC transport selected but no Sony Seeds keys configured")
        return None

    try:
        credentials = parse_credentials_json(keys_json)
    except (TypeError, json.JSONDecodeError):
        _LOGGER.exception("Invalid gRPC keys JSON in config entry")
        raise ConfigEntryAuthFailed(_INVALID_GRPC_KEYS_MSG) from None

    credentials, refreshed_proactively = await _maybe_refresh_credentials(
        hass, entry, credentials, proactive=True
    )
    keys_json = credentials_to_json(credentials)

    try:
        grpc_debug = bool(entry.options.get(CONF_GRPC_DEBUG, False))
        seeds_poll = bool(entry.options.get(CONF_GRPC_SEEDS_POLL, False))
        grpc_client = BraviaGrpcClientAsync.from_keys_json(
            entry.data["host"],
            keys_json,
            debug=grpc_debug,
            seeds_poll=seeds_poll,
            credentials=credentials,
            hass=hass,
        )
        if grpc_debug:
            _LOGGER.info("gRPC debug logging enabled for %s", entry.data["host"])
    except (ValueError, json.JSONDecodeError):
        _LOGGER.exception("Invalid gRPC keys JSON in config entry")
        raise ConfigEntryAuthFailed(_INVALID_GRPC_KEYS_MSG) from None

    async def _refresh_keys_callback() -> bool:
        nonlocal credentials, keys_json
        try:
            credentials, _ = await _maybe_refresh_credentials(
                hass, entry, credentials, proactive=False
            )
            keys_json = credentials_to_json(credentials)
            grpc_client.update_keys(credentials)
        except (
            CredentialsError,
            CredentialsRefreshError,
            ConfigEntryAuthFailed,
            ConfigEntryNotReady,
        ):
            # Never propagate into the connection manager: an escaping
            # exception killed the reconnect loop and left entities frozen
            # at their last state with no "connection down" ever shown.
            _LOGGER.exception(
                "Sony Seeds gRPC key refresh failed during reconnect for %s",
                entry.data["host"],
            )
            return False
        else:
            return True

    grpc_client.set_refresh_keys_callback(_refresh_keys_callback)

    async def _persist_feature_reasons(reasons: dict[str, str]) -> None:
        if entry.data.get(CONF_FEATURE_UNAVAILABLE_REASONS) == reasons:
            return
        # Data-only write must not trip options update_listener → reload.
        hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})[
            "_suppress_entry_reload"
        ] = True
        hass.config_entries.async_update_entry(
            entry,
            data={**entry.data, CONF_FEATURE_UNAVAILABLE_REASONS: reasons},
        )

    grpc_client.configure_feature_unavailable_persistence(
        entry.data.get(CONF_FEATURE_UNAVAILABLE_REASONS),
        _persist_feature_reasons,
    )

    try:
        connected = await grpc_client.async_connect()
        if (
            not connected
            and not refreshed_proactively
            and not grpc_client.is_transport_error
        ):
            credentials, _ = await _maybe_refresh_credentials(
                hass, entry, credentials, proactive=False
            )
            grpc_client.update_keys(credentials)
            connected = await grpc_client.async_connect()

        if not connected:
            raise ConfigEntryNotReady

        await grpc_client.async_fetch_capabilities()

        ensure_result = await async_ensure_external_control_enabled(
            entry.data["host"],
            name=entry.data.get(CONF_NAME, DEFAULT_NAME),
            grpc_client=grpc_client,
        )
        grpc_client.note_external_control_ensure(ensure_result)

        seeded = await _seed_snapshot_logged(grpc_client, entry.data["host"])
        await grpc_client.async_backfill_entity_paths()
        await grpc_client.async_start_notify()
        await grpc_client.async_warmup_notify()
        if not seeded:
            # _notify_stop is only cleared by async_start_notify; schedule after.
            grpc_client.schedule_seed_retry()
        _LOGGER.info("gRPC notify stream started for %s", entry.data["host"])
    except ConfigEntryAuthFailed:
        await grpc_client.async_disconnect()
        raise
    except CredentialsError as err:
        await grpc_client.async_disconnect()
        raise ConfigEntryAuthFailed(str(err)) from err
    except OSError as err:
        _LOGGER.exception("gRPC connection failed for %s", entry.data["host"])
        await grpc_client.async_disconnect()
        raise ConfigEntryNotReady from err
    else:
        return grpc_client
