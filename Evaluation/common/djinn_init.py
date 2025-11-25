"""
Helper to initialize Djinn just once for evaluation harnesses.

The goal is to mirror what real users do: call `djinn.init(server_address=...)`
before running workloads so remote Accel devices are available.  We keep init
outside of the measured workloads so timing aggregates remain untouched.
"""

from __future__ import annotations

from typing import Optional
import asyncio
import time

import logging

logger = logging.getLogger(__name__)


_initialized = False


def ensure_initialized(server_address: Optional[str]) -> None:
    """
    Initialize Djinn if not already done.

    Args:
        server_address: Remote Djinn server data port (e.g., "localhost:5556").
                        If None we rely on env vars or default config.
    """
    global _initialized
    if _initialized:
        return

    try:
        import djinn  # noqa: F401 - side effect init
    except ImportError as exc:
        logger.warning("Djinn module unavailable; skipping initialization: %s", exc)
        return

    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # In async context - can't use sync init, but initialization should already be done
            # Just verify it's initialized
            from djinn.backend.runtime.initialization import _runtime_state
            if not _runtime_state.initialized:
                logger.warning(
                    "In async context but Djinn not initialized. "
                    "Call ensure_initialized_before_async() before asyncio.run()"
                )
            else:
                _initialized = True
        except RuntimeError:
            # Not in async context - use sync init
            result = djinn.init(server_address=server_address, auto_connect=True, profiling=False)
            if result.get("status") == "success":
                logger.info("Djinn init succeeded (server=%s)", server_address or "default")
                _initialized = True
            else:
                logger.warning("Djinn init reported failure: %s", result.get("error"))
    except Exception as exc:
        logger.error("Djinn init() raised: %s", exc)


def ensure_initialized_before_async(server_address: Optional[str]) -> None:
    """
    Initialize Djinn synchronously before entering an async context.
    This should be called from main() or other sync contexts before
    asyncio.run() is called.

    Args:
        server_address: Remote Djinn server data port (e.g., "localhost:5556").
                        If None we rely on env vars or default config.
    """
    ensure_initialized(server_address)

