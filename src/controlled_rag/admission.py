"""Local FastAPI admission control: bounded in-flight + bounded waiting slots.

This module cannot alone guarantee distributed rate limits or global deadlines.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager


class BusyError(RuntimeError):
    pass


class AdmissionController:
    def __init__(self, max_active: int = 4, max_waiting: int = 8,
                 wait_timeout_s: float = 3.0):
        if max_active <= 0 or max_waiting < 0 or wait_timeout_s <= 0:
            raise ValueError("invalid bounded admission settings")
        self._pending_tokens = threading.BoundedSemaphore(max_active + max_waiting)
        self._active = asyncio.Semaphore(max_active)
        self._wait_timeout = wait_timeout_s

    @asynccontextmanager
    async def slot(self):
        if not self._pending_tokens.acquire(blocking=False):
            raise BusyError("all active and waiting slots are full")
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._active.acquire(), timeout=self._wait_timeout)
                acquired = True
            except asyncio.TimeoutError as exc:
                raise BusyError("waiting timeout exceeded") from exc
            yield
        finally:
            if acquired:
                self._active.release()
            self._pending_tokens.release()
