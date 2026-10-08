"""Atomic manifest pointer activation and request-snapshot pinning.

This is in-process policy code, NOT a full distributed index lifecycle manager.
The caller must perform real index/manifest data-integrity and regression checks.
"""

from __future__ import annotations

from threading import RLock
from typing import Callable

from .contracts import IndexManifest, IntegrityError


class IndexRegistry:
    def __init__(self):
        self._lock = RLock()
        self._verified: dict[str, str] = {}  # immutable serialized manifest
        self._active: str | None = None

    def stage(self, manifest: IndexManifest, validator: Callable[[IndexManifest], bool]) -> None:
        """Stage immutable bundle only if externally supplied validations pass."""
        if not validator(manifest):
            raise IntegrityError("index candidate rejected by validator")
        blob = manifest.model_dump_json()
        with self._lock:
            if manifest.index_version in self._verified and self._verified[manifest.index_version] != blob:
                raise IntegrityError("cannot modify immutable staged index version")
            self._verified[manifest.index_version] = blob

    def activate(self, index_version: str) -> None:
        with self._lock:
            if index_version not in self._verified:
                raise IntegrityError("index must be validated/staged before activation")
            self._active = index_version

    def pin(self) -> IndexManifest:
        """New request receives an independent immutable-in-practice snapshot."""
        with self._lock:
            if self._active is None:
                raise IntegrityError("no active index version")
            return IndexManifest.model_validate_json(self._verified[self._active])

    def active_version(self) -> str | None:
        with self._lock:
            return self._active
