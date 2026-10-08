import asyncio

import pytest

from controlled_rag.admission import AdmissionController, BusyError
from controlled_rag.contracts import IndexManifest, IntegrityError
from controlled_rag.index_registry import IndexRegistry


def manifest(version):
    return IndexManifest(knowledge_base_id="k", index_version=version,
                         document_versions={"doc": version})


def test_index_lifecycle_pin_does_not_switch_existing_request():
    registry = IndexRegistry()
    with pytest.raises(IntegrityError):
        registry.pin()
    registry.stage(manifest("v1"), lambda m: True)
    registry.activate("v1")
    old = registry.pin()
    registry.stage(manifest("v2"), lambda m: True)
    registry.activate("v2")
    assert old.index_version == "v1"
    assert registry.pin().index_version == "v2"
    with pytest.raises(IntegrityError):
        registry.stage(IndexManifest(knowledge_base_id="other", index_version="v1",
                                     document_versions={"doc": "v1"}), lambda m: True)


def test_unverified_index_cannot_activate():
    registry = IndexRegistry()
    with pytest.raises(IntegrityError):
        registry.stage(manifest("bad"), lambda m: False)
    with pytest.raises(IntegrityError):
        registry.activate("bad")


def test_admission_bounded_queue_and_cleanup():
    async def scenario():
        limiter = AdmissionController(max_active=1, max_waiting=0, wait_timeout_s=.01)
        async with limiter.slot():
            with pytest.raises(BusyError):
                async with limiter.slot():
                    pass
        async with limiter.slot():
            pass
    asyncio.run(scenario())


def test_admission_timeout():
    async def scenario():
        limiter = AdmissionController(max_active=1, max_waiting=1, wait_timeout_s=.01)
        async with limiter.slot():
            with pytest.raises(BusyError):
                async with limiter.slot():
                    pass
        async with limiter.slot():
            pass
    asyncio.run(scenario())
