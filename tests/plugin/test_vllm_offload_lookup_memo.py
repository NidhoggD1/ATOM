"""One external-tier lookup per frontier, not one per scheduler step.

vLLM's `schedule()` asks the connector how many tokens the tier can supply
before it knows whether the request can be admitted at all: the call is gated
only on `num_computed_tokens == 0` and runs ahead of `allocate_slots`. A
request that fails allocation returns to waiting with its frontier unchanged,
so the next step asks the identical question -- and that question copies the
prompt, hashes it a chunk at a time on the scheduler thread, and blocks on the
tier's reply. With the KV cache full this repeats every step until the step
rate collapses and the engine livelocks with the GPUs idle.

The memo that stops it lives in ATOM's own scheduler, which has the same retry
shape on the native path; these tests drive the real one through the plugin so
the two halves are exercised together. `tests/test_dense_offload_connector.py`
covers the native caller and the memo's lifetime rules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

connector_mod = pytest.importorskip(
    "atom.plugin.vllm.kv_transfer.connector",
    reason="the adapter imports vLLM's connector base",
)

CHUNK = 256
HIT = 16 * CHUNK


def _adapter(monkeypatch, calls, *, min_load):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=CHUNK),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    scheduler = DenseOffloadScheduler(
        SimpleNamespace(
            kv_transfer_config={"kv_role": "kv_consumer"},
            kv_cache_block_size=16,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        )
    )
    scheduler._min_load_tokens = min_load

    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        return HIT

    scheduler._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    # Built without __init__: constructing it for real needs a VllmConfig and
    # would pull the whole offload stack in.
    adapter = object.__new__(connector_mod.AtomLMCacheOffloadConnector)
    adapter._scheduler = scheduler
    adapter._seqs = SeqViewRegistry()
    adapter._promised_loads = {}
    return adapter


def _request(req_id: str = "r0", ntok: int = 150_000):
    return SimpleNamespace(request_id=req_id, prompt_token_ids=list(range(ntok)))


def _end_of_step(adapter):
    """What every scheduler step does after the connector has answered.

    This is the boundary the bug lived on: building the metadata dispatches the
    cleanup for any lookup no longer backed by a pending load, and a dispatched
    lookup is dropped. Without it a test cannot tell a memo from the in-step
    result that was always there.
    """

    adapter._scheduler.build_connector_meta()


def test_a_declined_request_stuck_in_waiting_is_looked_up_once():
    """The decline is what releases the lookup, and so what used to lose it.

    A hit below the transfer floor is dropped, which clears the pending load,
    which makes the lookup dispatchable -- and a dispatched lookup is gone. The
    request is still at the head of the waiting queue at the same frontier, and
    with the default 8192-token floor this is the common case, not the corner.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        answers = []
        for _ in range(64):
            answers.append(adapter.get_num_new_matched_tokens(request, 0))
            _end_of_step(adapter)

    assert calls == ["r0"]
    assert answers == [(0, False)] * 64
    assert adapter._promised_loads == {}


def test_a_promised_load_stuck_in_waiting_is_looked_up_once():
    """Parking is no protection: allocation can fail for a parked request too.

    vLLM still has to find blocks for the load to land in, and when it cannot
    the request goes back to waiting at the same frontier like any other.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=0)
        request = _request()

        answers = []
        for _ in range(64):
            answers.append(adapter.get_num_new_matched_tokens(request, 0))
            _end_of_step(adapter)

    assert calls == ["r0"]
    assert answers == [(HIT, True)] * 64


def test_the_repeated_answer_does_not_restart_the_promise_watchdog():
    """Re-answering the same question is not news: the promise is the old one.

    Reassigning the counter each step would hide a request parked forever in
    WAITING_FOR_REMOTE_KVS, which is the one failure the watchdog exists for.
    """

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=0)
        request = _request()

        adapter.get_num_new_matched_tokens(request, 0)
        _end_of_step(adapter)
        adapter._promised_loads[request.request_id] = 7
        adapter.get_num_new_matched_tokens(request, 0)

    assert adapter._promised_loads[request.request_id] == 7


def test_admission_spends_the_memo():
    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        adapter.get_num_new_matched_tokens(request, 0)
        adapter.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda: [[]]), 0
        )
        _end_of_step(adapter)
        adapter.get_num_new_matched_tokens(request, 0)

    assert calls == ["r0", "r0"]


def test_a_moved_frontier_is_a_new_question():
    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)
        request = _request()

        adapter.get_num_new_matched_tokens(request, 0)
        _end_of_step(adapter)
        adapter.get_num_new_matched_tokens(request, CHUNK)

    assert calls == ["r0", "r0"]


def test_a_reused_request_id_is_a_new_question():
    """Identity, not the id: a new vLLM Request under an old id is new work."""

    calls: list[str] = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        adapter = _adapter(monkeypatch, calls, min_load=1 << 20)

        adapter.get_num_new_matched_tokens(_request(), 0)
        _end_of_step(adapter)
        # Same id, different request: the memo is keyed on the sequence, so it
        # cannot answer for a lifecycle it was not filled by.
        adapter.get_num_new_matched_tokens(_request(), 0)

    assert calls == ["r0", "r0"]
