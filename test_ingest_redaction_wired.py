"""Is the ingest filter WIRED, or does it merely exist?

⛔⛔ WHY THIS FILE IS SEPARATE FROM test_redact.py. That suite proves the filter works:
87 checks, and 62 of them fail when the filter is neutered. It cannot prove the filter is
REACHED. `firm_hot_stack` once shipped complete, tested and with no possible caller, and
corroboration was built, reachable and used zero times in nine days. A filter nothing
calls and a filter that fails are indistinguishable from the corpus's point of view.

⇒ So this test does not test `redact`. It drives `memory_store` -- the real ingest tool,
  through its real code path -- with a fake credential and asserts the credential is not
  in what got stored.

The embedding backend is stubbed, because the question here is "did the text get filtered
on its way in", not "does Nomic work". Everything between the tool's first line and the
append to state["texts"] is the real thing.
"""

import sys

import numpy as np

import membot_server as ms


# A fake Anthropic-shaped key. Constructed here, contains FAKE, never a real credential.
FAKE_KEY = "sk-ant-api03-" + "FAKE" + ("q7Zx" * 20) + "AA"
SESSION = "redaction-wiring-probe"


def _setup(monkey_embed=True):
    """A session that is minimally mountable, so memory_store runs to the append."""
    ms._server_config["read_only"] = False
    state = ms._get_session(SESSION)
    state["cartridge_name"] = "probe"
    state["cartridge_path"] = None
    state["mount_read_only"] = False
    state["mempack_id"] = None
    state["texts"] = []
    state["embeddings"] = None
    state["binary_corpus"] = None
    state["metadata"] = []
    state.pop("_dedup_set", None)
    state.pop("_dedup_n", None)
    if monkey_embed:
        ms.embed_text = lambda text, prefix="search_document": np.ones(768,
                                                                       dtype=np.float32)
    return state


def _store(content):
    """Call the tool the way a client does, through its registered function."""
    fn = getattr(ms.memory_store, "fn", ms.memory_store)
    return fn(content=content, session_id=SESSION)


def test_a_credential_does_not_reach_the_corpus():
    """⭐⭐ THE ONE THAT MATTERS."""
    state = _setup()
    result = _store(f'export ANTHROPIC_API_KEY="{FAKE_KEY}" && echo deploying')

    stored = "\n".join(state["texts"])
    assert stored, f"nothing was stored, so this test proved nothing. tool said: {result}"
    assert FAKE_KEY not in stored, (
        "THE CREDENTIAL REACHED THE CORPUS. The filter exists but is not wired into "
        "memory_store, or runs after the text is captured."
    )
    assert "[REDACTED:" in stored, (
        f"the key is gone but no redaction marker is present; the passage may have been "
        f"dropped rather than redacted. stored: {stored[:200]!r}"
    )


def test_the_surrounding_passage_survives():
    """Context is useful memory and must not be destroyed to protect a string. If the
    whole passage vanished, the filter would be deleting evidence rather than masking a
    value -- and nobody would notice, because the secret is gone either way."""
    state = _setup()
    _store(f'we set the key to {FAKE_KEY} and deploys started working')

    stored = "\n".join(state["texts"])
    assert "deploys started working" in stored, f"context destroyed: {stored[:200]!r}"


def test_redaction_happens_before_the_embedding():
    """⛔ ORDERING IS THE WHOLE POINT. A vector computed from the original still encodes
    the secret, so masking the string while keeping that vector leaves a partial leak
    that LOOKS clean. We capture what the embedder was actually handed."""
    state = _setup(monkey_embed=False)
    seen = []

    def spy(text, prefix="search_document"):
        seen.append(text)
        return np.ones(768, dtype=np.float32)

    ms.embed_text = spy
    _store(f'token {FAKE_KEY} here')

    assert seen, "the embedder was never called, so the ordering was not exercised"
    assert FAKE_KEY not in "\n".join(seen), (
        "THE EMBEDDER SAW THE RAW CREDENTIAL. The string may be masked on disk while the "
        "vector still encodes the secret."
    )


def test_clean_text_is_stored_unchanged():
    """The negative control for the filter's blast radius. A filter that mangles ordinary
    text would pass every test above and be unusable -- and `disk-usage` contains the
    literal 'sk-'."""
    state = _setup()
    clean = "Reviewed the disk-usage-monitor-v2 task-oriented runbook; no changes needed."
    _store(clean)

    assert state["texts"], "nothing stored"
    assert clean in "\n".join(state["texts"]), (
        f"clean text was altered on ingest: {state['texts']!r}"
    )


def test_the_check_is_not_identical_in_both_states():
    """⛔ THE CONTROL FOR THIS FILE ITSELF. If `memory_store` ignored the filter, would
    the assertions above actually fail? Neuter `redact_text` to a pass-through and
    confirm the credential DOES reach the corpus. Without this, a `memory_store` that
    silently dropped all input would look like a working filter."""
    state = _setup()
    original = ms.redact.redact_text
    try:
        ms.redact.redact_text = lambda text, origin="": (text, [])
        _store(f'export KEY="{FAKE_KEY}"')
        stored = "\n".join(state["texts"])
        assert FAKE_KEY in stored, (
            "with the filter neutered the credential STILL did not land, so the passing "
            "tests above do not prove the filter is what stopped it"
        )
    finally:
        ms.redact.redact_text = original


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}")
            except Exception as e:                     # noqa: BLE001
                failures += 1
                print(f"ERROR {name}\n      {type(e).__name__}: {e}")
    print(f"\n{'FAIL' if failures else 'PASS'} -- {failures} failure(s)")
    sys.exit(1 if failures else 0)
