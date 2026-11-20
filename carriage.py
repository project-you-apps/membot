"""A carriage: what a reader brings with them onto a cart.

⭐ NAME IS A PLACEHOLDER (Andy, 2026-11-19): `carriage` is held *"until something better comes
along or it earns its place."* Same standing as `Chimera`. ⓘ It won on one argument: **`cart` and
`carriage` are the same word**, both from Latin *carrus*. Every other candidate was a metaphor
imported from web browsing, navigation or accounting and bolted on.

    The cart is the corpus you mount. The carriage is what you bring with you.

WHAT IT IS
----------
An **ask** is a query plus what you expected to find, recorded BEFORE the results come back.
An **outcome** is a separate, linked record saying what actually happened.

⭐⭐ WHY THE EXPECTATION RIDES ON THE QUERY. The `prediction` note kind has existed since before
2026-11-08 and has never once been scored. Not because the mechanism was missing -- because
scoring required somebody to come back later, and that never happens. Attaching the expectation
to a query you were already typing makes the prediction nearly free, and the outcome arrives
milliseconds later in the same breath. ⇒ **The loop closes itself.** That was Andy's reshape of a
design that would not have worked.

⛔⛔ A CARRIAGE IS NEVER AN INPUT TO RETRIEVAL. Two independent reasons:
  · It is the most sensitive artifact in the system -- MORE than the cart. The cart is documents
    someone already had. The carriage is what they were WORRIED about.
  · US11017047 claims cover using trails to establish search results and deeplinks. If a carriage
    never feeds ranking we are outside that purpose entirely, which is a far more robust position
    than arguing about where log data came from.
⇒ And never aggregate carriages across seats without an explicit grant.

⛔ IT LIVES WITH THE READER, NOT ON THE CART. A query is a fact about a reader, not about the
corpus. Write queries into the document being queried and the corpus GROWS BY BEING READ, which
changes what later searches return -- the observer modifying the observed.

⚠ THE POPULATION IS SELF-SELECTED. You record an expectation when you are unsure or testing, not
routinely. ⇒ **Never report a hit rate.** `misses()` exists and a `hit_rate()` deliberately does
not: an average over this population would flatter us, and the wrong ones are the informative ones
anyway.

Stdlib only, like `note_store`. No numpy, no server, no network.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

FORMAT = "vps-carriage"
VERSION = 1
SUFFIX = ".carriage.json"

# What happened, once something looked. Deliberately small.
#   hit        -- what was expected came back
#   miss       -- it did not. ⭐ THE INFORMATIVE ONE
#   mixed      -- partly, or something better came back instead
#   unresolved -- looked, could not tell. NOT the same as unjudged
VALID_DISPOSITIONS = ("hit", "miss", "mixed", "unresolved")

# How the verdict was reached. The PREFIX is load-bearing -- `strength()` parses it.
#   person:<who>    a human said so
#   peer:<who>      a different OWNER's agent said so  (⭐ the fleet)
#   model:<id>      a model under OUR control, pinned by id
#   lexical:<algo>  term overlap. ⭐ The only non-circular ALGORITHMIC judge we have
#   rank:<k>        the expectation named an identifiable target; no judge needed
JUDGE_PREFIXES = ("person", "peer", "model", "lexical", "rank")

MAX_TEXT = 2000


# ----------------------------------------------------------------------------
# ⛔⛔ STRENGTH IS DERIVED, NEVER STORED
# ----------------------------------------------------------------------------
# "Independent" is not a property of a judge. It is a RELATION between the judge and the party
# whose performance is in question. A person judging their own search is fine when they are the
# accountable principal -- their satisfaction IS the outcome. But a person judging whether THEIR
# OWN agent's prediction held is a self-report at the org level, however sincere.
#
# ⇒ So strength cannot be a number written into the record. It has to be computed from (who asked,
#   who judged). Same reason `superseded` is derived from the `supersedes` link and never stored:
#   ⭐ A STORED VERDICT IS AN ASSERTION. A DERIVED ONE IS CHECKABLE.
#
# ⚠ And the ladder is not about how smart the judge is. `rank` is the strongest thing here and
#   involves no intelligence at all -- it is a measurement. A frontier model ranks BELOW a tape
#   measure, because the question is independence, not capability.

STRENGTH_MEASURED = 4     # rank: no judgment was required, so none can be biased
STRENGTH_INDEPENDENT = 3  # a different accountable principal looked
STRENGTH_SELF = 2         # the party being measured, or something it controls, looked
STRENGTH_WEAK = 1         # lexical overlap only
STRENGTH_NONE = 0         # nothing has looked yet


def strength(ask: dict, outcome: dict) -> int:
    """How much this outcome is worth as evidence about this ask.

    ⛔ Takes BOTH records on purpose. Passing only the outcome would make this a property, and
    the whole point is that it is a relation. A caller that wants to rank evidence must hold the
    claim and the verdict together, which is the honest shape of the question.
    """
    if not outcome:
        return STRENGTH_NONE
    judged_by = (outcome.get("judged_by") or "").strip()
    kind, _, who = judged_by.partition(":")

    if kind == "rank":
        return STRENGTH_MEASURED
    if kind == "lexical":
        return STRENGTH_WEAK
    if kind in ("person", "peer", "model"):
        # ⭐ The actual test: is the judge a different principal from the asker?
        asker = _principal(ask.get("by", ""))
        judge = _principal(who)
        if kind == "model":
            # A model we run is OURS. It is a different MECHANISM, not a different principal,
            # so it can break circularity without conferring independence.
            return STRENGTH_SELF
        return STRENGTH_INDEPENDENT if (judge and judge != asker) else STRENGTH_SELF
    return STRENGTH_NONE


def _principal(seat: str) -> str:
    """The accountable party behind a seat id.

    ⚠ Deliberately crude: everything before the first `/`. A seat of `waving-cat/claude` and one
    of `waving-cat/andy` are the SAME principal, which is the answer we need -- our human
    confirming our agent's prediction is not independent corroboration. `legion/claude` is a
    different one.
    """
    return (seat or "").strip().split("/")[0].lower()


# ----------------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------------

def carriage_path(dirpath, seat: str) -> Path:
    """Where a seat's carriage lives. ⭐ Reader-side: the caller passes a READER directory.

    ⛔ Never derived from the cart path. That was the first design and it is wrong -- see the
    module docstring on the observer modifying the observed.
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (seat or "anon"))
    return Path(dirpath) / f"{safe}{SUFFIX}"


def load(dirpath, seat: str) -> dict:
    """Everything this seat has asked. NEVER raises.

    `readable` is load-bearing, copied from `note_store.load` for the same reason: "this reader
    asked nothing" and "we cannot read what they asked" are DIFFERENT FACTS. Collapsing them
    would turn a corrupt file into a confident, false "no expectations were recorded."
    """
    path = carriage_path(dirpath, seat)
    empty = {"format": FORMAT, "version": VERSION, "seat": seat,
             "asks": [], "outcomes": [], "readable": True}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {**empty, "readable": False, "error": str(e)}
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        fmt = data.get("format") if isinstance(data, dict) else type(data).__name__
        return {**empty, "readable": False, "error": f"unexpected format {fmt!r}"}
    for key in ("asks", "outcomes"):
        if not isinstance(data.get(key), list):
            return {**empty, "readable": False, "error": f"{key} is not a list"}
    data["readable"] = True
    return data


def _write(dirpath, seat: str, data: dict) -> None:
    """Atomic replace. A half-written carriage reads as unreadable, which blocks appends, so the
    write has to be all-or-nothing rather than merely usually fine."""
    path = carriage_path(dirpath, seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".carriage-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mint(prefix: str, *parts) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{h[:12]}"


def _text(value, field: str) -> str:
    value = (value or "").strip()
    if len(value) > MAX_TEXT:
        raise ValueError(f"{field} is {len(value)} chars, max {MAX_TEXT}")
    return value


# ----------------------------------------------------------------------------
# recording
# ----------------------------------------------------------------------------

def record_ask(dirpath, seat: str, surface: str, query: str,
               expectation: str = "", expect_ref: str = "",
               cart: str = "", now: float = 0.0) -> dict:
    """Record a query and what the asker expected, BEFORE results exist.

    ⚠ `expectation` is prose and needs a judge. `expect_ref` names something identifiable -- a
    source file, a passage key -- and needs NONE: the outcome is just where it landed. ⭐ Prefer
    `expect_ref` wherever the asker can supply it, because a measurement cannot be talked into
    agreeing with itself.

    ⛔ An ask with neither is refused. A query with no expectation is just a query, and recording
    it here would quietly turn a falsifiable instrument into a search log -- which is the thing
    the patent covers and the thing we promised readers we would not keep.
    """
    now = now or time.time()
    query = _text(query, "query")
    expectation = _text(expectation, "expectation")
    expect_ref = _text(expect_ref, "expect_ref")
    if not query:
        raise ValueError("an ask needs a query")
    if not expectation and not expect_ref:
        raise ValueError(
            "an ask needs an expectation or an expect_ref -- a query with neither is a search log")

    data = load(dirpath, seat)
    if not data.get("readable"):
        raise OSError(f"carriage unreadable, refusing to append: {data.get('error')}")

    ask = {
        "id": _mint("ask", query, expectation, expect_ref, seat, now),
        "query": query,
        "expectation": expectation,
        "expect_ref": expect_ref,
        "cart": cart,
        "by": seat,
        "surface": surface,
        "asked_at": now,
    }
    data["asks"].append(ask)
    _write(dirpath, seat, data)
    return ask


def record_outcome(dirpath, seat: str, surface: str, answers: str,
                   disposition: str, judged_by: str, detail: str = "",
                   now: float = 0.0) -> dict:
    """Record what actually happened, as a SEPARATE record linked to the ask.

    ⛔⛔ `answers` is a LINK, and the outcome is never written onto the ask. Settled 2026-11-18 for
    `superseded` and it binds here for the same reason: *"the link is what makes it checkable. A
    flag is just an assertion."* Store a verdict on the ask and you get expectations marked
    "confirmed" with nothing confirming them. Store it as a link and "show me every ask with no
    outcome yet" is a QUERY rather than a hope.

    ⭐ A verdict IS legitimate HERE, because this record is the confirming artifact. The rule is
    not "never store a verdict" -- it is **never store a verdict on the thing it judges.**
    """
    now = now or time.time()
    if disposition not in VALID_DISPOSITIONS:
        raise ValueError(f"disposition {disposition!r} not in {VALID_DISPOSITIONS}")
    judged_by = _text(judged_by, "judged_by")
    prefix = judged_by.partition(":")[0]
    if prefix not in JUDGE_PREFIXES:
        raise ValueError(f"judged_by must start with one of {JUDGE_PREFIXES}, got {judged_by!r}")
    if ":" not in judged_by:
        raise ValueError(f"judged_by needs a who/what after the colon, got {judged_by!r}")

    data = load(dirpath, seat)
    if not data.get("readable"):
        raise OSError(f"carriage unreadable, refusing to append: {data.get('error')}")
    if not any(a.get("id") == answers for a in data["asks"]):
        raise KeyError(f"no ask {answers!r} in this carriage")

    outcome = {
        "id": _mint("out", answers, disposition, judged_by, now),
        "answers": answers,
        "disposition": disposition,
        "judged_by": judged_by,
        "detail": _text(detail, "detail"),
        "by": seat,
        "surface": surface,
        "at": now,
    }
    data["outcomes"].append(outcome)
    _write(dirpath, seat, data)
    return outcome


def rank_outcome(dirpath, seat: str, surface: str, answers: str,
                 result_refs, now: float = 0.0) -> dict:
    """Score an `expect_ref` ask by WHERE the expected thing landed. No judge involved.

    ⭐⭐ This is the strongest tier precisely because nothing judges. `result_refs` is the ordered
    list of what came back; the outcome is the 1-based rank of the expected ref, or a miss. It is
    the same instrument we point at LongMemEval (`answer_session_rank`, R@1/R@5/R@15), turned on a
    real query against a real corpus.

    ⛔ Refuses an ask that has only prose. Guessing which result "looks like" the expectation is
    exactly the circular judgment this tier exists to avoid.
    """
    now = now or time.time()
    data = load(dirpath, seat)
    ask = next((a for a in data["asks"] if a.get("id") == answers), None)
    if ask is None:
        raise KeyError(f"no ask {answers!r} in this carriage")
    ref = (ask.get("expect_ref") or "").strip()
    if not ref:
        raise ValueError(
            f"ask {answers!r} has no expect_ref -- rank scoring needs an identifiable target, "
            "and inferring one from prose is the circularity this tier avoids")

    refs = [str(r).strip() for r in (result_refs or [])]
    if ref in refs:
        k = refs.index(ref) + 1
        return record_outcome(dirpath, seat, surface, answers, "hit",
                              f"rank:{k}", f"expected {ref!r} at rank {k} of {len(refs)}", now)
    return record_outcome(dirpath, seat, surface, answers, "miss",
                          f"rank:0", f"expected {ref!r} absent from {len(refs)} results", now)


# ----------------------------------------------------------------------------
# ⭐ derived views -- nothing below is stored
# ----------------------------------------------------------------------------

def outcomes_for(data: dict, ask_id: str) -> list:
    """Every outcome pointing at this ask, oldest first.

    ⚠ A list, not one record. An outcome can be revised -- a second look may disagree with the
    first -- and keeping both is what makes the revision auditable instead of invisible.
    """
    return [o for o in data.get("outcomes", []) if o.get("answers") == ask_id]


def open_asks(data: dict) -> list:
    """Asks nothing has looked at yet.

    ⭐⭐ THE POINT OF THE WHOLE MODULE. "We have never once scored a prediction" stops being a
    confession in an email and becomes a number the system reports about itself.
    """
    answered = {o.get("answers") for o in data.get("outcomes", [])}
    return [a for a in data.get("asks", []) if a.get("id") not in answered]


def misses(data: dict) -> list:
    """Every ask whose most recent outcome says it did not come back.

    ⭐ There is deliberately no `hit_rate()`. The population here is self-selected -- you record
    an expectation when you are unsure or testing -- so an average over it measures our mood, not
    our retrieval. ⇒ Surface the misses; they are the informative ones.
    """
    out = []
    for ask in data.get("asks", []):
        got = outcomes_for(data, ask.get("id"))
        if got and got[-1].get("disposition") in ("miss", "mixed"):
            out.append({"ask": ask, "outcome": got[-1],
                        "strength": strength(ask, got[-1])})
    return out


def corroborated(data: dict, min_strength: int = STRENGTH_INDEPENDENT) -> list:
    """Asks with an outcome that is worth something as evidence.

    ⚠ Default is STRENGTH_INDEPENDENT, so our own agent confirming our own prediction does NOT
    appear. That is the entire discipline: a signed self-report is still a self-report.
    """
    out = []
    for ask in data.get("asks", []):
        for got in outcomes_for(data, ask.get("id")):
            if strength(ask, got) >= min_strength:
                out.append({"ask": ask, "outcome": got,
                            "strength": strength(ask, got)})
                break
    return out


def summary(data: dict) -> dict:
    """What this carriage knows about itself. ⛔ Counts, never rates."""
    asks = data.get("asks", [])
    by_disp = {}
    for ask in asks:
        got = outcomes_for(data, ask.get("id"))
        key = got[-1].get("disposition") if got else "unjudged"
        by_disp[key] = by_disp.get(key, 0) + 1
    return {
        "readable": data.get("readable", True),
        "asks": len(asks),
        "outcomes": len(data.get("outcomes", [])),
        "open": len(open_asks(data)),
        "by_disposition": by_disp,
        "misses": len(misses(data)),
        "corroborated": len(corroborated(data)),
    }


def discard(dirpath, seat: str) -> bool:
    """Delete this carriage. ⭐ THE DEFAULT ON UNMOUNT unless the reader says otherwise.

    ⛔ This is the one destructive operation in the module and it is deliberately the easy path.
    Microsoft's trail work mines trails at population scale to improve ranking for other people.
    A carriage belongs to the reader, they are ASKED whether to keep it, and it is DELETED if they
    say no. Same object, opposite politics -- and the politics are the product.
    """
    path = carriage_path(dirpath, seat)
    if path.exists():
        path.unlink()
        return True
    return False
