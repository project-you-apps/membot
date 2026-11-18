"""`signals` on REST /api/search: the components that produced each score. 2026-11-15.

A customer asked for the score components back ("Please do return the signals"),
to use cosine BOTH for ordering and as a floor for "none found", calibrated against
test cartridges with planted gaps. Their prototype displays no score anywhere, so
this is machine-facing: it is read by their attribute check, not by a person who
would notice a number looking wrong.

That is what sets the bar here. A reported component nobody eyeballs has to be
checkable by arithmetic, so the test that matters is not "is `signals` present" but:

  => ARE THESE THE NUMBERS THAT PRODUCED `score`, or a plausible recomputation?

Reconstruction (SS2) is the check that separates those. Everything else supports it.

TWO TIERS.
  Tier 1  pure functions, no server, always runs. Plumbing and null discipline.
  Tier 2  live /api/search with a cart mounted. Real embeddings, real ranking.
          Reports INCONCLUSIVE and says so loudly when it cannot run -- it never
          reads as a pass. Set MEMBOT_SIGNALS_TEST_URL to point it somewhere.

WHAT THIS TEST DOES NOT ESTABLISH. That cosine is comparable ACROSS queries on an
absolute scale, which is what a fixed "none found" floor actually rests on. That is
a calibration claim and it needs the planted-gap cartridges. SS7 measures and prints
the gap rather than asserting it -- deliberately, because a pass line there would be
a verdict on a claim this test cannot reach.
"""
import json
import math
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import membot_server  # noqa: E402  (import after sys.path setup; precedent: test_bind_and_auth_defaults)

_signals_for = membot_server._signals_for
_signal_component = membot_server._signal_component

RESULTS = []
TIER2_RAN = False
# score is rounded to 4dp, each component to 6dp, so reconstruction can only ever
# agree to about 5e-5. 2e-4 leaves room without hiding a real arithmetic error.
TOL = 2e-4


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def note(label, detail):
    print(f"  ----  {label}   [{detail}]")


def reconstruct(sig, blend):
    """The score `signals` claims to explain, recomputed from the signals alone.

    This is the arithmetic a caller would do. It must be written the way a caller
    would write it -- from the documented contract, NOT by calling back into the
    server's own helper -- or it tests nothing but self-consistency.
    """
    cos, ham = sig["cosine"], sig["hamming"]
    if cos is not None and ham is not None:
        base = (1.0 - blend) * cos + blend * ham
    elif cos is not None:
        base = cos
    elif ham is not None:
        base = ham
    else:
        return None
    rec = 1.0 if sig["recency"] is None else sig["recency"]
    return (base + sig["keyword_boost"]) * rec


print("\nSS1 - the five components, and null where a component did not apply")

# Both similarity components present.
s = _signals_for(0, 0.8, 0.06, 0.75, [0.9], [0.5])
check("all five keys present", set(s) == {"cosine", "hamming", "keyword_boost", "recency", "base"},
      ",".join(sorted(s)))
check("cosine reported from the pre-blend array", s["cosine"] == 0.9)
check("hamming reported from the pre-blend array", s["hamming"] == 0.5)
check("keyword_boost reported", s["keyword_boost"] == 0.06)
check("recency reported as the multiplier", s["recency"] == 0.75)

# No embeddings: cosine must be null, NOT 0.0. This is the one that matters most --
# a caller thresholding on cosine reads 0.0 as "confidently irrelevant" and null as
# "this cart cannot answer that question". Those must not be the same value.
s = _signals_for(0, 0.42, 0.0, None, None, [0.42])
check("cosine is None on the hamming-only path, not 0.0",
      s["cosine"] is None, f"got {s['cosine']!r}")
check("and it survives JSON as null, not 0",
      json.loads(json.dumps(s))["cosine"] is None)
check("base equals hamming when cosine is null", s["base"] == 0.42)
check("recency is None when recency weighting did not apply",
      s["recency"] is None, f"got {s['recency']!r}")

# Index past the end of the sign-bit corpus: hamming did not reach this result.
s = _signals_for(5, 0.7, 0.0, None, [0.0] * 10, [0.3] * 3)
check("hamming is None past the end of the sign-bit corpus",
      s["hamming"] is None, f"got {s['hamming']!r}")

# A real 0.0 must still be reported as 0.0. Null means "did not apply"; it must not
# become the way a genuine zero gets expressed, or the distinction above is lost.
s = _signals_for(0, 0.0, 0.0, None, [0.0], None)
check("a genuine 0.0 cosine is reported as 0.0, not null",
      s["cosine"] == 0.0 and s["cosine"] is not None, f"got {s['cosine']!r}")

check("_signal_component: None array -> None", _signal_component(None, 0) is None)
check("_signal_component: index past end -> None", _signal_component([0.1], 7) is None)


blend = membot_server.HAMMING_BLEND


def server_base(cos, ham):
    """`scores[i]` as the server computes it, for a fixture with these components.

    The fixture must NOT state base_score independently of the components it hands
    in. The first draft of this test did, picked a base inconsistent with its own
    cosine and hamming, and SS2 duly failed -- on the fixture, not the code. In
    rest_search the two cannot disagree, because base_score IS scores[i] and
    scores[i] IS this blend. Deriving it here keeps the fixture honest about that.
    """
    if cos is not None and ham is not None:
        return (1.0 - blend) * cos + blend * ham
    return cos if cos is not None else ham


print("\nSS2 - RECONSTRUCTION: the reported components must produce the reported score")
# The real failure this guards against: `signals` computed by a second, independent
# pass that drifts from the one that actually ranked the results. Nobody looking at
# the JSON would catch it. The arithmetic does.
#                       label                        cos   ham  boost  recency
cases = [
    ("both components + recency",                    0.9,  0.5,  0.06,  0.75),
    ("both components, no recency",                  0.9,  0.5,  0.0,   None),
    ("cosine only",                                  0.9,  None, 0.03,  None),
    ("hamming only",                                 None, 0.42, 0.0,   0.9),
    ("boost at the 0.12 cap",                        0.5,  None, 0.12,  None),
    ("both components, boost 0, recency at floor",   0.6,  0.4,  0.0,   0.5),
]
for label, cos, ham, boost, rec in cases:
    base = server_base(cos, ham)
    sig = _signals_for(0, base, boost, rec,
                       None if cos is None else [cos],
                       None if ham is None else [ham])
    produced = (base + boost) * (1.0 if rec is None else rec)
    rebuilt = reconstruct(sig, blend)
    check(f"reconstructs: {label}",
          rebuilt is not None and math.isclose(rebuilt, produced, abs_tol=TOL),
          f"rebuilt={rebuilt:.6f} vs produced={produced:.6f}" if rebuilt is not None else "None")


print("\nSS3 - POSITIVE CONTROL: the reconstruction check can actually fail")
# A checker that never rejects anything is indistinguishable from a checker that
# matches nothing -- which is exactly how test_bind_and_auth_defaults passed for a
# month while the REST bridge sat on 0.0.0.0. So: hand SS2's own comparison a
# deliberately wrong signals block and require that it REJECTS it.
_base = server_base(0.9, 0.5)
good = _signals_for(0, _base, 0.06, 0.75, [0.9], [0.5])
produced = (_base + 0.06) * 0.75

tampered = dict(good, cosine=0.95)          # a component that did not produce the score
bad = reconstruct(tampered, blend)
check("rejects a tampered cosine",
      not math.isclose(bad, produced, abs_tol=TOL), f"rebuilt={bad:.6f} vs {produced:.6f}")

tampered = dict(good, keyword_boost=0.0)    # the shape of "boost reported as zero always"
bad = reconstruct(tampered, blend)
check("rejects a suppressed keyword_boost",
      not math.isclose(bad, produced, abs_tol=TOL), f"rebuilt={bad:.6f} vs {produced:.6f}")

tampered = dict(good, recency=None)         # the shape of "recency silently dropped"
bad = reconstruct(tampered, blend)
check("rejects a dropped recency",
      not math.isclose(bad, produced, abs_tol=TOL), f"rebuilt={bad:.6f} vs {produced:.6f}")

check("and still accepts the untampered block",
      math.isclose(reconstruct(good, blend), produced, abs_tol=TOL))


# ---------------------------------------------------------------------------
# Tier 2. Live /api/search.
#
# Runs in its OWN session slot, so a cart mounted here is invisible to the default
# slot and this never disturbs whatever the operator had mounted.
#
# ⛔ THE SESSION GOES IN THE BODY, NOT IN X-Membot-Session. The header route is
# broken for mounts on this revision, measured 2026-11-15: /api/mount derives
# `anon_<tab>` server-side and hands it to the mount_cartridge TOOL, where
# seat_identity.resolve_seat refuses the reserved `anon_` prefix as impersonation
# and seats the mount in `default` -- while /api/search reads _get_session()
# directly and honors it. Mount lands in one slot, search looks in another, and the
# caller is told "No cartridge mounted". See FINDING-rest-mount-session-split.
# ⇒ Using the header here would make this test fail for a reason that has nothing
#   to do with signals, which is how a test ends up being "fixed" by weakening it.
# ---------------------------------------------------------------------------
BASE = os.environ.get("MEMBOT_SIGNALS_TEST_BASE", "http://127.0.0.1:8000")
CART = os.environ.get("MEMBOT_SIGNALS_TEST_CART", "attention-is-all-you-need")
SESSION = "signalstest" + str(os.getpid())   # no reserved prefix: see above
Q1 = os.environ.get("MEMBOT_SIGNALS_Q1", "what does the provenance journal guarantee")
Q2_ENV = os.environ.get("MEMBOT_SIGNALS_Q2", "")
HDRS = {"Content-Type": "application/json"}


def post(path, payload, timeout=60):
    body = dict(payload, session_id=SESSION)
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 method="POST", headers=HDRS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def search(query, top_k=10):
    return post("/api/search", {"query": query, "top_k": top_k})


print(f"\nSS4 - live /api/search at {BASE}, cart '{CART}', session '{SESSION}'")
# ⛔ IDENTIFY THE SERVER BEFORE BELIEVING A NUMBER IT RETURNS.
# The first run of this test pointed at :8000, got a courteous JSON reply from
# /api/status and an empty result set from /api/search, and reported "no results --
# is a cart mounted?". The process on :8000 was Vector Plus Studio, which serves
# routes of the same names over a different engine. An endpoint that ANSWERS is not
# the endpoint you meant, and the misreading cost more than the check does: mount
# 404ing is what gave it away, so that is the probe.
r1 = None
try:
    m = post("/api/mount", {"name": CART}, timeout=180)
    # ⚠ `status: ok` is NOT evidence the mount happened. rest_mount wraps the tool's
    # return string and reports ok regardless of what that string says, so the
    # outcome has to be read out of `result`. Taking the envelope for the outcome is
    # how you end up searching an empty session and blaming the search.
    mres = str(m.get("result", ""))
    if m.get("status") != "ok" or "Mounted" not in mres:
        note("TIER 2 NOT RUN - mount did not take",
             (str(m.get("error")) or mres or "no result string")[:70])
    else:
        seat = mres.split("Session:")[-1].strip().rstrip(".") if "Session:" in mres else "?"
        print(f"  ----  mounted, seat={seat!r} (default slot untouched)")
        r1 = search(Q1)
except urllib.error.HTTPError as e:
    if e.code == 404:
        note("TIER 2 NOT RUN - NOT A MEMBOT SERVER",
             f"{BASE}/api/mount is 404. Something else is on this port.")
    else:
        note("TIER 2 NOT RUN - HTTP error", f"{e.code} {str(e.reason)[:50]}")
except (urllib.error.URLError, ConnectionError) as e:
    note("TIER 2 NOT RUN - nothing listening", f"{type(e).__name__}: {str(e)[:55]}")
except (TimeoutError, OSError) as e:
    note("TIER 2 NOT RUN - server did not answer in time", f"{type(e).__name__}: {str(e)[:50]}")
except Exception as e:  # noqa: BLE001
    note("TIER 2 NOT RUN - request failed", f"{type(e).__name__}: {str(e)[:55]}")

# ⇒ ORDER MATTERS. The empty/error check runs FIRST because the early-return paths
# in rest_search ("No cartridge mounted", "filter eliminated all") legitimately
# carry no `scoring` block. Checking for `scoring` first reported a mounted-state
# problem as "this build predates signals" -- a true statement about the response
# pointing at the wrong cause, which is the more expensive kind of wrong.
if r1 is not None and not r1.get("results"):
    note("TIER 2 NOT RUN - no results", r1.get("error") or f"'{Q1}' matched nothing in {CART}")
    r1 = None

if r1 is not None and "scoring" not in r1:
    # Results came back down the full path, so the absence is real: this build does
    # not have the signals change.
    note("TIER 2 NOT RUN - membot predates `signals`",
         "results but no `scoring` block; restart the server on this revision")
    r1 = None

if r1 is None:
    print("  ----  Tier 1 covers plumbing and null discipline ONLY.")
    print("  ----  The claims that need real embeddings are UNVERIFIED by this run.")
else:
    TIER2_RAN = True
    res1 = r1["results"]
    scoring = r1.get("scoring") or {}
    live_blend = scoring.get("hamming_blend")

    check("response carries a `scoring` block", bool(scoring), ",".join(sorted(scoring)))
    check("hamming_blend is in it", live_blend is not None, f"{live_blend!r}")
    check("every result carries `signals`", all("signals" in x for x in res1),
          f"{sum('signals' in x for x in res1)}/{len(res1)}")

    # PRECONDITIONS. Every assertion below is a verdict on this result set, so the
    # set has to be capable of failing them. A run where every boost is 0 and every
    # recency is null would "pass" reconstruction while proving nothing.
    print("\nSS5 - preconditions (asserted, not assumed)")
    sigs = [x["signals"] for x in res1 if "signals" in x]
    n_cos = sum(1 for s in sigs if s.get("cosine") is not None)
    n_ham = sum(1 for s in sigs if s.get("hamming") is not None)
    n_boost = sum(1 for s in sigs if s.get("keyword_boost"))
    n_rec = sum(1 for s in sigs if s.get("recency") is not None)
    check("at least one result has a non-null cosine", n_cos > 0, f"{n_cos}/{len(sigs)}")
    check("at least one result has a NONZERO keyword_boost", n_boost > 0,
          f"{n_boost}/{len(sigs)} -- reconstruction is degenerate without one")
    note("non-null hamming", f"{n_ham}/{len(sigs)}")

    # ⛔ RECENCY IS INERT REPO-WIDE, and the reason is not "this cart has no
    # timestamps" -- that was the first explanation written here and it is wrong.
    # `has_recency` reads state["hippocampus_meta"], which is READ at exactly one
    # line and WRITTEN NOWHERE: it is not a key in _new_session, and mount sets only
    # state["hippocampus"]. The read is a .get(), so a key that never existed comes
    # back None silently. => The recency multiplier in /api/search has never applied
    # on any cart, and the same line is in the 2026-06-10 droplet snapshot.
    # The field ships as null and is documented as reserved. See
    # FINDING-recency-never-applies.
    # If this check ever reports non-null, recency went live and SS6 now covers it.
    check("recency is null on every result, as expected while it is unwired",
          n_rec == 0, f"{n_rec}/{len(sigs)} non-null")

    print("\nSS6 - reconstruction against real scores")
    worst, worst_at = 0.0, None
    n_checked = 0
    for x in res1:
        sig, score = x.get("signals"), x.get("score")
        if not sig or score is None:
            continue
        rebuilt = reconstruct(sig, live_blend)
        if rebuilt is None:
            continue
        n_checked += 1
        d = abs(rebuilt - score)
        if d > worst:
            worst, worst_at = d, x.get("index")
    check(f"all {n_checked} live results reconstruct from their signals",
          n_checked > 0 and worst <= TOL,
          f"worst delta {worst:.2e} at index {worst_at} (tol {TOL:.0e})")

    print("\nSS7 - cross-query behaviour")
    # Q2 is derived from Q1's own top result when not given explicitly: real words
    # from a retrieved passage are GUARANTEED to overlap it, so the keyword boost
    # differs between the two queries by construction. Picking two queries by hand
    # and hoping they differ is how you end up measuring a degenerate pair.
    if Q2_ENV:
        Q2 = Q2_ENV
    else:
        words = [w.strip(".,:;()[]#|`*") for w in res1[0].get("full_text", "").split()]
        words = [w for w in words if len(w) >= 4 and w.isalpha()]
        Q2 = " ".join(words[:6]) or Q1
    print(f"  ----  Q1: {Q1!r}")
    print(f"  ----  Q2: {Q2!r}")
    r2 = search(Q2)
    res2 = {x["index"]: x for x in r2.get("results", []) if "signals" in x}
    common = [x for x in res1 if x.get("index") in res2]

    if not common:
        note("no documents common to both queries", "cross-query checks NOT RUN")
    else:
        # recency is a function of the DOCUMENT's age, not of the query. If it ever
        # differs for the same document across two queries, it is wired to the wrong
        # index -- a bug no amount of eyeballing the JSON would surface.
        drift = [(x["index"], x["signals"]["recency"], res2[x["index"]]["signals"]["recency"])
                 for x in common
                 if x["signals"]["recency"] != res2[x["index"]]["signals"]["recency"]]
        check(f"recency is query-independent across {len(common)} shared documents",
              not drift, f"{len(drift)} drifted: {drift[:2]}" if drift else "identical")

        # The headline claim, MEASURED and printed -- not asserted. Whether cosine is
        # stable enough across queries to carry an absolute "none found" floor is a
        # calibration question for the planted-gap cartridges. Printing the numbers is
        # honest; a PASS line here would be a verdict this run cannot support.
        dc = [abs(x["signals"]["cosine"] - res2[x["index"]]["signals"]["cosine"])
              for x in common
              if x["signals"]["cosine"] is not None
              and res2[x["index"]]["signals"]["cosine"] is not None]
        ds = [abs(x["score"] - res2[x["index"]]["score"]) for x in common]
        if dc:
            note("mean |change in cosine| across the two queries", f"{sum(dc)/len(dc):.4f}")
            note("mean |change in score|  across the two queries", f"{sum(ds)/len(ds):.4f}")
            note("MEASUREMENT, not a pass", "absolute-scale comparability needs the planted-gap carts")

    print("\nSS8 - determinism")
    r1b = search(Q1)
    again = {x["index"]: x["signals"] for x in r1b.get("results", []) if "signals" in x}
    same = [x for x in res1 if x.get("index") in again]
    mismatch = [x["index"] for x in same if x["signals"] != again[x["index"]]]
    check(f"identical signals on a repeat of the same query ({len(same)} compared)",
          bool(same) and not mismatch, f"{len(mismatch)} differed" if mismatch else "")


passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 68)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
if not TIER2_RAN:
    print("TIER 2 DID NOT RUN. Plumbing and null discipline are checked; the live")
    print("ranking path is NOT. This run does not establish that /api/search returns")
    print("correct signals -- only that the functions building them behave.")
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 68)
sys.exit(0 if passed == total else 1)
