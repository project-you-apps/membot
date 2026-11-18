"""The /v1 surface a consuming application builds against. 2026-11-15.

An external API contract governs these shapes. It marks
every endpoint SHIPPED / TO BUILD / DESIGNED so they can plan against what exists,
and two of its rows overstated:

    POST /v1/notes   marked SHIPPED with NO HTTP ROUTE AT ALL. The capability is
                     real as an MCP tool; the endpoint was not. The adjacent
                     provenance row carried the honest qualifier and this did not.
    rank_by          marked <TO BUILD, small> and not built.

Both are answered in code here rather than by a correction they have to read.

THE CHECKS THAT MATTER, in order:

  V3  the EMPTY paths return `matches`, not `results`. Early returns are where the
      wrong envelope leaks, because they are the paths nobody exercises while
      building the happy one -- and a /v1 caller handed {"results": []} on "no
      cartridge mounted" debugs a parse error instead of reading an empty answer.
  V4  absent facts are null, NEVER invented. document_id and source_target arrive
      with the ingest handoff, which is TO BUILD. A plausible stand-in is
      indistinguishable from a real one to their attribute check.
  V6  a note records the seat the request layer derived. note_add is an @mcp.tool()
      that resolves the seat again, so without the carry it would be refused as
      impersonation and the note attributed to `default`.

⛔ NOT TESTED HERE because it does not exist: GET /v1/documents/{id}/provenance.
   It keys on a document id, and ids are minted by POST /v1/documents (TO BUILD).
"""
import json
import os
import socket
import subprocess
import sys

# ⚠ The cart this runs against is NOT hardcoded. A live demo cart is named for
# whoever it was built for, and that name does not belong in a public test.
# ⭐ Defaults to the cart that SHIPS IN THIS REPO, so the live section actually runs
# for anyone who clones it. ⛔ It used to name a live demo cart built for a specific
# customer -- that name does not belong in a public test -- and a placeholder that
# exists nowhere is no better: the live checks SKIP and the suite still says "passed",
# just with a smaller number nobody reads. Measured 2026-11-18: 14/14 instead of
# 32/32, and 24/24 instead of 27/27.
_TEST_CART = os.environ.get("MEMBOT_TEST_CART", "attention-is-all-you-need")
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import membot_server as ms  # noqa: E402

RESULTS = []
LIVE_RAN = False
PORT = 8091
TAB = "v1test" + str(os.getpid())


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def note(label, detail):
    print(f"  ----  {label}   [{detail}]")


print("\nV1 - the filters translation, as a unit")
# Theirs: {"control_id","period","tags"} -> ours: {"meta", "tags_all"}
spec = ms._v1_filters_to_spec({"control_id": "FR-02", "period": "2026-Q3",
                               "tags": ["access-review"]})
check("control_id and period become meta predicates",
      spec.get("meta") == {"control_id": "FR-02", "period": "2026-Q3"}, str(spec.get("meta")))
check("tags become tags_all (AND)", spec.get("tags_all") == ["access-review"])
check("a bare string tag is accepted as a one-element list",
      ms._v1_filters_to_spec({"tags": "x"}).get("tags_all") == ["x"])
check("no filters -> None, so the engine skips filtering entirely",
      ms._v1_filters_to_spec(None) is None and ms._v1_filters_to_spec({}) is None)
check("empty-string values are dropped rather than matched literally",
      ms._v1_filters_to_spec({"control_id": ""}) is None,
      "an empty predicate would exclude everything")

print("\nV2 - rank_by refusals name the reason")
check("lexical is refused as NOT YET, naming BM25",
      "BM25" in ms._rank_by_error("lexical") and "not built yet" in ms._rank_by_error("lexical"))
check("an unknown value is refused as NOT SUPPORTED, listing what is",
      "not a supported ranking" in ms._rank_by_error("magic"))
check("lexical is absent from the supported set on purpose",
      "lexical" not in ms._RANK_BY_SUPPORTED and
      ms._RANK_BY_SUPPORTED == frozenset({"blend", "cosine", "hamming"}),
      ",".join(sorted(ms._RANK_BY_SUPPORTED)))

print("\nV3 - the empty envelope, as a unit")
for env, key, absent in (("v1", "matches", "results"), ("api", "results", "matches")):
    r = ms._empty_search(env, error="No cartridge mounted")
    body = json.loads(bytes(r.body).decode())
    check(f"envelope={env!r} returns {key!r} and not {absent!r}",
          key in body and absent not in body, ",".join(sorted(body)))
# POSITIVE CONTROL: the assertion above can actually fail.
_v1 = json.loads(bytes(ms._empty_search("v1").body).decode())
check("the check can tell the envelopes apart at all",
      "matches" in _v1 and "results" not in _v1,
      "if both keys appeared, V3 would pass vacuously")

# ---------------------------------------------------------------------------
print(f"\nV4 - live /v1 surface on 127.0.0.1:{PORT}")
proc = None
try:
    devnull = open(os.devnull, "w")
    proc = subprocess.Popen(
        [sys.executable, "membot_server.py", "--transport", "http", "--writable",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, stdout=devnull, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True)
    up = False
    for _ in range(90):
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                up = True
                break
        except OSError:
            time.sleep(2)

    if not up:
        note("LIVE NOT RUN - server did not start", "unit checks above still stand")
    else:
        H = {"Content-Type": "application/json", "X-Membot-Session": TAB}

        def req(method, path, payload=None):
            r = urllib.request.Request(
                f"http://127.0.0.1:{PORT}{path}", method=method, headers=H,
                data=json.dumps(payload).encode() if payload is not None else None)
            try:
                with urllib.request.urlopen(r, timeout=180) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as e:
                try:
                    return e.code, json.loads(e.read())
                except Exception:                           # noqa: BLE001
                    return e.code, {}

        # ⭐ BEFORE mounting: this is the early-return path, the one that leaks.
        code, pre = req("POST", "/v1/search", {"query": "x", "top_k": 2})
        check("empty /v1/search returns `matches` on the LIVE early-return path",
              "matches" in pre and "results" not in pre, ",".join(sorted(pre)))

        code, ver = req("GET", "/v1/version")
        check("/v1/version identifies the program and the build",
              code == 200 and ver.get("name") == "membot" and bool(ver.get("code_digest")),
              f"code_digest={ver.get('code_digest')}")

        code, hl = req("GET", "/v1/health")
        check("/v1/health reports whether the ENGINE is ready, not just liveness",
              "engine_ready" in hl, f"engine_ready={hl.get('engine_ready')}")

        m = req("POST", "/api/mount", {"name": _TEST_CART})[1]
        if "Mounted" not in str(m.get("result", "")):
            note("LIVE NOT RUN - mount did not take", str(m.get("result"))[:60])
        else:
            LIVE_RAN = True
            code, d = req("POST", "/v1/search", {
                "customer_id": "acme", "top_k": 3, "rank_by": "cosine",
                "query": "what does the provenance journal guarantee"})
            matches = d.get("matches", [])
            check("/v1/search returns their `matches` envelope", bool(matches),
                  f"{len(matches)} matches")
            check("ranked_by is echoed, so the ordering is never assumed",
                  d.get("ranked_by") == "cosine", repr(d.get("ranked_by")))
            check("customer_id is echoed AND its scoping stated, not silently ignored",
                  d.get("customer_id") == "acme" and "not enforced" in
                  (d.get("customer_scoping") or ""), d.get("customer_scoping"))

            top = matches[0]
            check("every field the spec names is present",
                  all(k in top for k in ("passage_id", "document_id", "text", "locator",
                                         "score", "signals", "source_target",
                                         "ingested_at")),
                  ",".join(sorted(top)))
            # ⛔ The null discipline. These are unknown until ingest exists.
            check("document_id is null, not invented", top["document_id"] is None)
            check("source_target is null, not invented", top["source_target"] is None)
            check("passage_id IS returned (cart-local until ingest mints ids)",
                  isinstance(top["passage_id"], str) and top["passage_id"].startswith("psg_"),
                  top["passage_id"])
            check("ingested_at is populated from the cart build time",
                  bool(top["ingested_at"]), top["ingested_at"])
            sig = top["signals"]
            check("signals uses THEIR names, incl. `lexical`",
                  set(sig) >= {"cosine", "hamming", "lexical"}, ",".join(sorted(sig)))
            check("`lexical` carries our keyword signal, not a fabricated BM25 score",
                  sig["lexical"] is not None and sig["lexical"] <= 0.12,
                  f"{sig['lexical']} (capped hit-count proxy, documented as such)")

            # Their filters block. Zero is the CORRECT answer here and must be
            # distinguishable from "search found nothing".
            code, f = req("POST", "/v1/search", {
                "query": "journal", "top_k": 3,
                "filters": {"control_id": "FR-02", "period": "2026-Q3"}})
            check("a filter that matches nothing reports filter_matched=0",
                  f.get("filter_matched") == 0,
                  "distinguishes 'filter excluded all' from 'search found nothing'")

            # ⭐ V6: the seat must survive into note_add, which re-resolves it.
            code, n = req("POST", "/v1/notes",
                          {"idx": 2, "kind": "why", "body": "v1 test note"})
            check("/v1/notes writes a note", code == 200 and n.get("status") == "ok",
                  str(n.get("error"))[:50])
            check("the note records the DERIVED seat, not 'default'",
                  n.get("seat") == f"anon_{TAB}", repr(n.get("seat")))
            check("and reports what the attribution is worth",
                  n.get("identity_source") == "claimed", repr(n.get("identity_source")))
            check("privacy is stated on the write, not just documented",
                  "private" in (n.get("visibility") or ""), n.get("visibility"))
            # ⚠ The first version of this asserted the SUBSTRING "share" was absent
            # from the response, and failed -- because note_add's own message tells
            # the caller about `note_share`, which is exactly the guidance it should
            # give. Checking for a word is not checking for a behaviour. ⇒ Send the
            # parameter and require that it changes nothing.
            code2, forced = req("POST", "/v1/notes",
                                {"idx": 2, "kind": "why", "body": "v1 test note, share attempt",
                                 "share": True, "scope": "shared"})
            check("a `share` parameter cannot make a note public through this route",
                  code2 == 200 and "private" in (forced.get("visibility") or ""),
                  f"visibility={forced.get('visibility')!r} -- sharing stays a separate act")

            code, bad = req("POST", "/v1/notes", {"idx": 2})
            check("an incomplete note is refused with 400 and the valid kinds",
                  code == 400 and "kinds" in bad, str(bad.get("error"))[:50])

            code, g = req("GET", "/v1/notes?idx=2")
            check("GET /v1/notes reads it back under the same seat",
                  code == 200 and TAB in str(g.get("notes_raw", "")),
                  f"seat={g.get('seat')!r}")
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 70)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
if not LIVE_RAN:
    print("LIVE SECTION DID NOT RUN. Unit behaviour is checked; the wire is not.")
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
