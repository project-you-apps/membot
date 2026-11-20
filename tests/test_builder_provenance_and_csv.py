"""CSV ingest and per-passage provenance in cartridge_builder.py.

WHY. `/v1/search` has returned `source_file` and `locator.page` since the
client API contract was written, and both read from `per_pattern_meta`.
cartridge_builder.py never wrote that array, so both came back null on every cart
it produced -- including the client demo cart, whose API reference shows a
filename as its example value. Verified 2026-11-18 by opening
demo-controls-evidence.cart.npz: six arrays, no per_pattern_meta.

So this file tests two things that are really one thing: that the provenance rides
with the passage, and that a CSV becomes searchable rows rather than being skipped.

THE CONTROL IS THE POINT. A test that only shows source_file populated cannot
tell "we fixed it" from "it was never broken" -- and the same test would have
passed against the old builder if it had been written loosely. Section G builds a
cart the OLD way (no per_pattern_meta) and requires source_file to be null there.
If G fails, every PASS above it is meaningless and the verdict says so.

No embedding model is needed: embeddings are random, because nothing here is about
ranking. Runs in about a second.

Run:  python tests/test_builder_provenance_and_csv.py
"""
import io
import json
import math
import os
import shutil
import sys
import tempfile

MEMBOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMBOT)

# A Windows console is cp1252. Keep every print ASCII AND wrap stdout, because a
# test that dies printing its own first result reads as a failing test.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

import cartridge_builder as cb  # noqa: E402
from cart_io import load_npz_cartridge  # noqa: E402

RESULTS = []
NOTES = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


# --------------------------------------------------------------- the fixture
CSV_ROWS = 95
HEADER = ["control_id", "owner", "reviewed_on", "exceptions", "amount_usd"]


def write_fixture(root):
    sub = os.path.join(root, "FY26-Q3", "ITGC-01")
    os.makedirs(sub, exist_ok=True)

    csv_path = os.path.join(sub, "access-review-export.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(HEADER) + "\n")
        for i in range(CSV_ROWS):
            fh.write(f"ITGC-01,control.owner@example.com,2026-09-{(i % 28) + 1:02d},{i % 3},{1000 + i * 7}\n")

    md_path = os.path.join(sub, "control-narrative.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# ITGC-01 user access review\n\n"
                 "The control owner reviews system access each quarter and evidences\n"
                 "the review with an export from the source system.\n")

    # Not in SUPPORTED_EXTS. Must be skipped, and its absence is itself a check:
    # if the walk started accepting everything, CSV support would look like it
    # worked for the wrong reason.
    with open(os.path.join(sub, "notes.rtf"), "w", encoding="utf-8") as fh:
        fh.write("{\\rtf1 ignore me}")

    return csv_path, md_path


def main():
    root = tempfile.mkdtemp(prefix="builder_prov_")
    try:
        csv_path, md_path = write_fixture(root)

        # ------------------------------------------------- A. the file walk
        print("\nA - the folder walk sees the CSV and still refuses the unsupported file")
        check("csv is in SUPPORTED_EXTS", ".csv" in cb.SUPPORTED_EXTS,
              f"{sorted(cb.SUPPORTED_EXTS)}")
        check("rtf is NOT (the walk is a whitelist, not a catch-all)",
              ".rtf" not in cb.SUPPORTED_EXTS)
        docs = cb.read_folder(root, recursive=True)
        names = sorted(os.path.basename(n) for n, _ in docs)
        check("walk returned exactly the csv and the md", names ==
              ["access-review-export.csv", "control-narrative.md"], f"{names}")
        rels = sorted(n.replace(os.sep, "/") for n, _ in docs)
        check("and the name carries the folder structure, not just the basename",
              all("FY26-Q3/ITGC-01/" in r for r in rels), f"{rels[0]}")

        # ------------------------------------------- B. the CSV becomes rows
        print("\nB - the CSV parses as a table, with the header on every passage")
        raw = cb.read_file(csv_path)
        check("read_file returned content", bool(raw.strip()), f"{len(raw)} chars")
        check("it uses the PASSAGE_BREAK sentinel, so chunk_text will not cut mid-row",
              "---PASSAGE_BREAK---" in raw)
        passages = cb.chunk_text(raw)
        expected = math.ceil(CSV_ROWS / cb.CSV_ROWS_PER_PASSAGE)
        check(f"{CSV_ROWS} rows at {cb.CSV_ROWS_PER_PASSAGE}/passage -> {expected} passages",
              len(passages) == expected, f"got {len(passages)}")
        header_line = " | ".join(HEADER)
        with_header = sum(1 for p in passages if header_line in p)
        check("EVERY passage carries the header row, not just the first",
              with_header == len(passages), f"{with_header}/{len(passages)}")
        check("and each one names its row range, so a reader knows where they are",
              all("rows " in p.split("\n")[0] for p in passages),
              passages[0].split("\n")[0])
        check("a cell value survived intact", "1000" in passages[0])

        # ------------------------- C. header-only and empty CSVs do not vanish
        print("\nC - degenerate CSVs are not silently dropped")
        hdr_only = os.path.join(root, "schema-only.csv")
        with open(hdr_only, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(HEADER) + "\n")
        out = cb.read_file(hdr_only)
        check("a header-only CSV still yields one passage (it documents the schema)",
              bool(out.strip()) and header_line in out, f"{out[:40]!r}")
        empty = os.path.join(root, "empty.csv")
        open(empty, "w").close()
        check("a truly empty CSV yields nothing, rather than a blank passage",
              cb.read_file(empty).strip() == "")

        # --------------------------------------- D. the provenance records
        print("\nD - per-passage provenance, and what each detail mode stores")
        doc_map = [("FY26-Q3/ITGC-01/access-review-export.csv", 0, 3),
                   ("FY26-Q3/ITGC-01/access-review-export.csv", 1, 3),
                   ("FY26-Q3/ITGC-01/access-review-export.csv", 2, 3),
                   ("FY26-Q3/ITGC-01/control-narrative.md", 0, 1)]

        modes = {}
        for mode in cb.SOURCE_DETAIL_CHOICES:
            modes[mode] = cb.build_per_pattern_meta(doc_map, source_root=root,
                                                    source_detail=mode)
        check("one record per entry, every mode",
              all(len(v) == len(doc_map) for v in modes.values()))
        check("source is ALWAYS the bare filename, in every mode",
              all(r["source"] == "access-review-export.csv"
                  for v in modes.values() for r in v[:3]),
              f"{[v[0]['source'] for v in modes.values()]}")
        check("filename mode stores no path",
              modes["filename"][0]["source_path"] == "access-review-export.csv",
              modes["filename"][0]["source_path"])
        check("relative mode stores the path within the evidence set",
              modes["relative"][0]["source_path"]
              == "FY26-Q3/ITGC-01/access-review-export.csv",
              modes["relative"][0]["source_path"])
        abs_path = modes["absolute"][0]["source_path"]
        check("absolute mode stores the full path on this machine",
              abs_path.endswith("FY26-Q3/ITGC-01/access-review-export.csv")
              and os.path.isabs(abs_path.replace("/", os.sep)), abs_path)
        check("relative mode does NOT leak the machine layout",
              root.replace(os.sep, "/") not in modes["relative"][0]["source_path"])
        check("chunk and chunks come through for the viewer's 'part 2 of 3'",
              [r["chunk"] for r in modes["relative"][:3]] == [0, 1, 2]
              and modes["relative"][0]["chunks"] == 3)
        check("page is None, not 0 -- this builder genuinely does not know it",
              all(r["page"] is None for r in modes["relative"]))
        check("the canonical keys the other two builders write are all present",
              set(["v", "content_type", "source", "page", "chunk", "chunks", "tags",
                   "created_at", "tombstone", "owner", "description"])
              <= set(modes["relative"][0].keys()),
              f"{sorted(modes['relative'][0].keys())}")
        bad = None
        try:
            cb.build_per_pattern_meta(doc_map, source_root=root, source_detail="everything")
        except ValueError as e:
            bad = str(e)
        check("an unknown detail mode is refused, not quietly defaulted", bad is not None,
              (bad or "")[:48])

        # ------------------------------- E. it survives the cart round trip
        print("\nE - the records survive the write and the real loader reads them")
        texts = [f"passage {i}" for i in range(len(doc_map))]
        embs = np.random.rand(len(texts), 768).astype(np.float32)
        ppm = modes["relative"]
        outdir = os.path.join(root, "carts")
        cart_path, _, _ = cb.save_cartridge(outdir, "prov-test", embs, texts,
                                            per_pattern_meta=ppm)
        check("cart written", os.path.exists(cart_path))
        with np.load(cart_path, allow_pickle=True) as z:
            check("per_pattern_meta IS a member of the npz now",
                  "per_pattern_meta" in list(z.keys()), f"{list(z.keys())}")
        loaded = load_npz_cartridge(cart_path)
        got = loaded.get("per_pattern_meta")
        check("the real loader returns a list of dicts, not a string",
              isinstance(got, list) and got and isinstance(got[0], dict),
              f"{type(got).__name__}")
        check("with one record per passage", got is not None and len(got) == len(texts),
              f"{len(got) if got else 0}")
        check("and source round-tripped unchanged",
              got and got[0]["source"] == "access-review-export.csv",
              got[0].get("source") if got else None)
        check("and source_path round-tripped unchanged",
              got and got[0]["source_path"] == "FY26-Q3/ITGC-01/access-review-export.csv",
              got[0].get("source_path") if got else None)
        mpath = os.path.join(outdir, "prov-test.cart_manifest.json")
        with open(mpath) as fh:
            manifest = json.load(fh)
        check("the manifest says the cart carries provenance, readably",
              manifest.get("has_per_pattern_meta") is True, f"{manifest.get('has_per_pattern_meta')}")

        # ----------------------- F. what /v1/search would actually return
        print("\nF - the field /v1/search populates is no longer null")
        try:
            from membot_server import _v1_match
            ran_live = True
        except Exception as e:  # noqa: BLE001
            _v1_match = None
            ran_live = False
            NOTES.append(f"could not import _v1_match ({type(e).__name__}: {e}); "
                         f"section F fell back to the documented lookup chain")

        if ran_live:
            state = {"per_pattern_meta": got, "hippocampus": None}
            m = _v1_match({"index": 0, "full_text": texts[0], "score": 0.5}, state, None)
            check("source_file is populated by the REAL _v1_match",
                  m.get("source_file") == "access-review-export.csv", f"{m.get('source_file')!r}")
            check("content_type comes through too", m.get("content_type") == "document",
                  f"{m.get('content_type')!r}")
            check("locator stays absent while page is unknown, rather than page 0",
                  m.get("locator") is None, f"{m.get('locator')!r}")
        else:
            print("    LIVE SECTION DID NOT RUN - membot_server could not be imported")
            rec = got[0] if got else {}
            resolved = rec.get("source") or rec.get("filename") or rec.get("source_file")
            check("source_file resolves via the documented chain (NOT the live function)",
                  resolved == "access-review-export.csv", f"{resolved!r}")

        # ------------------------------------------------- G. THE CONTROL
        print("\nG - CONTROL: a cart built the OLD way must still return null")
        old_path, _, _ = cb.save_cartridge(outdir, "prov-control", embs, texts)
        with np.load(old_path, allow_pickle=True) as z:
            check("the control cart has NO per_pattern_meta member",
                  "per_pattern_meta" not in list(z.keys()), f"{list(z.keys())}")
        old_loaded = load_npz_cartridge(old_path)
        check("and the loader reports it as absent", old_loaded.get("per_pattern_meta") is None,
              f"{type(old_loaded.get('per_pattern_meta')).__name__}")
        if ran_live:
            old_state = {"per_pattern_meta": old_loaded.get("per_pattern_meta"),
                         "hippocampus": None}
            om = _v1_match({"index": 0, "full_text": texts[0], "score": 0.5}, old_state, None)
            check("source_file IS null on the old shape - so section F measured the fix",
                  om.get("source_file") is None, f"{om.get('source_file')!r}")
        with open(os.path.join(outdir, "prov-control.cart_manifest.json")) as fh:
            check("and its manifest says so",
                  json.load(fh).get("has_per_pattern_meta") is False)

        # --------------------------------- H. the positional-mismatch guard
        print("\nH - a mismatched meta array is refused rather than written")
        err = None
        try:
            cb.save_cartridge(outdir, "prov-mismatch", embs, texts,
                              per_pattern_meta=ppm[:2])
        except ValueError as e:
            err = str(e)
        check("short per_pattern_meta raises instead of misattributing passages",
              err is not None, (err or "")[:60])
        check("and no cart was left behind",
              not os.path.exists(os.path.join(outdir, "prov-mismatch.cart.npz")))

        # ------------------------------- I. the read-time disclosure toggle
        print("\nI - source_path is disclosed only when the deployment says so")
        if ran_live:
            import membot_server as ms
            original = ms.SOURCE_DISCLOSURE
            try:
                state = {"per_pattern_meta": got, "hippocampus": None}
                entry = {"index": 0, "full_text": texts[0], "score": 0.5}

                ms.SOURCE_DISCLOSURE = "filename"
                m_tight = ms._v1_match(entry, state, None)
                check("filename mode withholds source_path even though the cart has one",
                      m_tight.get("source_path") is None, f"{m_tight.get('source_path')!r}")
                check("...and still returns source_file, so the viewer is not broken by it",
                      m_tight.get("source_file") == "access-review-export.csv",
                      f"{m_tight.get('source_file')!r}")

                ms.SOURCE_DISCLOSURE = "path"
                m_wide = ms._v1_match(entry, state, None)
                check("path mode discloses the stored path",
                      m_wide.get("source_path")
                      == "FY26-Q3/ITGC-01/access-review-export.csv",
                      f"{m_wide.get('source_path')!r}")
                check("the key is PRESENT in both modes, so withheld differs from unimplemented",
                      "source_path" in m_tight and "source_path" in m_wide)

                # ⭐ The narrowing guarantee. An older cart carries no source_path at
                #   all; `path` mode must not manufacture one out of the filename,
                #   which would make a tight build look like a loose one.
                legacy = [{"v": 1, "content_type": "document",
                           "source": "access-review-export.csv", "page": None}]
                m_legacy = ms._v1_match(entry, {"per_pattern_meta": legacy,
                                                "hippocampus": None}, None)
                check("CONTROL: path mode cannot invent a path a cart never stored",
                      m_legacy.get("source_path") is None, f"{m_legacy.get('source_path')!r}")
                check("...while still reporting the filename it does have",
                      m_legacy.get("source_file") == "access-review-export.csv")
            finally:
                ms.SOURCE_DISCLOSURE = original
        else:
            print("    LIVE SECTION DID NOT RUN - membot_server could not be imported")

        print("\nJ - a typo in the setting refuses to start, rather than guessing")
        import subprocess
        env = dict(os.environ)
        env["MEMBOT_SOURCE_DISCLOSURE"] = "paths"   # plural: the plausible typo
        proc = subprocess.run([sys.executable, "-c", "import membot_server"],
                              cwd=MEMBOT, env=env, capture_output=True, text=True)
        combined = (proc.stdout or "") + (proc.stderr or "")
        check("import fails with a nonzero exit on an unknown disclosure value",
              proc.returncode != 0, f"exit {proc.returncode}")
        check("and the message names the bad value and the valid ones",
              "paths" in combined and "filename" in combined,
              combined.strip().splitlines()[-1][:70] if combined.strip() else "(no output)")
        env["MEMBOT_SOURCE_DISCLOSURE"] = "path"
        ok_proc = subprocess.run([sys.executable, "-c", "import membot_server"],
                                 cwd=MEMBOT, env=env, capture_output=True, text=True)
        check("CONTROL: a VALID value still imports, so J is not just a broken import",
              ok_proc.returncode == 0, f"exit {ok_proc.returncode}")

    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 70)
    print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
    for label, ok, detail in RESULTS:
        if not ok:
            print(f"   FAILED: {label}   {detail}")
    for n in NOTES:
        print(f"   NOTE: {n}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
