"""A save through the server must not silently destroy what the mount loaded.

Found 2026-11-19, after legion-claude ran our provenance probe on their own carts and
reported back: "Our being stores memories through memory_store, not
cartridge_builder.py. So the gap isn't only in the builder."

⛔⛔ THE DEFECT. `save_as_npz` wrote embeddings, passages, compressed_texts and a
hardcoded version -- and nothing else. So a cart carrying `hippocampus`,
`pattern0`, `per_pattern_meta` or `sign_bits`, saved through the server, came back
without them. And `save_cartridge` REGENERATES the manifest immediately after, so
the stripped cart then PASSED integrity verification. The seal attested the damaged
version. Nothing anywhere reported a loss.

⇒ What that costs is not cosmetic. `per_pattern_meta` is where /v1/search reads
  `source_file` and `locator.page`. `hippocampus` carries prev/next episodic
  navigation AND the TOMBSTONE flag that hides a passage from every search.

⚠ A second, quieter defect in the same family: `memory_store` appended a passage
  without appending a provenance record, leaving N records against N+1 passages.
  Readers index that array BY POSITION, so passage N+1 took record N's filename.
  Harmless until 2026-11-18 because no cart had the array; live the moment the
  builder started writing it.

⭐⭐ SECTION E IS THE POINT. It saves the OLD way and requires the arrays to be
LOST. Without it, "nothing was lost" could mean the fix works or could mean the
test never had anything to lose. A round-trip test that passes in both states
measures nothing.

Run:  python tests/test_save_roundtrip_is_lossless.py
"""
import io
import os
import sys
import tempfile

MEMBOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMBOT)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from cart_io import load_npz_cartridge  # noqa: E402

RESULTS = []
NOTES = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


N = 6
HIPPO_SIZE = 64


def build_full_cart(path):
    """A cart carrying every member the loader reads, written the way the builder does."""
    rng = np.random.default_rng(11)
    embeddings = rng.standard_normal((N, 768)).astype(np.float32)
    texts = [f"passage {i} body text" for i in range(N)]
    import zlib
    compressed = [np.void(zlib.compress(t.encode("utf-8"), level=9)) for t in texts]

    # Hippocampus rows: (N, 64) uint8. Byte 0 is non-zero so unpack sees real data,
    # and we set a distinctive value per row so a swap would be visible.
    hippo = np.zeros((N, HIPPO_SIZE), dtype=np.uint8)
    for i in range(N):
        hippo[i, 0] = i + 1
        hippo[i, 63] = 0xAB

    pattern0 = np.frombuffer(b"PATTERN0-HEADER" + b"\x00" * 100, dtype=np.uint8)
    ppm = [{"v": 1, "content_type": "document", "source": f"file_{i}.md",
            "source_path": f"FY/CTRL/file_{i}.md", "page": None,
            "chunk": 0, "chunks": 1, "tombstone": False} for i in range(N)]
    sign_bits = np.packbits((embeddings > 0).astype(np.uint8), axis=1)

    np.savez_compressed(
        path,
        embeddings=embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed, dtype=object),
        hippocampus=hippo,
        pattern0=pattern0,
        per_pattern_meta=np.array(ppm, dtype=object),
        sign_bits=sign_bits,
        version="mcp-v4",
    )
    return embeddings, texts, hippo, pattern0, ppm


def members(path):
    with np.load(path, allow_pickle=True) as z:
        return set(z.keys())


def main():
    tmp = tempfile.mkdtemp(prefix="roundtrip_")
    src = os.path.join(tmp, "full.cart.npz")
    embeddings, texts, hippo, pattern0, ppm = build_full_cart(src)

    import membot_server as M

    print("\nA - the fixture really does carry everything")
    m = members(src)
    for k in ("hippocampus", "pattern0", "per_pattern_meta", "sign_bits"):
        check(f"source cart has {k}", k in m)

    print("\nB - the loader now hands back the RAW members, not only the decoded ones")
    d = load_npz_cartridge(src)
    check("hippocampus_raw returned", d.get("hippocampus_raw") is not None,
          str(getattr(d.get("hippocampus_raw"), "shape", None)))
    check("and it is the (N, 64) rows, not the unpacked list",
          getattr(d.get("hippocampus_raw"), "shape", None) == (N, HIPPO_SIZE))
    check("pattern0 returned", d.get("pattern0") is not None)
    check("the decoded hippocampus is still there too", isinstance(d.get("hippocampus"), list))
    check("version preserved on load", d.get("version") == "mcp-v4", str(d.get("version")))

    print("\nC - a save that passes what the load gave loses nothing")
    out = os.path.join(tmp, "resaved.cart.npz")
    M.save_as_npz(
        out, d["embeddings"], d["texts"],
        hippocampus=d.get("hippocampus_raw"),
        pattern0=d.get("pattern0"),
        per_pattern_meta=d.get("per_pattern_meta"),
        sign_bits=np.packbits((d["embeddings"] > 0).astype(np.uint8), axis=1),
        version=d.get("version") or "mcp-v3",
    )
    m2 = members(out)
    for k in ("hippocampus", "pattern0", "per_pattern_meta", "sign_bits"):
        check(f"resaved cart still has {k}", k in m2)

    d2 = load_npz_cartridge(out)
    check("passage count unchanged", len(d2["texts"]) == N, f"{len(d2['texts'])}")
    check("hippocampus bytes are IDENTICAL, not re-packed",
          np.array_equal(d2.get("hippocampus_raw"), hippo))
    check("pattern0 bytes identical", np.array_equal(d2.get("pattern0"), pattern0))
    check("per_pattern_meta round-tripped with its fields",
          d2.get("per_pattern_meta") and
          d2["per_pattern_meta"][3].get("source") == "file_3.md",
          str(d2.get("per_pattern_meta", [{}])[3].get("source") if d2.get("per_pattern_meta") else None))
    check("version NOT downgraded to mcp-v3", d2.get("version") == "mcp-v4", str(d2.get("version")))

    print("\nD - the positional guard refuses a mismatched array rather than writing it")
    err = None
    try:
        M.save_as_npz(os.path.join(tmp, "bad.cart.npz"), d["embeddings"], d["texts"],
                      per_pattern_meta=d["per_pattern_meta"][:-1])
    except ValueError as e:
        err = str(e)
    check("short per_pattern_meta raises", err is not None, (err or "")[:58])
    check("and no file was written",
          not os.path.exists(os.path.join(tmp, "bad.cart.npz")))

    print("\nE - CONTROL: the OLD call shape must still lose everything")
    print("    (if this passes, sections A-D prove nothing)")
    old = os.path.join(tmp, "old_style.cart.npz")
    M.save_as_npz(old, d["embeddings"], d["texts"])   # exactly the pre-fix call
    m3 = members(old)
    lost = [k for k in ("hippocampus", "pattern0", "per_pattern_meta", "sign_bits")
            if k not in m3]
    check("saving the old way DOES lose all four", len(lost) == 4, f"lost: {lost}")
    d3 = load_npz_cartridge(old)
    check("...and the stripped cart still loads without error", d3 is not None)
    check("...with provenance gone", d3.get("per_pattern_meta") is None)
    check("...and the hippocampus gone", d3.get("hippocampus") is None)

    print("\nF - memory_store appends a record, so the array cannot desync")
    sid = f"roundtrip-{os.getpid()}"
    try:
        st = M._get_session(sid)
        st["cartridge_name"] = "roundtrip-test"
        st["texts"] = list(d["texts"])
        st["embeddings"] = d["embeddings"]
        st["binary_corpus"] = None
        st["has_embeddings"] = True
        st["per_pattern_meta"] = [dict(r) for r in d["per_pattern_meta"]]
        st["mount_read_only"] = False
        before_t, before_m = len(st["texts"]), len(st["per_pattern_meta"])
        # ⚠ `memory_store` is wrapped by @mcp.tool() into a FunctionTool, which is
        #   not callable. Reach the underlying function rather than skipping the
        #   section -- a store path left UNVERIFIED is exactly the gap this file
        #   exists to close.
        _store = getattr(M.memory_store, "fn", None) or getattr(
            M.memory_store, "__wrapped__", None) or M.memory_store
        _store("a freshly stored memory", tags="test", session_id=sid)
        after_t, after_m = len(st["texts"]), len(st["per_pattern_meta"])
        check("one passage added", after_t == before_t + 1, f"{before_t}->{after_t}")
        check("AND one record added", after_m == before_m + 1, f"{before_m}->{after_m}")
        check("lengths stay equal, which is what the readers assume",
              after_t == after_m, f"texts={after_t} meta={after_m}")
        check("the new record says where it came from",
              st["per_pattern_meta"][-1].get("source") == "memory_store",
              str(st["per_pattern_meta"][-1].get("source")))
    except Exception as e:  # noqa: BLE001
        NOTES.append(f"section F could not run ({type(e).__name__}: {e}) -- the store "
                     f"path is UNVERIFIED by this run")
        print(f"    SECTION F DID NOT RUN: {type(e).__name__}: {e}")

    print("\nG - hippocampus rows are padded to the passage count, never silently short")
    short = hippo[:4]                      # as if 2 passages were appended after mount
    padded = M._hippo_padded(short, N)
    check("a short array is padded, not dropped", padded is not None and len(padded) == N,
          str(getattr(padded, "shape", None)))
    check("the original rows are untouched", np.array_equal(padded[:4], short))
    check("the padding is zero rows — no links, not tombstoned",
          bool(np.all(padded[4:] == 0)))
    check("None stays None", M._hippo_padded(None, N) is None)
    check("an array LONGER than the passages is refused, not truncated",
          M._hippo_padded(hippo, 3) is None,
          "truncating would guess which rows still match")
    # ⭐ CONTROL: the padded array must still survive a real save/load cycle,
    #    otherwise "padded" only means "the right length in memory".
    pth = os.path.join(tmp, "padded.cart.npz")
    M.save_as_npz(pth, d["embeddings"], d["texts"], hippocampus=padded)
    check("and a padded array round-trips through a save",
          np.array_equal(load_npz_cartridge(pth).get("hippocampus_raw"), padded))

    print("\nH - a materialized sub-cart carries provenance, sliced to its own passages")
    import materialize as _mat
    import subcart as _sc
    # ⚠ A real manifest, not a sketch. The first version of this section used a
    #   made-up dict, materialize refused it, and the section silently reported
    #   "DID NOT RUN" -- which is honest but leaves the path unverified, and an
    #   unverified path is what this whole file exists to object to.
    WANT = [0, 2, 4]
    man = {
        "format": _sc.FORMAT,
        "version": _sc.VERSION,
        "id": "t1",
        "name": "slice",
        "parent": {"cart": "full"},
        "membership": {"mode": "explicit",
                       "keys": [_sc.content_key(texts[i], "") for i in WANT]},
    }
    parent = {"version": 1, "passages": texts, "embeddings": embeddings,
              "links": [], "metadata": {}, "per_pattern_meta": ppm}
    try:
        res = _mat.materialize(man, parent)
        check("the slice has the passages the manifest asked for",
              len(res["passages"]) == len(WANT), f"{len(res['passages'])} of {len(WANT)}")
        got = res.get("per_pattern_meta")
        if res["passages"]:
            check("the slice carries per_pattern_meta", got is not None)
            check("one record per exported passage",
                  got is not None and len(got) == len(res["passages"]),
                  f"{len(got) if got else 0} vs {len(res['passages'])}")
            check("and each record still names ITS OWN passage's file",
                  got is not None and all(
                      r.get("source") in {p.get("source") for p in ppm} for r in got))
        else:
            NOTES.append("materialize resolved 0 passages with this synthetic manifest; "
                         "section H's slicing assertions DID NOT RUN")
            print("    SECTION H DID NOT RUN - 0 passages resolved")
        # ⭐ CONTROL: a parent whose array is the wrong length must be dropped, not sliced.
        bad_parent = dict(parent, per_pattern_meta=ppm[:-1])
        res2 = _mat.materialize(man, bad_parent,
                                key_index={t: i for i, t in enumerate(texts)})
        check("CONTROL: a short parent array is dropped rather than mis-sliced",
              res2.get("per_pattern_meta") is None)
    except Exception as e:  # noqa: BLE001
        NOTES.append(f"section H could not run ({type(e).__name__}: {e}) -- sub-cart "
                     f"export provenance is UNVERIFIED by this run")
        print(f"    SECTION H DID NOT RUN: {type(e).__name__}: {e}")

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
