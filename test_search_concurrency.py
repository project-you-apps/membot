"""Search concurrency ceiling (S7). 2026-10-18.

Nothing bounded how many searches ran at once. Each is a transformer pass plus an
argsort allocating 8 bytes per pattern per caller -- 272 MB on a 34M corpus, counted
by no budget.

⛔ THE TRAP THIS TEST IS BUILT AGAINST: a ceiling set above the offered load is a
knob at its do-nothing default. It will pass every "does it work" check while
bounding nothing. So §2 drives MORE load than the limit and asserts observed
concurrency never exceeded it, and §3 asserts the limit was actually REACHED -- a
run where peak < limit proves only that the test was too gentle.

Uses a stub workload, not the real embedder: the question is whether the gate holds,
not how fast numpy is. ~20 lines of output.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import membot_server as M          # noqa: E402

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


LIMIT = 3
M._set_search_slots(LIMIT)
M._search_stats["peak_inflight"] = 0
M._search_stats["rejected"] = 0

observed = []
obs_lock = threading.Lock()
live = {"n": 0}


def worker(hold=0.25, timeout=20.0, errors=None):
    try:
        with M.search_slot(timeout=timeout):
            with obs_lock:
                live["n"] += 1
                observed.append(live["n"])
            time.sleep(hold)
            with obs_lock:
                live["n"] -= 1
    except M.SearchBusy:
        if errors is not None:
            errors.append("busy")


print(f"\n§1 · the ceiling is derived, not hardcoded")
d = M._default_search_slots()
check("default is between 2 and 8", 2 <= d <= 8, f"{d} on {os.cpu_count()} cores")
check("limit is settable", M._search_stats["limit"] == LIMIT, f"set to {LIMIT}")

print(f"\n§2 · 12 callers against a limit of {LIMIT}")
threads = [threading.Thread(target=worker) for _ in range(12)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.perf_counter() - t0
peak = max(observed) if observed else 0
check("all 12 completed", len(observed) == 12, f"{len(observed)} ran")
check(f"concurrency NEVER exceeded {LIMIT}", peak <= LIMIT, f"peak observed {peak}")
check("serialized into waves, not all at once", elapsed >= 0.25 * (12 / LIMIT) * 0.8,
      f"{elapsed:.2f}s for 12 x 0.25s at limit {LIMIT}")

print(f"\n§3 · CONTROL — the ceiling must actually BIND")
# A limit never reached bounds nothing. If peak < LIMIT the load was too light and
# every check above is vacuous.
check(f"the limit was REACHED (peak == {LIMIT})", peak == LIMIT,
      f"peak {peak} of {LIMIT}")
check("server-reported peak agrees", M._search_stats["peak_inflight"] == LIMIT,
      f"reported {M._search_stats['peak_inflight']}")

print("\n§4 · a caller that cannot get a slot is TOLD, not hung forever")
M._set_search_slots(1)
errs = []
blocker = threading.Thread(target=worker, kwargs={"hold": 2.0})
blocker.start()
time.sleep(0.2)
t0 = time.perf_counter()
worker(hold=0.0, timeout=0.5, errors=errs)
waited = time.perf_counter() - t0
blocker.join()
check("rejected rather than blocked indefinitely", errs == ["busy"], str(errs))
check("gave up near the timeout, not later", 0.4 <= waited <= 1.5, f"waited {waited:.2f}s")
check("rejection was counted", M._search_stats["rejected"] >= 1,
      f"rejected={M._search_stats['rejected']}")

print("\n§5 · slots are returned on the ERROR path")
M._set_search_slots(2)
before = M._search_stats["inflight"]
for _ in range(5):
    try:
        with M.search_slot(timeout=2.0):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
check("inflight back to baseline after 5 exceptions",
      M._search_stats["inflight"] == before, f"inflight={M._search_stats['inflight']}")
# If slots leaked, this acquire would fail.
ok = True
try:
    with M.search_slot(timeout=1.0):
        pass
except M.SearchBusy:
    ok = False
check("slots still acquirable (no leak)", ok)

passed = sum(1 for _, ok_, _ in RESULTS if ok_)
total = len(RESULTS)
print("\n" + "=" * 62)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok_, detail in RESULTS:
    if not ok_:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
