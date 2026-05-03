# Membot Bugs

Severity classes:
- **A** — blocks shipping, data loss, security, demo-breaking
- **B** — visible product defect, wrong output, broken UX
- **C** — cosmetic, edge-case, low-traffic

Open bugs listed first. Add new entries at the top of "Open"; move to "Closed" with a date and resolution note when fixed.

---

## Open

### C · Cart-mounting UX: highlight should move immediately + disable search during mount
**Reported:** 2026-05-03 (Andy, post-provenance-shipping)
**Affects:** all carts, demo polish.
**Symptom:** When the user clicks a cart chip to mount it, the active highlight stays on the *previously* mounted cart until the mount completes. The search input remains enabled during mount, allowing queries against an in-flight state.
**Desired behavior:**
1. On click: immediately move the highlight to the clicked cart chip (and remove from prior). Optimistic UI.
2. Disable the search input + Search button while mount is in progress.
3. Re-enable on mount-complete (success or fail). On fail, revert highlight.
**Fix scope:** purely client-side in `mountCart()` in both `_APP_HTML` (membot_server.py) and `membot_app.html`. Toggle a `.mounting` class, set `disabled` on the input + button, restore in the success/error branches. Existing `.cart-chip.mounting` opacity rule already exists; just need to wire the highlight + input disable.

### B-high · Gutenberg-classics chunking: PREV/NEXT passages discontinuous, possible tail truncation
**Reported:** 2026-05-03 (Andy, during membot demo polish session)
**Cart:** `gutenberg-classics.cart.npz`
**Symptom:** Mounting the cart and navigating with PREV/NEXT in the passage modal produces passages that don't flow — there appears to be missing text at the boundaries between sibling chunks. Reads as if the tail of each chunk is being dropped at cart-build time, so consecutive `chunk_n` and `chunk_n+1` aren't actually contiguous in the source text.
**Affects:** Reader trust on the public demo. PREV/NEXT navigation is supposed to read like the original source; right now it reads like glitches between chunks.
**Hypothesis:** `build_gutenberg_cartridge.py` (or the underlying chunker) has an off-by-one or boundary-handling issue — splitting on a delimiter and not rejoining the delimiter character, or computing a chunk end position that excludes the last word/sentence.
**Repro:** Mount `gutenberg-classics`, search any common term, open a result, click PREV/NEXT a few times, compare consecutive passage tails/heads.
**Fix scope:** investigate the chunker; if confirmed, rebuild `gutenberg-classics.cart.npz` with the corrected chunk boundaries. (Not urgent enough to redo today; logging here so it's not forgotten.)

---

## Closed

(empty)
