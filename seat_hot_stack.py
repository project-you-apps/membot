"""A seat's Hot Stack: score THIS seat's attention over ONLY what this seat can resolve.

Step 4 of `docs/SPEC-SUBCARTRIDGE-MANIFEST.md` §9, and the piece that makes the rest real.

    cart (shared content)  +  manifest (what I can see)  +  overlay (what I've touched)
                                        |
                                  this seat's Hot Stack

FILTER AT THE INPUT TO SCORING, NEVER AT THE OUTPUT
---------------------------------------------------
Scoring the whole parent and then hiding rows the seat may not read leaks the hierarchy:
a payroll passage still shapes what ranks around it, and gaps in a ranked list are
inferable. So the resolved set is computed FIRST and nothing outside it is ever scored.

The one deliberate exception is the FIRM view (`firm_hot_stack`), which is computed
cart-wide by design -- being firm-wide is that view's entire purpose -- and masked at
presentation. That trade is recorded in the spec rather than defaulted into, and the two
paths are separate functions here so neither can be reached by accident.

THE SCORER IS INJECTED
----------------------
`score_fn` takes the link dict that `overlay.as_link()` already produces, so the existing
`compute_hot_score()` drops in with no change. Passing it rather than importing it keeps
this module stdlib-only and importable by both the VPS API and the Membot server -- and,
more usefully, means the decay model has exactly one implementation instead of a second
copy that must be kept in agreement.
"""

from __future__ import annotations

import time

import overlay as _ov
import subcart as _sc


def scoring_link(cart_link: dict, seat_link: dict) -> dict:
    """Content from the CART, attention from the OVERLAY, merged into one scorer input.

    THIS FUNCTION EXISTS BECAUSE THE FIRST VERSION DID NOT HAVE IT. `overlay.as_link()`
    emits attention as flat keys, but `compute_hot_score` reads
    `link["_reserved"]["canonical"]` for truth_status / kind / important / urgent. Feeding it
    the overlay alone silently fell back to defaults for every one of them: seat #1's Hot
    Stack topped out at 0.92 against the live stack's 2.14, and shared ZERO of its top 8.
    Nothing raised an error -- the scorer simply scored a passage that had no properties.

    The content/attention split is still right. What was wrong was assuming the SCORER
    consumes only one side of it. It consumes both, so the join has to be explicit and has
    to happen somewhere; here is that somewhere.

    Starts from the cart link and overrides only the attention fields, so any property the
    scorer reads -- including ones added later -- is present by default rather than by
    enumeration. Enumerating them is how this broke the first time.
    """
    merged = dict(cart_link)
    merged["access_count"] = seat_link["access_count"]
    merged["accessed_at"] = seat_link["accessed_at"]

    reserved = dict(cart_link.get("_reserved") or {})
    canonical = dict(reserved.get("canonical") or {})
    flags = dict(reserved.get("flags") or {})

    canonical["important"] = seat_link["important"]
    canonical["urgent"] = seat_link["urgent"]
    canonical["to_consolidate"] = seat_link["to_consolidate"]
    if seat_link["agency"]:
        flags["agency"] = seat_link["agency"]

    reserved["canonical"] = canonical
    reserved["flags"] = flags
    merged["_reserved"] = reserved
    return merged


def seat_hot_stack(manifest, texts, links, seat_overlay, score_fn, *,
                   top_k: int = 8, salt: str = "", now: float = 0.0,
                   key_index=None) -> dict:
    """Rank a seat's own attention across the passages that seat can actually resolve.

    Returns {items, unresolved, stats}. `items` are dicts carrying the parent index, the
    content key, the score, and the seat's link -- everything a surface needs, and nothing
    about passages outside the seat's view.
    """
    now = now or time.time()
    resolved = _sc.resolve(manifest, texts, salt, key_index)

    scored = []
    hidden = 0
    for idx in resolved["indices"]:
        key = _sc.content_key(texts[idx], salt)
        # Hidden is this seat saying "noise, to me". Filtered here rather than scored to
        # zero, because a hidden passage should not merely rank low -- it should be absent.
        # Counted, never silently dropped: a hot stack that quietly omits things is exactly
        # as untrustworthy as one that quietly includes them.
        if _ov.is_hidden(seat_overlay, key):
            hidden += 1
            continue
        seat_link = _ov.get(seat_overlay, key)
        # An untouched passage is cold, not absent. Skipping it keeps the seat's view
        # honest: things they could read but never have simply do not rise.
        if not seat_link["access_count"] and not seat_link["accessed_at"]:
            continue
        cart_link = links[idx] if idx < len(links) and isinstance(links[idx], dict) else {}
        scored.append({
            "index": idx,
            "content_key": key,
            "score": float(score_fn(scoring_link(cart_link, seat_link), now)),
            "link": seat_link,
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    return {
        "items": scored[:top_k],
        "unresolved": resolved["unresolved"],
        "stats": {
            "seat": seat_overlay.get("seat"),
            "resolvable": len(resolved["indices"]),
            "scoreable": len(scored),
            "hidden": hidden,
            "unresolved": len(resolved["unresolved"]),
            "returned": min(top_k, len(scored)),
        },
    }


def firm_hot_stack(overlays, texts, links, score_fn, *, top_k: int = 8, salt: str = "",
                   now: float = 0.0, viewer_manifest=None, key_index=None) -> dict:
    """The company brain's own Hot Stack: what the FIRM is warm on, not any one person.

    Andy, 2026-08-02 -- this is the requirement, not a nicety:

        "It keeps an episodic record of the firm's evolution over time as an emergent
         property of the Hot Stack system within the main cart."

    Fifty personal overlays do not sum to an institutional record unless the roll-up is
    deliberately kept, so per-seat attention alone would silently delete the firm's memory
    of itself.

    `seats` -- how many distinct people touched a passage -- is the signal that only exists
    here. One person reading something forty times and forty people reading it once produce
    the same access_count and mean very different things about a company.

    MASKING: pass `viewer_manifest` to restrict the RESULT to what a given viewer may see.
    The ranking is still computed firm-wide, so a viewer's list is shaped by items they
    cannot read. Accepted for this view specifically; see the spec. Omit `viewer_manifest`
    only for an unmasked firm-wide report.
    """
    now = now or time.time()
    firm = _ov.rollup(overlays, now)
    if key_index is None:
        key_index = _sc.build_key_index(texts, salt)

    visible = None
    if viewer_manifest is not None:
        resolved = _sc.resolve(viewer_manifest, texts, salt, key_index)
        visible = {_sc.content_key(texts[i], salt) for i in resolved["indices"]}

    scored = []
    for key, entry in firm["entries"].items():
        if visible is not None and key not in visible:
            continue
        seat_link = _ov.as_link(entry)
        if not seat_link["access_count"] and not seat_link["accessed_at"]:
            continue
        hits = key_index.get(key) or []
        cart_link = links[hits[0]] if hits and isinstance(links[hits[0]], dict) else {}
        scored.append({
            "content_key": key,
            "index": hits[0] if hits else None,
            "score": float(score_fn(scoring_link(cart_link, seat_link), now)),
            "seats": entry.get("seats", 0),
            "link": seat_link,
        })

    scored.sort(key=lambda r: (r["score"], r["seats"]), reverse=True)
    return {
        "items": scored[:top_k],
        "stats": {
            "seats_rolled_up": len(overlays),
            "keys": len(firm["entries"]),
            "visible": len(scored),
            "masked": viewer_manifest is not None,
            "returned": min(top_k, len(scored)),
        },
    }
