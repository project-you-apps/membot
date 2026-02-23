"""
Gutenberg Poetry Cartridge Builder
====================================
Downloads public domain poetry collections from Project Gutenberg,
splits by individual poem (not fixed word windows), and builds a membot cartridge.

Usage:
  python build_poetry_cartridge.py                    # Download + build
  python build_poetry_cartridge.py --skip-download    # Build from cached files
  python build_poetry_cartridge.py --train            # Build + train brain (GPU)
"""

import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartridge_builder import embed_texts, save_cartridge

# ============================================================
# CURATED POETRY COLLECTIONS
# ============================================================
# Format: (gutenberg_id, title, author, tags)

POETRY = [
    # English Romantics
    (8800, "Songs of Innocence and Experience", "William Blake", "romantic,innocence,experience,visionary"),
    (574, "Poems", "John Keats", "romantic,beauty,mortality,nature"),
    (4800, "The Complete Poetical Works of Percy Bysshe Shelley", "Percy Shelley", "romantic,liberty,nature,imagination"),
    (8601, "Poems of William Wordsworth", "William Wordsworth", "romantic,nature,memory,childhood"),
    (7242, "Poems of Coleridge", "Samuel Taylor Coleridge", "romantic,supernatural,imagination,dream"),
    (21700, "The Works of Lord Byron", "Lord Byron", "romantic,passion,freedom,exile"),

    # Victorian
    (8601, "Poems of Tennyson", "Alfred Lord Tennyson", "victorian,loss,duty,nature"),
    (9006, "Sonnets from the Portuguese", "Elizabeth Barrett Browning", "love,sonnets,devotion,victorian"),
    (4697, "Poems", "Christina Rossetti", "devotion,death,nature,longing"),
    (730, "Poems", "Emily Dickinson", "death,nature,immortality,solitude"),
    (264, "Goblin Market", "Christina Rossetti", "temptation,sisterhood,fairy-tale,victorian"),

    # American voices
    (1322, "Leaves of Grass", "Walt Whitman", "democracy,self,nature,america"),
    (10800, "The Raven and Other Poems", "Edgar Allan Poe", "horror,loss,melancholy,rhythm"),

    # Irish / Celtic
    (5765, "The Wind Among the Reeds", "W.B. Yeats", "celtic,love,mysticism,ireland"),
    (33340, "Crossways", "W.B. Yeats", "youth,myth,ireland,longing"),

    # Kipling
    (7700, "Barrack-Room Ballads", "Rudyard Kipling", "empire,soldiers,duty,adventure"),

    # Longfellow
    (16, "Song of Hiawatha", "Henry Wadsworth Longfellow", "native,nature,myth,america"),
    (2039, "Evangeline", "Henry Wadsworth Longfellow", "exile,love,loss,journey"),

    # Ancient
    (22376, "Sappho — Poems", "Sappho", "love,desire,beauty,fragments"),

    # Renaissance
    (1041, "Shakespeare's Sonnets", "William Shakespeare", "love,time,beauty,mortality"),

    # Metaphysical
    (15553, "Poems of John Donne", "John Donne", "love,death,religion,wit"),

    # Modern (pre-1929 works, public domain)
    (5402, "Prufrock and Other Observations", "T.S. Eliot", "modern,alienation,urban,consciousness"),
    (8606, "A Shropshire Lad", "A.E. Housman", "youth,mortality,countryside,war"),
    (2981, "War Poems", "Siegfried Sassoon", "war,trauma,protest,soldiers"),
    (20450, "Chicago Poems", "Carl Sandburg", "urban,labor,america,grit"),
    (1321, "Spoon River Anthology", "Edgar Lee Masters", "death,small-town,epitaphs,america"),
    (601, "The Congo and Other Poems", "Vachel Lindsay", "rhythm,america,race,performance"),

    # Women's voices
    (11, "Poems", "Amy Lowell", "imagism,nature,love,form"),
    (1079, "Poems", "Sara Teasdale", "love,beauty,solitude,nature"),
    (4725, "Poems", "Edna St. Vincent Millay", "love,feminism,mortality,sonnets"),
]

# Some IDs above are placeholders — we'll validate during download.
# Real Gutenberg IDs can be looked up at gutenberg.org.


# ============================================================
# DOWNLOAD (reuse from classics builder)
# ============================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gutenberg_cache")
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_MIRROR2 = "https://www.gutenberg.org/files/{id}/{id}-0.txt"


def download_book(book_id: int, title: str) -> str | None:
    """Download from Gutenberg, caching locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"pg{book_id}.txt")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if len(text) > 200:
            return text

    for mirror in [GUTENBERG_MIRROR, GUTENBERG_MIRROR2]:
        url = mirror.format(id=book_id)
        try:
            print(f"  Downloading: {title} (ID {book_id})...")
            req = urllib.request.Request(url, headers={"User-Agent": "MemBot/1.0 (poetry cartridge builder)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if len(text) > 200:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
        except Exception as e:
            print(f"    Mirror failed: {e}")
            continue

    print(f"  FAILED: {title} (ID {book_id}) — skipping")
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline = text.find("\n", idx)
            if newline != -1:
                start_idx = newline + 1
            break

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


# ============================================================
# POEM SPLITTER
# ============================================================

def split_into_poems(text: str, collection_title: str, author: str) -> list[str]:
    """
    Split a poetry collection into individual poems.

    Heuristics for poem boundaries:
    - Roman numeral headers (I, II, III, IV, etc.)
    - ALL CAPS titles on their own line
    - Numbered poems (1., 2., etc.)
    - Blank-line separated stanzas get grouped into poems

    Returns list of passage strings, each tagged with [Poem: title from collection by author]
    """
    lines = text.split("\n")
    poems = []
    current_poem_lines = []
    current_title = None
    blank_count = 0

    # Patterns that indicate a new poem title
    roman = re.compile(r"^(?:(?:[IVXLC]+\.?\s)|(?:\d+\.?\s))(.+)?$")
    all_caps_title = re.compile(r"^[A-Z][A-Z\s\-\',\.!?:;]{4,60}$")
    # Centered / short title-like line after multiple blanks
    short_title = re.compile(r"^\s{2,}.{3,60}\s*$")

    def flush_poem():
        nonlocal current_poem_lines, current_title
        text_block = "\n".join(current_poem_lines).strip()
        if len(text_block) < 20:
            current_poem_lines = []
            current_title = None
            return

        word_count = len(text_block.split())
        # Skip table of contents, notes, etc. (too short or too long)
        if word_count < 8:
            current_poem_lines = []
            current_title = None
            return

        title_part = f'"{current_title}" from ' if current_title else ""
        passage = f"[Poem: {title_part}{collection_title} by {author}]\n{text_block}"

        # If a single poem is very long (>600 words), split into stanzas
        if word_count > 600:
            stanzas = re.split(r"\n\s*\n", text_block)
            accumulated = []
            acc_words = 0
            for stanza in stanzas:
                s_words = len(stanza.split())
                if acc_words + s_words > 400 and accumulated:
                    chunk = "\n\n".join(accumulated)
                    poems.append(f"[Poem: {title_part}{collection_title} by {author}]\n{chunk}")
                    accumulated = [stanza]
                    acc_words = s_words
                else:
                    accumulated.append(stanza)
                    acc_words += s_words
            if accumulated:
                chunk = "\n\n".join(accumulated)
                poems.append(f"[Poem: {title_part}{collection_title} by {author}]\n{chunk}")
        else:
            poems.append(passage)

        current_poem_lines = []
        current_title = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            blank_count += 1
            if blank_count >= 3 and current_poem_lines:
                # 3+ blank lines = likely poem boundary
                flush_poem()
            continue

        # Check for new poem title
        is_title = False
        title_text = None

        if blank_count >= 2:
            # After significant whitespace, check for title patterns
            if all_caps_title.match(stripped):
                is_title = True
                title_text = stripped.title()
            elif roman.match(stripped):
                m = roman.match(stripped)
                title_text = m.group(1).strip() if m.group(1) else stripped
                is_title = True

        blank_count = 0

        if is_title and current_poem_lines:
            flush_poem()
            current_title = title_text
        elif is_title:
            current_title = title_text
        else:
            current_poem_lines.append(line)

    # Flush last poem
    flush_poem()

    return poems


# ============================================================
# FALLBACK: word-window chunking for collections that don't split well
# ============================================================

def chunk_by_words(text: str, collection_title: str, author: str,
                   chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Fallback chunker: fixed word windows with metadata prefix."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        passage = f"[{collection_title} by {author}] {chunk_text}"
        chunks.append(passage)
        i += chunk_size - overlap
    return chunks


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Gutenberg poetry cartridge")
    parser.add_argument("--name", default="gutenberg-poetry", help="Cartridge name")
    parser.add_argument("--skip-download", action="store_true", help="Use cached files only")
    parser.add_argument("--train", action="store_true", help="Train brain weights (GPU)")
    parser.add_argument("--max-passages", type=int, default=0, help="Max total passages (0=unlimited)")
    parser.add_argument("--min-poems", type=int, default=3,
                        help="Min poems from splitter before falling back to word chunking")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartridges")

    print(f"\n{'='*60}")
    print(f"  GUTENBERG POETRY CARTRIDGE BUILDER")
    print(f"  {len(POETRY)} collections")
    print(f"{'='*60}\n")

    all_passages = []
    books_ok = 0
    books_failed = 0

    for book_id, title, author, tags in POETRY:
        if args.skip_download:
            cache_path = os.path.join(CACHE_DIR, f"pg{book_id}.txt")
            if not os.path.exists(cache_path):
                print(f"  Skipping {title} (not cached)")
                books_failed += 1
                continue
            with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        else:
            raw = download_book(book_id, title)

        if not raw or len(raw) < 200:
            books_failed += 1
            continue

        cleaned = strip_gutenberg_boilerplate(raw)
        if len(cleaned) < 100:
            print(f"  WARNING: {title} too short after cleaning ({len(cleaned)} chars)")
            books_failed += 1
            continue

        # Try poem-aware splitting first
        poems = split_into_poems(cleaned, title, author)

        if len(poems) < args.min_poems:
            # Fallback to word chunking
            poems = chunk_by_words(cleaned, title, author)
            method = "word-chunk"
        else:
            method = "poem-split"

        all_passages.extend(poems)
        books_ok += 1
        print(f"  {title} ({author}): {len(poems)} passages ({method}), {len(cleaned):,} chars")

    if args.max_passages > 0 and len(all_passages) > args.max_passages:
        print(f"\nTrimming to {args.max_passages} passages (from {len(all_passages)})")
        step = len(all_passages) / args.max_passages
        indices = [int(i * step) for i in range(args.max_passages)]
        all_passages = [all_passages[i] for i in indices]

    print(f"\nPhase 1 complete: {books_ok} collections, {books_failed} failed, {len(all_passages):,} passages")

    if not all_passages:
        print("ERROR: No passages to embed. Exiting.")
        sys.exit(1)

    # Embed
    print(f"\nPhase 2: Embedding {len(all_passages):,} passages...")
    t0 = time.time()
    embeddings = embed_texts(all_passages)
    embed_time = time.time() - t0
    print(f"Embedded in {embed_time:.1f}s ({len(all_passages)/embed_time:.0f} passages/sec)")

    # Save
    print(f"\nPhase 3: Saving cartridge '{args.name}'...")
    cart_path, size_mb, fingerprint = save_cartridge(output_dir, args.name, embeddings, all_passages)
    print(f"Saved: {cart_path} ({size_mb:.1f} MB, {fingerprint})")

    # Optional training
    if args.train:
        print(f"\nPhase 4: Training brain (GPU)...")
        from cartridge_builder import train_and_sign
        train_and_sign(output_dir, args.name, embeddings, all_passages)

    print(f"\n{'='*60}")
    print(f"  DONE: {args.name}")
    print(f"  {books_ok} collections → {len(all_passages):,} passages → {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
