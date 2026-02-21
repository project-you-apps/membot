"""
Gutenberg Classics Cartridge Builder
=====================================
Downloads curated public domain books from Project Gutenberg,
strips boilerplate, chunks into passages, and builds a membot cartridge.

Usage:
  python build_gutenberg_cartridge.py                    # Download + build
  python build_gutenberg_cartridge.py --skip-download    # Build from cached files
  python build_gutenberg_cartridge.py --train            # Build + train brain (GPU)

Books are cached in gutenberg_cache/ so subsequent runs skip downloads.
"""

import os
import re
import sys
import time
import urllib.request

# Add parent dir for cartridge_builder imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartridge_builder import chunk_text, embed_texts, save_cartridge

# ============================================================
# CURATED BOOK LIST
# ============================================================
# Format: (gutenberg_id, title, author, theme_tags)
# Chosen for Associate search demos — diverse themes that cross-reference.

BOOKS = [
    # Tragedy & ambition
    (1524, "Hamlet", "Shakespeare", "tragedy,ambition,revenge,madness"),
    (2265, "Macbeth", "Shakespeare", "tragedy,ambition,power,guilt"),
    (1533, "Julius Caesar", "Shakespeare", "tragedy,politics,betrayal,power"),
    (1532, "King Lear", "Shakespeare", "tragedy,madness,family,power"),
    (1120, "Romeo and Juliet", "Shakespeare", "tragedy,love,fate,family"),

    # Power & politics
    (1232, "The Prince", "Machiavelli", "politics,power,strategy,leadership"),
    (132, "The Art of War", "Sun Tzu", "strategy,war,leadership,tactics"),
    (5827, "The Problems of Philosophy", "Bertrand Russell", "philosophy,knowledge,truth,reason"),
    (1497, "Republic", "Plato", "philosophy,justice,politics,education"),

    # Dystopia & freedom
    (174, "The Picture of Dorian Gray", "Wilde", "beauty,corruption,morality,art"),
    (1661, "The Adventures of Sherlock Holmes", "Doyle", "mystery,logic,crime,detective"),

    # Science & exploration
    (84, "Frankenstein", "Shelley", "science,creation,monster,responsibility"),
    (36, "The War of the Worlds", "Wells", "science,invasion,survival,fear"),
    (35, "The Time Machine", "Wells", "science,future,evolution,class"),
    (164, "Twenty Thousand Leagues Under the Sea", "Verne", "science,ocean,exploration,technology"),
    (103, "Around the World in 80 Days", "Verne", "adventure,travel,wager,determination"),

    # Mythology & the supernatural
    (6130, "The Iliad", "Homer", "war,gods,honor,fate"),
    (1727, "The Odyssey", "Homer", "journey,home,gods,cunning"),
    (345, "Dracula", "Stoker", "supernatural,evil,blood,fear"),
    (11, "Alice's Adventures in Wonderland", "Carroll", "fantasy,logic,absurdity,childhood"),
    (46, "A Christmas Carol", "Dickens", "redemption,ghosts,greed,compassion"),

    # Nature of humanity
    (1342, "Pride and Prejudice", "Austen", "love,class,prejudice,wit"),
    (98, "A Tale of Two Cities", "Dickens", "revolution,sacrifice,justice,love"),
    (76, "Adventures of Huckleberry Finn", "Twain", "freedom,race,river,conscience"),
    (74, "The Adventures of Tom Sawyer", "Twain", "childhood,adventure,mischief,freedom"),
    (1260, "Jane Eyre", "Bronte", "independence,love,class,morality"),
    (768, "Wuthering Heights", "Bronte", "passion,revenge,nature,obsession"),

    # Philosophy & morality
    (600, "Notes from the Underground", "Dostoevsky", "alienation,consciousness,rebellion,suffering"),
    (2554, "Crime and Punishment", "Dostoevsky", "guilt,morality,punishment,redemption"),
    (28054, "The Brothers Karamazov", "Dostoevsky", "faith,doubt,family,morality"),

    # Transformation & identity
    (5200, "Metamorphosis", "Kafka", "alienation,transformation,family,absurdity"),
    (1184, "The Count of Monte Cristo", "Dumas", "revenge,justice,patience,identity"),
    (1257, "The Three Musketeers", "Dumas", "honor,friendship,adventure,loyalty"),

    # Exploration & survival
    (120, "Treasure Island", "Stevenson", "adventure,piracy,treasure,loyalty"),
    (43, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Stevenson", "duality,evil,science,identity"),
    (2701, "Moby-Dick", "Melville", "obsession,nature,whale,fate"),

    # Political philosophy
    (7370, "Second Treatise of Government", "Locke", "government,rights,liberty,property"),
    (3207, "Leviathan", "Hobbes", "government,power,social-contract,authority"),

    # Eastern wisdom
    (2680, "Meditations", "Marcus Aurelius", "stoicism,virtue,duty,mortality"),

    # Horror & the unknown
    (215, "The Call of Cthulhu", "Lovecraft", "horror,cosmic,unknown,madness"),
    (209, "The Turn of the Screw", "Henry James", "ghosts,ambiguity,children,evil"),

    # American classics
    (64317, "The Great Gatsby", "Fitzgerald", "wealth,illusion,american-dream,tragedy"),
    (1322, "Leaves of Grass", "Whitman", "poetry,nature,democracy,self"),
    (1952, "The Yellow Wallpaper", "Gilman", "madness,gender,confinement,identity"),
]


# ============================================================
# DOWNLOAD
# ============================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gutenberg_cache")
GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
# Fallback mirror
GUTENBERG_MIRROR2 = "https://www.gutenberg.org/files/{id}/{id}-0.txt"


def download_book(book_id: int, title: str) -> str | None:
    """Download a book from Gutenberg, caching locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"pg{book_id}.txt")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if len(text) > 500:
            return text

    for mirror in [GUTENBERG_MIRROR, GUTENBERG_MIRROR2]:
        url = mirror.format(id=book_id)
        try:
            print(f"  Downloading: {title} (ID {book_id})...")
            req = urllib.request.Request(url, headers={"User-Agent": "MemBot/1.0 (cartridge builder)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if len(text) > 500:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return text
        except Exception as e:
            print(f"    Mirror failed: {e}")
            continue

    print(f"  FAILED: {title} (ID {book_id}) — skipping")
    return None


# ============================================================
# TEXT CLEANING
# ============================================================

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find start of actual text
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
            # Skip past the marker line
            newline = text.find("\n", idx)
            if newline != -1:
                start_idx = newline + 1
            break

    # Find end of actual text
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

    cleaned = text[start_idx:end_idx].strip()

    # Remove excessive blank lines
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)

    return cleaned


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Gutenberg classics cartridge")
    parser.add_argument("--name", default="gutenberg-classics", help="Cartridge name")
    parser.add_argument("--chunk-size", type=int, default=300, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks")
    parser.add_argument("--skip-download", action="store_true", help="Use cached files only")
    parser.add_argument("--train", action="store_true", help="Train brain weights (GPU)")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max total chunks (0=unlimited)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: cartridges/)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartridges")

    print(f"\n{'='*60}")
    print(f"  GUTENBERG CLASSICS CARTRIDGE BUILDER")
    print(f"  {len(BOOKS)} books, chunk_size={args.chunk_size}, overlap={args.overlap}")
    print(f"{'='*60}\n")

    # 1. Download books
    print("Phase 1: Downloading books...")
    all_chunks = []
    books_ok = 0
    books_failed = 0

    for book_id, title, author, tags in BOOKS:
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

        if not raw or len(raw) < 500:
            books_failed += 1
            continue

        # 2. Clean
        cleaned = strip_gutenberg_boilerplate(raw)
        if len(cleaned) < 200:
            print(f"  WARNING: {title} too short after cleaning ({len(cleaned)} chars)")
            books_failed += 1
            continue

        # 3. Chunk with title prefix
        chunks = chunk_text(cleaned, chunk_size=args.chunk_size, overlap=args.overlap)
        for chunk in chunks:
            # Prefix each chunk with book metadata for better search
            passage = f"[{title} by {author}] {chunk}"
            all_chunks.append(passage)

        books_ok += 1
        print(f"  {title} ({author}): {len(chunks)} chunks, {len(cleaned):,} chars")

    if args.max_chunks > 0 and len(all_chunks) > args.max_chunks:
        print(f"\nTrimming to {args.max_chunks} chunks (from {len(all_chunks)})")
        # Sample evenly across books rather than truncating
        step = len(all_chunks) / args.max_chunks
        indices = [int(i * step) for i in range(args.max_chunks)]
        all_chunks = [all_chunks[i] for i in indices]

    print(f"\nPhase 1 complete: {books_ok} books, {books_failed} failed, {len(all_chunks):,} chunks")

    if not all_chunks:
        print("ERROR: No chunks to embed. Exiting.")
        sys.exit(1)

    # 4. Embed
    print(f"\nPhase 2: Embedding {len(all_chunks):,} passages...")
    t0 = time.time()
    embeddings = embed_texts(all_chunks)
    embed_time = time.time() - t0
    print(f"Embedded in {embed_time:.1f}s ({len(all_chunks)/embed_time:.0f} passages/sec)")

    # 5. Save cartridge
    print(f"\nPhase 3: Saving cartridge '{args.name}'...")
    cart_path, size_mb, fingerprint = save_cartridge(output_dir, args.name, embeddings, all_chunks)
    print(f"Saved: {cart_path} ({size_mb:.1f} MB, {fingerprint})")

    # 6. Optional training
    if args.train:
        print(f"\nPhase 4: Training brain (GPU)...")
        from cartridge_builder import train_and_sign
        train_and_sign(output_dir, args.name, embeddings, all_chunks)

    print(f"\n{'='*60}")
    print(f"  DONE: {args.name}")
    print(f"  {books_ok} books → {len(all_chunks):,} passages → {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
