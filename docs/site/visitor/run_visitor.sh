#!/usr/bin/env bash
# ─────────────────────────────────────────────────
# Membot Explainer Site — Visitor Track
# 4 personas browse the site, report friction & drift
# ─────────────────────────────────────────────────
set -euo pipefail

SITE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$SITE_DIR/visitor/logs"
TODAY=$(date -u +%Y-%m-%d)
LOG="$LOG_DIR/$TODAY.md"

# Site files on disk (for local auditing)
SITE_PATH="$SITE_DIR"

mkdir -p "$LOG_DIR"

cat > "$LOG" <<HEADER
# Membot Explainer — Visitor Report $TODAY

**Site path**: $SITE_PATH
**Generated**: $(date -u +%H:%M) UTC
**Personas**: 4 sequential passes (isolated — no cross-reading)

---

HEADER

echo "═══ Membot Visitor Track — $TODAY ═══"

# ── Persona definitions ──────────────────────────

COMMON_CONSTRAINTS="
CONSTRAINTS (apply to ALL personas):
- Read the site HTML files from disk at $SITE_PATH/ using the Read tool.
  The main page is $SITE_PATH/index.html.
  Concept pages are at $SITE_PATH/concepts/*.html.
  Styles are at $SITE_PATH/style.css.
- Read the raw HTML as a visitor would see it rendered. Pay attention to
  tooltips (span class='tip'), links, structure, and content.
- Stay strictly in character — do not use knowledge above your ceiling.
- Be honest — if something confuses your persona, say so.
- Check that deep-dive concept pages are reachable and make sense.
- Verify all acronyms are explained (tooltip or inline) on first use.

OUTPUT FORMAT (use exactly this structure):

## Pass N: [Persona Name]

### Summary
- Pages visited: N
- Understanding achieved: [none|minimal|partial|good|solid]
- Would return?: [yes|maybe|no]
- Overall impression: [one sentence]

### Journey
[Narrative of browsing experience — what you clicked, what you understood, where you got stuck]

### Friction Log
| Location | Issue | Severity | Suggestion |
|----------|-------|----------|------------|

### Concept Accuracy Check
| Page | Concept | Status | Notes |
|------|---------|--------|-------|

### Unanswered Questions
1. ...

### Verdict
[2-3 sentence honest assessment]
"

# ── Pass 1: AI Agent Developer ───────────────────
echo "▸ Pass 1: AI Agent Developer"

claude --print --allowedTools Read -p "
You are an AI AGENT DEVELOPER browsing the Membot explainer site for the first time.

BACKGROUND: You build AI agents with Claude Code and have used MCP servers before.
You heard about Membot on social media — someone said it gives agents 'brain cartridges.'
You want to know: what is this, should I care, and how fast can I try it?

KNOWLEDGE CEILING: You know MCP, embeddings at a high level, Python. You do NOT know
Hamming distance, Hopfield networks, neuromorphic computing, or sign-zero encoding.

WHAT TO TEST:
- Is the value proposition clear within 30 seconds?
- Can you understand the getting-started section without external docs?
- Do the tooltips actually help, or do they use more jargon to explain jargon?
- Would you actually try this after reading the site?
- Are the code examples copy-pasteable?

Read these pages:
1. $SITE_PATH/index.html (main page — read it thoroughly)
2. At least 2 concept deep-dive pages that you'd naturally click as this persona

$COMMON_CONSTRAINTS
" >> "$LOG" 2>/dev/null

echo "" >> "$LOG"
echo "---" >> "$LOG"
echo "" >> "$LOG"

# ── Pass 2: Technical Writer ─────────────────────
echo "▸ Pass 2: Technical Writer"

claude --print --allowedTools Read -p "
You are a TECHNICAL WRITER doing a documentation audit of the Membot explainer site.

BACKGROUND: You review developer docs professionally. You care about structure,
consistency, progressive disclosure, and whether readers can self-serve.

KNOWLEDGE CEILING: You can follow technical content if explained. You are not a domain
expert in neuromorphic computing or information retrieval.

WHAT TO TEST:
- Are acronyms defined on first use (via tooltip or inline)?
- Is terminology consistent (same concept, same word everywhere)?
- Does the page flow logically for a newcomer? (what → how → why → start)
- Are concept deep-dive pages self-contained? Can you land on one directly and understand it?
- Do cross-links between concept pages form a coherent web, or dead ends?
- Is the reading level appropriate? Too academic? Too hand-wavy?

Read these pages:
1. $SITE_PATH/index.html (full audit)
2. At least 3 concept deep-dive pages, checking cross-links
3. Check at least one concept page as if you landed on it directly (no context from main page)

$COMMON_CONSTRAINTS
" >> "$LOG" 2>/dev/null

echo "" >> "$LOG"
echo "---" >> "$LOG"
echo "" >> "$LOG"

# ── Pass 3: Information Retrieval Researcher ─────
echo "▸ Pass 3: IR Researcher"

claude --print --allowedTools Read -p "
You are an INFORMATION RETRIEVAL RESEARCHER evaluating the Membot explainer site.

BACKGROUND: You work on search systems. You know cosine similarity, LSH, Hamming
distance, embedding models, and vector databases inside out. You've published papers
on approximate nearest neighbor search.

KNOWLEDGE CEILING: Expert on IR and search. Less familiar with neuromorphic computing,
Hopfield networks, and Hebbian learning (you know OF them but haven't worked with them).

WHAT TO TEST:
- Are the technical claims accurate? (SimHash attribution, correlation figures, scale claims)
- Is the cosine/Hamming/keyword blend described correctly?
- Is the SimHash/sign-zero encoding explained accurately? (Charikar 2002 attribution)
- Are the comparison tables fair, or do they strawman 'typical AI memory'?
- Does the neuromorphic layer description hold up to scrutiny?
- Are the benchmark numbers (R@1=1.000, Pearson r=0.891) presented with appropriate context?

Read these pages:
1. $SITE_PATH/index.html (focus on technical claims)
2. $SITE_PATH/concepts/cosine.html
3. $SITE_PATH/concepts/hamming.html
4. $SITE_PATH/concepts/signzero.html
5. $SITE_PATH/concepts/neuromorphic.html

$COMMON_CONSTRAINTS
" >> "$LOG" 2>/dev/null

echo "" >> "$LOG"
echo "---" >> "$LOG"
echo "" >> "$LOG"

# ── Pass 4: Skeptical CTO ───────────────────────
echo "▸ Pass 4: Skeptical CTO"

claude --print --allowedTools Read -p "
You are a SKEPTICAL CTO evaluating whether Membot is worth adopting for your team.

BACKGROUND: You run a 20-person AI startup. Your agents currently use Pinecone for
vector search and you're paying \$200/month. Someone on your team sent you this site.
You have 5 minutes.

KNOWLEDGE CEILING: Strong engineering background. You know vector DBs, embeddings,
and cloud infra well. You don't know neuromorphic computing or Hebbian learning.
You are naturally skeptical of 'we're different' claims.

WHAT TO TEST:
- What's the actual differentiator vs. Pinecone/Weaviate/Qdrant?
- Is the '\$12/month for 4.8M entries' claim credible? What are the trade-offs?
- Is there vendor lock-in? Can you leave easily?
- What about reliability, uptime, backups?
- Is this production-ready or a research project?
- Would you actually switch, or is this interesting but not practical?
- Security story — is it adequate?

Read these pages:
1. $SITE_PATH/index.html (skim for decision-relevant info)
2. 1-2 concept pages if needed to evaluate claims

$COMMON_CONSTRAINTS
" >> "$LOG" 2>/dev/null

echo "" >> "$LOG"
echo "---" >> "$LOG"
echo "" >> "$LOG"

# ── Cross-persona synthesis ──────────────────────
echo "▸ Synthesis"

claude --print --allowedTools Read -p "
Read the visitor log at $LOG. It contains 4 persona passes auditing the Membot explainer site.

Write a CROSS-PERSONA SYNTHESIS section with:

## Cross-Persona Synthesis

### Merged Impressions
| Persona | Understanding | Would Return? | Key Friction |
|---------|--------------|---------------|--------------|

### Concept Accuracy Summary
List any technical inaccuracies or misleading claims found across all passes.

### Priority Fixes
Merge and deduplicate friction items from all passes. Rank by severity (high/medium/low).
Format as a numbered list with: [severity] location — issue — suggestion.

### What's Working
List things that multiple personas praised or found effective.

### Verdict
3-4 sentence overall assessment of the site's readiness.
" >> "$LOG" 2>/dev/null

echo ""
echo "═══ Done. Log: $LOG ═══"
