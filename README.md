# Membot

**Brain cartridge server for AI agents.**

Membot is an MCP server that gives AI agents swappable, physics-enhanced memory. Mount a brain cartridge, search it with real neural physics, store new memories, swap to a different domain --all through standard [Model Context Protocol](https://modelcontextprotocol.io/) tool calls.

Built on the [Vector+ Lattice Engine](https://github.com/project-you-apps/vector-plus-studio), Membot blends traditional embedding similarity with a 16-million neuron Hopfield network to find **associative relationships** that pure cosine similarity misses.

![Membot Physics Demo](docs/garfield-physics-demo.png)

## The Physics Difference

Standard vector search returns the closest embeddings by cosine distance. Membot does that too--but it also settles your query through a neural lattice with Mexican hat inhibition, Hebbian weights, and energy dynamics. The physics layer surfaces **contextual connections** across domains.

Here we searched 100,000 Wikipedia articles with the query *"Is Garfield an American President or a Cat?"*:

| Query: "Is Garfield an American President or a Cat?" | Embedding-Only | Physics Blend (70/30) |
|---|---|---|
| #1 | Garfield (cat) | Garfield (cat) |
| #2 | James A. Garfield | James A. Garfield |
| #3 | Charles J. Guiteau | Charles J. Guiteau |
| #4 | Gerald Ford | Gerald Ford |
| **#5** | **Barack Obama** | **Assassination of Garfield** |

The top 4 stay the same (accuracy preserved), but rank #5 changes from a generic "president" match to a contextually meaningful connection. The physics found the assassination--a relationship that lives in the attractor dynamics, not in the embedding geometry.

## Quick Start

### Prerequisites

- Python 3.10+
- An MCP-compatible agent ([OpenClaw](https://github.com/anthropics/openclaw), [Claude Code](https://claude.com/claude-code), etc.)

Optional (for physics-enhanced search):
- NVIDIA GPU with CUDA 11.0+
- Pre-built CUDA engine (`lattice_cuda_v7.dll` / `.so`)

### Install

```bash
git clone https://github.com/project-you-apps/membot.git
cd membot
pip install -r requirements.txt
```

### Run

```bash
# Local (stdio)--for OpenClaw, Claude Desktop, local agents
python membot_server.py

# Remote (HTTP)--for any MCP client over the network
python membot_server.py --transport http --port 8000

# Read-only (disables store and save)--for public-facing servers
python membot_server.py --transport http --port 8000 --read-only
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--transport` | `stdio` | Transport mode: `stdio`, `http`, or `sse` |
| `--host` | `0.0.0.0` | Bind address (HTTP/SSE mode) |
| `--port` | `8000` | Listen port (HTTP/SSE mode) |
| `--read-only` | off | Disable `memory_store` and `save_cartridge` |

**stdio mode**: JSON-RPC over stdin/stdout, designed for MCP agent frameworks that launch Membot as a subprocess.

**HTTP mode**: Streamable HTTP on `http://{host}:{port}/mcp`. Any MCP client can connect remotely. Includes rate limiting (60 req/min per IP) and optional API key auth via `MEMBOT_API_KEY` environment variable.

### Agent Configuration

**OpenClaw** (`~/.openclaw/openclaw.json`):

```json
{
  "plugins": {
    "entries": {
      "mcp-adapter": {
        "enabled": true,
        "config": {
          "servers": [
            {
              "name": "membot",
              "transport": "stdio",
              "command": "python",
              "args": ["/path/to/membot/membot_server.py"]
            }
          ]
        }
      }
    }
  }
}
```

**Claude Code** (local, stdio):

```json
{
  "mcpServers": {
    "membot": {
      "command": "python",
      "args": ["/path/to/membot/membot_server.py"]
    }
  }
}
```

**Claude Code** (remote, HTTP):

```json
{
  "mcpServers": {
    "membot": {
      "type": "http",
      "url": "http://your-server:8000/mcp"
    }
  }
}
```

Tools will appear prefixed with `membot_` (e.g., `membot_memory_search`).

**OpenClaw agent dispatch** (headless):

OpenClaw agent dispatch doesn't load MCP adapter tools. Use [mcporter](https://github.com/steipete/mcporter) + a SOUL.md that instructs the agent to call Membot via Bash:

```bash
mcporter call membot.memory_search query="your query" top_k=5
```

See [SOUL-research-bot-merged.md](SOUL-research-bot-merged.md) for a working example.

## Tools

| Tool | Description |
|------|-------------|
| `list_cartridges` | Browse available brain cartridges with size and capabilities |
| `mount_cartridge` | Load a cartridge into memory (embeddings + optional GPU brain) |
| `memory_search` | Semantic search with physics+embedding blend and keyword reranking |
| `memory_store` | Store new text into the mounted cartridge |
| `save_cartridge` | Persist the current cartridge to disk (secure NPZ format) |
| `unmount` | Free memory and unload the current cartridge |
| `get_status` | Server diagnostics (mounted cartridge, memory count, GPU status) |

## How It Works

1. **Mount** a brain cartridge--embeddings, text, L2 signatures, and Hebbian weights load into memory
2. **Search**--your query is embedded (Nomic v1.5, 768-dim), then:
   - Cosine similarity against stored embeddings (fast, always available)
   - Lattice settle: query is imprinted → physics runs → L2 signature extracted → cosine against stored signatures (GPU, when available)
   - **70/30 blend**: 70% embedding + 30% physics similarity
   - Keyword reranking boosts results that contain query terms
3. **Store**--new text is embedded and added to the cartridge; optionally imprinted into the lattice with Hebbian learning
4. **Save**--cartridge persists as secure `.npz` with SHA256 integrity manifest

### Search Modes

| Mode | When | Speed (100k) |
|------|------|------|
| Embedding-only | No GPU or no signatures | ~200ms |
| Physics + Embedding (70/30) | GPU + signatures loaded | ~10s |

Physics search is slower but finds cross-domain associative connections that embedding search can't. The confidence gating automatically falls back to embedding-only when the physics signal is degenerate.

## Brain Cartridges

A brain cartridge is a self-contained memory unit:

| File | Contents | Required |
|------|----------|----------|
| `name.pkl` or `name.cart.npz` | Embeddings + text | Yes |
| `name_signatures.npz` | L2 hierarchy vectors (4096-dim) | For physics search |
| `name_brain.npy` | Hebbian weight matrix (128 MB) | For physics search |
| `name_manifest.json` | SHA256 integrity fingerprint | Recommended |

Cartridges are compatible with [Vector+ Studio](https://github.com/project-you-apps/vector-plus-studio) v8.2+. Build them in Studio or with the CLI builder, serve them with Membot.

### Building Cartridges

Use the included `cartridge_builder.py` to create cartridges from local documents:

```bash
# Embed a folder of documents (fast, no GPU needed)
python cartridge_builder.py ./my-docs/ --name my-knowledge

# Full build with lattice training (GPU required, enables physics search)
python cartridge_builder.py ./my-docs/ --name my-knowledge --train

# Single file, custom chunk size
python cartridge_builder.py research-paper.pdf --name paper --chunk-size 500
```

Supports `.txt`, `.md`, `.pdf`, and `.docx`. Long documents are automatically chunked with overlap.

Place cartridges in `cartridges/` or `data/` directories relative to the server.

### Sample Cartridge

The repo includes a pre-built cartridge of [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)--the paper that introduced the Transformer architecture. 24 chunks with pre-computed embeddings, ready for immediate embedding-only search.

```bash
# Mount it and start searching right away
> mount_cartridge("attention-is-all-you-need")
> memory_search("how does multi-head attention work")
```

To enable physics-enhanced search, rebuild the cartridge with `--train` (requires GPU). This generates the brain weights and L2 signatures that the physics blend needs:

```bash
python cartridge_builder.py attention-paper.pdf --name attention-is-all-you-need --train
```

Build your own from any PDF, markdown, or text file in seconds with `cartridge_builder.py`.

## Deployment

For remote hosting, run Membot in HTTP mode behind a reverse proxy or directly:

```bash
# Set API key (clients must send Authorization: Bearer <key>)
export MEMBOT_API_KEY="your-secret-key"

# Start read-only public server
python membot_server.py --transport http --port 8000 --read-only
```

**Architecture**: Public server runs read-only (search only). Build and curate cartridges locally, then upload finished ones to the server. Users and agents connect to browse and search, not write.

**systemd** (Linux):

```ini
[Service]
Environment="MEMBOT_API_KEY=your-secret-key"
ExecStart=/opt/membot/venv/bin/python /opt/membot/membot_server.py --transport http --port 8000 --read-only
Restart=always
```

**Requirements**: Python 3.10+ and ~2 GB RAM (SentenceTransformer model). No GPU needed for embedding-only search.

## Security

- **NPZ-first**: New cartridges are always saved as `.npz` (NumPy archive --no code execution)
- **PKL sandboxing**: Legacy `.pkl` files are only loaded from trusted directories (configurable)
- **Integrity verification**: SHA256 manifest checked on mount; tampered cartridges are rejected
- **Input sanitization**: Cartridge names validated against path traversal; text and query lengths capped
- **Resource limits**: Max 100,000 entries per cartridge, 10,000 chars per store, 2,000 chars per query

## Embedding Model

Membot uses [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) via SentenceTransformers. This matches the embedder used by Vector+ Studio to build cartridges.

The model downloads automatically on first run (~270 MB). Subsequent starts load from cache.

**Important**: The embedding model used to build cartridges must match the one used to query them. Membot and Vector+ Studio both use the same model, so cartridges are interchangeable.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| RAM | 4 GB | 16+ GB |
| GPU | None (embedding-only mode) | NVIDIA RTX 3080+ |
| VRAM | -- | 8+ GB |
| CUDA | -- | 12.0+ |

## Project Structure

```
membot/
├── membot_server.py              # MCP server entry point
├── cartridge_builder.py          # CLI tool to build cartridges from documents
├── multi_lattice_wrapper_v7.py   # Python wrapper for CUDA engine
├── requirements.txt
├── bin/
│   └── lattice_cuda_v7.dll       # Pre-built CUDA physics engine (Windows)
├── cartridges/                   # Your brain cartridges go here
└── data/                         # Alternative cartridge directory
```

## License

**Dual-Licensed:**

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Python code (`.py` files) | MIT | Yes |
| CUDA Engine (`bin/*.dll`) | Proprietary | [Contact for license](mailto:andy@project-you.app) |

The server code and utilities are open source under MIT. The compiled CUDA physics engine is free for personal, educational, and non-commercial use. Commercial use requires a separate license--see [bin/LICENSE](bin/LICENSE).

## Links

- [Vector+ Studio](https://github.com/project-you-apps/vector-plus-studio) -- GUI for building and searching brain cartridges
- [Project You](https://project-you.app) -- Parent project
- [Licensing](https://project-you.app/licensing) -- Commercial licensing options

---

*Patterns stored holographically, not as records. Memory served as cartridges, not as databases.*
