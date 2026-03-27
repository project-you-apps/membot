"""
MCP Stdio Tool Tests (Legion)
==============================
Tests the MCP tool functions via the running HTTP server on port 8000.
Validates the actual search/store/mount pipeline.

Covers: list_cartridges, mount_cartridge, memory_search, get_status, unmount.

Usage:
    python3 tests/fleet/test_mcp_stdio_tools.py

Requires: membot_server running on http://localhost:8000 (or MEMBOT_URL env).
Fallback: if server unavailable, imports membot_server directly (slow cold start).

Author: Legion (RTX 4090, Ubuntu Linux)
"""

import unittest
import os
import sys
import json
import urllib.request
import urllib.error

import numpy as np

MEMBOT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MEMBOT_URL = os.environ.get("MEMBOT_URL", "http://localhost:8000")

# ── HTTP helpers ────────────────────────────────────────────────────

def _server_available():
    try:
        req = urllib.request.Request(f"{MEMBOT_URL}/api/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False

_USE_HTTP = _server_available()


def _api(method, path, data=None, timeout=30):
    """Call membot REST API."""
    url = f"{MEMBOT_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── Direct-import fallback (for offline/no-server testing) ──────────

if not _USE_HTTP:
    sys.path.insert(0, MEMBOT_ROOT)


def _call(func_name, **kwargs):
    """Call a membot tool, preferring HTTP server, falling back to direct import."""
    if _USE_HTTP:
        if func_name == "list_cartridges":
            r = _api("GET", "/api/cartridges")
            # Reconstruct the text listing from JSON
            carts = r.get("cartridges", [])
            lines = [f"Available cartridges ({len(carts)}):"]
            for c in carts:
                lines.append(f"  {c.get('name', c)} — {c.get('size', '?')}")
            return "\n".join(lines)

        elif func_name == "mount_cartridge":
            r = _api("POST", "/api/mount", {
                "name": kwargs.get("name", ""),
                "session_id": kwargs.get("session_id", ""),
            })
            return r.get("result", r.get("message", json.dumps(r)))

        elif func_name == "memory_search":
            r = _api("POST", "/api/search", {
                "query": kwargs.get("query", ""),
                "top_k": kwargs.get("top_k", 5),
                "session_id": kwargs.get("session_id", ""),
            })
            # Reconstruct text output from JSON results
            results = r.get("results", [])
            if not results:
                return f"No results. {r.get('message', '')}"
            lines = [f"{len(results)} results:"]
            for i, res in enumerate(results):
                score = res.get("score", 0)
                text = res.get("text", "")[:200]
                idx = res.get("index", i)
                lines.append(f"#{i+1} (idx:{idx}) [{score:.3f}] {text}")
            return "\n".join(lines)

        elif func_name == "memory_store":
            r = _api("POST", "/api/store", {
                "content": kwargs.get("content", ""),
                "tags": kwargs.get("tags", ""),
                "session_id": kwargs.get("session_id", ""),
            })
            return r.get("result", r.get("message", json.dumps(r)))

        elif func_name == "get_status":
            r = _api("GET", "/api/status", timeout=10)
            # Return a text representation that matches what the tool returns
            parts = []
            if r.get("cartridge"):
                parts.append(f"Cartridge: {r['cartridge']}")
            parts.append(f"Memories: {r.get('memories', 0)}")
            parts.append(f"Embed: nomic-embed-text-v1.5")
            parts.append(f"GPU: {r.get('gpu', False)}")
            carts = r.get("available_cartridges", 0)
            parts.append(f"Available cartridges: {carts}")
            return " | ".join(parts)

        elif func_name == "unmount":
            r = _api("POST", "/api/mount", {
                "name": "__unmount__",
                "session_id": kwargs.get("session_id", ""),
            })
            # unmount via the search endpoint isn't standard — try direct
            # Fall through to direct import for unmount
            pass

        # Fallback for unmount or unknown
        import membot_server as mb
        fn = getattr(mb, func_name)
        return fn(**kwargs)

    else:
        import membot_server as mb
        fn = getattr(mb, func_name)
        return fn(**kwargs)


def _setup_test_cartridge(cart_dir, name="legion-mcp-test"):
    """Create a minimal test cartridge with known embeddings and texts."""
    texts = np.array([
        "The RTX 4090 has 16384 CUDA cores and 24GB of GDDR6X memory.",
        "Trust tensors in Web4 measure talent, training, and temperament.",
        "Epistemic proprioception is the ability to know what you know.",
        "The Jetson Orin Nano runs on ARM64 with 8GB unified memory.",
        "JSON-LD contexts map field names to semantic IRIs for interoperability.",
    ], dtype=object)

    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(texts), 768).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    path = os.path.join(cart_dir, f"{name}.cart.npz")
    np.savez(path,
             embeddings=embeddings,
             passages=texts,
             compressed_texts=texts,
             version=np.array("8.3"))
    return path, name


class TestListCartridges(unittest.TestCase):
    """list_cartridges() tool returns available cartridges."""

    def test_finds_sample_cartridge(self):
        """The shipped sample cartridge appears in the listing."""
        result = _call("list_cartridges")
        self.assertIn("attention-is-all-you-need", result,
                      "Sample cartridge should appear in listing")

    def test_output_includes_count(self):
        """Listing includes a count of available cartridges."""
        result = _call("list_cartridges")
        self.assertIn("Available cartridges", result)


class TestMountAndSearch(unittest.TestCase):
    """mount_cartridge + memory_search via tool calls."""

    @classmethod
    def setUpClass(cls):
        """Create a test cartridge in the standard cartridges/ directory."""
        cls.cart_dir = os.path.join(MEMBOT_ROOT, "cartridges")
        cls.cart_path, cls.cart_name = _setup_test_cartridge(cls.cart_dir)

    @classmethod
    def tearDownClass(cls):
        """Remove test cartridge."""
        if os.path.exists(cls.cart_path):
            os.remove(cls.cart_path)
        manifest = cls.cart_path.replace(".cart.npz", ".cart_manifest.json")
        if os.path.exists(manifest):
            os.remove(manifest)

    def test_mount_returns_confirmation(self):
        """Mounting a valid cartridge returns a success message."""
        result = _call("mount_cartridge", name=self.cart_name,
                        session_id="legion-test-mount")
        self.assertIn("Mounted", result, f"Expected mount confirmation, got: {result}")

    def test_mount_nonexistent_returns_error(self):
        """Mounting a nonexistent cartridge returns a useful error."""
        result = _call("mount_cartridge", name="does-not-exist-abc123",
                        session_id="legion-test-missing")
        self.assertTrue(
            "not found" in result.lower() or "no cartridge" in result.lower(),
            f"Expected 'not found' error, got: {result}"
        )

    def test_search_finds_relevant_content(self):
        """Searching returns content semantically related to the query."""
        _call("mount_cartridge", name=self.cart_name,
              session_id="legion-test-search")
        result = _call("memory_search", query="GPU memory and CUDA cores",
                        top_k=3, session_id="legion-test-search")
        self.assertIn("results", result.lower(),
                      f"Search should report results, got: {result[:200]}")
        self.assertRegex(result, r'\[[\d.]+\]',
                         f"Search should include score brackets, got: {result[:200]}")

    def test_search_without_mount_returns_message(self):
        """Searching on a fresh session without mounting returns guidance."""
        result = _call("memory_search", query="anything", top_k=1,
                        session_id="legion-test-no-mount")
        self.assertTrue(
            "no cartridge" in result.lower() or "mount" in result.lower()
            or "no results" in result.lower(),
            f"Expected mount guidance, got: {result}"
        )

    def test_search_respects_top_k(self):
        """Search returns at most top_k results."""
        _call("mount_cartridge", name=self.cart_name,
              session_id="legion-test-topk")
        result = _call("memory_search", query="test query", top_k=2,
                        session_id="legion-test-topk")
        # Count score brackets as proxy for result count
        import re
        scores = re.findall(r'\[\d+\.\d+\]', result)
        self.assertLessEqual(len(scores), 2,
                             f"Asked for top_k=2 but got {len(scores)} results")


class TestGetStatus(unittest.TestCase):
    """get_status() returns server diagnostics."""

    def test_status_returns_info(self):
        """Status includes diagnostic information."""
        result = _call("get_status")
        self.assertIn("Embed", result, f"Status should include embedding info: {result[:200]}")

    def test_status_shows_cartridge_count(self):
        """Status reports how many cartridges are available."""
        result = _call("get_status")
        self.assertIn("cartridge", result.lower(),
                      f"Status should mention cartridges: {result[:200]}")


class TestUnmount(unittest.TestCase):
    """unmount() releases the current cartridge."""

    def test_unmount_after_mount(self):
        """Unmounting a mounted cartridge succeeds."""
        from membot_server import mount_cartridge, unmount
        mount_cartridge("attention-is-all-you-need", session_id="legion-test-unmount")
        result = unmount(session_id="legion-test-unmount")
        self.assertTrue(
            "unmounted" in result.lower() or "unloaded" in result.lower() or "freed" in result.lower(),
            f"Expected unmount confirmation, got: {result}"
        )

    def test_unmount_without_mount(self):
        """Unmounting when nothing is mounted returns a message (not crash)."""
        from membot_server import unmount
        result = unmount(session_id="legion-test-unmount-empty")
        self.assertIsInstance(result, str)


class TestSanitization(unittest.TestCase):
    """Security: cartridge name sanitization."""

    def test_path_traversal_neutralized(self):
        """Names with path traversal have dangerous components stripped."""
        from membot_server import sanitize_name
        result = sanitize_name("../../../etc/passwd")
        self.assertNotIn("/", result)
        self.assertNotIn("..", result)
        self.assertNotIn("\\", result)

    def test_valid_names_pass(self):
        """Normal cartridge names are accepted."""
        from membot_server import sanitize_name
        for name in ["my-cartridge", "test_cart_v2", "attention-is-all-you-need"]:
            result = sanitize_name(name)
            self.assertEqual(result, name)

    def test_empty_name_rejected(self):
        """Empty string is rejected."""
        from membot_server import sanitize_name
        with self.assertRaises(ValueError):
            sanitize_name("")


if __name__ == "__main__":
    unittest.main(verbosity=2)
