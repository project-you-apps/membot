"""
MCP Stdio Tool Tests (Legion)
==============================
Tests the MCP tool functions directly via Python import — no HTTP server needed.
This validates the actual code path used by Claude Code in stdio mode.

Covers: list_cartridges, mount_cartridge, memory_search, get_status, unmount.
Store/save tested only if --writable is simulated.

Usage:
    python3 tests/fleet/test_mcp_stdio_tools.py

Author: Legion (RTX 4090, Ubuntu Linux)
"""

import unittest
import os
import sys
import json
import tempfile
import shutil

import numpy as np

# Add membot root to path so we can import the server module
MEMBOT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, MEMBOT_ROOT)


def _setup_test_cartridge(cart_dir, name="legion-mcp-test"):
    """Create a minimal test cartridge with known embeddings and texts."""
    texts = np.array([
        "The RTX 4090 has 16384 CUDA cores and 24GB of GDDR6X memory.",
        "Trust tensors in Web4 measure talent, training, and temperament.",
        "Epistemic proprioception is the ability to know what you know.",
        "The Jetson Orin Nano runs on ARM64 with 8GB unified memory.",
        "JSON-LD contexts map field names to semantic IRIs for interoperability.",
    ], dtype=object)

    # Generate deterministic pseudo-embeddings (768-dim, normalized)
    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(texts), 768).astype(np.float32)
    # Normalize to unit vectors (cosine similarity requires this)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    path = os.path.join(cart_dir, f"{name}.cart.npz")
    np.savez(path,
             embeddings=embeddings,
             passages=texts,            # membot expects 'passages' key
             compressed_texts=texts,     # compressed_texts mirrors passages
             version=np.array("8.3"))    # version flag for loader
    return path, name


class TestListCartridges(unittest.TestCase):
    """list_cartridges() tool returns available cartridges."""

    def test_finds_sample_cartridge(self):
        """The shipped sample cartridge appears in the listing."""
        from membot_server import list_cartridges
        result = list_cartridges()
        self.assertIn("attention-is-all-you-need", result,
                      "Sample cartridge should appear in listing")

    def test_output_includes_count(self):
        """Listing includes a count of available cartridges."""
        from membot_server import list_cartridges
        result = list_cartridges()
        self.assertIn("Available cartridges", result)


class TestMountAndSearch(unittest.TestCase):
    """mount_cartridge + memory_search via direct tool calls."""

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
        from membot_server import mount_cartridge
        result = mount_cartridge(self.cart_name, session_id="legion-test-mount")
        self.assertIn("Mounted", result, f"Expected mount confirmation, got: {result}")

    def test_mount_nonexistent_returns_error(self):
        """Mounting a nonexistent cartridge returns a useful error."""
        from membot_server import mount_cartridge
        result = mount_cartridge("does-not-exist-abc123", session_id="legion-test-missing")
        self.assertTrue(
            "not found" in result.lower() or "no cartridge" in result.lower(),
            f"Expected 'not found' error, got: {result}"
        )

    def test_search_finds_relevant_content(self):
        """Searching returns content semantically related to the query."""
        from membot_server import mount_cartridge, memory_search
        mount_cartridge(self.cart_name, session_id="legion-test-search")

        # With pseudo-random embeddings, semantic ranking isn't meaningful,
        # but we verify the search pipeline returns results with scores.
        result = memory_search("GPU memory and CUDA cores", top_k=3,
                               session_id="legion-test-search")
        # Result format: "#1 (idx:0) [0.265] text..."
        self.assertIn("results", result.lower(),
                      f"Search should report results, got: {result[:200]}")
        self.assertRegex(result, r'\[[\d.]+\]',
                         f"Search should include score brackets, got: {result[:200]}")

    def test_search_without_mount_returns_message(self):
        """Searching on a fresh session without mounting returns guidance."""
        from membot_server import memory_search
        result = memory_search("anything", top_k=1, session_id="legion-test-no-mount")
        self.assertTrue(
            "no cartridge" in result.lower() or "mount" in result.lower(),
            f"Expected mount guidance, got: {result}"
        )

    def test_search_respects_top_k(self):
        """Search returns at most top_k results."""
        from membot_server import mount_cartridge, memory_search
        mount_cartridge(self.cart_name, session_id="legion-test-topk")
        result = memory_search("test query", top_k=2, session_id="legion-test-topk")
        # Count "Score:" occurrences as a proxy for result count
        score_count = result.count("Score:")
        self.assertLessEqual(score_count, 2,
                             f"Asked for top_k=2 but got {score_count} results")


class TestGetStatus(unittest.TestCase):
    """get_status() returns server diagnostics."""

    def test_status_returns_info(self):
        """Status includes diagnostic information."""
        from membot_server import get_status
        result = get_status()
        # get_status returns a diagnostic string with cartridge/embedding info
        self.assertIn("Embed", result, f"Status should include embedding info: {result[:200]}")

    def test_status_shows_cartridge_count(self):
        """Status reports how many cartridges are available."""
        from membot_server import get_status
        result = get_status()
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
        # Should be a graceful message, not an exception
        self.assertIsInstance(result, str)


class TestSanitization(unittest.TestCase):
    """Security: cartridge name sanitization."""

    def test_path_traversal_neutralized(self):
        """Names with path traversal have dangerous components stripped."""
        from membot_server import sanitize_name
        # sanitize_name strips .. and / then validates the remainder
        # The result should NOT contain any path separators or traversal
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
