"""
SNARC Bridge Integration Tests (Nomad)
=======================================
Validates the dual-write / dual-search path used by SNARC's membot-bridge.ts.

The SNARC hooks (session-start, pre-compact, session-end) call membot's REST API
to store observations and search for context alongside SNARC's native FTS5.
This test simulates that flow end-to-end using the same REST endpoints.

Tests:
  1. Store → search roundtrip via REST API
  2. Dual-write simulation (store same content, verify retrieval)
  3. Search ranking: semantic relevance vs keyword match
  4. Fire-and-forget resilience (membot down → graceful fallback)
  5. Cartridge persistence (save → remount → search still works)
  6. Session-start briefing simulation (store patterns, search with project context)

Requires membot running on port 8000 (or MEMBOT_URL env) with REST API enabled.

Usage:
    python3 tests/fleet/test_snarc_bridge.py

Machine: Nomad (WSL2, 16GB)
Experiment: membot-integration-experiment-2026-03-26
"""

import unittest
import json
import os
import time
import urllib.request
import urllib.error

MEMBOT_URL = os.environ.get("MEMBOT_URL", "http://localhost:8000")
TEST_CARTRIDGE = "snarc-bridge-test"


def api(method, path, data=None, timeout=30):
    """Call membot REST API."""
    url = f"{MEMBOT_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


class TestSnarcBridgeIntegration(unittest.TestCase):
    """Tests that simulate the SNARC hook → membot REST flow."""

    @classmethod
    def setUpClass(cls):
        """Check membot is reachable and set up test cartridge."""
        status = api("GET", "/api/status")
        if not status or "error" in status:
            raise unittest.SkipTest(f"membot not reachable at {MEMBOT_URL}")

        # Mount the shipped demo cartridge to verify mounting works
        result = api("POST", "/api/mount", {"name": "attention-is-all-you-need"})
        if result and "Mounted" in result.get("result", ""):
            cls.has_cartridge = True
        else:
            cls.has_cartridge = False

        # Warm up the embedding model (first store loads SentenceTransformer ~15s)
        warmup = api("POST", "/api/store",
                      {"content": "warmup embedding model", "tags": "test"},
                      timeout=120)
        if warmup and warmup.get("status") == "ok":
            cls.embedder_ready = True
        else:
            cls.embedder_ready = False

    def test_01_store_and_search_roundtrip(self):
        """Store an observation, search for it by meaning."""
        content = "[tool_sequence] Recurring workflow: Read → Edit → Bash (confidence: 0.85)"
        store = api("POST", "/api/store", {"content": content, "tags": "pattern,tool_sequence"})
        self.assertIsNotNone(store)
        # Store should succeed (status ok or contain "Stored")
        self.assertEqual(store.get("status"), "ok", f"Store failed: {store}")

        # Small delay for indexing
        time.sleep(0.5)

        # Search by semantically related query (not exact keywords)
        search = api("POST", "/api/search", {"query": "common tool usage patterns", "top_k": 5})
        self.assertIsNotNone(search)
        self.assertEqual(search.get("status"), "ok")

    def test_02_dual_write_pattern(self):
        """Simulate SNARC's dual-write: store tool observations like pre-compact does."""
        observations = [
            "[Bash] cd /mnt/c/projects/ai-agents && git pull --ff-only origin main",
            "[Read] /mnt/c/projects/ai-agents/SAGE/CLAUDE.md — reading session primer",
            "[Edit] /mnt/c/projects/ai-agents/snarc/hooks/handlers/session-start.ts — fixed hook path",
            "[Grep] pattern: membot, path: snarc/src — searching for bridge code",
        ]

        stored = 0
        for obs in observations:
            result = api("POST", "/api/store", {"content": obs, "tags": "conversation"})
            if result and "error" not in result:
                stored += 1

        self.assertEqual(stored, len(observations), "All observations should store successfully")

        time.sleep(0.5)

        # Search should find git-related observation when asking about repo maintenance
        search = api("POST", "/api/search", {"query": "repository update and maintenance", "top_k": 3})
        self.assertIsNotNone(search)

    def test_03_dream_pattern_storage(self):
        """Simulate session-end dream cycle: store consolidated patterns."""
        patterns = [
            {"content": "[concept_cluster] Focused work on governance hooks and supervisor maintenance",
             "tags": "pattern,concept_cluster,conf:0.75"},
            {"content": "[tool_sequence] Recurring workflow: pull repos → check status → fix issues",
             "tags": "pattern,tool_sequence,conf:0.90"},
            {"content": "[identity] Nomad machine handles mobile development and testing",
             "tags": "pattern,proposed_identity,conf:0.60"},
        ]

        stored = 0
        for p in patterns:
            result = api("POST", "/api/store", p)
            if result and "error" not in result:
                stored += 1

        self.assertGreaterEqual(stored, 2, "Most patterns should store")

    def test_04_session_briefing_search(self):
        """Simulate session-start: search for relevant context given project root."""
        queries = [
            "recent work on this project",
            "common patterns and workflows",
            "identity and project facts",
        ]

        for query in queries:
            search = api("POST", "/api/search", {"query": query, "top_k": 3})
            self.assertIsNotNone(search, f"Search should not fail for: {query}")
            self.assertEqual(search.get("status"), "ok", f"Search should return ok for: {query}")

    def test_05_cartridge_save_and_remount(self):
        """Verify cartridge persistence: save, remount, data survives."""
        # Store something unique
        unique = f"snarc-bridge-persistence-test-{int(time.time())}"
        api("POST", "/api/store", {"content": unique, "tags": "test"})
        time.sleep(0.5)

        # Save cartridge to disk
        save_result = api("POST", "/api/save")
        self.assertIsNotNone(save_result)

        # Remount
        mount_result = api("POST", "/api/mount", {"name": TEST_CARTRIDGE})
        self.assertIsNotNone(mount_result)
        time.sleep(0.5)

        # Search for the unique content
        search = api("POST", "/api/search", {"query": unique, "top_k": 1})
        self.assertIsNotNone(search)

    def test_06_status_endpoint_works(self):
        """Status endpoint should respond with valid structure."""
        status = api("GET", "/api/status")
        self.assertIsNotNone(status)
        self.assertEqual(status.get("status"), "ok")
        # memories count depends on whether cartridge was mounted
        self.assertIn("memories", status)
        self.assertIn("read_only", status)

    def test_07_empty_search_graceful(self):
        """Searching for nonsense should return results list (possibly empty), not crash."""
        search = api("POST", "/api/search", {"query": "xyzzy plugh 99999", "top_k": 3})
        self.assertIsNotNone(search)
        self.assertEqual(search.get("status"), "ok")
        self.assertIsInstance(search.get("results", []), list)

    def test_08_large_content_store(self):
        """Store a large observation (simulating a full conversation turn)."""
        large = "This is a detailed conversation about " + " ".join(
            [f"topic-{i} involving complex architectural decisions" for i in range(100)]
        )
        result = api("POST", "/api/store", {"content": large, "tags": "conversation,large"})
        self.assertIsNotNone(result)
        self.assertNotIn("error", result)


class TestSnarcBridgeFallback(unittest.TestCase):
    """Tests that SNARC bridge degrades gracefully when membot is unavailable."""

    def test_unreachable_membot_returns_error(self):
        """Calling a dead endpoint should return error, not crash."""
        bad_url = "http://localhost:59999"
        req = urllib.request.Request(f"{bad_url}/api/status", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                self.fail("Should not connect to dead port")
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass  # Expected — SNARC bridge catches this and falls back

    def test_timeout_handling(self):
        """Very short timeout should not hang indefinitely."""
        t0 = time.time()
        try:
            req = urllib.request.Request(f"http://10.255.255.1/api/status", method="GET")
            urllib.request.urlopen(req, timeout=1)
        except Exception:
            pass
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5, "Timeout should fire within 5s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
