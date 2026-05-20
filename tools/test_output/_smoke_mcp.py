"""Quick smoke test for the mempack_local_agent MCP plumbing.
Runs against the live membot MCP endpoint; expects ~7-10 tools back."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mempack_local_agent import MempackMCPClient

m = MempackMCPClient("https://project-you.app/membot/mcp")
print("Connecting...", end=" ")
init = m.initialize()
print("OK")
print(f"  protocolVersion: {m.protocol_version}")
print(f"  sessionId: {m.session_id}")
si = init.get("serverInfo", {}) or {}
print(f"  serverInfo: {si.get('name', '<unknown>')} {si.get('version', '')}")
print()
print("Fetching tools/list...", end=" ")
tools = m.list_tools()
print(f"OK ({len(tools)} tools)")
print()
for t in tools[:20]:
    desc = (t.get("description", "") or "").split(".")[0][:80]
    print(f"  {t['name']:35s} {desc}")
print()
print("=== MCP plumbing works end-to-end. ===")
