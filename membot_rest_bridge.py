"""
Thin REST bridge for membot — exposes store/search/mount as plain HTTP POST.
Used by SNARC's membot-bridge.ts experiment. Runs alongside the MCP server.

Usage:
    .venv/bin/python membot_rest_bridge.py --port 8001

Endpoints:
    POST /store   {"content": "...", "tags": "..."}  → {"ok": true, "msg": "..."}
    POST /search  {"query": "...", "top_k": 5}       → {"results": [...]}
    POST /mount   {"name": "..."}                     → {"ok": true, "msg": "..."}
    POST /save    {}                                  → {"ok": true, "msg": "..."}
    GET  /status                                      → {"mounted": "...", "count": N}
"""

import argparse
import json
import sys
import os

# Add membot to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import HTTPServer, BaseHTTPRequestHandler
import membot_server as mb
import service_identity

# Auto-mount default cartridge on startup
_default_session = "snarc-bridge"

# What this process actually bound to, filled in at startup. Reported by /status so
# exposure is read off the running process rather than inferred from a unit file --
# the committed unit is not the running unit.
_bound_host = None
_bound_port = None


class BridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/store':
            result = mb.memory_store(
                content=body.get('content', ''),
                tags=body.get('tags', ''),
                session_id=_default_session,
            )
            self._json({'ok': 'Stored' in result, 'msg': result})

        elif self.path == '/search':
            result = mb.memory_search(
                query=body.get('query', ''),
                top_k=body.get('top_k', 5),
                verbose=body.get('verbose', False),
                session_id=_default_session,
            )
            self._json({'ok': True, 'raw': result})

        elif self.path == '/mount':
            result = mb.mount_cartridge(
                name=body.get('name', ''),
                session_id=_default_session,
            )
            self._json({'ok': 'Mounted' in result, 'msg': result})

        elif self.path == '/save':
            result = mb.save_cartridge(session_id=_default_session)
            self._json({'ok': 'Saved' in result, 'msg': result})

        else:
            self._json({'error': 'unknown endpoint'}, 404)

    def do_GET(self):
        if self.path == '/status':
            result = mb.get_status(session_id=_default_session)
            # ⭐ WHICH PROGRAM AND WHICH BUILD, and here it matters more than on the
            # MCP server. This bridge is a SECOND server in the same directory with
            # none of that one's protections: no authentication, unconditionally
            # writable, one shared session id for every caller. On 2026-11-14 it was
            # found on two fleet seats bound to all interfaces because the loopback
            # default had been applied to membot_server.py and nobody looked here.
            # ⇒ role="rest-bridge" is how a caller or a sweep tells the two apart
            #   from the outside, which is exactly what nobody could do that day.
            self._json({
                'ok': True,
                'service': service_identity.service_block(
                    'rest-bridge',
                    host=_bound_host, port=_bound_port,
                    # Stated rather than left to be assumed. Every one of these is a
                    # fact an operator has been surprised by.
                    authentication=None,
                    writable=True,
                    shared_session=_default_session,
                ),
                'raw': result,
            })
        else:
            self._json({'error': 'unknown endpoint'}, 404)

    def _json(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Suppress default logging — membot already logs
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument(
        '--host', default='127.0.0.1',
        help="Bind address. Default 127.0.0.1, loopback only. Pass 0.0.0.0 to accept "
             "connections from other machines. WARNING: this bridge has no "
             "authentication of any kind and is always writable, so exposing it "
             "gives anyone who can reach the port read and write access to the "
             "mounted cartridge. From 2026-11-18 a non-loopback bind will be "
             "REFUSED rather than warned about.")
    args = parser.parse_args()

    # Was hard-coded to 0.0.0.0, with no flag. Found 2026-11-14 by cbp-claude during
    # the fleet listener check, on two seats.
    #
    # The fix that created the gap: 9d7c471 (2026-10-18) made 127.0.0.1 the default
    # for membot_server.py and nobody looked at the second server in the same
    # directory. test_bind_and_auth_defaults.py checks that 'default="0.0.0.0"' does
    # not appear in membot_server.py's source, which a hard-coded HTTPServer tuple in
    # another file passes trivially. A check matching one pattern in one file, read as
    # covering the behaviour.
    #
    # Why this is worse here than on the server, and why the posture differs:
    #   - No auth exists. MEMBOT_API_KEY cannot help: the middleware that enforces it
    #     is a FastMCP protocol hook, and this bridge calls the tool functions
    #     in-process, so the key is never consulted.
    #   - Always writable. _server_config defaults to read_only=False and this entry
    #     point never runs the main() that would set it from --writable. /store and
    #     /save therefore work regardless of any flag.
    #   - No caller identity. _default_session is a constant, so every caller shares
    #     one session.
    # => On the server, exposure can be made safe by requiring a key. Here there is
    #    nothing to require, so exposure has to be refused instead. Staged: warn now,
    #    refuse from 2026-11-18, so fleet units can add --host first.
    bind_loopback = mb._bind_is_loopback(args.host)
    _bound_host, _bound_port = args.host, args.port

    print(f"Membot REST bridge on {args.host}:{args.port}")
    # Identity on the boot line, same as the MCP server. ASCII only -- cp1252 console.
    print("Service: " + service_identity.describe(
        "rest-bridge", host=args.host, port=args.port))
    if bind_loopback:
        print("Bound to loopback - not reachable from other machines.")
    else:
        print("WARNING: bound to a non-loopback address.")
        print("  This bridge has NO authentication and is ALWAYS writable.")
        print("  Anyone who can reach this port can read the mounted cartridge,")
        print("  write into it with /store, and persist that with /save.")
        print("  From 2026-11-18 this will refuse to start. Put a proxy in front,")
        print("  or bind loopback and reach it through that.")
    print("Writable: yes, unconditionally. This bridge has no read-only mode.")

    HTTPServer((args.host, args.port), BridgeHandler).serve_forever()
