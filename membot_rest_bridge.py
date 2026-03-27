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

# Auto-mount default cartridge on startup
_default_session = "snarc-bridge"


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
            self._json({'ok': True, 'raw': result})
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
    args = parser.parse_args()

    print(f"Membot REST bridge on port {args.port}")
    HTTPServer(('0.0.0.0', args.port), BridgeHandler).serve_forever()
