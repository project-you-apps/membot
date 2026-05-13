#!/usr/bin/env python3
"""
Admin tool — provision a Mempack for a user by calling /api/mempack/create.

Usage:
    python provision_mempack.py --owner-id <uuid> --name primary
    python provision_mempack.py --owner-id <uuid> --name primary --server http://137.184.227.79:8040
    python provision_mempack.py --owner-id <uuid> --name primary --briefing "custom briefing text"

Until auto-provision-on-signup lands (needs Supabase JWT verification on Membot
side), this script handles per-user Mempack creation manually. Andy or an admin
script calls it after a new user signs up via Vector+ Studio.

Andy 2026-05-12.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error


def main():
    p = argparse.ArgumentParser(description="Provision a Mempack for a Supabase user.")
    p.add_argument("--owner-id", required=True, help="Supabase user UUID")
    p.add_argument("--name", default="primary", help="Mempack name (default: primary)")
    p.add_argument("--server", default="http://localhost:8040",
                   help="Membot write-tier URL (default: http://localhost:8040). "
                        "Use http://137.184.227.79:8040 for the droplet.")
    p.add_argument("--api-key", default=None,
                   help="X-API-Key header value (defaults to MEMBOT_API_KEY env or no auth)")
    p.add_argument("--briefing", default=None, help="Override the default briefing template")
    p.add_argument("--pattern-i", default=None, help="Override the default Pattern I template")
    args = p.parse_args()

    body = {"owner_id": args.owner_id, "name": args.name}
    if args.briefing:
        body["briefing"] = args.briefing
    if args.pattern_i:
        body["pattern_i"] = args.pattern_i

    url = f"{args.server}/api/mempack/create"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if args.api_key:
        req.add_header("X-API-Key", args.api_key)
    elif "MEMBOT_API_KEY" in __import__("os").environ:
        req.add_header("X-API-Key", __import__("os").environ["MEMBOT_API_KEY"])

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            print(json.dumps(result, indent=2))
            return 0
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code}: {e.read().decode()}\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write(f"Connection failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
