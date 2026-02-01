#!/usr/bin/env python3
"""
Submit solution to mlpuzzles.com and retrieve results.

Usage:
    python submit.py                    # Submit prefix_sum.py
    python submit.py --file solution.py # Submit custom file
    python submit.py --leaderboard      # View leaderboard only
    python submit.py --challenge        # View challenge details
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

API_BASE = "https://puzzle.metr-dev.org/api"


def submit_code(code: str) -> dict:
    """Submit code to the API and return response."""
    import urllib.request
    import urllib.error

    data = json.dumps({"code": code}).encode('utf-8')

    req = urllib.request.Request(
        f"{API_BASE}/submit",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}", "body": e.read().decode('utf-8')}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def get_leaderboard() -> dict:
    """Fetch the current leaderboard."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{API_BASE}/leaderboard", timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}


def get_challenge() -> dict:
    """Fetch challenge details."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{API_BASE}/challenge", timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}


def print_result(result: dict):
    """Pretty print submission result."""
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        if "body" in result:
            print(f"   {result['body']}")
        return False

    print("\n" + "=" * 60)
    print("SUBMISSION RESULT")
    print("=" * 60)

    # Core metrics
    correct = result.get("correct", "unknown")
    score = result.get("score", "N/A")
    time_ms = result.get("time_ms", "N/A")

    status = "✅ PASS" if correct else "❌ FAIL"
    print(f"\nCorrectness: {status}")
    print(f"Score:       {score}")
    print(f"Time:        {time_ms} ms")

    # Additional details
    if "message" in result:
        print(f"\nMessage: {result['message']}")

    if "ai_review" in result:
        print(f"\nAI Review: {result['ai_review']}")

    if "details" in result:
        print(f"\nDetails:")
        for k, v in result["details"].items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)

    return correct


def print_leaderboard(data):
    """Pretty print leaderboard."""
    if "error" in data:
        print(f"Error fetching leaderboard: {data['error']}")
        return

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)

    if isinstance(data, list):
        print(f"\n{'Rank':<6} {'Score':<12} {'Time (ms)':<12}")
        print("-" * 30)
        for i, entry in enumerate(data[:20], 1):
            score = entry.get("score", "N/A")
            time = entry.get("time_ms", "N/A")
            print(f"{i:<6} {score:<12} {time:<12}")
    else:
        print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Submit to mlpuzzles.com")
    parser.add_argument("--file", "-f", type=str, default="prefix_sum.py",
                        help="File to submit (default: prefix_sum.py)")
    parser.add_argument("--leaderboard", "-l", action="store_true",
                        help="Show leaderboard only")
    parser.add_argument("--challenge", "-c", action="store_true",
                        help="Show challenge details")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON")

    args = parser.parse_args()

    # Leaderboard only
    if args.leaderboard:
        data = get_leaderboard()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print_leaderboard(data)
        return

    # Challenge details
    if args.challenge:
        data = get_challenge()
        print(json.dumps(data, indent=2))
        return

    # Submit code
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Reading {filepath}...")
    code = filepath.read_text()

    print(f"Submitting to {API_BASE}/submit...")
    result = submit_code(code)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        success = print_result(result)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
