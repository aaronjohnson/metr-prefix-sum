#!/usr/bin/env python3
"""
MCP Server for MLPuzzles submission.
Wraps the puzzle.metr-dev.org API for use with Claude Code.

Usage:
  python server.py

Add to ~/.claude/claude_desktop_config.json:
{
  "mcpServers": {
    "mlpuzzles": {
      "command": "python",
      "args": ["/path/to/mcp-submit-server/server.py"]
    }
  }
}
"""

import json
import sys
import requests
from pathlib import Path

# MCP Protocol constants
JSONRPC_VERSION = "2.0"

def send_response(id, result=None, error=None):
    response = {"jsonrpc": JSONRPC_VERSION, "id": id}
    if error:
        response["error"] = error
    else:
        response["result"] = result
    print(json.dumps(response), flush=True)

def send_notification(method, params=None):
    notification = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params:
        notification["params"] = params
    print(json.dumps(notification), flush=True)

def handle_initialize(id, params):
    send_response(id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "mlpuzzles-submit", "version": "1.0.0"}
    })

def handle_tools_list(id, params):
    send_response(id, {
        "tools": [
            {
                "name": "submit_solution",
                "description": "Submit a solution to MLPuzzles.com and get the score",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the solution.py file to submit"
                        },
                        "code": {
                            "type": "string",
                            "description": "Alternative: raw Python code to submit (if file_path not provided)"
                        }
                    }
                }
            },
            {
                "name": "check_api_status",
                "description": "Check if the MLPuzzles API is reachable",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    })

def submit_to_mlpuzzles(code: str) -> dict:
    """Submit code to MLPuzzles API and return result."""
    api_url = "https://puzzle.metr-dev.org/api/submit"

    try:
        response = requests.post(
            api_url,
            json={"code": code},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code == 200:
            return {"success": True, "result": response.json()}
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (60s)"}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Connection failed: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def handle_tools_call(id, params):
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "submit_solution":
        # Get code from file or direct input
        code = arguments.get("code")
        file_path = arguments.get("file_path")

        if file_path:
            try:
                code = Path(file_path).read_text()
            except Exception as e:
                send_response(id, {"content": [{"type": "text", "text": f"Error reading file: {e}"}]})
                return

        if not code:
            send_response(id, {"content": [{"type": "text", "text": "Error: No code provided (use file_path or code parameter)"}]})
            return

        result = submit_to_mlpuzzles(code)
        send_response(id, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

    elif tool_name == "check_api_status":
        try:
            response = requests.get("https://puzzle.metr-dev.org/", timeout=10)
            send_response(id, {"content": [{"type": "text", "text": f"API reachable: HTTP {response.status_code}"}]})
        except Exception as e:
            send_response(id, {"content": [{"type": "text", "text": f"API unreachable: {e}"}]})

    else:
        send_response(id, error={"code": -32601, "message": f"Unknown tool: {tool_name}"})

def main():
    handlers = {
        "initialize": handle_initialize,
        "initialized": lambda id, p: None,  # Notification, no response
        "tools/list": handle_tools_list,
        "tools/call": handle_tools_call,
    }

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method")
            id = request.get("id")
            params = request.get("params", {})

            handler = handlers.get(method)
            if handler:
                handler(id, params)
            elif id:  # Only respond to requests, not notifications
                send_response(id, error={"code": -32601, "message": f"Method not found: {method}"})
        except json.JSONDecodeError:
            pass
        except Exception as e:
            if 'id' in dir() and id:
                send_response(id, error={"code": -32603, "message": str(e)})

if __name__ == "__main__":
    main()
