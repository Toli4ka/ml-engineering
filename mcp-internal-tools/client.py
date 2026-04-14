from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat


SYSTEM_PROMPT = """You are a careful data-quality assistant.
You can use MCP tools to inspect timesheet imports that are already stored in Postgres.
When a tool is useful, call it.
When you get tool results, summarize them clearly for a non-technical employee.
Do not invent file paths or tool outputs.
You may only describe tool results that were actually returned by a tool call.
Do not claim that CSV validation, fixes, or imports were run by MCP; the MCP server is read-only.
If a tool exists but has not been called, say that the capability may be available but no tool result has been produced yet.
"""


def mcp_tool_to_ollama_tool(tool: Any) -> dict[str, Any]:
    """
    Convert an MCP tool description into the schema Ollama expects for tool calling.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {
                "type": "object",
                "properties": {},
            },
        },
    }


def extract_tool_result(result: Any) -> str:
    """
    Prefer structured JSON if available; otherwise join text content.
    """
    if hasattr(result, "structuredContent") and result.structuredContent:
        return json.dumps(result.structuredContent, ensure_ascii=False)

    texts: list[str] = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text:
            texts.append(text)

    return "\n".join(texts) if texts else "{}"


@dataclass(slots=True) #TODO: what is slots=True?
class McpToolClient:
    server_script: Path
    session: ClientSession | None = None
    _stdio_context: Any = None
    _session_context: Any = None
    tool_schemas: list[dict[str, Any]] | None = None
    tool_names: set[str] | None = None

    async def __aenter__(self) -> "McpToolClient": #NOTE: this will be called when async with MCPToolClient obj is created and returned
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env={**os.environ},
        )

        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read, write) #NOTE: create i/o streams 
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        await self.refresh_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._session_context is not None:
            await self._session_context.__aexit__(exc_type, exc, tb)
        if self._stdio_context is not None:
            await self._stdio_context.__aexit__(exc_type, exc, tb)

    async def refresh_tools(self) -> None:
        if self.session is None:
            raise RuntimeError("MCP session has not been initialized.")

        tools_response = await self.session.list_tools()
        self.tool_schemas = [mcp_tool_to_ollama_tool(tool) for tool in tools_response.tools]
        self.tool_names = {tool.name for tool in tools_response.tools}

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError("MCP session has not been initialized.")

        if self.tool_names is None or tool_name not in self.tool_names:
            return json.dumps(
                {"status": "error", "message": f"Unknown tool: {tool_name}"},
                ensure_ascii=False,
            )

        try:
            #NOTE: How the tool knows about the path to the timesheet table? 
            # A: path to the csv file is a part of the user prompt. Model extracts it and saves into the variable {path}
            # Q: How to make it better? Define file pathes so the model knows just gives a request to the path
            result = await self.session.call_tool(tool_name, tool_args) 
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool call failed: {tool_name}",
                    "details": str(exc),
                },
                ensure_ascii=False,
            )

        return extract_tool_result(result)

from datetime import datetime, timezone
@dataclass(slots=True)
class AgentRunner:
    mcp_client: McpToolClient
    model: str = "qwen3"
    system_prompt: str = SYSTEM_PROMPT
    messages: list[dict[str, Any]] | None = None

    # _________ DEBUG________________
    think: bool | str | None = None
    debug_log_path: Path | None = None

    def _log_debug(self, event: str, payload: dict[str, Any]) -> None:
        if self.debug_log_path is None:
            return

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # _________ DEBUG________________
    

    async def run(self, user_message: str) -> str:
        if self.mcp_client.tool_schemas is None:
            raise RuntimeError("MCP tools have not been loaded.")

        if self.messages is None:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        self.messages.append({"role": "user", "content": user_message})

        i = 1
        while True:
            response = chat(
                model=self.model,
                messages=self.messages,
                tools=self.mcp_client.tool_schemas,
                # think=None
            )

            assistant_message = response.message.model_dump(exclude_none=True) #NOTE: Model output with reasoning to call a tool
            self.messages.append(assistant_message)

            tool_calls = getattr(response.message, "tool_calls", None) or [] #NOTE: here model will tell which model to call
            if not tool_calls:
                return response.message.content or ""

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments or {}

                # print(f"\n[tool call] {tool_name}({tool_args})") #TODO: make log
                tool_output = await self.mcp_client.call_tool(tool_name, tool_args)
                # print(f"[tool result] {tool_output}")

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": tool_output,
                    }
                )
            self._log_debug(
                "chat_request",
                {
                    "iteration": i + 1,
                    "model": self.model,
                    "messages": self.messages
                },
            )
            i += 1


async def run_agent(
    user_message: str,
    model: str,
    server_script: Path,
    debug_log_path: Path | None = None,
) -> None:

    async with McpToolClient(server_script) as mcp_client:
        print("Connected MCP tools:", ", ".join(sorted(mcp_client.tool_names or [])) or "(none)")
        
        runner = AgentRunner(
            mcp_client=mcp_client,
            model=model,
            debug_log_path=debug_log_path,
        )

        while True:
            user_message = input("\nYou: ").strip()

            if user_message.lower() in {"exit", "quit", "q", "/exit", "/quit"}:
                print("Bye.")
                return

            if not user_message:
                continue

            response = await runner.run(user_message)
            print("\nAssistant:")
            print(response)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama + read-only timesheet DB MCP client")
    parser.add_argument(
        "message",
        nargs="?",
        default="List recent timesheet imports and summarize the latest one.",
        help="User message to send to the local agent.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:e4b",
        help="Local Ollama model name",
    )
    parser.add_argument(
        "--server",
        default=str(Path(__file__).with_name("server.py")),
        help="Path to the MCP server script.",
    )
    parser.add_argument(
        "--think",
        choices=["low", "medium", "high"],
        help="Enable model thinking mode if the selected model supports it.",
    )
    parser.add_argument(
        "--debug-log",
        help="Write JSONL debug logs with assistant responses and tool results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asyncio.run(
        run_agent(
            args.message,
            model=args.model,
            server_script=Path(args.server),
            debug_log_path=Path(args.debug_log).resolve() if args.debug_log else None,
        )
    )


if __name__ == "__main__":
    main()
