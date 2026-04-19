"""Command Line Interface for NKit.

Provides standard entrypoints to test, orchestrate, and audit agents
directly from the terminal.

Usage:
    nkit run "Summarize this PDF" --llm ollama:llama3 --tools web,file
    nkit replay ./logs/audit.jsonl --session <session_id>
    nkit health
    nkit tools list
    nkit chat --llm ollama:llama3
"""

import argparse
import sys
import uuid
import asyncio
from typing import Optional

# Internal imports
from NKit.agent.core import Agent
from NKit.llms import OllamaLLM, OpenAILLM, AnthropicLLM, OpenRouterLLM
from NKit.observer import LiveObserver
from NKit.audit import WhyLog
from NKit.tools import ToolRegistry

def resolve_llm(model_tuple: str):
    """Instantiate proper LLM based on string like 'ollama:llama3'."""
    if ":" not in model_tuple:
        provider, model = "openai", model_tuple
    else:
        provider, model = model_tuple.split(":", 1)
        
    provider = provider.lower()
    if provider == "ollama":
        return OllamaLLM(model=model)
    elif provider == "openai":
        return OpenAILLM(model=model)
    elif provider == "anthropic":
        return AnthropicLLM(model=model)
    elif provider == "openrouter":
        return OpenRouterLLM(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def cmd_run(args):
    print(f"Starting agent run: '{args.task}'")
    llm = resolve_llm(args.llm)
    observer = LiveObserver()
    
    @observer.on("agent.reasoning")
    def _print_thought(e):
        print(f"🤔 {e['thought']}")
        
    @observer.on("tool.before")
    def _print_tool(e):
        print(f"🛠  {e['tool_name']}({e['args']})")
        
    @observer.on("tool.after")
    def _print_res(e):
        print(f"✅ Executed. Success: {e['success']}")
        
    agent = Agent(llm=llm.complete, observer=observer)
    try:
        final = agent.run(args.task)
        print(f"\n🎉 FINAL ANSWER:\n{final}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def cmd_replay(args):
    log = WhyLog(args.file)
    log.replay(args.session)

def cmd_health(args):
    print("Running health checks...")
    print("Ollama: Reachable" if OllamaLLM("test").health_check() else "Ollama: UNREACHABLE")
    # Don't natively test cloud ones to avoid empty auth errors by default
    print("Health check complete.")

def cmd_tools_list(args):
    reg = ToolRegistry()
    print("Registered Built-in Tools:")
    for name, tool in reg.tools.items():
        print(f" - {name}: {tool.description}")

def cmd_chat(args):
    print("Starting NKit REPL Chat... (Type 'exit' to quit)")
    llm = resolve_llm(args.llm)
    agent = Agent(llm=llm.complete)
    
    while True:
        try:
            q = input("\nUser> ")
            if q.strip().lower() in ['exit', 'quit']:
                break
            res = agent.run(q)
            print(f"\nAgent> {res}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="NKit - Production Safety Layer for Agentic AI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    p_run = subparsers.add_parser("run", help="Run an agent task")
    p_run.add_argument("task", type=str)
    p_run.add_argument("--llm", type=str, default="ollama:llama3", help="provider:model string")
    p_run.add_argument("--tools", type=str, help="Comma separated tools")

    # replay command
    p_replay = subparsers.add_parser("replay", help="Replay a WhyLog audit trail")
    p_replay.add_argument("file", type=str)
    p_replay.add_argument("--session", type=str, required=True)

    # health command
    p_health = subparsers.add_parser("health", help="Check system health")

    # tools list command
    p_tools = subparsers.add_parser("tools")
    p_tools.add_argument("action", choices=["list"], help="List tools")

    # chat command
    p_chat = subparsers.add_parser("chat", help="Start agent REPL")
    p_chat.add_argument("--llm", type=str, default="ollama:llama3")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "tools" and args.action == "list":
        cmd_tools_list(args)
    elif args.command == "chat":
        cmd_chat(args)

if __name__ == "__main__":
    main()
