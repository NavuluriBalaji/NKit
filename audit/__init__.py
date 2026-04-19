"""Audit trail module providing the WhyLog system.

This module implements the WhyLog class, writing structured JSON Lines (JSONL)
representing an agent's internal reasoning ('Why' and 'Thought') tied to every
external action. This creates a full, replayable narrative of agent behavior
for debugging and enterprise compliance.
"""

import json
import uuid
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

class WhyLog:
    """Structured JSONL audit log for NKit agents.
    
    Writes a structured execution narrative for later querying or replaying,
    ensuring that every decision the agent makes is stored with its reasoning.
    Rotates the log file when it reaches 10MB.
    """
    
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
    
    def __init__(self, path: str = "./logs/audit.jsonl"):
        """Initialize the WhyLog.
        
        Args:
            path: Target file path for the JSONL log.
        """
        self.path = Path(path)
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
    def _check_rotation(self) -> None:
        """Rotate the file if it exceeds the maximum size limit."""
        if self.path.exists() and self.path.stat().st_size >= self.MAX_FILE_SIZE_BYTES:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            archive_path = self.path.with_name(f"{self.path.stem}_{timestamp}{self.path.suffix}")
            self.path.rename(archive_path)

    def _get_timestamp(self) -> str:
        """Get current ISO 8601 timestamp."""
        return datetime.datetime.utcnow().isoformat() + "Z"

    def log(
        self,
        session_id: str,
        event_type: str,
        goal: str,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        why: Optional[str] = None,
        was_blocked: bool = False,
        human_approved: Optional[bool] = None
    ) -> None:
        """Write an entry to the WhyLog.
        
        Args:
            session_id: The UUID4 identifier grouping this agent run.
            event_type: e.g. 'agent.start', 'tool.before', 'safety.block', 'agent.end'.
            goal: The original task requested by the user.
            thought: The reasoning state leading up to this moment.
            action: The tool call or general action triggered.
            result: The returned value from a tool or LLM parsing.
            why: The specific deduction explaining this action.
            was_blocked: True if the SafetyGate intercepted it.
            human_approved: True/False if HITL was involved, None otherwise.
        """
        self._check_rotation()
        
        entry = {
            "timestamp": self._get_timestamp(),
            "session_id": session_id,
            "event_type": event_type,
            "goal": goal,
            "thought": thought,
            "action": action,
            "result": result,
            "why": why,
            "was_blocked": was_blocked,
            "human_approved": human_approved
        }
        
        # Write to JSONL
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")

    def query(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all log entries for a given session ID.
        
        Args:
            session_id: The session string to query.
            
        Returns:
            List of dictionary entries.
        """
        if not self.path.exists():
            return []
            
        results = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("session_id") == session_id:
                        results.append(data)
                except json.JSONDecodeError:
                    continue
        return results

    def replay(self, session_id: str) -> None:
        """Print a human-readable narrative of the agent's session payload.
        
        Args:
            session_id: The session string to replay.
        """
        entries = self.query(session_id)
        if not entries:
            print(f"No log entries found for session: {session_id}")
            return
            
        print(f"\n{'='*60}")
        print(f"WHY LOG REPLAY: Session {session_id}")
        
        # Goal is universally tracked
        goal = entries[0].get("goal", "Unknown")
        print(f"Original Goal: {goal}")
        print(f"{'='*60}\n")
        
        for e in entries:
            event_type = e.get("event_type")
            time_str = e.get("timestamp", "").split("T")[1][:8] # Extract just HH:MM:SS
            
            print(f"[{time_str}] :: {event_type.upper()}")
            
            if event_type == "agent.start":
                print(f"   Agent initialized task.")
            
            elif event_type == "agent.reasoning":
                print(f"   Thought: {e.get('thought')}")
                
            elif event_type == "tool.before":
                print(f"   Intent : {e.get('why')}")
                print(f"   Action : {e.get('action')}")
                
            elif event_type == "safety.block":
                print(f"   [BLOCKED] : {e.get('action')}")
                if e.get("human_approved") is not None:
                    print(f"   HITL Override: {e.get('human_approved')}")
                    
            elif event_type == "tool.after":
                res = str(e.get('result'))
                print(f"   Result : {res[:200]}{'...' if len(res) > 200 else ''}")
                
            elif event_type == "agent.end":
                print(f"   Final Answer: {e.get('result')}")
                
            print("-" * 60)

__all__ = ["WhyLog"]
