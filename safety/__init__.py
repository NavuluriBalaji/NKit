"""Safety Gate module for pre-execution verification.

This module provides the SafetyGate class, which is designed to sit between the 
agent's reasoning and the actual tool execution. It evaluates the agent's intent
against a set of strict rules and the original goal, preventing destructive or 
off-topic actions before they happen.
"""

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from NKit.security import PathValidator

class SafetyViolation(Exception):
    """Raised when an agent attempts an action that violates safety policies."""
    pass

class SafetyGate:
    """Pre-execution safety gate for NKit agents.
    
    Validates tool calls before they happen, checking for destructive actions,
    path escapes, uncontrolled domain access, and alignment with the goal.
    Supports a Human-In-The-Loop (HITL) fallback for risky actions.
    """
    
    def __init__(
        self,
        allowed_dirs: Optional[List[str]] = None,
        allowed_domains: Optional[List[str]] = None,
        risk_threshold: float = 0.7,
        hitl: bool = False
    ):
        """Initialize the SafetyGate.
        
        Args:
            allowed_dirs: List of allowed directories for file operations.
            allowed_domains: List of allowed domains for network operations.
            risk_threshold: 0.0 to 1.0 (Currently qualitative tracking via keywords).
            hitl: If True, halts and requests human input before risky actions.
        """
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or ["."])]
        self.allowed_domains = allowed_domains or []
        self.risk_threshold = risk_threshold
        self.hitl = hitl
        
        self.path_validator = PathValidator(allowed_dirs=allowed_dirs)
        
        self.destructive_keywords = {"delete", "drop", "remove", "truncate", "rm", "format"}

    def _request_human_approval(self, action: str, reason: str) -> bool:
        """Pauses execution to ask the user for approval via CLI."""
        print(f"\n[SAFETY GATE: HITL APPROVAL REQUIRED]")
        print(f"Reason: {reason}")
        print(f"Action: {action}")
        
        while True:
            choice = input("Allow this action? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'.")

    def _check_destructive_action(self, tool_name: str, args: Dict[str, Any], goal: str) -> Optional[str]:
        """Check if action is destructive and whether the goal explicitly permits it."""
        action_str = f"{tool_name} {args}".lower()
        goal_lower = goal.lower()
        
        for keyword in self.destructive_keywords:
            if keyword in action_str:
                # If goal doesn't explicitly mention the destructive keyword, flag it.
                if keyword not in goal_lower:
                    return f"Destructive action '{keyword}' detected but not requested in original goal."
        return None

    def _check_file_writes(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Ensure file path arguments don't break outside allowed directories."""
        # This is a heuristic check looking for common path keys
        for key, val in args.items():
            if isinstance(val, str) and ("/" in val or "\\" in val or "path" in key.lower() or "file" in key.lower()):
                # Attempt to validate path safely
                try:
                    self.path_validator.validate_path(val)
                except ValueError:
                    return f"Path access violation: {val} is outside allowed directories."
        return None

    def _check_domain_whitelist(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Ensure URL arguments comply with the domain whitelist."""
        if not self.allowed_domains:
            return None # No whitelist enforced
            
        for key, val in args.items():
            if isinstance(val, str) and ("http://" in val or "https://" in val):
                # Extract domain rudimentarily
                domain_match = re.search(r"https?://([^/]+)", val)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    
                    # Check if domain or any of its suffixes are in whitelist
                    allowed = False
                    for allowed_domain in self.allowed_domains:
                        allowed_domain = allowed_domain.lower()
                        if domain == allowed_domain or domain.endswith("." + allowed_domain):
                            allowed = True
                            break
                    
                    if not allowed:
                        return f"Network violation: Domain '{domain}' is not in allowed_domains."
        return None

    def evaluate(self, tool_name: str, args: Dict[str, Any], goal: str, why: str) -> None:
        """Run all safety checks before allowing the tool to execute.
        
        Args:
            tool_name: The tool about to be executed.
            args: The arguments passed to the tool.
            goal: The original agent task/goal.
            why: The agent's thought leading up to this choice.
            
        Raises:
            SafetyViolation: If the action is determined to be critically unsafe
                             and no HITL approval was granted.
        """
        violations = []
        
        # 1. Hard blocked items (Files outside allowed_dirs)
        file_violation = self._check_file_writes(tool_name, args)
        if file_violation:
            raise SafetyViolation(file_violation)
            
        # 2. Hard blocked items (Non-whitelisted domains)
        domain_violation = self._check_domain_whitelist(tool_name, args)
        if domain_violation:
            raise SafetyViolation(domain_violation)
            
        # 3. Risky/destructive items
        destruction_risk = self._check_destructive_action(tool_name, args, goal)
        if destruction_risk:
            violations.append(destruction_risk)
            
        # Optional NLP alignment could go here using self.risk_threshold
        # Note: True semantic thresholding requires an LLM call. 
        # This implementation uses fast heuristic checks.

        if violations:
            violation_summary = " | ".join(violations)
            
            if self.hitl:
                approved = self._request_human_approval(f"Call `{tool_name}` with `{args}`", violation_summary)
                if not approved:
                    raise SafetyViolation(f"Action blocked by user operator: {violation_summary}")
            else:
                raise SafetyViolation(f"Action blocked by SafetyGate: {violation_summary}")

__all__ = ["SafetyGate", "SafetyViolation"]
