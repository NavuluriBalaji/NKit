"""Security utilities for input validation and sanitization.

This module provides validators and sanitizers to prevent common attacks:
- Path traversal in file operations
- Command injection in shell tools
- Prompt injection in LLM inputs
- Resource exhaustion (limits on sizes, counts)

Architecture:
    Validators implement ToolValidator interface for use in ToolRegistry.
    Can be chained for multi-layer defense.

Security Principle:
    **Defense in Depth** - validate at multiple layers:
    1. Input validation (this module)
    2. Tool execution isolation
    3. Output sanitization
    4. Rate limiting/resource quotas
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class PathValidator:
    """Validates and sanitizes file paths.
    
    Purpose:
        Prevents path traversal attacks (../../../etc/passwd).
        Restricts file operations to allowed directories.
    
    Reuse Patterns:
        - File read/write tools
        - Document upload systems
        - Log file access
        - Backup/restore operations
    
    Security:
        - Resolves symlinks to detect traversal
        - Validates against whitelist of allowed directories
        - Rejects absolute paths to sensitive locations
        - Checks file extensions if configured
    
    Example:
        ```python
        validator = PathValidator(allowed_dirs=["/data", "/tmp"])
        
        # Safe path
        safe = validator.validate_path("/data/user/file.txt")
        
        # Rejected paths
        validator.validate_path("/etc/passwd")  # raises ValueError
        validator.validate_path("/data/../etc/passwd")  # raises ValueError
        ```
    """
    
    def __init__(
        self,
        allowed_dirs: Optional[List[str]] = None,
        allowed_extensions: Optional[List[str]] = None,
        max_path_length: int = 4096,
    ):
        """Initialize path validator.
        
        Args:
            allowed_dirs: Whitelist of allowed parent directories.
                         If None, allows current directory only.
            allowed_extensions: Whitelist of file extensions (e.g., [".txt", ".json"]).
                               If None, allows all extensions.
            max_path_length: Maximum path length (default 4096)
        """
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or ["."])]
        self.allowed_extensions = allowed_extensions
        self.max_path_length = max_path_length
    
    def validate_path(self, path: str) -> Path:
        """Validate and resolve a file path.
        
        Args:
            path: File path to validate
        
        Returns:
            Resolved absolute Path object
        
        Raises:
            ValueError: If path is unsafe
        
        Security Checks:
            1. Length limit (prevent buffer overflows in C extensions)
            2. Null byte rejection (prevent string truncation attacks)
            3. Path traversal detection (resolve and check parent)
            4. Extension whitelist (if configured)
            5. Directory whitelist
        """
        if not isinstance(path, str) or not path:
            raise ValueError("Path must be non-empty string")
        
        if len(path) > self.max_path_length:
            raise ValueError(f"Path too long: {len(path)} chars (max {self.max_path_length})")
        
        if '\x00' in path:
            raise ValueError("Path contains null byte")
        
        try:
            resolved = Path(path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid path: {e}")
        
        # Check extension
        if self.allowed_extensions:
            if resolved.suffix.lower() not in self.allowed_extensions:
                raise ValueError(f"File extension {resolved.suffix} not allowed")
        
        # Check against allowed directories
        allowed = False
        for allowed_dir in self.allowed_dirs:
            try:
                # Check if path is within allowed directory
                resolved.relative_to(allowed_dir)
                allowed = True
                break
            except ValueError:
                continue
        
        if not allowed:
            raise ValueError(f"Path not in allowed directories: {path}")
        
        return resolved


class StringValidator:
    """Validates string inputs for length, characters, patterns.
    
    Purpose:
        Prevents injection attacks and resource exhaustion via:
        - Length limits
        - Character whitelisting
        - Pattern blacklisting (SQL keywords, shell metacharacters)
    
    Reuse Patterns:
        - LLM prompt inputs
        - Search queries
        - User-provided text
        - Configuration values
    
    Example:
        ```python
        validator = StringValidator(max_length=1000, allowed_chars="alphanumeric_space")
        safe_input = validator.validate("User query here")
        ```
    """
    
    # Predefined character sets
    ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9]+$')
    ALPHANUMERIC_SPACE = re.compile(r'^[a-zA-Z0-9\s]+$')
    ALPHANUMERIC_SPACE_PUNCT = re.compile(r'^[a-zA-Z0-9\s\.\,\!\?\-]+$')
    
    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 0,
        allowed_chars: Optional[str] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ):
        """Initialize string validator.
        
        Args:
            max_length: Maximum string length
            min_length: Minimum string length
            allowed_chars: Character set name ("alphanumeric", "alphanumeric_space", etc.)
                          or regex pattern. If None, allows all.
            forbidden_patterns: List of regex patterns to reject
        """
        self.max_length = max_length
        self.min_length = min_length
        self.forbidden_patterns = [re.compile(p, re.IGNORECASE) for p in (forbidden_patterns or [])]
        
        # Map named character sets
        if allowed_chars == "alphanumeric":
            self.allowed_pattern = self.ALPHANUMERIC
        elif allowed_chars == "alphanumeric_space":
            self.allowed_pattern = self.ALPHANUMERIC_SPACE
        elif allowed_chars == "alphanumeric_space_punct":
            self.allowed_pattern = self.ALPHANUMERIC_SPACE_PUNCT
        elif allowed_chars:
            self.allowed_pattern = re.compile(allowed_chars)
        else:
            self.allowed_pattern = None
    
    def validate(self, text: str) -> str:
        """Validate a string.
        
        Args:
            text: String to validate
        
        Returns:
            Original string (if valid)
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError("Input must be string")
        
        if len(text) < self.min_length:
            raise ValueError(f"String too short: {len(text)} < {self.min_length}")
        
        if len(text) > self.max_length:
            raise ValueError(f"String too long: {len(text)} > {self.max_length}")
        
        if self.allowed_pattern and not self.allowed_pattern.match(text):
            raise ValueError("String contains disallowed characters")
        
        for pattern in self.forbidden_patterns:
            if pattern.search(text):
                raise ValueError(f"String contains forbidden pattern: {pattern.pattern}")
        
        return text


class ToolInputValidator:
    """Composite validator for tool inputs.
    
    Purpose:
        Validates entire tool input dictionaries by:
        - Checking required parameters
        - Validating parameter types
        - Applying per-parameter validators
    
    Reuse Patterns:
        - Tool execution pipeline
        - API endpoint validation
        - User input sanitization
    
    Example:
        ```python
        validator = ToolInputValidator({
            "file_path": PathValidator(allowed_dirs=["/data"]),
            "query": StringValidator(max_length=500),
        }, required=["file_path"])
        
        inputs = validator.validate("read_file", {
            "file_path": "/data/doc.txt",
            "query": "search term"
        })
        ```
    """
    
    def __init__(
        self,
        param_validators: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ):
        """Initialize tool input validator.
        
        Args:
            param_validators: Dict mapping param names to validators.
                             Each validator should have a validate() method.
            required: List of required parameter names
        """
        self.param_validators = param_validators or {}
        self.required = set(required or [])
    
    def validate(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool inputs.
        
        Args:
            tool_name: Tool identifier (for error messages)
            inputs: Input dictionary
        
        Returns:
            Validated (possibly transformed) inputs
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"Tool '{tool_name}' inputs must be dict")
        
        # Check required parameters
        missing = self.required - set(inputs.keys())
        if missing:
            raise ValueError(f"Tool '{tool_name}' missing required params: {missing}")
        
        # Validate each parameter
        validated = {}
        for key, value in inputs.items():
            if key in self.param_validators:
                validator = self.param_validators[key]
                try:
                    if hasattr(validator, 'validate'):
                        validated[key] = validator.validate(value)
                    elif hasattr(validator, 'validate_path'):
                        validated[key] = str(validator.validate_path(value))
                    else:
                        validated[key] = value
                except Exception as e:
                    raise ValueError(f"Tool '{tool_name}' param '{key}' validation failed: {e}")
            else:
                validated[key] = value
        
        return validated


# Predefined validators for common use cases
FILE_PATH_VALIDATOR = PathValidator(allowed_dirs=[".", "/tmp", "/data"])
SEARCH_QUERY_VALIDATOR = StringValidator(
    max_length=500,
    allowed_chars="alphanumeric_space_punct",
    forbidden_patterns=[
        r'<script',  # XSS
        r'javascript:',  # XSS
        r'DROP\s+TABLE',  # SQL injection
        r';\s*--',  # SQL comment
    ]
)


__all__ = [
    "PathValidator",
    "StringValidator",
    "ToolInputValidator",
    "FILE_PATH_VALIDATOR",
    "SEARCH_QUERY_VALIDATOR",
]
