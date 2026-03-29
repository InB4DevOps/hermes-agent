#!/usr/bin/env python3
"""
Read Tracker Module - Consecutive Read/Search Loop Detection

This module provides tracking and prevention mechanisms for detecting when
the agent gets stuck in re-read or re-search loops. This commonly happens
after context compression when the agent forgets it already read a file.

Block/Warn Policy:
------------------
The tracker monitors consecutive identical read/search operations and applies
thresholds to prevent wasted API calls and token usage:

1. **Normal Mode** (default):
   - Warn after 3 consecutive identical reads/searches
   - Block after 4 consecutive identical reads/searches

2. **Investigation Mode** (stricter):
   - Warn after 2 consecutive identical reads/searches
   - Block after 2 consecutive identical reads/searches
   
   Investigation mode is intended for debugging sessions where the agent
   might be more prone to loops. Stricter thresholds catch problems earlier.

3. **Counter Reset**:
   - The consecutive counter resets whenever a DIFFERENT tool is called
   - This ensures we only block on truly consecutive repeated operations
   - If the agent does anything else in between (write, patch, terminal, etc.)
     the next read is treated as fresh

Threshold Configuration:
------------------------
Thresholds can be customized via environment variables:
  - HERMES_READ_BLOCK_THRESHOLD: Block threshold (default: 4)
  - HERMES_READ_WARN_THRESHOLD: Warn threshold (default: 3)

Investigation mode thresholds are hardcoded to 2 for both warn and block,
as this is the appropriate strictness for debugging scenarios.

Usage:
------
    from tools.read_tracker import ReadTracker
    
    tracker = ReadTracker()
    
    # Before calling read_file or search_files:
    result = tracker.check_and_increment("read", "/path/to/file.py", offset=1, limit=500)
    if result.blocked:
        return result.block_message()
    
    # After a different tool is called:
    tracker.reset_counter()
    
    # Enable stricter thresholds for debugging:
    tracker.set_investigation_mode(True)
"""

import os
import threading
from typing import Dict, Any, Optional, Tuple, Set


# =========================================================================
# Threshold Constants
# =========================================================================
# Default thresholds for consecutive read/search blocking
# Can be overridden via environment variables
DEFAULT_BLOCK_THRESHOLD = int(os.getenv("HERMES_READ_BLOCK_THRESHOLD", "4"))
DEFAULT_WARN_THRESHOLD = int(os.getenv("HERMES_READ_WARN_THRESHOLD", "3"))

# Investigation mode thresholds (stricter - block/warn earlier)
# These are hardcoded as investigation mode is specifically for debugging
# where we want to catch loops as early as possible
INVESTIGATION_BLOCK_THRESHOLD = 2
INVESTIGATION_WARN_THRESHOLD = 2


class ReadTracker:
    """
    Track consecutive file read/search operations to detect and prevent loops.
    
    This class maintains per-task state to detect when the agent is
    repeatedly reading the same file region or running the same search
    without making progress.
    
    Thread-safe: all public methods acquire the internal lock.
    
    Attributes:
        _lock: Thread lock for concurrent access safety
        _tracker: Dict mapping task_id to tracking state:
            - last_key: Tuple of the last operation key
            - consecutive: Count of consecutive identical operations
            - read_history: Set of (path, offset, limit) tuples read
            - investigation_mode: Boolean for stricter thresholds
    """
    
    def __init__(self):
        """Initialize the read tracker with empty state."""
        self._lock = threading.Lock()
        self._tracker: Dict[str, Dict[str, Any]] = {}
    
    def _get_task_data(self, task_id: str) -> Dict[str, Any]:
        """
        Get or create tracking data for a task.
        
        Must be called while holding the lock.
        
        Args:
            task_id: The task/session identifier
            
        Returns:
            Dict with tracking state for this task
        """
        if task_id not in self._tracker:
            self._tracker[task_id] = {
                "last_key": None,
                "consecutive": 0,
                "read_history": set(),
                "investigation_mode": False,
            }
        return self._tracker[task_id]
    
    def check_and_increment(
        self,
        operation: str,
        task_id: str = "default",
        **operation_args
    ) -> "CheckResult":
        """
        Check thresholds and increment the consecutive counter.
        
        This is the main entry point for read_file and search_files tools.
        It should be called BEFORE executing the actual operation.
        
        Args:
            operation: "read" or "search"
            task_id: Task/session identifier
            **operation_args: Operation-specific arguments used to build
                            the tracking key (e.g., path, offset, limit for reads)
        
        Returns:
            CheckResult with:
            - consecutive: Current consecutive count
            - blocked: True if operation should be blocked
            - warned: True if operation should include a warning
            - block_threshold: Current block threshold
            - warn_threshold: Current warn threshold
        """
        with self._lock:
            task_data = self._get_task_data(task_id)
            
            # Build the operation key for comparison
            key = self._build_key(operation, **operation_args)
            
            # Track read history separately (for context compression)
            if operation == "read":
                path = operation_args.get("path", "")
                offset = operation_args.get("offset", 1)
                limit = operation_args.get("limit", 500)
                task_data["read_history"].add((path, offset, limit))
            
            # Increment or reset consecutive counter
            if task_data["last_key"] == key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = key
                task_data["consecutive"] = 1
            
            count = task_data["consecutive"]
            
            # Determine thresholds based on investigation mode
            is_investigation = task_data.get("investigation_mode", False)
            if is_investigation:
                block_threshold = INVESTIGATION_BLOCK_THRESHOLD
                warn_threshold = INVESTIGATION_WARN_THRESHOLD
            else:
                block_threshold = DEFAULT_BLOCK_THRESHOLD
                warn_threshold = DEFAULT_WARN_THRESHOLD
            
            # Check if we should block or warn
            blocked = count >= block_threshold
            warned = count >= warn_threshold and not blocked
            
            return CheckResult(
                consecutive=count,
                blocked=blocked,
                warned=warned,
                block_threshold=block_threshold,
                warn_threshold=warn_threshold,
                is_investigation=is_investigation,
            )
    
    def _build_key(self, operation: str, **operation_args) -> Tuple:
        """
        Build a hashable key for an operation.
        
        The key must include all parameters that would make the operation
        return different results. For reads, this is (path, offset, limit).
        For searches, this includes pattern, target, path, file_glob, limit, offset.
        
        Args:
            operation: "read" or "search"
            **operation_args: Operation arguments
            
        Returns:
            Tuple key for comparison
        """
        if operation == "read":
            return (
                "read",
                operation_args.get("path", ""),
                operation_args.get("offset", 1),
                operation_args.get("limit", 500),
            )
        elif operation == "search":
            return (
                "search",
                operation_args.get("pattern", ""),
                operation_args.get("target", "content"),
                str(operation_args.get("path", ".")),
                operation_args.get("file_glob") or "",
                operation_args.get("limit", 50),
                operation_args.get("offset", 0),
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def reset_counter(self, task_id: str = "default") -> None:
        """
        Reset the consecutive counter for a task.
        
        Call this when a tool OTHER than read_file/search_files is executed.
        This ensures we only block on truly consecutive repeated operations.
        
        Args:
            task_id: Task/session identifier
        """
        with self._lock:
            task_data = self._tracker.get(task_id)
            if task_data:
                task_data["last_key"] = None
                task_data["consecutive"] = 0
    
    def set_investigation_mode(self, task_id: str, enabled: bool = True) -> None:
        """
        Set investigation mode for a specific task/session.
        
        In investigation mode, stricter thresholds are applied to prevent
        the agent from getting stuck in re-read loops during debugging.
        
        Block threshold: 2 (instead of 4)
        Warn threshold: 2 (instead of 3)
        
        This is per-session (per task_id), not a global flag.
        
        Args:
            task_id: The task/session ID to modify
            enabled: True to enable stricter thresholds, False for normal mode
        """
        with self._lock:
            task_data = self._get_task_data(task_id)
            task_data["investigation_mode"] = enabled
    
    def get_investigation_mode(self, task_id: str = "default") -> bool:
        """
        Get the investigation mode status for a specific task/session.
        
        Args:
            task_id: The task/session ID to query
            
        Returns:
            True if investigation mode is enabled (stricter thresholds), False otherwise
        """
        with self._lock:
            task_data = self._tracker.get(task_id, {})
            return task_data.get("investigation_mode", False)
    
    def get_read_history(self, task_id: str = "default") -> list:
        """
        Get a summary of files read in this session for the given task.
        
        Used by context compression to preserve file-read history across
        compression boundaries.
        
        Args:
            task_id: Task/session identifier
            
        Returns:
            List of dicts with "path" and "regions" keys:
            [
                {"path": "/path/to/file.py", "regions": ["lines 1-500", "lines 100-200"]},
                ...
            ]
        """
        with self._lock:
            task_data = self._tracker.get(task_id, {})
            read_history = task_data.get("read_history", set())
            
            # Group by path
            seen_paths: Dict[str, list] = {}
            for (path, offset, limit) in read_history:
                if path not in seen_paths:
                    seen_paths[path] = []
                seen_paths[path].append(f"lines {offset}-{offset + limit - 1}")
            
            return [
                {"path": p, "regions": regions}
                for p, regions in sorted(seen_paths.items())
            ]
    
    def clear(self, task_id: Optional[str] = None) -> None:
        """
        Clear the read tracker.
        
        Call with a task_id to clear just that task, or without to clear all.
        Should be called when a session is destroyed to prevent memory leaks
        in long-running gateway processes.
        
        Args:
            task_id: Task ID to clear, or None to clear all
        """
        with self._lock:
            if task_id:
                self._tracker.pop(task_id, None)
            else:
                self._tracker.clear()


class CheckResult:
    """
    Result from ReadTracker.check_and_increment().
    
    Attributes:
        consecutive: Current consecutive operation count
        blocked: True if the operation should be blocked
        warned: True if the operation should include a warning
        block_threshold: Current block threshold being used
        warn_threshold: Current warn threshold being used
        is_investigation: True if investigation mode is active
    """
    
    def __init__(
        self,
        consecutive: int,
        blocked: bool,
        warned: bool,
        block_threshold: int,
        warn_threshold: int,
        is_investigation: bool = False,
    ):
        self.consecutive = consecutive
        self.blocked = blocked
        self.warned = warned
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold
        self.is_investigation = is_investigation
    
    def block_message(self, operation: str, operation_key: dict = None) -> dict:
        """
        Generate a block message for when an operation is blocked.
        
        Args:
            operation: "read" or "search"
            operation_key: Dict with operation-specific info for the message
            
        Returns:
            Dict with "error" key containing the block message
        """
        if operation == "read":
            path = operation_key.get("path", "") if operation_key else ""
            return {
                "error": (
                    f"BLOCKED: You have read this exact file region "
                    f"{self.consecutive} times in a row. "
                    "The content has NOT changed. You already have this information. "
                    "STOP re-reading and proceed with your task."
                ),
                "path": path,
                "already_read": self.consecutive,
            }
        elif operation == "search":
            pattern = operation_key.get("pattern", "") if operation_key else ""
            return {
                "error": (
                    f"BLOCKED: You have run this exact search "
                    f"{self.consecutive} times in a row. "
                    "The results have NOT changed. You already have this information. "
                    "STOP re-searching and proceed with your task."
                ),
                "pattern": pattern,
                "already_searched": self.consecutive,
            }
        else:
            return {
                "error": (
                    f"BLOCKED: You have performed this operation "
                    f"{self.consecutive} times in a row. "
                    "STOP repeating and proceed with your task."
                ),
            }
    
    def warn_message(self, operation: str) -> str:
        """
        Generate a warning message for when an operation exceeds warn threshold.
        
        Args:
            operation: "read" or "search"
            
        Returns:
            Warning message string
        """
        if operation == "read":
            return (
                f"You have read this exact file region {self.consecutive} times "
                "consecutively. The content has not changed since your last read. "
                "Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with "
                "writing or responding."
            )
        elif operation == "search":
            return (
                f"You have run this exact search {self.consecutive} times "
                "consecutively. The results have not changed. "
                "Use the information you already have."
            )
        else:
            return (
                f"You have performed this operation {self.consecutive} times "
                "consecutively. Consider whether you are stuck in a loop."
            )


# Global tracker instance for backward compatibility
# New code should instantiate ReadTracker directly
_tracker_instance = ReadTracker()


def get_tracker() -> ReadTracker:
    """Get the global read tracker instance."""
    return _tracker_instance
