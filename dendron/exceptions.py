"""
Dendron Exceptions hierarchy.
Defines domain-specific exceptions for tree navigation, tool execution,
validation, transitions, and security clearances.
"""

from __future__ import annotations


class DendronError(Exception):
    """Base exception class for all Dendron-related errors."""


class NodeNotFoundError(DendronError, KeyError, ValueError):
    """
    Raised when a requested node or tool cannot be found in the tree.
    Inherits from KeyError and ValueError for backwards compatibility.
    """


class InvalidTransitionError(DendronError, ValueError):
    """Raised when an invalid transition path is attempted or configured."""


class ToolValidationError(DendronError, ValueError):
    """Raised when tool arguments fail validation, e.g. unfulfilled placeholders."""


class SecurityClearanceError(DendronError, PermissionError):
    """Raised when an agent attempts to invoke a destructive or guarded tool without confirmation."""


class CycleDetectedError(DendronError, ValueError):
    """Raised when an operation would introduce an illegal cycle into a tree hierarchy."""


class ToolExecutionError(DendronError, RuntimeError):
    """Raised when a tool handler fails during execution."""
