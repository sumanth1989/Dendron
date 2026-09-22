"""
Dendron: An adaptive, tree-based tool execution and dynamic discovery library for AI agents.
Each node is a smart tool call along an execution path, allowing agents to build out
their tool tree based on experience, evaluate transitions, store prompt injection variables,
perform BFS/DFS and RAG semantic searches, and instantly access frequently used tools.
"""

from .exceptions import (
    DendronError,
    NodeNotFoundError,
    InvalidTransitionError,
    ToolValidationError,
    SecurityClearanceError,
    CycleDetectedError,
    ToolExecutionError,
)
from .models import (
    ToolDefinition,
    CompositeToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    PromptContext,
)
from .node import DendronNode, ToolNode
from .tree import Dendron, DendronTree, SmartToolTree
from .mcp_adapter import MCPAdapter
from .langchain_adapter import LangChainAdapter, DendronLangChainTool
from .retriever import DendronRetriever, RAGToolRetriever, RAGSearchResult
from .planner import ActionStep, ActionPlan, AutonomousActionPlanner

__version__ = "0.1.0"
__all__ = [
    "Dendron",
    "DendronTree",
    "DendronNode",
    "DendronRetriever",
    "SmartToolTree",
    "ToolNode",
    "RAGToolRetriever",
    "RAGSearchResult",
    "ToolDefinition",
    "CompositeToolDefinition",
    "ToolParameter",
    "ToolResult",
    "TransitionCondition",
    "PromptContext",
    "MCPAdapter",
    "LangChainAdapter",
    "DendronLangChainTool",
    "ActionStep",
    "ActionPlan",
    "AutonomousActionPlanner",
    "DendronError",
    "NodeNotFoundError",
    "InvalidTransitionError",
    "ToolValidationError",
    "SecurityClearanceError",
    "CycleDetectedError",
    "ToolExecutionError",
]
