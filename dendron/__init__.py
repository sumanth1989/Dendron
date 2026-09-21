"""
Dendron: An adaptive, tree-based tool execution and dynamic discovery library for AI agents.
Each node is a smart tool call along an execution path, allowing agents to build out
their tool tree based on experience, evaluate transitions, store prompt injection variables,
perform BFS/DFS and RAG semantic searches, and instantly access frequently used tools.
"""

from .models import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    PromptContext,
)
from .node import DendronNode, ToolNode
from .tree import Dendron, DendronTree, SmartToolTree
from .mcp_adapter import MCPAdapter
from .retriever import DendronRetriever, RAGToolRetriever, RAGSearchResult

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
    "ToolParameter",
    "ToolResult",
    "TransitionCondition",
    "PromptContext",
    "MCPAdapter",
]
