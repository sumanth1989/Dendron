"""
MCP (Model Context Protocol) Adapter for Dendron.
Facilitates importing tools from standard MCP servers into Dendron,
and exporting tree tools to MCP client/server responses.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from .models import ToolDefinition
from .node import DendronNode, ToolNode
from .tree import Dendron, SmartToolTree


class MCPAdapter:
    """
    Adapter between Dendron trees and the Model Context Protocol (MCP).
    """

    @staticmethod
    def to_mcp_tools_list(tree: Dendron) -> Dict[str, Any]:
        """
        Formats all tools in the tree into the standard MCP `tools/list` response:
        {
            "tools": [
                {
                    "name": "...",
                    "description": "...",
                    "inputSchema": {...}
                }, ...
            ]
        }
        """
        return {
            "tools": tree.export_all_mcp_tools()
        }

    @staticmethod
    def from_mcp_tools_list(
        name: str,
        tools_data: List[Dict[str, Any]],
        root_tool_name: Optional[str] = None,
        discovery_instructions: str = ""
    ) -> Dendron:
        """
        Builds a Dendron tree from an MCP tools list.
        If `root_tool_name` is provided, that tool becomes the root;
        otherwise, the first tool in the list serves as the root.
        """
        if not tools_data:
            raise ValueError("tools_data list cannot be empty.")

        # Parse definitions
        tool_defs = [ToolDefinition.from_mcp_dict(t) for t in tools_data]
        
        # Pick root
        root_def: ToolDefinition
        if root_tool_name:
            matching = [t for t in tool_defs if t.name == root_tool_name]
            if not matching:
                raise ValueError(f"Root tool '{root_tool_name}' not found in provided MCP tools.")
            root_def = matching[0]
            remaining = [t for t in tool_defs if t.name != root_tool_name]
        else:
            root_def = tool_defs[0]
            remaining = tool_defs[1:]

        tree = Dendron(
            name=name,
            root_tool=root_def,
            discovery_instructions=discovery_instructions
        )

        # Attach remaining tools as children of root by default
        for t in remaining:
            tree.add_node(
                parent_id=tree.root.id,
                tool=t,
                branch_label=t.name
            )

        return tree

