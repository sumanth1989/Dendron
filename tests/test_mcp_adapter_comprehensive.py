"""
Comprehensive unit tests for dendron.mcp_adapter.MCPAdapter.
Tests conversion between Dendron execution trees and MCP tools/list schemas.
"""

import unittest
from dendron import Dendron, ToolDefinition, ToolParameter, MCPAdapter


class TestMCPAdapterComprehensive(unittest.TestCase):

    def setUp(self):
        self.root_tool = ToolDefinition(
            name="fetch_user",
            description="Fetches user details",
            parameters={"user_id": ToolParameter(name="user_id", type="string", required=True)}
        )
        self.tree = Dendron(name="UserTree", root_tool=self.root_tool)
        self.child_tool = ToolDefinition(
            name="update_user",
            description="Updates user profile",
            parameters={"email": ToolParameter(name="email", type="string")}
        )
        self.tree.add_node(parent_id=self.tree.root.id, tool=self.child_tool)

    def test_to_mcp_tools_list(self):
        mcp_payload = MCPAdapter.to_mcp_tools_list(self.tree)
        self.assertIn("tools", mcp_payload)
        self.assertEqual(len(mcp_payload["tools"]), 2)

        names = [t["name"] for t in mcp_payload["tools"]]
        self.assertIn("fetch_user", names)
        self.assertIn("update_user", names)

    def test_from_mcp_tools_list_default_root(self):
        tools_data = [
            {"name": "tool_a", "description": "Tool A", "inputSchema": {"type": "object"}},
            {"name": "tool_b", "description": "Tool B", "inputSchema": {"type": "object"}}
        ]
        imported_tree = MCPAdapter.from_mcp_tools_list(name="ImportedTree", tools_data=tools_data)
        self.assertEqual(imported_tree.name, "ImportedTree")
        self.assertEqual(imported_tree.root.tool.name, "tool_a")
        self.assertEqual(len(imported_tree), 2)
        self.assertIsNotNone(imported_tree.find_by_name("tool_b"))

    def test_from_mcp_tools_list_explicit_root(self):
        tools_data = [
            {"name": "tool_a", "description": "Tool A", "inputSchema": {"type": "object"}},
            {"name": "tool_b", "description": "Tool B", "inputSchema": {"type": "object"}}
        ]
        imported_tree = MCPAdapter.from_mcp_tools_list(
            name="ImportedTree",
            tools_data=tools_data,
            root_tool_name="tool_b"
        )
        self.assertEqual(imported_tree.root.tool.name, "tool_b")
        self.assertIsNotNone(imported_tree.find_by_name("tool_a"))

    def test_from_mcp_tools_list_errors(self):
        # Empty list
        with self.assertRaises(ValueError):
            MCPAdapter.from_mcp_tools_list(name="Empty", tools_data=[])

        # Missing root tool
        tools_data = [{"name": "tool_a", "description": "Tool A"}]
        with self.assertRaises(ValueError):
            MCPAdapter.from_mcp_tools_list(name="Test", tools_data=tools_data, root_tool_name="non_existent")


if __name__ == "__main__":
    unittest.main()
