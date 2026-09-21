"""
Tests for Model Context Protocol (MCP) tool format compliance and adapter.
"""

import unittest
from dendron.models import ToolDefinition, ToolParameter
from dendron.tree import Dendron
from dendron.mcp_adapter import MCPAdapter


class TestMCPAdapter(unittest.TestCase):
    def setUp(self):
        self.tool_root = ToolDefinition(
            name="fetch_data",
            description="Fetch data from remote API",
            parameters={
                "endpoint": ToolParameter(name="endpoint", type="string", description="API endpoint", required=True),
                "timeout": ToolParameter(name="timeout", type="integer", description="Timeout in seconds", default=30)
            }
        )
        self.tree = Dendron(name="MCPTree", root_tool=self.tool_root)

        self.tool_child = ToolDefinition(
            name="process_data",
            description="Process fetched data",
            parameters={
                "format": ToolParameter(name="format", type="string", enum=["json", "xml"], default="json")
            }
        )
        self.tree.add_node(self.tree.root.id, self.tool_child)

    def test_tool_definition_to_mcp_dict(self):
        mcp_dict = self.tool_root.to_mcp_dict()
        self.assertEqual(mcp_dict["name"], "fetch_data")
        self.assertEqual(mcp_dict["description"], "Fetch data from remote API")
        
        input_schema = mcp_dict["inputSchema"]
        self.assertEqual(input_schema["type"], "object")
        self.assertIn("endpoint", input_schema["properties"])
        self.assertEqual(input_schema["properties"]["endpoint"]["type"], "string")
        self.assertIn("endpoint", input_schema["required"])
        self.assertEqual(input_schema["properties"]["timeout"]["default"], 30)

    def test_tool_definition_from_mcp_dict(self):
        raw_mcp = {
            "name": "send_alert",
            "description": "Send alert message",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "description": "Alert severity", "enum": ["info", "error"]},
                    "message": {"type": "string", "description": "Alert body"}
                },
                "required": ["level", "message"]
            }
        }
        tool_def = ToolDefinition.from_mcp_dict(raw_mcp)
        self.assertEqual(tool_def.name, "send_alert")
        self.assertEqual(tool_def.description, "Send alert message")
        self.assertEqual(len(tool_def.parameters), 2)
        self.assertTrue(tool_def.parameters["level"].required)
        self.assertEqual(tool_def.parameters["level"].enum, ["info", "error"])

    def test_mcp_adapter_to_and_from_tools_list(self):
        tools_list_response = MCPAdapter.to_mcp_tools_list(self.tree)
        self.assertIn("tools", tools_list_response)
        tools = tools_list_response["tools"]
        self.assertEqual(len(tools), 2)
        names = [t["name"] for t in tools]
        self.assertIn("fetch_data", names)
        self.assertIn("process_data", names)

        # Import back from MCP tools list
        restored_tree = MCPAdapter.from_mcp_tools_list(
            name="RestoredMCPTree",
            tools_data=tools,
            root_tool_name="fetch_data",
            discovery_instructions="Restored from MCP list"
        )
        self.assertEqual(restored_tree.root.tool.name, "fetch_data")
        self.assertEqual(len(restored_tree.root.children), 1)
        self.assertEqual(restored_tree.root.children[0].tool.name, "process_data")


if __name__ == "__main__":
    unittest.main()
