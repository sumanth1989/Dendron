"""
Tests for progressive token-tiered tool views (Level 1, Level 2, Level 3).
"""

import unittest
from dendron import Dendron, ToolDefinition, ToolParameter


class TestTieredViews(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(
            name="search_database",
            description="Searches relational database for customer records.",
            parameters={
                "query": ToolParameter(name="query", type="string", description="SQL query string", required=True),
                "limit": ToolParameter(name="limit", type="integer", description="Max records", required=False, default=50),
            },
            tags=["database", "sql", "search"]
        )
        self.tree = Dendron(name="DatabaseTree", root_tool=self.root_tool)

        self.child_tool = ToolDefinition(
            name="export_csv",
            description="Exports retrieved records to a CSV file.",
            parameters={
                "records": ToolParameter(name="records", type="array", description="List of rows", required=True),
                "filename": ToolParameter(name="filename", type="string", description="File name", required=False, default="export.csv"),
            },
            tags=["export", "csv"]
        )
        self.child_node = self.tree.add_node(parent_id=self.tree.root.id, tool=self.child_tool, branch_label="export")

    def test_level_1_compact_summary(self):
        summary = self.tree.root.to_compact_summary()
        self.assertIn("search_database(query, [limit])", summary)
        self.assertIn("Searches relational database", summary)
        self.assertIn("#database,sql,search", summary)

    def test_level_1_compact_dict(self):
        compact = self.tree.root.to_compact_dict()
        self.assertEqual(compact["name"], "search_database")
        self.assertEqual(compact["required_inputs"], ["query"])
        self.assertEqual(compact["optional_inputs"], ["limit"])
        self.assertEqual(compact["tags"], ["database", "sql", "search"])

    def test_level_2_parameter_summary(self):
        summary = self.tree.root.to_parameter_summary()
        self.assertEqual(summary["name"], "search_database")
        self.assertIn("parameters", summary)
        self.assertEqual(summary["parameters"]["query"]["type"], "string")
        self.assertTrue(summary["parameters"]["query"]["required"])
        self.assertEqual(summary["parameters"]["limit"]["default"], 50)
        self.assertFalse(summary["parameters"]["limit"]["required"])

    def test_level_3_mcp_dict(self):
        mcp = self.tree.root.to_mcp_dict()
        self.assertEqual(mcp["name"], "search_database")
        self.assertIn("inputSchema", mcp)
        self.assertEqual(mcp["inputSchema"]["type"], "object")
        self.assertIn("query", mcp["inputSchema"]["properties"])

    def test_to_view_unified(self):
        view_1 = self.tree.root.to_view(level=1)
        self.assertIsInstance(view_1, str)
        self.assertIn("search_database", view_1)

        view_2 = self.tree.root.to_view(level=2)
        self.assertIsInstance(view_2, dict)
        self.assertIn("parameters", view_2)

        view_3 = self.tree.root.to_view(level=3)
        self.assertIsInstance(view_3, dict)
        self.assertIn("inputSchema", view_3)

        with self.assertRaises(ValueError):
            self.tree.root.to_view(level=4)

    def test_export_tool_views(self):
        views_lvl1 = self.tree.export_tool_views(level=1)
        self.assertEqual(len(views_lvl1), 2)
        self.assertTrue(all(isinstance(v, str) for v in views_lvl1))

        views_lvl2 = self.tree.export_tool_views(level=2)
        self.assertEqual(len(views_lvl2), 2)
        self.assertTrue(all(isinstance(v, dict) and "parameters" in v for v in views_lvl2))

        views_lvl3 = self.tree.export_tool_views(level=3)
        self.assertEqual(len(views_lvl3), 2)
        self.assertTrue(all(isinstance(v, dict) and "inputSchema" in v for v in views_lvl3))

    def test_inspect_tool(self):
        inspected = self.tree.inspect_tool("export_csv")
        self.assertEqual(inspected["name"], "export_csv")
        self.assertIn("inputSchema", inspected)

        inspected_by_id = self.tree.inspect_tool(self.child_node.id)
        self.assertEqual(inspected_by_id["name"], "export_csv")

        with self.assertRaises(KeyError):
            self.tree.inspect_tool("nonexistent_tool")


if __name__ == "__main__":
    unittest.main()
