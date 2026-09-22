"""
Comprehensive unit tests for dendron.node.DendronNode.
Covers hierarchy, execution tracking, progressive views, prompt injection,
DAG transitions, and serialization.
"""

import unittest
from dendron.models import (
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
)
from dendron.node import DendronNode


class TestNodeComprehensive(unittest.TestCase):

    def setUp(self):
        self.tool = ToolDefinition(
            name="search_users",
            description="Searches users by email or name",
            parameters={
                "query": ToolParameter(name="query", type="string", description="Search query", required=True),
                "limit": ToolParameter(name="limit", type="integer", description="Max results", default=10, required=False),
            },
            tags=["users", "search"],
            handler=lambda query, limit=10: [{"id": 1, "name": "Alice"}]
        )
        self.node = DendronNode(
            tool=self.tool,
            branch_label="search_branch",
            system_prompt_template="Search agent for {org}.",
            user_prompt_template="Execute search for {user_query}.",
            prompt_variables={"org": "Acme Corp"}
        )

    def test_node_initialization(self):
        self.assertEqual(self.node.tool.name, "search_users")
        self.assertEqual(self.node.branch_label, "search_branch")
        self.assertEqual(self.node.access_count, 0)
        self.assertEqual(self.node.success_count, 0)
        self.assertEqual(self.node.failure_count, 0)
        self.assertIsNone(self.node.parent)
        self.assertEqual(len(self.node.children), 0)
        self.assertEqual(len(self.node.transitions), 0)

    def test_tree_hierarchy(self):
        child_tool = ToolDefinition(name="view_profile", description="View profile")
        child_node = DendronNode(tool=child_tool)
        
        self.node.add_child(child_node, branch_label="profile_branch")
        self.assertEqual(len(self.node.children), 1)
        self.assertEqual(child_node.parent, self.node)
        self.assertEqual(child_node.branch_label, "profile_branch")

        # Execution path
        self.assertEqual(self.node.get_execution_path(), ["search_users"])
        self.assertEqual(child_node.get_execution_path(), ["search_users", "view_profile"])

        # Remove child
        removed = self.node.remove_child(child_node.id)
        self.assertTrue(removed)
        self.assertEqual(len(self.node.children), 0)
        self.assertIsNone(child_node.parent)

        # Remove non-existent child
        self.assertFalse(self.node.remove_child("fake_id"))

    def test_experience_and_frequency_tracking(self):
        # Access
        self.node.record_access()
        self.assertEqual(self.node.access_count, 1)
        self.assertIsNotNone(self.node.last_accessed_at)

        # Execution success
        res_success = ToolResult(tool_name="search_users", status="success")
        self.node.record_execution(res_success, note="Found 1 user")
        self.assertEqual(self.node.access_count, 2)
        self.assertEqual(self.node.success_count, 1)
        self.assertEqual(self.node.failure_count, 0)
        self.assertIn("Found 1 user", self.node.experience_notes)

        # Execution failure
        res_fail = ToolResult(tool_name="search_users", status="error", error_message="Timeout")
        self.node.record_execution(res_fail)
        self.assertEqual(self.node.failure_count, 1)

        # Negative feedback
        self.node.record_negative_feedback()
        self.assertEqual(self.node.negative_feedback_count, 1)

    def test_search_document_generation(self):
        doc = self.node.to_search_document()
        self.assertIn("Tool: search_users", doc)
        self.assertIn("Description: Searches users", doc)
        self.assertIn("Tags: users, search", doc)
        self.assertIn("Parameters:", doc)
        self.assertIn("query (string)", doc)

        # With experience notes
        self.node.add_experience_note("Frequent queries use email format")
        doc2 = self.node.to_search_document()
        self.assertIn("Learned Experience: Frequent queries use email format", doc2)

    def test_progressive_views(self):
        # Level 1: compact summary string
        v1 = self.node.to_view(level=1)
        self.assertIsInstance(v1, str)
        self.assertIn("search_users(query, [limit])", v1)
        self.assertIn("#users,search", v1)

        # Level 2: parameter summary dict
        v2 = self.node.to_view(level=2)
        self.assertIsInstance(v2, dict)
        self.assertEqual(v2["name"], "search_users")
        self.assertIn("parameters", v2)
        self.assertEqual(v2["parameters"]["query"]["type"], "string")

        # Level 3: full MCP dict
        v3 = self.node.to_view(level=3)
        self.assertIsInstance(v3, dict)
        self.assertEqual(v3["name"], "search_users")
        self.assertIn("inputSchema", v3)

        # Invalid level
        with self.assertRaises(ValueError):
            self.node.to_view(level=99)

    def test_node_execution_and_dag_transitions(self):
        # Execute node
        result = self.node.execute(query="Alice")
        self.assertTrue(result.is_success)
        self.assertEqual(self.node.success_count, 1)

        # DAG transition
        target_node = DendronNode(tool=ToolDefinition(name="target", description="Target"))
        self.node.add_transition_to(target_node.id)
        self.assertEqual(len(self.node.transitions), 1)
        self.assertEqual(self.node.transitions[0][0], target_node.id)

    def test_node_serialization(self):
        child = DendronNode(tool=ToolDefinition(name="child_tool", description="Child"))
        self.node.add_child(child)
        self.node.add_transition_to("external_node_123")

        node_dict = self.node.to_dict(include_children=True)
        self.assertEqual(node_dict["tool"]["name"], "search_users")
        self.assertEqual(len(node_dict["children"]), 1)
        self.assertEqual(len(node_dict["transitions"]), 1)

        # Deserialization
        restored = DendronNode.from_dict(node_dict)
        self.assertEqual(restored.tool.name, "search_users")
        self.assertEqual(len(restored.children), 1)
        self.assertEqual(restored.children[0].tool.name, "child_tool")
        self.assertEqual(len(restored.transitions), 1)
        self.assertEqual(restored.transitions[0][0], "external_node_123")


if __name__ == "__main__":
    unittest.main()
