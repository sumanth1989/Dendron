"""
Tests for Dendron structure, hierarchy, and serialization.
"""

import unittest
from dendron.models import ToolDefinition, ToolParameter
from dendron.node import DendronNode, ToolNode
from dendron.tree import Dendron, DendronTree, SmartToolTree


class TestDendron(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(
            name="get_all_emails",
            description="Fetch emails from inbox",
            parameters={"limit": ToolParameter(name="limit", type="integer", default=10)}
        )
        self.tree = Dendron(
            name="EmailTree",
            root_tool=self.root_tool,
            discovery_instructions="Start at get_all_emails and choose unread or respond paths.",
            description="Smart email agent execution tree"
        )

    def test_tree_initialization(self):
        self.assertEqual(self.tree.name, "EmailTree")
        self.assertEqual(self.tree.description, "Smart email agent execution tree")
        self.assertEqual(self.tree.root.tool.name, "get_all_emails")
        self.assertEqual(self.tree.root.branch_label, "root")
        self.assertIn("Start at get_all_emails", self.tree.discovery_instructions)

    def test_add_child_node(self):
        respond_tool = ToolDefinition(
            name="respond_to_email",
            description="Send a response to an email",
            parameters={"email_id": ToolParameter(name="email_id", type="string")}
        )
        child = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=respond_tool,
            branch_label="respond"
        )
        self.assertEqual(len(self.tree.root.children), 1)
        self.assertEqual(self.tree.root.children[0].id, child.id)
        self.assertEqual(child.parent.id, self.tree.root.id)
        self.assertEqual(child.branch_label, "respond")

    def test_add_node_invalid_parent(self):
        tool = ToolDefinition(name="invalid_tool", description="test")
        with self.assertRaises(ValueError):
            self.tree.add_node(parent_id="nonexistent-id", tool=tool)

    def test_remove_child(self):
        respond_tool = ToolDefinition(name="respond_to_email", description="Send response")
        child = self.tree.add_node(parent_id=self.tree.root.id, tool=respond_tool)
        self.assertEqual(len(self.tree.root.children), 1)
        
        removed = self.tree.root.remove_child(child.id)
        self.assertTrue(removed)
        self.assertEqual(len(self.tree.root.children), 0)
        self.assertIsNone(child.parent)

    def test_serialization_dict_and_json(self):
        respond_tool = ToolDefinition(
            name="respond_to_email",
            description="Send a response",
            parameters={"email_id": ToolParameter(name="email_id", type="string")}
        )
        self.tree.add_node(parent_id=self.tree.root.id, tool=respond_tool, branch_label="respond")

        # Serialize to dict and json
        tree_dict = self.tree.to_dict()
        tree_json = self.tree.to_json()
        self.assertIsInstance(tree_dict, dict)
        self.assertIsInstance(tree_json, str)

        # Deserialize back
        restored_from_dict = Dendron.from_dict(tree_dict)
        restored_from_json = Dendron.from_json(tree_json)

        for restored in (restored_from_dict, restored_from_json):
            self.assertEqual(restored.name, self.tree.name)
            self.assertEqual(restored.root.tool.name, "get_all_emails")
            self.assertEqual(len(restored.root.children), 1)
            self.assertEqual(restored.root.children[0].tool.name, "respond_to_email")
            self.assertEqual(restored.root.children[0].branch_label, "respond")

    def test_backwards_compatibility_aliases(self):
        self.assertIs(DendronTree, Dendron)
        self.assertIs(SmartToolTree, Dendron)
        self.assertIs(ToolNode, DendronNode)


if __name__ == "__main__":
    unittest.main()
