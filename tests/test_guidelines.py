"""
Tests for LLM node addition guidelines.
"""

import unittest
from dendron import Dendron, ToolDefinition


class TestGuidelines(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(name="base_tool", description="Base tool")
        self.tree = Dendron(name="GuidelineTree", root_tool=self.root_tool)

    def test_get_node_addition_guidelines(self):
        guidelines = self.tree.get_node_addition_guidelines()
        self.assertIsInstance(guidelines, str)
        self.assertIn("When to Add a Node", guidelines)
        self.assertIn("How to Add a Node", guidelines)
        self.assertIn("tree.add_node", guidelines)
        self.assertIn("DendronNode", guidelines)


if __name__ == "__main__":
    unittest.main()
