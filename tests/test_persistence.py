"""
Tests for Dendron tree file persistence (save and load).
"""

import os
import tempfile
import unittest
from dendron import Dendron, ToolDefinition, ToolParameter


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(
            name="root_tool",
            description="Root tool for persistence test",
            parameters={"api_key": ToolParameter(name="api_key", type="string")}
        )
        self.tree = Dendron(name="PersistentTree", root_tool=self.root_tool)
        self.child_tool = ToolDefinition(
            name="child_tool",
            description="Child tool",
            parameters={"arg": ToolParameter(name="arg", type="integer")}
        )
        self.tree.add_node(parent_id=self.tree.root.id, tool=self.child_tool, branch_label="child_branch")

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "tree.json")
            self.tree.save(file_path)
            self.assertTrue(os.path.exists(file_path))

            loaded_tree = Dendron.load(file_path)
            self.assertEqual(loaded_tree.name, self.tree.name)
            self.assertEqual(loaded_tree.root.tool.name, "root_tool")
            self.assertEqual(len(loaded_tree.root.children), 1)
            self.assertEqual(loaded_tree.root.children[0].tool.name, "child_tool")

            # Verify inverted registries work after loading
            params = loaded_tree.find_by_input_param("api_key")
            self.assertEqual(len(params), 1)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            Dendron.load("/path/to/nonexistent/dendron_tree_12345.json")


if __name__ == "__main__":
    unittest.main()
