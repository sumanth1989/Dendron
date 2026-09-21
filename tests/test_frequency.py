"""
Tests for fast lookup registry and MFU (Most Frequently Used) caching.
"""

import unittest
from dendron.models import ToolDefinition
from dendron.tree import Dendron


class TestFrequencyAndCaching(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(name="tool_root", description="Root tool")
        self.tree = Dendron(name="FrequencyTree", root_tool=self.root_tool)

        self.tool_a = ToolDefinition(name="tool_a", description="Tool A")
        self.tool_b = ToolDefinition(name="tool_b", description="Tool B")
        self.tool_c = ToolDefinition(name="tool_c", description="Tool C")

        self.node_a = self.tree.add_node(self.tree.root.id, self.tool_a)
        self.node_b = self.tree.add_node(self.tree.root.id, self.tool_b)
        self.node_c = self.tree.add_node(self.node_b.id, self.tool_c)

    def test_fast_lookup_by_id_and_name(self):
        found_id = self.tree.find_by_id(self.node_a.id)
        self.assertIsNotNone(found_id)
        self.assertEqual(found_id.tool.name, "tool_a")

        found_name = self.tree.find_by_name("tool_b")
        self.assertIsNotNone(found_name)
        self.assertEqual(found_name.id, self.node_b.id)

        search_fast_res = self.tree.search_fast("tool_c")
        self.assertIsNotNone(search_fast_res)
        self.assertEqual(search_fast_res.id, self.node_c.id)

    def test_access_count_increments(self):
        initial_count = self.node_a.access_count
        self.tree.find_by_name("tool_a")
        self.assertEqual(self.node_a.access_count, initial_count + 1)
        self.tree.find_by_id(self.node_a.id)
        self.assertEqual(self.node_a.access_count, initial_count + 2)

    def test_mfu_caching_ranking(self):
        # Access tool_b 5 times
        for _ in range(5):
            self.tree.find_by_name("tool_b")

        # Access tool_c 3 times
        for _ in range(3):
            self.tree.find_by_name("tool_c")

        # Access tool_a 1 time
        self.tree.find_by_name("tool_a")

        frequent_tools = self.tree.get_frequently_accessed_tools(limit=3)
        self.assertEqual(len(frequent_tools), 3)
        # Most frequent should be tool_b (5 accesses), followed by tool_c (3 accesses)
        self.assertEqual(frequent_tools[0].tool.name, "tool_b")
        self.assertEqual(frequent_tools[1].tool.name, "tool_c")

    def test_mfu_limit(self):
        frequent_1 = self.tree.get_frequently_accessed_tools(limit=1)
        self.assertEqual(len(frequent_1), 1)


if __name__ == "__main__":
    unittest.main()
