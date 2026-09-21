"""
Tests for BFS and DFS search traversals in Dendron.
"""

import unittest
from dendron.models import ToolDefinition
from dendron.tree import Dendron


class TestSearchTraversals(unittest.TestCase):
    def setUp(self):
        # Build tree:
        #           root (get_all_emails)
        #          /                     \
        #    left (respond_to_email)   right (extract_unread_emails)
        #                               /               \
        #                       read_email          extract_vital_info
        self.root_tool = ToolDefinition(name="get_all_emails", description="Fetch all emails", tags=["inbox", "root"])
        self.tree = Dendron(name="EmailSearchTree", root_tool=self.root_tool)

        self.left_tool = ToolDefinition(name="respond_to_email", description="Send response to an email", tags=["reply", "action"])
        self.right_tool = ToolDefinition(name="extract_unread_emails", description="Filter out unread emails", tags=["filter"])
        self.sub_left_tool = ToolDefinition(name="read_email", description="Read email body", tags=["reader"])
        self.sub_right_tool = ToolDefinition(name="extract_vital_information", description="Parse vital information from text", tags=["nlp", "extract"])

        self.left_node = self.tree.add_node(self.tree.root.id, self.left_tool, branch_label="respond_branch")
        self.right_node = self.tree.add_node(self.tree.root.id, self.right_tool, branch_label="unread_branch")
        self.sub_left_node = self.tree.add_node(self.right_node.id, self.sub_left_tool, branch_label="read_branch")
        self.sub_right_node = self.tree.add_node(self.right_node.id, self.sub_right_tool, branch_label="vital_branch")

    def test_bfs_order(self):
        # BFS should visit: root -> left -> right -> sub_left -> sub_right
        bfs_nodes = self.tree.search_bfs()
        names = [n.tool.name for n in bfs_nodes]
        expected = [
            "get_all_emails",
            "respond_to_email",
            "extract_unread_emails",
            "read_email",
            "extract_vital_information"
        ]
        self.assertEqual(names, expected)

    def test_dfs_order(self):
        # DFS should visit: root -> left -> right -> sub_left -> sub_right (deep branch first)
        dfs_nodes = self.tree.search_dfs()
        names = [n.tool.name for n in dfs_nodes]
        # Under root, left has no children, right has children:
        # root -> left -> right -> sub_left -> sub_right
        expected = [
            "get_all_emails",
            "respond_to_email",
            "extract_unread_emails",
            "read_email",
            "extract_vital_information"
        ]
        self.assertEqual(names, expected)

    def test_search_by_query_name(self):
        results = self.tree.search_bfs(query="unread")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool.name, "extract_unread_emails")

    def test_search_by_query_description(self):
        results = self.tree.search_bfs(query="vital information")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool.name, "extract_vital_information")

    def test_search_by_tag(self):
        results = self.tree.search_bfs(query="nlp")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool.name, "extract_vital_information")

    def test_search_by_branch_label(self):
        results = self.tree.search_bfs(query="respond_branch")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool.name, "respond_to_email")

    def test_search_by_predicate(self):
        # Match all tools that have 'extract' in name
        results = self.tree.search_dfs(predicate=lambda n: "extract" in n.tool.name)
        names = [n.tool.name for n in results]
        self.assertIn("extract_unread_emails", names)
        self.assertIn("extract_vital_information", names)
        self.assertEqual(len(names), 2)


if __name__ == "__main__":
    unittest.main()
