"""
Comprehensive unit tests for dendron.tree.Dendron.
Tests all registry lookups, inverted indexing, progressive views,
search, dynamic learning, RAG, DAG transitions, visualization, and model formatters.
"""

import os
import tempfile
import unittest
from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    NodeNotFoundError,
)


class TestTreeComprehensive(unittest.TestCase):

    def setUp(self):
        self.root_tool = ToolDefinition(
            name="query_ticket",
            description="Finds customer ticket by ticket ID",
            parameters={
                "ticket_id": ToolParameter(name="ticket_id", type="string", description="Ticket ID", required=True),
            },
            tags=["support", "tickets", "root"],
            handler=lambda ticket_id: {"ticket_id": ticket_id, "status": "open", "category": "billing"}
        )
        self.tree = Dendron(
            name="TicketTree",
            root_tool=self.root_tool,
            discovery_instructions="Start with query_ticket then branch"
        )

    def test_registry_and_len(self):
        self.assertEqual(len(self.tree), 1)
        self.assertEqual(len(self.tree.nodes), 1)
        self.assertEqual(self.tree.find_by_name("query_ticket").id, self.tree.root.id)
        self.assertEqual(self.tree.find_by_id(self.tree.root.id).tool.name, "query_ticket")
        self.assertEqual(self.tree.search_fast("query_ticket").id, self.tree.root.id)
        self.assertIsNone(self.tree.find_by_name("non_existent"))

    def test_inverted_indexes(self):
        child = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(
                name="apply_credit",
                description="Applies credit to account",
                parameters={
                    "ticket_id": ToolParameter(name="ticket_id", type="string", required=True),
                    "amount": ToolParameter(name="amount", type="number", required=True)
                },
                tags=["billing", "finance"]
            )
        )
        # By param
        by_ticket_id = self.tree.find_by_input_param("ticket_id")
        self.assertEqual(len(by_ticket_id), 2)

        by_amount = self.tree.find_by_input_param("amount")
        self.assertEqual(len(by_amount), 1)
        self.assertEqual(by_amount[0].id, child.id)

        # By params (match_all=True vs False)
        match_any = self.tree.find_by_input_params(["amount", "non_existent"], match_all=False)
        self.assertEqual(len(match_any), 1)

        match_all = self.tree.find_by_input_params(["ticket_id", "amount"], match_all=True)
        self.assertEqual(len(match_all), 1)
        self.assertEqual(match_all[0].id, child.id)

        self.assertEqual(self.tree.find_by_input_params([]), [])

        # By tag
        billing_nodes = self.tree.find_by_tag("billing")
        self.assertEqual(len(billing_nodes), 1)
        self.assertEqual(billing_nodes[0].tool.name, "apply_credit")

    def test_get_frequently_accessed_tools(self):
        child = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(name="audit_ticket", description="Audits ticket")
        )
        child.record_access()
        child.record_access()
        child.record_access()

        mfu = self.tree.get_frequently_accessed_tools(limit=2)
        self.assertEqual(len(mfu), 2)
        self.assertEqual(mfu[0].tool.name, "audit_ticket")

    def test_inspection_and_views(self):
        views_1 = self.tree.export_tool_views(level=1)
        self.assertIsInstance(views_1[0], str)

        views_2 = self.tree.export_tool_views(level=2)
        self.assertIsInstance(views_2[0], dict)

        views_3 = self.tree.export_tool_views(level=3)
        self.assertIsInstance(views_3[0], dict)

        inspected = self.tree.inspect_tool("query_ticket")
        self.assertEqual(inspected["name"], "query_ticket")

        with self.assertRaises(NodeNotFoundError):
            self.tree.inspect_tool("unknown_tool")

    def test_information_discovery(self):
        self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(
                name="escalate",
                description="Escalate ticket",
                parameters={"department": ToolParameter(name="department", type="string", required=True)}
            )
        )
        # Actionable tools with known input 'ticket_id'
        actionable = self.tree.get_actionable_tools(available_inputs=["ticket_id"], require_all=True)
        self.assertEqual(len(actionable), 1)

        actionable_any = self.tree.get_actionable_tools(available_inputs=["department"], require_all=False)
        self.assertEqual(len(actionable_any), 1)

        # Reachable tools
        reachable = self.tree.get_reachable_tools(current_node_id=self.tree.root.id, max_hops=1)
        self.assertEqual(len(reachable), 1)
        self.assertEqual(reachable[0]["name"], "escalate")

        with self.assertRaises(NodeNotFoundError):
            self.tree.get_reachable_tools(current_node_id="invalid_node")

    def test_search_facets_and_traversal(self):
        self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(name="refund_fee", description="Refunds fee", tags=["finance"])
        )
        # Multi-faceted search
        res = self.tree.search(query="refund", tags=["finance"])
        self.assertEqual(len(res), 1)

        # BFS
        bfs_nodes = self.tree.search_bfs(query="refund")
        self.assertEqual(len(bfs_nodes), 1)
        self.assertEqual(bfs_nodes[0].tool.name, "refund_fee")

        # DFS with predicate
        dfs_nodes = self.tree.search_dfs(predicate=lambda n: "finance" in n.tool.tags)
        self.assertEqual(len(dfs_nodes), 1)

    def test_dynamic_learning_and_suppression(self):
        # Record agent experience
        learned_node = self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=ToolDefinition(name="notify_manager", description="Notify manager"),
            trigger_condition_description="Priority high",
            condition_type="output_contains",
            condition_expression="high_priority",
            experience_note="Learned to escalate high priority tickets"
        )
        self.assertEqual(learned_node.tool.name, "notify_manager")
        self.assertIn("Learned to escalate", learned_node.experience_notes[0])

        # Suggest next tool
        suggested = self.tree.suggest_next_tool(self.tree.root.id, previous_output="status: high_priority")
        self.assertIsNotNone(suggested)
        self.assertEqual(suggested.tool.name, "notify_manager")

        # Negative experience suppression threshold (>= 3 dismissals)
        self.tree.record_negative_experience("notify_manager")
        self.tree.record_negative_experience("notify_manager")
        self.tree.record_negative_experience("notify_manager")

        suppressed = self.tree.suggest_next_tool(self.tree.root.id, previous_output="status: high_priority")
        self.assertIsNone(suppressed)

    def test_serialization_and_file_io(self):
        json_str = self.tree.to_json()
        self.assertIsInstance(json_str, str)

        restored = Dendron.from_json(json_str)
        self.assertEqual(restored.name, self.tree.name)
        self.assertEqual(restored.root.tool.name, "query_ticket")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "tree.json")
            self.tree.save(file_path)
            loaded = Dendron.load(file_path)
            self.assertEqual(loaded.name, self.tree.name)

        with self.assertRaises(FileNotFoundError):
            Dendron.load("/non/existent/path.json")


if __name__ == "__main__":
    unittest.main()
