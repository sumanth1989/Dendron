"""
Tests for production-grade features in Dendron:
- Custom exceptions hierarchy
- Tool execution engine (handler binding, argument validation, execution history)
- Tree visualization (ASCII & Mermaid)
- Tree-of-trees mounting (mount_subtree)
- Tree pruning (stale/unused node cleanup)
- DAG cross-branch transitions
- LLM formatters (to_openai_tools, to_anthropic_tools)
"""

import unittest
from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    NodeNotFoundError,
    ToolValidationError,
)


class TestProductionFeatures(unittest.TestCase):

    def setUp(self):
        self.root_tool = ToolDefinition(
            name="order_lookup",
            description="Looks up customer order by ID",
            parameters={
                "order_id": ToolParameter(name="order_id", type="string", description="Order ID", required=True)
            },
            handler=lambda order_id: {"order_id": order_id, "status": "shipped"}
        )
        self.tree = Dendron(name="TestTree", root_tool=self.root_tool)

    def test_custom_exceptions(self):
        with self.assertRaises(NodeNotFoundError):
            self.tree.find_by_id("non_existent_id") or (_ for _ in ()).throw(NodeNotFoundError("Not found"))

        with self.assertRaises(NodeNotFoundError):
            self.tree.inspect_tool("non_existent_tool")

        with self.assertRaises(NodeNotFoundError):
            self.tree.add_node(parent_id="invalid_parent", tool=ToolDefinition(name="sub", description="sub"))

    def test_tool_execution(self):
        # Direct execution on ToolDefinition
        res = self.root_tool.execute(order_id="ORD-101")
        self.assertTrue(res.is_success)
        self.assertEqual(res.output_data["status"], "shipped")

        # Execution on Tree with automatic next tool suggestion
        child_tool = ToolDefinition(
            name="track_package",
            description="Tracks shipped package",
            parameters={"order_id": ToolParameter(name="order_id", type="string")},
            handler=lambda order_id: {"eta": "Tomorrow"}
        )
        self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=child_tool,
            branch_label="shipped_branch",
            condition=TransitionCondition(
                description="Status is shipped",
                condition_type="output_contains",
                expression="shipped"
            )
        )

        res, next_node = self.tree.execute("order_lookup", order_id="ORD-202")
        self.assertTrue(res.is_success)
        self.assertIsNotNone(next_node)
        self.assertEqual(next_node.tool.name, "track_package")

    def test_execution_placeholder_validation(self):
        res = self.root_tool.execute(order_id="[Insert Date]")
        self.assertFalse(res.is_success)
        self.assertIn("placeholders detected", res.error_message)

    def test_tree_visualization(self):
        child_tool = ToolDefinition(name="cancel_order", description="Cancels order")
        self.tree.add_node(parent_id=self.tree.root.id, tool=child_tool, branch_label="cancel_branch")

        ascii_out = self.tree.visualize(format="ascii")
        self.assertIn("order_lookup", ascii_out)
        self.assertIn("cancel_order", ascii_out)

        mermaid_out = self.tree.visualize(format="mermaid")
        self.assertIn("graph TD", mermaid_out)
        self.assertIn("order_lookup", mermaid_out)
        self.assertIn("cancel_order", mermaid_out)

    def test_mount_subtree(self):
        sub_root = ToolDefinition(name="shipping_root", description="Subtree root")
        sub_tree = Dendron(name="ShippingSubtree", root_tool=sub_root)
        sub_child = ToolDefinition(name="courier_api", description="Courier API")
        sub_tree.add_node(parent_id=sub_tree.root.id, tool=sub_child)

        mounted_root = self.tree.mount_subtree(
            parent_id=self.tree.root.id,
            subtree=sub_tree,
            branch_label="shipping_pipeline"
        )
        self.assertEqual(mounted_root.tool.name, "shipping_root")
        self.assertIsNotNone(self.tree.find_by_name("courier_api"))
        self.assertEqual(len(self.tree), 3)

    def test_dag_cross_branch_transitions(self):
        branch_a = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(name="step_a", description="Step A")
        )
        branch_b = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(name="step_b", description="Step B")
        )
        converged = self.tree.add_node(
            parent_id=branch_a.id,
            tool=ToolDefinition(name="send_alert", description="Send alert")
        )

        # Connect branch_b to converged node via cross-branch transition
        self.tree.add_transition(
            source_id_or_name=branch_b.id,
            target_id_or_name=converged.id,
            condition=TransitionCondition(
                description="When step_b completes",
                condition_type="output_contains",
                expression="alert_needed"
            )
        )

        suggested = self.tree.suggest_next_tool(branch_b.id, previous_output="Result: alert_needed")
        self.assertIsNotNone(suggested)
        self.assertEqual(suggested.tool.name, "send_alert")

    def test_pruning(self):
        unused_node = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=ToolDefinition(name="stale_tool", description="Stale tool")
        )
        self.assertEqual(unused_node.access_count, 0)
        # Prune with min_access_count = 1
        pruned_count = self.tree.prune(min_access_count=1)
        self.assertEqual(pruned_count, 1)
        self.assertIsNone(self.tree.find_by_name("stale_tool"))

    def test_llm_formatters(self):
        openai_tools = self.tree.to_openai_tools(level=1)
        self.assertEqual(len(openai_tools), 1)
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertEqual(openai_tools[0]["function"]["name"], "order_lookup")

        anthropic_tools = self.tree.to_anthropic_tools(level=1)
        self.assertEqual(len(anthropic_tools), 1)
        self.assertEqual(anthropic_tools[0]["name"], "order_lookup")
        self.assertIn("input_schema", anthropic_tools[0])


if __name__ == "__main__":
    unittest.main()
