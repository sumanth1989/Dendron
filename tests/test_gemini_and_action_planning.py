"""
Tests for Google Gemini tool formatting, fetch model integration,
and Autonomous Action Planning across Dendron trees.
"""

import unittest
from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    ActionStep,
    ActionPlan,
    AutonomousActionPlanner,
)


class TestGeminiAndActionPlanning(unittest.TestCase):

    def setUp(self):
        self.root_tool = ToolDefinition(
            name="lookup_order",
            description="Retrieves customer order by ID",
            parameters={
                "order_id": ToolParameter(name="order_id", type="string", description="Order ID", required=True),
                "include_history": ToolParameter(name="include_history", type="boolean", description="Include order history", default=False, required=False),
            },
            handler=lambda order_id, include_history=False: {
                "order_id": order_id,
                "status": "delivered",
                "item": "Wireless Headphones",
                "price": 49.99
            }
        )
        self.tree = Dendron(name="SupportTree", root_tool=self.root_tool)

        # Attach return tool
        self.return_tool = ToolDefinition(
            name="process_return",
            description="Processes return for a delivered order",
            parameters={
                "order_id": ToolParameter(name="order_id", type="string", required=True),
                "reason": ToolParameter(name="reason", type="string", required=True),
            },
            handler=lambda order_id, reason: {
                "return_id": f"RET-{order_id}",
                "status": "return_processed",
                "reason": reason
            }
        )
        self.return_node = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=self.return_tool,
            branch_label="delivered_branch",
            condition=TransitionCondition(
                description="Status is delivered",
                condition_type="output_contains",
                expression="delivered"
            )
        )

        # Attach instant refund tool
        self.refund_tool = ToolDefinition(
            name="issue_refund",
            description="Issues immediate monetary refund to customer",
            parameters={
                "order_id": ToolParameter(name="order_id", type="string", required=True),
                "amount": ToolParameter(name="amount", type="number", required=True),
            },
            handler=lambda order_id, amount: {
                "refund_id": f"REF-{order_id}",
                "amount": amount,
                "status": "refund_issued"
            }
        )
        self.refund_node = self.tree.add_node(
            parent_id=self.return_node.id,
            tool=self.refund_tool,
            branch_label="refund_branch",
            condition=TransitionCondition(
                description="When customer is eligible for refund",
                condition_type="always"
            )
        )

    def test_to_gemini_tools_formatting(self):
        """Verify Google Gemini tool schema complies with FunctionDeclaration format."""
        gemini_tools = self.tree.to_gemini_tools()
        self.assertEqual(len(gemini_tools), 3)

        root_gemini = gemini_tools[0]
        self.assertEqual(root_gemini["name"], "lookup_order")
        self.assertEqual(root_gemini["description"], "Retrieves customer order by ID")
        self.assertIn("parameters", root_gemini)
        params = root_gemini["parameters"]

        # Gemini requires UPPERCASE OpenAPI types
        self.assertEqual(params["type"], "OBJECT")
        self.assertEqual(params["properties"]["order_id"]["type"], "STRING")
        self.assertEqual(params["properties"]["include_history"]["type"], "BOOLEAN")
        self.assertIn("order_id", params["required"])

    def test_fetch_tools_for_model(self):
        """Verify fetch_tools_for_model correctly routes to Gemini, OpenAI, Claude, and MCP."""
        # Gemini
        gemini_tools = self.tree.fetch_tools_for_model("gemini")
        self.assertEqual(len(gemini_tools), 3)
        self.assertEqual(gemini_tools[0]["parameters"]["type"], "OBJECT")

        # Google alias
        google_tools = self.tree.fetch_tools_for_model("google")
        self.assertEqual(len(google_tools), 3)

        # OpenAI
        openai_tools = self.tree.fetch_tools_for_model("openai")
        self.assertEqual(len(openai_tools), 3)
        self.assertEqual(openai_tools[0]["type"], "function")

        # Anthropic
        claude_tools = self.tree.fetch_tools_for_model("claude")
        self.assertEqual(len(claude_tools), 3)
        self.assertIn("input_schema", claude_tools[0])

        # MCP
        mcp_tools = self.tree.fetch_tools_for_model("mcp")
        self.assertEqual(len(mcp_tools), 3)
        self.assertIn("inputSchema", mcp_tools[0])

        # Invalid provider
        with self.assertRaises(ValueError):
            self.tree.fetch_tools_for_model("unsupported_provider")

    def test_autonomous_action_planner_plan(self):
        """Test autonomous plan formulation for a user goal."""
        planner = AutonomousActionPlanner(self.tree)
        plan = planner.plan(goal="Customer wants a refund for damaged item ORD-999")

        self.assertIsInstance(plan, ActionPlan)
        self.assertEqual(plan.goal, "Customer wants a refund for damaged item ORD-999")
        self.assertGreater(len(plan.steps), 0)
        self.assertEqual(plan.status, "planned")

        # First step should be order lookup or refund path
        tool_names = [s.tool_name for s in plan.steps]
        self.assertIn("lookup_order", tool_names)

    def test_action_plan_fetch_tools_for_model(self):
        """Test ActionPlan.fetch_tools_for_model for Gemini and other providers."""
        plan = self.tree.create_action_plan(goal="Process return for customer")
        self.assertIsInstance(plan, ActionPlan)

        # Fetch for Gemini
        gemini_plan_tools = plan.fetch_tools_for_model("gemini")
        self.assertIsInstance(gemini_plan_tools, list)
        for t in gemini_plan_tools:
            self.assertEqual(t["parameters"]["type"], "OBJECT")

        # Fetch for OpenAI
        openai_plan_tools = plan.fetch_tools_for_model("openai")
        self.assertIsInstance(openai_plan_tools, list)
        for t in openai_plan_tools:
            self.assertEqual(t["type"], "function")

    def test_autonomous_action_planning_and_execution(self):
        """Test end-to-end plan and execute loop passing context across steps."""
        input_context = {
            "order_id": "ORD-5501",
            "reason": "Damaged on arrival",
            "amount": 49.99
        }
        results = self.tree.plan_and_execute(
            goal="Issue refund for order ORD-5501",
            input_context=input_context
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for res in results:
            self.assertTrue(res.is_success)

    def test_action_plan_step_by_step_execution(self):
        """Test ActionPlan step-by-step manual execution and completion detection."""
        plan = self.tree.create_action_plan(goal="Lookup order and return")
        self.assertFalse(plan.is_complete)

        # Execute step 1
        res1, finished1 = plan.execute_next(order_id="ORD-777")
        self.assertIsNotNone(res1)
        self.assertTrue(res1.is_success)

        # Step through until complete
        while not plan.is_complete:
            step = plan.get_current_step()
            res, finished = plan.execute_next(order_id="ORD-777", reason="Wrong item", amount=25.00)
            if finished:
                break

        self.assertTrue(plan.is_complete)
        plan_dict = plan.to_dict()
        self.assertIn("steps", plan_dict)
        self.assertEqual(plan_dict["status"], "completed")


if __name__ == "__main__":
    unittest.main()
