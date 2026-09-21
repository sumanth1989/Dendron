"""
Tests for dynamic tree building from agent experience and transition conditions.
"""

import unittest
from dendron.models import ToolDefinition, ToolResult, TransitionCondition
from dendron.tree import Dendron


class TestAgentExperience(unittest.TestCase):
    def setUp(self):
        self.root_tool = ToolDefinition(name="fetch_tickets", description="Fetch support tickets")
        self.tree = Dendron(name="SupportAgentTree", root_tool=self.root_tool)

    def test_record_agent_experience_always(self):
        triage_tool = ToolDefinition(name="triage_ticket", description="Triage incoming ticket")
        node = self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=triage_tool,
            trigger_condition_description="Always triage after fetching tickets",
            condition_type="always",
            experience_note="Learned that fetching tickets must immediately be followed by triage"
        )
        self.assertEqual(node.tool.name, "triage_ticket")
        self.assertEqual(len(self.tree.root.children), 1)
        self.assertEqual(node.experience_notes[0], "Learned that fetching tickets must immediately be followed by triage")

        # Test suggest_next_tool
        suggested = self.tree.suggest_next_tool(self.tree.root.id, previous_output={"tickets": [1, 2]})
        self.assertIsNotNone(suggested)
        self.assertEqual(suggested.tool.name, "triage_ticket")

    def test_transition_output_contains(self):
        urgent_tool = ToolDefinition(name="escalate_urgent", description="Escalate urgent tickets")
        standard_tool = ToolDefinition(name="process_standard", description="Process normal tickets")

        self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=urgent_tool,
            trigger_condition_description="Output mentions urgent or high priority",
            condition_type="output_contains",
            condition_expression="urgent"
        )
        self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=standard_tool,
            trigger_condition_description="Output is standard",
            condition_type="output_contains",
            condition_expression="standard"
        )

        # Output with urgent
        urgent_next = self.tree.suggest_next_tool(self.tree.root.id, previous_output="Found 1 urgent ticket: server down")
        self.assertIsNotNone(urgent_next)
        self.assertEqual(urgent_next.tool.name, "escalate_urgent")

        # Output with standard
        standard_next = self.tree.suggest_next_tool(self.tree.root.id, previous_output="All items are standard requests")
        self.assertIsNotNone(standard_next)
        self.assertEqual(standard_next.tool.name, "process_standard")

    def test_transition_key_equals(self):
        vip_tool = ToolDefinition(name="vip_support", description="Handle VIP customer")
        self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=vip_tool,
            trigger_condition_description="Customer tier is VIP",
            condition_type="key_equals",
            condition_expression="tier:vip"
        )

        match = self.tree.suggest_next_tool(self.tree.root.id, previous_output={"tier": "VIP", "user": "Alice"})
        self.assertIsNotNone(match)
        self.assertEqual(match.tool.name, "vip_support")

        no_match = self.tree.suggest_next_tool(self.tree.root.id, previous_output={"tier": "free", "user": "Bob"})
        self.assertIsNone(no_match)

    def test_custom_callable_condition(self):
        bulk_tool = ToolDefinition(name="bulk_process", description="Process bulk items")
        self.tree.record_agent_experience(
            parent_id=self.tree.root.id,
            next_tool=bulk_tool,
            trigger_condition_description="More than 5 tickets returned",
            condition_type="custom",
            condition_expression=lambda output: isinstance(output, list) and len(output) > 5
        )

        match = self.tree.suggest_next_tool(self.tree.root.id, previous_output=[1, 2, 3, 4, 5, 6])
        self.assertIsNotNone(match)
        self.assertEqual(match.tool.name, "bulk_process")

        no_match = self.tree.suggest_next_tool(self.tree.root.id, previous_output=[1, 2])
        self.assertIsNone(no_match)

    def test_record_execution(self):
        tool = ToolDefinition(name="test_tool", description="test")
        node = self.tree.add_node(self.tree.root.id, tool)

        success_res = ToolResult(tool_name="test_tool", status="success", output_data="ok")
        node.record_execution(success_res, note="Completed with 200ms latency")

        self.assertEqual(node.success_count, 1)
        self.assertEqual(node.failure_count, 0)
        self.assertEqual(len(node.experience_notes), 1)

        fail_res = ToolResult(tool_name="test_tool", status="error", error_message="timeout")
        node.record_execution(fail_res)
        self.assertEqual(node.failure_count, 1)


if __name__ == "__main__":
    unittest.main()
