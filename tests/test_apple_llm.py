"""
Tests for Apple LLM integration and adapter in Dendron.
"""

import unittest
from dendron.models import ToolDefinition, ToolParameter
from dendron.tree import Dendron

try:
    from dendron.examples.apple_llm_test import AppleLLMClient, build_email_tree
except ImportError:
    AppleLLMClient = None
    build_email_tree = None


@unittest.skipIf(AppleLLMClient is None, "apple_llm_test.py is not present")
class TestAppleLLMIntegration(unittest.TestCase):
    def setUp(self):
        if AppleLLMClient is None:
            self.skipTest("apple_llm_test.py is not present")
        self.client = AppleLLMClient()
        self.tree = build_email_tree()

    def test_client_generation(self):
        # Generates response either via native Apple Foundation Models or simulator fallback
        response = self.client.generate(
            prompt="Which tool should I call to filter unread emails?",
            system_prompt="You are a tool router."
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_tree_with_apple_llm_flow(self):
        # 1. Inspect root discovery instructions
        self.assertIn("HOW TO NAVIGATE", self.tree.discovery_instructions)

        # 2. Injected system and user prompts
        unread_node = self.tree.find_by_name("extract_unread_emails")
        self.assertIsNotNone(unread_node)
        sys_prompt = unread_node.get_system_prompt()
        self.assertIn("Priority threshold: HIGH", sys_prompt)

        # 3. Dynamic experience recording
        archive_tool = ToolDefinition(
            name="archive_email",
            description="Archive email",
            parameters={"email_id": ToolParameter(name="email_id", type="string")}
        )
        self.tree.record_agent_experience(
            parent_id=unread_node.id,
            next_tool=archive_tool,
            trigger_condition_description="Invoice or receipt detected",
            condition_type="output_contains",
            condition_expression="invoice"
        )

        # 4. Suggestion prioritizing specific condition
        match = self.tree.suggest_next_tool(unread_node.id, previous_output="Received invoice #12345")
        self.assertIsNotNone(match)
        self.assertEqual(match.tool.name, "archive_email")

        # 5. Non-matching output falls back to unconditioned child (read_email)
        fallback = self.tree.suggest_next_tool(unread_node.id, previous_output="Regular unread message")
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.tool.name, "read_email")


if __name__ == "__main__":
    unittest.main()
