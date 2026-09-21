"""
Tests for prompt injection variables and prompt template rendering.
"""

import unittest
from dendron.models import ToolDefinition
from dendron.node import DendronNode


class TestPromptInjection(unittest.TestCase):
    def setUp(self):
        self.tool = ToolDefinition(name="respond_tool", description="Respond to user")
        self.node = DendronNode(
            tool=self.tool,
            system_prompt_template="You are a support agent for {company_name}. Always maintain a {tone} tone.",
            user_prompt_template="Customer {customer_name} wrote: '{message}'. Please respond.",
            prompt_variables={
                "company_name": "Acme Corp",
                "tone": "professional and empathetic"
            }
        )

    def test_render_system_prompt_stored_variables(self):
        prompt = self.node.get_system_prompt()
        self.assertEqual(prompt, "You are a support agent for Acme Corp. Always maintain a professional and empathetic tone.")

    def test_render_system_prompt_runtime_override(self):
        prompt = self.node.get_system_prompt(tone="concise")
        self.assertEqual(prompt, "You are a support agent for Acme Corp. Always maintain a concise tone.")

    def test_render_user_prompt_runtime_vars(self):
        prompt = self.node.get_user_prompt(customer_name="Alice", message="Can you help me reset my password?")
        self.assertEqual(prompt, "Customer Alice wrote: 'Can you help me reset my password?'. Please respond.")

    def test_set_prompt_variable(self):
        self.node.set_prompt_variable("company_name", "Globex Inc")
        prompt = self.node.get_system_prompt()
        self.assertIn("Globex Inc", prompt)

    def test_missing_variable_fallback(self):
        # When template has unfilled variables, fallback returns the raw template without crashing
        unfilled_node = DendronNode(
            tool=self.tool,
            system_prompt_template="Hello {missing_variable}!"
        )
        prompt = unfilled_node.get_system_prompt()
        self.assertEqual(prompt, "Hello {missing_variable}!")


if __name__ == "__main__":
    unittest.main()
