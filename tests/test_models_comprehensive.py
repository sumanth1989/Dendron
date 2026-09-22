"""
Comprehensive unit tests for dendron.models.
Covers ToolParameter, ToolDefinition, CompositeToolDefinition,
ToolResult, TransitionCondition, and PromptContext.
"""

import unittest
from dendron.models import (
    ToolParameter,
    ToolDefinition,
    CompositeToolDefinition,
    ToolResult,
    TransitionCondition,
    PromptContext,
)


class TestModelsComprehensive(unittest.TestCase):

    def test_tool_parameter_schema(self):
        param = ToolParameter(
            name="count",
            type="integer",
            description="Number of items",
            required=True,
            default=10,
            enum=[5, 10, 20]
        )
        schema = param.to_json_schema()
        self.assertEqual(schema["type"], "integer")
        self.assertEqual(schema["description"], "Number of items")
        self.assertEqual(schema["default"], 10)
        self.assertEqual(schema["enum"], [5, 10, 20])

    def test_tool_definition_argument_validation(self):
        tool = ToolDefinition(name="test_tool", description="Test tool")
        
        # Valid arguments
        valid, placeholders = tool.validate_arguments({"query": "python", "limit": 10})
        self.assertTrue(valid)
        self.assertEqual(placeholders, [])

        # Placeholders detection
        valid, placeholders = tool.validate_arguments({"date": "[Insert Date]", "notes": "normal text"})
        self.assertFalse(valid)
        self.assertIn("[Insert Date]", placeholders)

        valid, placeholders = tool.validate_arguments({"task": "[TODO] write tests"})
        self.assertFalse(valid)

        valid, placeholders = tool.validate_arguments({"task": "<TODO> fix bug"})
        self.assertFalse(valid)

        valid, placeholders = tool.validate_arguments({"name": "{{user_name}}"})
        self.assertFalse(valid)

    def test_tool_definition_execution(self):
        # 1. Tool with handler
        tool = ToolDefinition(
            name="add",
            description="Add numbers",
            parameters={
                "a": ToolParameter(name="a", type="number"),
                "b": ToolParameter(name="b", type="number")
            },
            handler=lambda a, b: a + b
        )
        result = tool.execute(a=3, b=5)
        self.assertTrue(result.is_success)
        self.assertEqual(result.output_data, 8)

        # 2. Tool without handler
        tool_no_handler = ToolDefinition(name="noop", description="No handler")
        result_no = tool_no_handler.execute(val=1)
        self.assertFalse(result_no.is_success)
        self.assertIn("No execution handler registered", result_no.error_message)

        # 3. Tool with placeholder in arguments
        result_placeholder = tool.execute(a=3, b="[TODO]")
        self.assertFalse(result_placeholder.is_success)
        self.assertIn("unresolved placeholders", result_placeholder.error_message)

        # 4. Tool handler raising exception
        def failing_handler(**kwargs):
            raise RuntimeError("Database connection failed")

        failing_tool = ToolDefinition(name="fail", description="Fails", handler=failing_handler)
        result_fail = failing_tool.execute()
        self.assertFalse(result_fail.is_success)
        self.assertIn("Database connection failed", result_fail.error_message)

    def test_tool_definition_mcp_serialization(self):
        tool = ToolDefinition(
            name="query_db",
            description="Queries database",
            parameters={
                "query": ToolParameter(name="query", type="string", description="SQL query", required=True),
                "timeout": ToolParameter(name="timeout", type="integer", description="Timeout in sec", required=False, default=30)
            },
            tags=["sql", "db"]
        )
        mcp_dict = tool.to_mcp_dict()
        self.assertEqual(mcp_dict["name"], "query_db")
        self.assertEqual(mcp_dict["description"], "Queries database")
        self.assertIn("query", mcp_dict["inputSchema"]["properties"])
        self.assertIn("query", mcp_dict["inputSchema"]["required"])
        self.assertNotIn("timeout", mcp_dict["inputSchema"]["required"])

        # Reconstruct from MCP dict
        reconstructed = ToolDefinition.from_mcp_dict(mcp_dict)
        self.assertEqual(reconstructed.name, tool.name)
        self.assertEqual(reconstructed.description, tool.description)
        self.assertIn("query", reconstructed.parameters)
        self.assertTrue(reconstructed.parameters["query"].required)
        self.assertFalse(reconstructed.parameters["timeout"].required)

    def test_composite_tool_definition(self):
        t1 = ToolDefinition(name="git_add", description="Stage files")
        t2 = ToolDefinition(name="git_commit", description="Commit staged files")
        composite = CompositeToolDefinition(
            name="git_save",
            description="Stage and commit files",
            sub_tools=[t1, t2],
            execution_mode="sequential"
        )
        self.assertEqual(len(composite.sub_tools), 2)
        self.assertEqual(composite.execution_mode, "sequential")

    def test_tool_result(self):
        res = ToolResult(
            tool_name="test_tool",
            input_args={"x": 10},
            output_data="Done",
            status="success",
            metadata={"latency_ms": 12}
        )
        self.assertTrue(res.is_success)
        self.assertEqual(res.metadata["latency_ms"], 12)
        self.assertIsNotNone(res.call_id)

        res_err = ToolResult(tool_name="test_tool", status="error", error_message="Fatal error")
        self.assertFalse(res_err.is_success)
        self.assertEqual(res_err.error_message, "Fatal error")

    def test_transition_condition_evaluation(self):
        # 1. 'always'
        c_always = TransitionCondition(description="Always proceed", condition_type="always")
        self.assertTrue(c_always.evaluate(None))
        self.assertTrue(c_always.evaluate("random output"))

        # 2. 'output_contains'
        c_contains = TransitionCondition(
            description="Check for error",
            condition_type="output_contains",
            expression="error_503"
        )
        self.assertTrue(c_contains.evaluate("Received ERROR_503: Service Unavailable"))
        self.assertFalse(c_contains.evaluate("Status 200 OK"))

        # 3. 'key_equals'
        c_key = TransitionCondition(
            description="Check status key",
            condition_type="key_equals",
            expression="status:shipped"
        )
        self.assertTrue(c_key.evaluate({"status": "shipped", "tracking": "123"}))
        self.assertTrue(c_key.evaluate({"status": "SHIPPED"}))  # case insensitive
        self.assertFalse(c_key.evaluate({"status": "delivered"}))
        self.assertFalse(c_key.evaluate("not a dict"))

        # 4. 'custom' lambda
        c_custom = TransitionCondition(
            description="Check value > 100",
            condition_type="custom",
            expression=lambda out: isinstance(out, (int, float)) and out > 100
        )
        self.assertTrue(c_custom.evaluate(150))
        self.assertFalse(c_custom.evaluate(50))
        self.assertFalse(c_custom.evaluate("invalid"))

        # 5. 'custom' lambda raising exception returns False
        def bad_lambda(out):
            raise ValueError("Boom")

        c_bad = TransitionCondition(description="Bad lambda", condition_type="custom", expression=bad_lambda)
        self.assertFalse(c_bad.evaluate(10))

        # 6. 'always' with expression converts to 'output_contains'
        c_auto = TransitionCondition(description="Auto convert", condition_type="always", expression="keyword")
        self.assertEqual(c_auto.condition_type, "output_contains")
        self.assertTrue(c_auto.evaluate("this has keyword"))

    def test_prompt_context_rendering(self):
        ctx = PromptContext(
            system_prompt_template="You are an assistant for {domain}. Model: {model}.",
            user_prompt_template="Help customer {customer_id}.",
            variables={"domain": "Banking", "model": "GPT-4"}
        )
        # Render system prompt
        sys_p = ctx.render_system_prompt()
        self.assertEqual(sys_p, "You are an assistant for Banking. Model: GPT-4.")

        # Render with override
        sys_override = ctx.render_system_prompt(extra_vars={"model": "Claude-3.5"})
        self.assertEqual(sys_override, "You are an assistant for Banking. Model: Claude-3.5.")

        # Render user prompt
        usr_p = ctx.render_user_prompt(extra_vars={"customer_id": "C-991"})
        self.assertEqual(usr_p, "Help customer C-991.")

        # Missing variable fallback
        ctx_missing = PromptContext(system_prompt_template="Hello {missing_var}")
        self.assertEqual(ctx_missing.render_system_prompt(), "Hello {missing_var}")

        # Empty templates
        ctx_empty = PromptContext()
        self.assertIsNone(ctx_empty.render_system_prompt())
        self.assertIsNone(ctx_empty.render_user_prompt())


if __name__ == "__main__":
    unittest.main()
