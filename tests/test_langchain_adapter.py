"""
Comprehensive unit tests for dendron.langchain_adapter and LangChain interoperability.
Tests bidirectional conversion between LangChain tools and Dendron trees/nodes.
"""

import unittest
from unittest.mock import patch
from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    LangChainAdapter,
    DendronLangChainTool,
)

try:
    from langchain_core.tools import tool, StructuredTool, BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class TestLangChainAdapter(unittest.TestCase):

    def setUp(self):
        def add_numbers(a: int, b: int) -> int:
            """Adds two integers together."""
            return a + b

        self.add_handler = add_numbers
        self.add_tool_def = ToolDefinition(
            name="add_numbers",
            description="Adds two integers together.",
            parameters={
                "a": ToolParameter(name="a", type="integer", required=True, description="First integer"),
                "b": ToolParameter(name="b", type="integer", required=True, description="Second integer"),
            },
            handler=self.add_handler
        )
        self.tree = Dendron(name="MathTree", root_tool=self.add_tool_def)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_from_langchain_tool_decorated(self):
        @tool
        def multiply(x: int, y: int = 2) -> int:
            """Multiplies x by y."""
            return x * y

        tool_def = LangChainAdapter.from_langchain_tool(multiply)
        self.assertEqual(tool_def.name, "multiply")
        self.assertIn("Multiplies x by y", tool_def.description)
        self.assertIn("x", tool_def.parameters)
        self.assertIn("y", tool_def.parameters)
        self.assertTrue(tool_def.parameters["x"].required)
        self.assertFalse(tool_def.parameters["y"].required)
        self.assertEqual(tool_def.parameters["y"].default, 2)

        # Execution
        result = tool_def.execute(x=7, y=3)
        self.assertTrue(result.is_success)
        self.assertEqual(result.output_data, 21)

        # Default argument execution
        result_default = tool_def.execute(x=7)
        self.assertTrue(result_default.is_success)
        self.assertEqual(result_default.output_data, 14)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_from_langchain_tool_structured(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        st = StructuredTool.from_function(
            func=greet,
            name="greet_user",
            description="Greets a user by name."
        )

        tool_def = ToolDefinition.from_langchain(st)
        self.assertEqual(tool_def.name, "greet_user")
        self.assertEqual(tool_def.description, "Greets a user by name.")
        self.assertIn("name", tool_def.parameters)

        res = tool_def.execute(name="Alice")
        self.assertTrue(res.is_success)
        self.assertEqual(res.output_data, "Hello, Alice!")

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_from_langchain_tool_error_handling(self):
        @tool
        def failing_tool(val: str) -> str:
            """Always fails."""
            raise ValueError(f"Fatal error with {val}")

        tool_def = LangChainAdapter.from_langchain_tool(failing_tool)
        res = tool_def.execute(val="test")
        self.assertFalse(res.is_success)
        self.assertIn("Fatal error with test", res.error_message)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_from_langchain_tools_builds_tree(self):
        @tool
        def step_one(query: str) -> str:
            """Initial lookup."""
            return f"Data for {query}"

        @tool
        def step_two(data: str) -> str:
            """Process data."""
            return f"Processed: {data}"

        tree = Dendron.from_langchain_tools(
            tools=[step_one, step_two],
            name="PipelineTree",
            discovery_instructions="Run step_one then step_two"
        )
        self.assertEqual(tree.name, "PipelineTree")
        self.assertEqual(tree.root.tool.name, "step_one")
        self.assertEqual(len(tree), 2)
        self.assertIsNotNone(tree.find_by_name("step_two"))

        # Test execution through tree
        res = tree.root.execute(query="Agent Alpha")
        self.assertTrue(res.is_success)
        self.assertEqual(res.output_data, "Data for Agent Alpha")

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_from_langchain_tools_explicit_root(self):
        @tool
        def tool_a() -> str:
            """Tool A"""
            return "A"

        @tool
        def tool_b() -> str:
            """Tool B"""
            return "B"

        tree = LangChainAdapter.from_langchain_tools(
            tools=[tool_a, tool_b],
            name="ExplicitRootTree",
            root_tool_name="tool_b"
        )
        self.assertEqual(tree.root.tool.name, "tool_b")
        self.assertIsNotNone(tree.find_by_name("tool_a"))

    def test_from_langchain_tools_validation(self):
        with self.assertRaises(ValueError):
            LangChainAdapter.from_langchain_tools(tools=[])

        dummy_tool = ToolDefinition(name="dummy", description="d")
        with self.assertRaises(ValueError):
            LangChainAdapter.from_langchain_tools(tools=[dummy_tool], root_tool_name="non_existent")

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_to_langchain_tool_from_node_and_definition(self):
        # 1. From DendronNode
        lc_tool_node = self.tree.root.to_langchain()
        self.assertEqual(lc_tool_node.name, "add_numbers")
        self.assertEqual(lc_tool_node.description, "Adds two integers together.")
        self.assertIsNotNone(lc_tool_node.args_schema)

        res = lc_tool_node.invoke({"a": 10, "b": 20})
        self.assertEqual(res, 30)

        # 2. From ToolDefinition directly
        lc_tool_def = self.add_tool_def.to_langchain()
        self.assertEqual(lc_tool_def.name, "add_numbers")
        res2 = lc_tool_def.invoke({"a": 15, "b": 25})
        self.assertEqual(res2, 40)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_to_langchain_tools_tree(self):
        child_tool = ToolDefinition(
            name="subtract",
            description="Subtracts b from a",
            parameters={
                "a": ToolParameter(name="a", type="integer", required=True),
                "b": ToolParameter(name="b", type="integer", required=True)
            },
            handler=lambda a, b: a - b
        )
        self.tree.add_node(parent_id=self.tree.root.id, tool=child_tool)

        lc_tools = self.tree.to_langchain_tools()
        self.assertEqual(len(lc_tools), 2)
        names = [t.name for t in lc_tools]
        self.assertIn("add_numbers", names)
        self.assertIn("subtract", names)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_fetch_tools_for_model_langchain(self):
        tools_lc = self.tree.fetch_tools_for_model("langchain")
        self.assertEqual(len(tools_lc), 1)
        self.assertEqual(tools_lc[0].name, "add_numbers")

        tools_short = self.tree.fetch_tools_for_model("lc")
        self.assertEqual(len(tools_short), 1)

    def test_dendron_langchain_tool_duck_typed(self):
        tool_wrapper = DendronLangChainTool(self.add_tool_def)
        self.assertEqual(tool_wrapper.name, "add_numbers")
        self.assertIn("a", tool_wrapper.args)
        self.assertIn("b", tool_wrapper.args)

        # invoke with dict
        res1 = tool_wrapper.invoke({"a": 5, "b": 10})
        self.assertEqual(res1, 15)

        # run with kwargs
        res2 = tool_wrapper.run(a=20, b=30)
        self.assertEqual(res2, 50)

        # __call__
        res3 = tool_wrapper({"a": 100, "b": 200})
        self.assertEqual(res3, 300)

        # repr
        self.assertIn("<DendronLangChainTool", repr(tool_wrapper))

    def test_dendron_langchain_tool_single_arg_run(self):
        single_tool = ToolDefinition(
            name="uppercase",
            description="Converts to uppercase",
            parameters={"text": ToolParameter(name="text", type="string", required=True)},
            handler=lambda text: text.upper()
        )
        wrapper = DendronLangChainTool(single_tool)
        self.assertEqual(wrapper.run("hello"), "HELLO")
        self.assertEqual(wrapper("world"), "WORLD")

    def test_dendron_langchain_tool_error(self):
        failing_def = ToolDefinition(
            name="crash",
            description="Always crashes",
            parameters={},
            handler=lambda: 1 / 0
        )
        wrapper = DendronLangChainTool(failing_def)
        with self.assertRaises(RuntimeError):
            wrapper.invoke({})

    def test_fallback_when_langchain_not_installed(self):
        # Force ImportError on langchain_core.tools import
        with patch.dict("sys.modules", {"langchain_core.tools": None, "langchain_core": None}):
            tool_obj = LangChainAdapter.to_langchain_tool(self.add_tool_def)
            self.assertIsInstance(tool_obj, DendronLangChainTool)
            self.assertEqual(tool_obj.invoke({"a": 4, "b": 6}), 10)

    @unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core is not installed")
    def test_roundtrip_langchain_to_dendron_to_langchain(self):
        @tool
        def format_currency(amount: float, currency: str = "USD") -> str:
            """Formats amount as currency."""
            return f"{currency} {amount:.2f}"

        # 1. LangChain -> Dendron ToolDefinition
        dendron_tool = LangChainAdapter.from_langchain_tool(format_currency)
        self.assertEqual(dendron_tool.name, "format_currency")

        # 2. Build tree and add transition
        tree = Dendron(name="FinanceTree", root_tool=dendron_tool)
        res = tree.root.execute(amount=123.456, currency="EUR")
        self.assertTrue(res.is_success)
        self.assertEqual(res.output_data, "EUR 123.46")

        # 3. Dendron -> LangChain StructuredTool
        converted_back = tree.root.to_langchain()
        self.assertEqual(converted_back.name, "format_currency")
        invoked_result = converted_back.invoke({"amount": 99.9, "currency": "GBP"})
        self.assertEqual(invoked_result, "GBP 99.90")


if __name__ == "__main__":
    unittest.main()
