"""
LangChain Adapter for Dendron.
Enables seamless bidirectional interoperability between Dendron trees/tools
and the LangChain ecosystem (BaseTool, StructuredTool, @tool, AgentExecutor).
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING
from .models import ToolDefinition, ToolParameter, ToolResult

if TYPE_CHECKING:
    from .node import DendronNode
    from .tree import Dendron


class DendronLangChainTool:
    """
    Duck-typed LangChain-compatible tool wrapper for Dendron tools.
    Provides invoke(), run(), and args interface compatible with LangChain's BaseTool.
    Used when langchain-core is not installed, or as a standalone lightweight bridge.
    """

    def __init__(
        self,
        tool_definition: ToolDefinition,
        node: Optional[DendronNode] = None,
    ):
        self._tool = tool_definition
        self._node = node
        self.name: str = tool_definition.name
        self.description: str = tool_definition.description
        self.args: Dict[str, Any] = {
            p_name: p.to_json_schema()
            for p_name, p in tool_definition.parameters.items()
        }

    def invoke(self, input_dict: Any, config: Optional[Any] = None) -> Any:
        """LangChain standard execution method."""
        args = input_dict if isinstance(input_dict, dict) else {"input": input_dict}
        if self._node:
            result = self._node.execute(**args)
        else:
            result = self._tool.execute(**args)

        if result.is_success:
            return result.output_data
        raise RuntimeError(result.error_message or f"Execution failed for tool '{self.name}'")

    def run(self, tool_input: Any = None, **kwargs: Any) -> Any:
        """LangChain legacy execution method."""
        if isinstance(tool_input, dict):
            args = {**tool_input, **kwargs}
        elif tool_input is not None:
            param_names = list(self._tool.parameters.keys())
            if len(param_names) == 1:
                args = {param_names[0]: tool_input, **kwargs}
            else:
                args = {"input": tool_input, **kwargs}
        else:
            args = kwargs
        return self.invoke(args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tool_input = args[0] if args else None
        return self.run(tool_input=tool_input, **kwargs)

    def __repr__(self) -> str:
        return f"<DendronLangChainTool(name='{self.name}', description='{self.description}')>"


class LangChainAdapter:
    """
    Adapter between Dendron tool execution trees and LangChain tools.
    Supports converting LangChain tools into Dendron ToolDefinitions/Trees,
    and converting Dendron trees/nodes into LangChain BaseTool/StructuredTool instances.
    """

    @classmethod
    def from_langchain_tool(cls, tool: Any, tags: Optional[List[str]] = None) -> ToolDefinition:
        """
        Converts any LangChain tool (BaseTool, StructuredTool, @tool function)
        into a Dendron ToolDefinition.
        """
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "langchain_tool")
        description = getattr(tool, "description", "") or (tool.__doc__ or "")

        # Extract parameters
        parameters: Dict[str, ToolParameter] = {}
        required_list: List[str] = []

        # 1. Inspect args_schema if present
        args_schema = getattr(tool, "args_schema", None)
        schema_dict: Dict[str, Any] = {}
        if args_schema:
            if hasattr(args_schema, "model_json_schema"):
                schema_dict = args_schema.model_json_schema()
            elif hasattr(args_schema, "schema"):
                schema_dict = args_schema.schema()

        if schema_dict:
            props = schema_dict.get("properties", {})
            required_list = schema_dict.get("required", [])
            for p_name, p_info in props.items():
                parameters[p_name] = ToolParameter(
                    name=p_name,
                    type=p_info.get("type", "string"),
                    description=p_info.get("description", ""),
                    required=(p_name in required_list),
                    default=p_info.get("default"),
                    enum=p_info.get("enum")
                )

        # 2. Fallback to tool.args
        if not parameters and hasattr(tool, "args") and isinstance(tool.args, dict):
            for p_name, p_info in tool.args.items():
                if isinstance(p_info, dict):
                    parameters[p_name] = ToolParameter(
                        name=p_name,
                        type=p_info.get("type", "string"),
                        description=p_info.get("description", ""),
                        required=(p_name in required_list or "default" not in p_info),
                        default=p_info.get("default"),
                        enum=p_info.get("enum")
                    )

        # 3. Create bound execution handler
        def _langchain_handler(**kwargs: Any) -> Any:
            # Try invoke first (modern LangChain standard)
            if hasattr(tool, "invoke"):
                try:
                    return tool.invoke(kwargs)
                except Exception:
                    # If single param, try primitive
                    if len(kwargs) == 1:
                        val = next(iter(kwargs.values()))
                        try:
                            return tool.invoke(val)
                        except Exception:
                            pass

            # Try run (LangChain legacy)
            if hasattr(tool, "run"):
                try:
                    return tool.run(kwargs)
                except TypeError:
                    return tool.run(**kwargs)

            # Fallback to direct call
            if callable(tool):
                return tool(**kwargs)

            raise RuntimeError(f"Unable to invoke LangChain tool '{name}'.")

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=_langchain_handler,
            tags=tags or ["langchain"]
        )

    @classmethod
    def from_langchain_tools(
        cls,
        tools: List[Any],
        name: str = "LangChainTree",
        root_tool_name: Optional[str] = None,
        discovery_instructions: str = ""
    ) -> Dendron:
        """
        Builds a Dendron tree from a list of LangChain tools.
        """
        from .tree import Dendron

        if not tools:
            raise ValueError("tools list cannot be empty.")

        tool_defs = [cls.from_langchain_tool(t) for t in tools]

        # Select root
        root_def: ToolDefinition
        if root_tool_name:
            matching = [t for t in tool_defs if t.name == root_tool_name]
            if not matching:
                raise ValueError(f"Root tool '{root_tool_name}' not found in provided LangChain tools.")
            root_def = matching[0]
            remaining = [t for t in tool_defs if t.name != root_tool_name]
        else:
            root_def = tool_defs[0]
            remaining = tool_defs[1:]

        tree = Dendron(
            name=name,
            root_tool=root_def,
            discovery_instructions=discovery_instructions
        )

        for t in remaining:
            tree.add_node(
                parent_id=tree.root.id,
                tool=t,
                branch_label=t.name
            )

        return tree

    @classmethod
    def to_langchain_tool(cls, node_or_tool: Union[DendronNode, ToolDefinition]) -> Any:
        """
        Converts a Dendron ToolDefinition or DendronNode into a LangChain tool.
        Returns a StructuredTool if langchain_core is installed, otherwise DendronLangChainTool.
        """
        tool_def = node_or_tool.tool if hasattr(node_or_tool, "tool") else node_or_tool
        node = node_or_tool if hasattr(node_or_tool, "tool") else None

        try:
            from langchain_core.tools import StructuredTool
            from pydantic import create_model, Field
            from typing import Optional as Opt

            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict,
            }

            fields: Dict[str, Any] = {}
            for p_name, param in tool_def.parameters.items():
                py_type = type_mapping.get(param.type.lower(), Any)
                if param.required:
                    fields[p_name] = (py_type, Field(description=param.description or ""))
                else:
                    fields[p_name] = (Opt[py_type], Field(default=param.default, description=param.description or ""))

            args_model = create_model(f"{tool_def.name}Args", **fields) if fields else None

            def _execution_proxy(**kwargs: Any) -> Any:
                if node:
                    res = node.execute(**kwargs)
                else:
                    res = tool_def.execute(**kwargs)
                if res.is_success:
                    return res.output_data
                raise RuntimeError(res.error_message or f"Execution failed for tool '{tool_def.name}'")

            kwargs_to_pass: Dict[str, Any] = {
                "func": _execution_proxy,
                "name": tool_def.name,
                "description": tool_def.description,
            }
            if args_model:
                kwargs_to_pass["args_schema"] = args_model

            return StructuredTool.from_function(**kwargs_to_pass)

        except ImportError:
            # Fallback to duck-typed tool
            return DendronLangChainTool(tool_definition=tool_def, node=node)

    @classmethod
    def to_langchain_tools(cls, tree_or_nodes: Union[Dendron, Sequence[DendronNode]]) -> List[Any]:
        """
        Converts all nodes in a Dendron tree or a sequence of nodes into LangChain tools.
        """
        if hasattr(tree_or_nodes, "search_bfs"):
            nodes = tree_or_nodes.search_bfs()
        else:
            nodes = list(tree_or_nodes)

        return [cls.to_langchain_tool(n) for n in nodes]
