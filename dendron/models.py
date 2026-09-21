"""
Data models for Dendron, representing tool definitions, schemas,
results, transitions, and prompt context. Designed with full compatibility
for the Model Context Protocol (MCP) tool specification.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolParameter:
    """Represents a parameter/argument for a tool call."""
    name: str
    type: str = "string"  # string, number, integer, boolean, object, array
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert parameter to JSON Schema format compatible with MCP."""
        schema: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.enum is not None:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolDefinition:
    """
    Complete tool definition matching the Model Context Protocol (MCP)
    tool specification standard.
    """
    name: str
    description: str
    parameters: Dict[str, ToolParameter] = field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = self._generate_mcp_input_schema()

    def _generate_mcp_input_schema(self) -> Dict[str, Any]:
        """Generates standard MCP inputSchema dictionary."""
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in self.parameters.items():
            properties[param_name] = param.to_json_schema()
            if param.required:
                required.append(param_name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def to_mcp_dict(self) -> Dict[str, Any]:
        """Export tool definition in exact MCP server schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema or self._generate_mcp_input_schema(),
        }

    @classmethod
    def from_mcp_dict(cls, data: Dict[str, Any]) -> ToolDefinition:
        """Construct a ToolDefinition from an MCP tool dictionary."""
        name = data.get("name", "")
        description = data.get("description", "")
        input_schema = data.get("inputSchema", {})
        
        parameters: Dict[str, ToolParameter] = {}
        props = input_schema.get("properties", {})
        required_list = input_schema.get("required", [])

        for p_name, p_val in props.items():
            parameters[p_name] = ToolParameter(
                name=p_name,
                type=p_val.get("type", "string"),
                description=p_val.get("description", ""),
                required=(p_name in required_list),
                default=p_val.get("default"),
                enum=p_val.get("enum")
            )

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            input_schema=input_schema
        )


@dataclass
class ToolResult:
    """Encapsulates the execution output of a tool call."""
    tool_name: str
    input_args: Dict[str, Any] = field(default_factory=dict)
    output_data: Any = None
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "success"


@dataclass
class TransitionCondition:
    """
    Defines when an agent should transition from the parent tool to this
    sub-tool based on the previous tool's output or execution context.
    """
    description: str
    condition_type: str = "always"  # "always", "output_contains", "key_equals", "custom"
    expression: Optional[Union[str, Callable[[Any], bool]]] = None

    def evaluate(self, previous_output: Any) -> bool:
        """Evaluates whether this transition path matches the output."""
        if self.condition_type == "always":
            return True
        
        if callable(self.expression):
            try:
                return bool(self.expression(previous_output))
            except Exception:
                return False

        if self.condition_type == "output_contains" and isinstance(self.expression, str):
            output_str = str(previous_output).lower()
            return self.expression.lower() in output_str

        if self.condition_type == "key_equals" and isinstance(previous_output, dict):
            # expression format: "key:value"
            if isinstance(self.expression, str) and ":" in self.expression:
                key, expected_val = self.expression.split(":", 1)
                actual_val = str(previous_output.get(key.strip(), "")).strip()
                return actual_val.lower() == expected_val.strip().lower()

        return False


@dataclass
class PromptContext:
    """
    Holds prompt templates that are dynamically injected into system or
    user prompts by the agent as it navigates tool execution paths.
    """
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    def render_system_prompt(self, extra_vars: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Renders system prompt with stored and extra variables."""
        if not self.system_prompt_template:
            return None
        combined = {**self.variables, **(extra_vars or {})}
        try:
            return self.system_prompt_template.format(**combined)
        except KeyError:
            # Fallback if any key is unfulfilled
            return self.system_prompt_template

    def render_user_prompt(self, extra_vars: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Renders user prompt with stored and extra variables."""
        if not self.user_prompt_template:
            return None
        combined = {**self.variables, **(extra_vars or {})}
        try:
            return self.user_prompt_template.format(**combined)
        except KeyError:
            return self.user_prompt_template
