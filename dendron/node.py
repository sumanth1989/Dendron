"""
DendronNode representation for Dendron.
Each node represents a tool call along a path of execution within the tree,
carrying tool specifications, transition rules, experience history, and
dynamic prompt injection variables.
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional
from .models import ToolDefinition, ToolParameter, ToolResult, TransitionCondition, PromptContext


class DendronNode:
    """
    A single node in a Dendron tree.
    Represents an actionable tool along a path of execution.
    """

    def __init__(
        self,
        tool: ToolDefinition,
        node_id: Optional[str] = None,
        branch_label: Optional[str] = None,
        transition_condition: Optional[TransitionCondition] = None,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        parent: Optional[DendronNode] = None,
    ) -> None:
        self.id: str = node_id or str(uuid.uuid4())
        self.tool: ToolDefinition = tool
        self.branch_label: Optional[str] = branch_label
        self.transition_condition: Optional[TransitionCondition] = transition_condition
        self.parent: Optional[DendronNode] = parent
        self.children: List[DendronNode] = []

        # Prompt injection context
        self.prompt_context: PromptContext = PromptContext(
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            variables=prompt_variables or {}
        )

        # Execution frequency and experience tracking
        self.access_count: int = 0
        self.last_accessed_at: Optional[float] = None
        self.success_count: int = 0
        self.failure_count: int = 0
        self.execution_history: List[ToolResult] = []
        self.experience_notes: List[str] = []

    # MARK: - Tree Hierarchy

    def add_child(
        self,
        child: DendronNode,
        branch_label: Optional[str] = None,
        condition: Optional[TransitionCondition] = None
    ) -> DendronNode:
        """Attaches a child node representing a subsequent execution step."""
        child.parent = self
        if branch_label is not None:
            child.branch_label = branch_label
        if condition is not None:
            child.transition_condition = condition
        self.children.append(child)
        return child

    def remove_child(self, child_id: str) -> bool:
        """Removes a child node by its ID."""
        for i, c in enumerate(self.children):
            if c.id == child_id:
                removed = self.children.pop(i)
                removed.parent = None
                return True
        return False

    # MARK: - Experience & Frequency Tracking

    def record_access(self) -> None:
        """Increments access count and updates timestamp for frequency tracking."""
        self.access_count += 1
        self.last_accessed_at = time.time()

    def record_execution(self, result: ToolResult, note: Optional[str] = None) -> None:
        """Records an execution outcome and optional agent experience note."""
        self.record_access()
        self.execution_history.append(result)
        if result.is_success:
            self.success_count += 1
        else:
            self.failure_count += 1
        if note:
            self.experience_notes.append(note)

    def add_experience_note(self, note: str) -> None:
        """Records a learned rule or insight based on agent experience."""
        self.experience_notes.append(note)

    # MARK: - RAG & Path Representation

    def get_execution_path(self) -> List[str]:
        """Returns the full execution path from root to this node."""
        path: List[str] = []
        curr: Optional[DendronNode] = self
        while curr:
            path.append(curr.tool.name)
            curr = curr.parent
        return list(reversed(path))

    def to_search_document(self) -> str:
        """
        Builds a comprehensive text document representing this tool for RAG indexing.
        Includes tool name, description, tags, parameter specifications,
        execution path, and learned experience notes.
        """
        parts = [
            f"Tool: {self.tool.name}",
            f"Description: {self.tool.description}",
        ]
        if self.branch_label:
            parts.append(f"Branch: {self.branch_label}")
        if self.tool.tags:
            parts.append(f"Tags: {', '.join(self.tool.tags)}")

        path = self.get_execution_path()
        if len(path) > 1:
            parts.append(f"Execution Path: {' -> '.join(path)}")

        if self.tool.parameters:
            param_strs = [
                f"{p.name} ({p.type}): {p.description}"
                for p in self.tool.parameters.values()
            ]
            parts.append(f"Parameters: {'; '.join(param_strs)}")

        if self.experience_notes:
            parts.append(f"Learned Experience: {'; '.join(self.experience_notes)}")

        return "\n".join(parts)

    # MARK: - Prompt Variables & Rendering

    def set_prompt_variable(self, key: str, value: Any) -> None:
        """Sets or updates a prompt variable stored on this node."""
        self.prompt_context.variables[key] = value

    def get_system_prompt(self, **kwargs: Any) -> Optional[str]:
        """Renders system prompt with node variables and runtime kwargs."""
        self.record_access()
        return self.prompt_context.render_system_prompt(kwargs)

    def get_user_prompt(self, **kwargs: Any) -> Optional[str]:
        """Renders user prompt with node variables and runtime kwargs."""
        self.record_access()
        return self.prompt_context.render_user_prompt(kwargs)

    # MARK: - Transition Evaluation

    def has_specific_condition(self) -> bool:
        """Returns True if this node has an explicit condition other than 'always'."""
        if not self.transition_condition:
            return False
        return self.transition_condition.condition_type != "always"

    def can_transition(self, previous_output: Any) -> bool:
        """Evaluates whether this node's transition condition matches previous output."""
        if not self.transition_condition:
            return True
        return self.transition_condition.evaluate(previous_output)

    # MARK: - Serialization

    def to_mcp_format(self) -> Dict[str, Any]:
        """Export tool definition in standard MCP tool format."""
        return self.tool.to_mcp_dict()

    # MARK: - Progressive Token-Tiered Views

    def get_required_parameter_names(self) -> List[str]:
        """Returns the list of parameter names that are required for this tool."""
        return [p_name for p_name, p in self.tool.parameters.items() if p.required]

    def get_all_parameter_names(self) -> List[str]:
        """Returns all parameter names for this tool."""
        return list(self.tool.parameters.keys())

    def to_compact_summary(self) -> str:
        """
        Level 1: Ultra-compact tool signature (approx. 10-20 tokens).
        Format: name(req_param, [opt_param]) - Description
        """
        params_str_list = []
        for p_name, p in self.tool.parameters.items():
            if p.required:
                params_str_list.append(p_name)
            else:
                params_str_list.append(f"[{p_name}]")
        params_formatted = ", ".join(params_str_list)
        tags_str = f" #{','.join(self.tool.tags)}" if self.tool.tags else ""
        return f"{self.tool.name}({params_formatted}) - {self.tool.description}{tags_str}"

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        Level 1 dict: Compact structured representation of tool essentials.
        """
        return {
            "id": self.id,
            "name": self.tool.name,
            "description": self.tool.description,
            "required_inputs": self.get_required_parameter_names(),
            "optional_inputs": [p_name for p_name, p in self.tool.parameters.items() if not p.required],
            "tags": self.tool.tags,
            "branch_label": self.branch_label,
        }

    def to_parameter_summary(self) -> Dict[str, Any]:
        """
        Level 2: Parameter summary with types and constraints, omitting JSON Schema boilerplate (~50 tokens).
        """
        params_detail: Dict[str, Any] = {}
        for p_name, p in self.tool.parameters.items():
            entry: Dict[str, Any] = {
                "type": p.type,
                "required": p.required,
                "description": p.description,
            }
            if p.default is not None:
                entry["default"] = p.default
            if p.enum:
                entry["enum"] = p.enum
            params_detail[p_name] = entry

        return {
            "id": self.id,
            "name": self.tool.name,
            "description": self.tool.description,
            "branch_label": self.branch_label,
            "tags": self.tool.tags,
            "parameters": params_detail,
        }

    def to_mcp_dict(self) -> Dict[str, Any]:
        """
        Level 3: Full Model Context Protocol (MCP) JSON Schema definition (~200+ tokens).
        """
        return self.tool.to_mcp_dict()

    def to_view(self, level: int = 1) -> Any:
        """
        Returns the tool view at the requested detail level:
        - Level 1: Compact signature string (~10-20 tokens)
        - Level 2: Parameter summary dictionary (~50 tokens)
        - Level 3: Full MCP JSON Schema dictionary (~200+ tokens)
        """
        if level == 1:
            return self.to_compact_summary()
        elif level == 2:
            return self.to_parameter_summary()
        elif level == 3:
            return self.to_mcp_dict()
        else:
            raise ValueError(f"detail_level must be 1, 2, or 3 (got {level}).")

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Serializes node to a JSON-compatible dictionary."""
        data: Dict[str, Any] = {
            "id": self.id,
            "tool": self.tool.to_mcp_dict(),
            "branch_label": self.branch_label,
            "system_prompt_template": self.prompt_context.system_prompt_template,
            "user_prompt_template": self.prompt_context.user_prompt_template,
            "prompt_variables": self.prompt_context.variables,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "experience_notes": self.experience_notes,
        }

        if self.transition_condition:
            data["transition_condition"] = {
                "description": self.transition_condition.description,
                "condition_type": self.transition_condition.condition_type,
                "expression": (
                    self.transition_condition.expression
                    if isinstance(self.transition_condition.expression, str)
                    else str(self.transition_condition.expression)
                )
            }

        if include_children:
            data["children"] = [child.to_dict(include_children=True) for child in self.children]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DendronNode:
        """Deserializes a DendronNode from a dictionary."""
        tool_def = ToolDefinition.from_mcp_dict(data["tool"])
        cond_data = data.get("transition_condition")
        condition = None
        if cond_data:
            condition = TransitionCondition(
                description=cond_data.get("description", ""),
                condition_type=cond_data.get("condition_type", "always"),
                expression=cond_data.get("expression")
            )

        node = cls(
            tool=tool_def,
            node_id=data.get("id"),
            branch_label=data.get("branch_label"),
            transition_condition=condition,
            system_prompt_template=data.get("system_prompt_template"),
            user_prompt_template=data.get("user_prompt_template"),
            prompt_variables=data.get("prompt_variables", {})
        )

        node.access_count = data.get("access_count", 0)
        node.last_accessed_at = data.get("last_accessed_at")
        node.success_count = data.get("success_count", 0)
        node.failure_count = data.get("failure_count", 0)
        node.experience_notes = data.get("experience_notes", [])

        for child_data in data.get("children", []):
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)

        return node

    def __repr__(self) -> str:
        return f"<DendronNode(id='{self.id}', tool='{self.tool.name}', branch='{self.branch_label}', access={self.access_count})>"


# Backwards compatibility alias
ToolNode = DendronNode
