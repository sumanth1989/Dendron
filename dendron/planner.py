"""
Autonomous Action Planning for Dendron.
Provides structured plan formulation, dependency resolution,
autonomous execution loops, and model-specific tool fetching (e.g. Google Gemini, OpenAI, Claude).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from .models import ToolResult

if TYPE_CHECKING:
    from .node import DendronNode
    from .tree import Dendron


@dataclass
class ActionStep:
    """Represents an individual planned step in an action plan."""
    step_number: int
    node: DendronNode
    tool_name: str
    rationale: str
    required_inputs: List[str]
    input_args: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    result: Optional[ToolResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "rationale": self.rationale,
            "required_inputs": self.required_inputs,
            "input_args": self.input_args,
            "executed": self.executed,
            "success": self.result.is_success if self.result else None
        }


@dataclass
class ActionPlan:
    """Represents an autonomous multi-step execution plan."""
    goal: str
    steps: List[ActionStep] = field(default_factory=list)
    status: str = "planned"  # "planned", "in_progress", "completed", "failed"
    current_step_index: int = 0
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status == "completed" or self.status == "failed"

    def get_current_step(self) -> Optional[ActionStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def execute_next(self, **runtime_args: Any) -> Tuple[Optional[ToolResult], bool]:
        """
        Executes the current step of the plan with provided arguments.
        Returns (ToolResult, is_finished).
        """
        if self.current_step_index >= len(self.steps):
            self.status = "completed"
            return None, True

        step = self.steps[self.current_step_index]
        self.status = "in_progress"

        combined_args = {**step.input_args, **runtime_args}
        result = step.node.execute(**combined_args)
        step.executed = True
        step.result = result

        if not result.is_success:
            self.status = "failed"
            self.error_message = result.error_message
            return result, True

        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "completed"
            return result, True

        return result, False

    def fetch_tools_for_model(self, model_provider: str = "gemini") -> Any:
        """
        Fetches the tool schemas for all steps in this plan formatted
        for the target model (e.g. 'gemini', 'openai', 'anthropic').
        """
        nodes = [s.node for s in self.steps]
        if not nodes:
            return []

        type_map = {
            "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
            "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT"
        }
        provider = model_provider.lower().strip()
        if provider in ["gemini", "google"]:
            tools = []
            for n in nodes:
                properties = {
                    p_name: {"type": type_map.get(p.type.lower(), "STRING"), "description": p.description}
                    for p_name, p in n.tool.parameters.items()
                }
                tools.append({
                    "name": n.tool.name,
                    "description": n.tool.description,
                    "parameters": {
                        "type": "OBJECT",
                        "properties": properties,
                        "required": n.get_required_parameter_names()
                    }
                })
            return tools
        elif provider in ["openai", "gpt"]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": n.tool.name,
                        "description": n.tool.description,
                        "parameters": n.tool.input_schema or n.tool._generate_mcp_input_schema()
                    }
                }
                for n in nodes
            ]
        elif provider in ["anthropic", "claude"]:
            return [
                {
                    "name": n.tool.name,
                    "description": n.tool.description,
                    "input_schema": n.tool.input_schema or n.tool._generate_mcp_input_schema()
                }
                for n in nodes
            ]
        else:
            return [n.to_mcp_format() for n in nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "current_step_index": self.current_step_index,
            "error_message": self.error_message,
            "steps": [s.to_dict() for s in self.steps]
        }


class AutonomousActionPlanner:
    """
    Autonomous planning engine that constructs and executes action sequences
    across a Dendron tree for a given user goal or query.
    """

    def __init__(self, tree: Dendron):
        self.tree = tree

    def plan(
        self,
        goal: str,
        start_node_id: Optional[str] = None,
        available_inputs: Optional[List[str]] = None,
        max_steps: int = 5
    ) -> ActionPlan:
        """
        Formulates an autonomous action plan to achieve `goal`.
        Identifies relevant tools using RAG retrieval and tree traversal paths.
        """
        # 1. RAG search to find most relevant target tools for the goal
        rag_matches = self.tree.retrieve_tools(query=goal, top_k=max_steps)
        if not rag_matches:
            return ActionPlan(goal=goal, steps=[], status="failed", error_message="No matching tools found for goal")

        # 2. Determine start node
        start_node = self.tree.find_by_id(start_node_id) if start_node_id else self.tree.root
        if not start_node:
            start_node = self.tree.root

        # 3. Build path from start_node to the best target node
        best_target = rag_matches[0].node
        path_nodes = self._find_path(start_node, best_target, max_steps=max_steps)

        # 4. If path couldn't reach target, include start_node and best_target
        if not path_nodes:
            path_nodes = [start_node]
            if best_target.id != start_node.id:
                path_nodes.append(best_target)

        # 5. Create action steps
        steps: List[ActionStep] = []
        for i, node in enumerate(path_nodes):
            reqs = node.get_required_parameter_names()
            rationale = f"Step {i+1}: Invoke '{node.tool.name}' ({node.tool.description})"
            steps.append(ActionStep(
                step_number=i + 1,
                node=node,
                tool_name=node.tool.name,
                rationale=rationale,
                required_inputs=reqs
            ))

        return ActionPlan(goal=goal, steps=steps, status="planned")

    def _find_path(self, start: DendronNode, target: DendronNode, max_steps: int = 5) -> List[DendronNode]:
        """Finds directed path from start to target via BFS over children and DAG transitions."""
        from collections import deque
        if start.id == target.id:
            return [start]

        queue = deque([(start, [start])])
        visited = {start.id}

        while queue:
            curr, path = queue.popleft()
            if curr.id == target.id:
                return path

            if len(path) < max_steps:
                # Check children
                for child in curr.children:
                    if child.id not in visited:
                        visited.add(child.id)
                        queue.append((child, path + [child]))

                # Check DAG transitions
                for trans_id, _ in curr.transitions:
                    target_node = self.tree.find_by_id(trans_id)
                    if target_node and target_node.id not in visited:
                        visited.add(target_node.id)
                        queue.append((target_node, path + [target_node]))

        return []

    def plan_and_execute(
        self,
        goal: str,
        input_context: Dict[str, Any],
        start_node_id: Optional[str] = None,
        max_steps: int = 5
    ) -> List[ToolResult]:
        """
        Plans and executes an action sequence to achieve `goal`.
        Accumulates and passes outputs to downstream steps.
        """
        plan = self.plan(goal=goal, start_node_id=start_node_id, available_inputs=list(input_context.keys()), max_steps=max_steps)
        results: List[ToolResult] = []
        accumulated_context = dict(input_context)

        while not plan.is_complete:
            step = plan.get_current_step()
            if not step:
                break
            res, finished = plan.execute_next(**accumulated_context)
            if res:
                results.append(res)
                if isinstance(res.output_data, dict):
                    accumulated_context.update(res.output_data)
                elif res.output_data is not None:
                    accumulated_context[f"{step.tool_name}_output"] = res.output_data
            if finished:
                break

        return results
