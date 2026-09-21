"""
Dendron: The adaptive, tree-based tool execution and dynamic discovery library for AI agents.
Manages the hierarchical tree of tool execution paths, dynamic tree growth based
on agent experience, BFS/DFS searching, discovery instructions, RAG semantic retrieval,
and fast-path lookup for most frequently accessed tools (MFU).
"""

from __future__ import annotations
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from .models import ToolDefinition, ToolResult, TransitionCondition
from .node import DendronNode, ToolNode
from .retriever import RAGToolRetriever, RAGSearchResult


class Dendron:
    """
    Core Dendron tree object.
    Manages an execution path tree where each node is a smart tool call,
    allowing agents to dynamically grow the tree based on experience,
    search tools using BFS/DFS or RAG semantic retrieval, access discovery instructions,
    and quickly retrieve most frequently accessed tools.
    """

    def __init__(
        self,
        name: str,
        root_tool: ToolDefinition,
        discovery_instructions: str = "",
        description: str = "",
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> None:
        self.name: str = name
        self.description: str = description or root_tool.description
        self.discovery_instructions: str = discovery_instructions

        self.root: DendronNode = DendronNode(
            tool=root_tool,
            branch_label="root",
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            prompt_variables=prompt_variables or {}
        )

        # RAG semantic / lexical retriever
        self._retriever: RAGToolRetriever = RAGToolRetriever(embedding_fn=embedding_fn)

        # Fast lookup registry mapping tool_name and node_id to DendronNode
        self._registry_by_id: Dict[str, DendronNode] = {}
        self._registry_by_name: Dict[str, DendronNode] = {}
        self._registry_by_param: Dict[str, List[DendronNode]] = {}
        self._registry_by_tag: Dict[str, List[DendronNode]] = {}
        self._register_node(self.root)

    # MARK: - Registry & Fast Lookup (O(1))

    def _register_node(self, node: DendronNode) -> None:
        """Registers node in internal fast lookup caches, inverted indexes, and RAG index."""
        self._registry_by_id[node.id] = node
        self._registry_by_name[node.tool.name] = node
        
        # Inverted index for parameters
        for param_name in node.tool.parameters.keys():
            if param_name not in self._registry_by_param:
                self._registry_by_param[param_name] = []
            if node not in self._registry_by_param[param_name]:
                self._registry_by_param[param_name].append(node)

        # Inverted index for tags
        for tag in node.tool.tags:
            tag_clean = tag.lower()
            if tag_clean not in self._registry_by_tag:
                self._registry_by_tag[tag_clean] = []
            if node not in self._registry_by_tag[tag_clean]:
                self._registry_by_tag[tag_clean].append(node)

        self._retriever.index_node(node)
        for child in node.children:
            self._register_node(child)

    def _unregister_node(self, node: DendronNode) -> None:
        """Removes node and its children from fast lookup caches, inverted indexes, and RAG index."""
        self._registry_by_id.pop(node.id, None)
        self._registry_by_name.pop(node.tool.name, None)

        for param_name in node.tool.parameters.keys():
            if param_name in self._registry_by_param and node in self._registry_by_param[param_name]:
                self._registry_by_param[param_name].remove(node)

        for tag in node.tool.tags:
            tag_clean = tag.lower()
            if tag_clean in self._registry_by_tag and node in self._registry_by_tag[tag_clean]:
                self._registry_by_tag[tag_clean].remove(node)

        self._retriever.remove_node(node.id)
        for child in node.children:
            self._unregister_node(child)

    def find_by_id(self, node_id: str) -> Optional[DendronNode]:
        """O(1) fast lookup by unique node ID."""
        node = self._registry_by_id.get(node_id)
        if node:
            node.record_access()
        return node

    def find_by_name(self, tool_name: str) -> Optional[DendronNode]:
        """O(1) fast lookup by tool name."""
        node = self._registry_by_name.get(tool_name)
        if node:
            node.record_access()
        return node

    def search_fast(self, tool_name: str) -> Optional[DendronNode]:
        """Readily searches and returns tool by name using the fast lookup cache."""
        return self.find_by_name(tool_name)

    @property
    def nodes(self) -> List[DendronNode]:
        """Returns all nodes registered in the tree."""
        return list(self._registry_by_id.values())

    def __len__(self) -> int:
        """Returns the total number of nodes in the tree."""
        return len(self._registry_by_id)

    def get_frequently_accessed_tools(self, limit: int = 5) -> List[DendronNode]:
        """
        Returns the most frequently accessed tools (MFU) sorted by access_count
        and recency, ensuring commonly needed tools are readily available.
        """
        all_nodes = list(self._registry_by_id.values())
        # Sort descending by access_count, then descending by last_accessed_at
        sorted_nodes = sorted(
            all_nodes,
            key=lambda n: (n.access_count, n.last_accessed_at or 0.0),
            reverse=True
        )
        return sorted_nodes[:limit]

    def find_by_input_param(self, param_name: str) -> List[DendronNode]:
        """Returns all nodes in the tree that accept the given input parameter."""
        nodes = self._registry_by_param.get(param_name, [])
        for n in nodes:
            n.record_access()
        return list(nodes)

    def find_by_input_params(self, param_names: List[str], match_all: bool = False) -> List[DendronNode]:
        """
        Returns nodes matching input parameter names.
        :param match_all: If True, nodes must accept all parameter names in param_names.
                          If False, nodes accepting any of the parameter names are returned.
        """
        if not param_names:
            return []
        if match_all:
            result_sets = [set(self._registry_by_param.get(p, [])) for p in param_names]
            matched = list(set.intersection(*result_sets)) if result_sets else []
        else:
            matched_set: Set[DendronNode] = set()
            for p in param_names:
                matched_set.update(self._registry_by_param.get(p, []))
            matched = list(matched_set)
        for n in matched:
            n.record_access()
        return matched

    def find_by_tag(self, tag: str) -> List[DendronNode]:
        """Returns all nodes in the tree tagged with the given tag."""
        nodes = self._registry_by_tag.get(tag.lower(), [])
        for n in nodes:
            n.record_access()
        return list(nodes)

    # MARK: - Progressive Token-Tiered Views & Inspection

    def export_tool_views(
        self,
        nodes: Optional[List[DendronNode]] = None,
        level: int = 1
    ) -> List[Any]:
        """
        Exports tool views at the specified detail level to minimize LLM token consumption:
        - Level 1: Compact signature string (~10-20 tokens/tool)
        - Level 2: Parameter summary dictionary (~50 tokens/tool)
        - Level 3: Full Model Context Protocol (MCP) JSON Schema dictionary (~200+ tokens/tool)
        """
        target_nodes = nodes if nodes is not None else list(self._registry_by_id.values())
        return [node.to_view(level=level) for node in target_nodes]

    def inspect_tool(self, name_or_id: str) -> Dict[str, Any]:
        """
        Expands a specific tool to its full Level 3 MCP JSON Schema on demand.
        The LLM can inspect compact views (Level 1) to select a tool,
        then call `inspect_tool()` to view its complete inputSchema before execution.
        """
        node = self.find_by_name(name_or_id) or self.find_by_id(name_or_id)
        if not node:
            raise KeyError(f"Tool '{name_or_id}' not found in tree '{self.name}'.")
        node.record_access()
        return node.to_mcp_dict()

    # MARK: - Information-State & Reachability Discovery

    def get_actionable_tools(
        self,
        available_inputs: List[str],
        detail_level: int = 1,
        require_all: bool = True,
        level: Optional[int] = None
    ) -> List[Any]:
        """
        Returns tools whose required parameters match the information the LLM currently possesses.
        Allows the LLM to identify immediate next steps or bypass intermediate exploratory steps.
        
        :param available_inputs: List of parameter/input names the LLM currently has in hand.
        :param detail_level: 1 (compact), 2 (parameter summary), or 3 (full MCP schema). (Alias: level)
        :param require_all: If True, all required parameters of a tool must be in available_inputs.
                            If False, any overlap satisfies the filter.
        """
        eff_level = level if level is not None else detail_level
        known_set = set(available_inputs)
        matching_nodes: List[DendronNode] = []
        for node in self._registry_by_id.values():
            req_params = set(node.get_required_parameter_names())
            all_params = set(node.get_all_parameter_names())
            if require_all:
                if req_params.issubset(known_set):
                    matching_nodes.append(node)
            else:
                if all_params.intersection(known_set) or len(req_params) == 0:
                    matching_nodes.append(node)

        return [node.to_view(level=eff_level) for node in matching_nodes]

    def get_reachable_tools(
        self,
        current_node_id: str,
        max_hops: int = 2,
        detail_level: int = 1,
        level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns all tools reachable from current_node_id within max_hops steps,
        showing step distance, path breadcrumbs, and branch conditions.
        """
        eff_level = level if level is not None else detail_level
        start_node = self.find_by_id(current_node_id) or self.find_by_name(current_node_id)
        if not start_node:
            raise ValueError(f"Node '{current_node_id}' not found in tree '{self.name}'.")

        reachable: List[Dict[str, Any]] = []
        queue: deque[Tuple[DendronNode, int, List[str]]] = deque([(start_node, 0, [start_node.tool.name])])
        visited: Set[str] = {start_node.id}

        while queue:
            node, hops, path = queue.popleft()
            if 1 <= hops <= max_hops:
                cond_desc = node.transition_condition.description if node.transition_condition else None
                reachable.append({
                    "node": node.to_view(level=eff_level),
                    "name": node.tool.name,
                    "hops": hops,
                    "path": path,
                    "branch_label": node.branch_label,
                    "condition": cond_desc
                })

            if hops < max_hops:
                for child in node.children:
                    if child.id not in visited:
                        visited.add(child.id)
                        queue.append((child, hops + 1, path + [child.tool.name]))

        return reachable

    def search(
        self,
        query: Optional[str] = None,
        required_inputs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        detail_level: int = 1,
        top_k: int = 5,
        level: Optional[int] = None
    ) -> List[Any]:
        """
        Multi-faceted fast search combining natural language query, required inputs, and tags.
        Returns results formatted at the requested detail_level to conserve tokens. (Alias: level)
        """
        eff_level = level if level is not None else detail_level
        if query:
            rag_results = self._retriever.retrieve(query=query, top_k=top_k * 3)
            candidates = [r.node for r in rag_results]
        else:
            candidates = list(self._registry_by_id.values())

        if required_inputs:
            known_set = set(required_inputs)
            candidates = [n for n in candidates if set(n.get_required_parameter_names()).issubset(known_set)]

        if tags:
            tag_set = set(t.lower() for t in tags)
            candidates = [n for n in candidates if any(t.lower() in tag_set for t in n.tool.tags)]

        return [n.to_view(level=eff_level) for n in candidates[:top_k]]

    def get_node_addition_guidelines(self) -> str:
        """
        Returns structured guidelines for an LLM on when and how to dynamically add
        nodes to the tree during execution.
        """
        return (
            "### Dendron: Guidelines for Dynamically Adding Nodes\n\n"
            "Every node in Dendron is a standard `DendronNode`. The LLM can dynamically expand\n"
            "the tree at runtime using `tree.add_node(...)` or `tree.record_agent_experience(...)`.\n\n"
            "#### When to Add a Node:\n"
            "1. **New Capability / Sub-Workflow**:\n"
            "   When a new external tool, API endpoint, or sub-task is discovered that logically\n"
            "   belongs under an existing parent step.\n"
            "2. **Niche / Specialized Presets**:\n"
            "   When you learn that a recurring niche scenario (e.g. VIP refunds, newsletter archiving,\n"
            "   Kubernetes OOM errors) requires specific pre-filled parameters, flags, or constraints.\n"
            "   Spawning a specialized node avoids re-computing arguments and saves context tokens.\n"
            "3. **Learned Transition Path**:\n"
            "   When an observed output pattern (e.g. 'error_code: 503') consistently transitions into\n"
            "   a specific next action (e.g. 'restart_service'), attach it with a `TransitionCondition`\n"
            "   so future turns immediately recognize the optimal path.\n\n"
            "#### How to Add a Node:\n"
            "```python\n"
            "tree.add_node(\n"
            "    parent_id=parent_node.id,\n"
            "    tool=ToolDefinition(\n"
            "        name='specialized_tool_name',\n"
            "        description='Specific purpose description',\n"
            "        parameters={'param': ToolParameter(name='param', type='string')},\n"
            "        tags=['niche', 'category']\n"
            "    ),\n"
            "    branch_label='specialized_branch',\n"
            "    condition=TransitionCondition(\n"
            "        description='When output contains target keyword',\n"
            "        condition_type='output_contains',\n"
            "        expression='target_keyword'\n"
            "    ),\n"
            "    system_prompt_template='You are specialized in {domain}.',\n"
            "    prompt_variables={'domain': 'Finance'}\n"
            ")\n"
            "```\n"
            "Once added, the node is immediately indexed in $O(1)$ registries, RAG search, and MFU caching.\n"
        )

    # MARK: - Dynamic Tree Growth (Agent Experience)

    def add_node(
        self,
        parent_id: str,
        tool: ToolDefinition,
        branch_label: Optional[str] = None,
        condition: Optional[TransitionCondition] = None,
        system_prompt_template: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None
    ) -> DendronNode:
        """
        Attaches a new tool node to a designated parent node.
        Used by the agent as it builds out the tree structure.
        """
        parent = self._registry_by_id.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node with ID '{parent_id}' does not exist in tree '{self.name}'.")

        new_node = DendronNode(
            tool=tool,
            branch_label=branch_label,
            transition_condition=condition,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            prompt_variables=prompt_variables or {}
        )

        parent.add_child(new_node, branch_label=branch_label, condition=condition)
        self._register_node(new_node)
        return new_node

    def record_agent_experience(
        self,
        parent_id: str,
        next_tool: ToolDefinition,
        trigger_condition_description: str,
        condition_type: str = "always",
        condition_expression: Optional[Any] = None,
        branch_label: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        experience_note: Optional[str] = None
    ) -> DendronNode:
        """
        Dynamically builds out the tree based on the agent's real-time experience.
        When an agent encounters a situation and realizes that `next_tool` should follow
        `parent_id`, it records this path so subsequent workflows automatically know
        which next tool call to make with the output from the previous call.
        """
        condition = TransitionCondition(
            description=trigger_condition_description,
            condition_type=condition_type,
            expression=condition_expression
        )

        node = self.add_node(
            parent_id=parent_id,
            tool=next_tool,
            branch_label=branch_label or next_tool.name,
            condition=condition,
            system_prompt_template=system_prompt,
            user_prompt_template=user_prompt,
            prompt_variables=prompt_variables
        )

        if experience_note:
            node.add_experience_note(experience_note)

        return node

    def suggest_next_tool(self, current_node_id: str, previous_output: Any) -> Optional[DendronNode]:
        """
        Evaluates children of the current node against the output of the previous
        tool call, returning the best matching next tool node to execute.
        Prioritizes specific transition conditions over unconditional fallbacks.
        """
        current_node = self.find_by_id(current_node_id)
        if not current_node:
            return None

        # First pass: evaluate children with specific conditions
        for child in current_node.children:
            if child.has_specific_condition() and child.can_transition(previous_output):
                child.record_access()
                return child

        # Second pass: evaluate unconditional or default children
        for child in current_node.children:
            if not child.has_specific_condition() and child.can_transition(previous_output):
                child.record_access()
                return child

        return None

    # MARK: - Tree Search: BFS & DFS

    def search_bfs(
        self,
        query: Optional[str] = None,
        predicate: Optional[Callable[[DendronNode], bool]] = None
    ) -> List[DendronNode]:
        """
        Breadth-First Search (BFS) across the tool execution tree.
        Finds all nodes matching the query string or custom predicate,
        exploring level by level.
        """
        matches: List[DendronNode] = []
        queue: deque[DendronNode] = deque([self.root])
        visited: Set[str] = set()

        lower_query = query.lower() if query else None

        while queue:
            node = queue.popleft()
            if node.id in visited:
                continue
            visited.add(node.id)

            node.record_access()

            # Evaluate match criteria
            is_match = False
            if predicate is not None:
                is_match = predicate(node)
            elif lower_query:
                is_match = (
                    lower_query in node.tool.name.lower() or
                    lower_query in node.tool.description.lower() or
                    any(lower_query in tag.lower() for tag in node.tool.tags) or
                    (node.branch_label and lower_query in node.branch_label.lower())
                )
            else:
                is_match = True

            if is_match:
                matches.append(node)

            for child in node.children:
                if child.id not in visited:
                    queue.append(child)

        return matches

    def search_dfs(
        self,
        query: Optional[str] = None,
        predicate: Optional[Callable[[DendronNode], bool]] = None
    ) -> List[DendronNode]:
        """
        Depth-First Search (DFS) along execution paths of the tool tree.
        Finds all nodes matching the query string or custom predicate,
        exploring deeply down each execution branch before backtracking.
        """
        matches: List[DendronNode] = []
        stack: List[DendronNode] = [self.root]
        visited: Set[str] = set()

        lower_query = query.lower() if query else None

        while stack:
            node = stack.pop()
            if node.id in visited:
                continue
            visited.add(node.id)

            node.record_access()

            # Evaluate match criteria
            is_match = False
            if predicate is not None:
                is_match = predicate(node)
            elif lower_query:
                is_match = (
                    lower_query in node.tool.name.lower() or
                    lower_query in node.tool.description.lower() or
                    any(lower_query in tag.lower() for tag in node.tool.tags) or
                    (node.branch_label and lower_query in node.branch_label.lower())
                )
            else:
                is_match = True

            if is_match:
                matches.append(node)

            # Reverse children on stack to preserve left-to-right DFS order
            for child in reversed(node.children):
                if child.id not in visited:
                    stack.append(child)

        return matches

    # MARK: - RAG Tool Retrieval

    def retrieve_tools(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[RAGSearchResult]:
        """
        RAG-based tool retrieval: queries the execution tree semantically to find
        the best available tools for a given user or agent task description.
        Returns top-k matches with relevance scores and match rationales.
        """
        return self._retriever.retrieve(query=query, top_k=top_k, min_score=min_score)

    def retrieve_best_tool(
        self,
        query: str,
        min_score: float = 0.0
    ) -> Optional[DendronNode]:
        """
        Convenience method that returns the single best DendronNode for a given query,
        or None if no tool matches above min_score.
        """
        return self._retriever.retrieve_best(query=query, min_score=min_score)

    def set_embedding_function(
        self,
        embedding_fn: Callable[[List[str]], List[List[float]]]
    ) -> None:
        """
        Configures an external dense embedding function (e.g. OpenAI, Hugging Face)
        and re-indexes all nodes in the tree for hybrid retrieval.
        """
        self._retriever.embedding_fn = embedding_fn
        self._retriever.index_tree(self)

    # MARK: - MCP Server Export & Serialization

    def export_all_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Exports all tools defined in this tree in standard MCP tool format,
        ready to be served by any Model Context Protocol server.
        """
        all_nodes = self.search_bfs()
        return [node.to_mcp_format() for node in all_nodes]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire tree structure to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "discovery_instructions": self.discovery_instructions,
            "root": self.root.to_dict(include_children=True)
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the entire tree structure to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dendron:
        """Deserializes a Dendron tree from a dictionary."""
        root_data = data["root"]
        root_tool = ToolDefinition.from_mcp_dict(root_data["tool"])
        
        tree = cls(
            name=data.get("name", "Dendron"),
            root_tool=root_tool,
            discovery_instructions=data.get("discovery_instructions", ""),
            description=data.get("description", ""),
            system_prompt_template=root_data.get("system_prompt_template"),
            user_prompt_template=root_data.get("user_prompt_template"),
            prompt_variables=root_data.get("prompt_variables", {})
        )

        tree.root = DendronNode.from_dict(root_data)
        tree._registry_by_id.clear()
        tree._registry_by_name.clear()
        tree._registry_by_param.clear()
        tree._registry_by_tag.clear()
        tree._register_node(tree.root)
        return tree

    @classmethod
    def from_json(cls, json_str: str) -> Dendron:
        """Deserializes a Dendron tree from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves the entire Dendron tree structure to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Dendron:
        """Loads and deserializes a Dendron tree structure from a JSON file."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Dendron file not found: {filepath}")
        return cls.from_json(p.read_text())

    def __repr__(self) -> str:
        total_nodes = len(self._registry_by_id)
        return f"<Dendron(name='{self.name}', nodes={total_nodes}, root='{self.root.tool.name}')>"


# Backwards compatibility aliases
DendronTree = Dendron
SmartToolTree = Dendron
