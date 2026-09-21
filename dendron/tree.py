"""
Dendron: The adaptive, tree-based tool execution and dynamic discovery library for AI agents.
Manages the hierarchical tree of tool execution paths, dynamic tree growth based
on agent experience, BFS/DFS searching, discovery instructions, RAG semantic retrieval,
and fast-path lookup for most frequently accessed tools (MFU).
"""

from __future__ import annotations
import json
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set
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
        self._register_node(self.root)

    # MARK: - Registry & Fast Lookup (O(1))

    def _register_node(self, node: DendronNode) -> None:
        """Registers node in internal fast lookup caches and RAG index."""
        self._registry_by_id[node.id] = node
        self._registry_by_name[node.tool.name] = node
        self._retriever.index_node(node)
        for child in node.children:
            self._register_node(child)

    def _unregister_node(self, node: DendronNode) -> None:
        """Removes node and its children from fast lookup caches and RAG index."""
        self._registry_by_id.pop(node.id, None)
        self._registry_by_name.pop(node.tool.name, None)
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
        tree._register_node(tree.root)
        return tree

    @classmethod
    def from_json(cls, json_str: str) -> Dendron:
        """Deserializes a Dendron tree from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        total_nodes = len(self._registry_by_id)
        return f"<Dendron(name='{self.name}', nodes={total_nodes}, root='{self.root.tool.name}')>"


# Backwards compatibility aliases
DendronTree = Dendron
SmartToolTree = Dendron
