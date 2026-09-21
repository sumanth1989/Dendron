# Dendron 🌳🧠

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib%20Only)-green.svg?style=flat)](#installation)
[![MCP](https://img.shields.io/badge/MCP-Compliant-8A2BE2.svg?style=flat)](https://modelcontextprotocol.io)

An **adaptive, tree-based tool execution and dynamic discovery library** for AI agents.

> **Framework-Agnostic & Zero Dependencies**: Built purely on the Python standard library. Works with any LLM (OpenAI, Claude, Gemini, Apple Foundation Models, MLX, Ollama) and any agent framework (LangChain, CrewAI, AutoGen, or vanilla Python).

---

## Why Dendron?

Standard agent architectures dump 20–50 tools into every LLM prompt turn. This wastes context tokens, increases hallucinations, and loses the natural flow of execution. For example, an agent shouldn't try to issue a refund or print a return label before it has even looked up the customer's order!

**Dendron** structures tool use as an **adaptive execution tree**:
- **Context-Efficient**: The agent only sees relevant next-step tools along its current branch (e.g. `track_shipment` or `process_return` only after `lookup_order`).
- **Dynamic Learning**: Agents discover and attach new execution paths at runtime as they encounter novel tasks.
- **Deterministic Transitions**: Evaluates tool outputs against rules (`output_contains`, `key_equals`, or custom predicates) to suggest next steps.
- **RAG Semantic Discovery**: When an agent doesn't know what tool to use, it queries the tree using natural language intent.

```
                      [ Root Tool Node ]
                   (lookup_order: order_id)
                   /                      \
      status: "delivered"            status: "in_transit"
                 /                          \
   [ DendronNode: process_return ]     [ DendronNode: track_shipment ]
          /                                   |
condition: "defective"                 (Learned dynamically)
        /                                     |
[ DendronNode: issue_instant_refund ]  [ DendronNode: send_sms_alert ]
```

---

## Installation

```bash
pip install -e .
```
*(Or copy the `dendron/` folder directly into your project — zero external dependencies required.)*

---

## Quickstart

```python
from dendron import Dendron, ToolDefinition, ToolParameter, TransitionCondition

# 1. Define Root Tool (Look up customer order)
root_tool = ToolDefinition(
    name="lookup_order",
    description="Retrieves customer order details and fulfillment status by order ID.",
    parameters={
        "order_id": ToolParameter(name="order_id", type="string", description="Order ID e.g. ORD-12345", required=True),
    },
    tags=["orders", "lookup", "root"]
)

# 2. Create the Tree with Discovery Instructions
tree = Dendron(
    name="OrderSupportTree",
    root_tool=root_tool,
    discovery_instructions="Start at 'lookup_order'. If delivered, branch to 'process_return'. If in transit, branch to 'track_shipment'."
)

# 3. Attach Execution Branches with Prompt Templates
track_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(name="track_shipment", description="Fetches live courier tracking and ETA.", tags=["shipping", "tracking"]),
    branch_label="in_transit_branch",
    condition=TransitionCondition(
        description="Order status is in_transit",
        condition_type="output_contains",
        expression="in_transit"
    )
)

return_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(
        name="process_return",
        description="Initiates a return for a delivered order.",
        parameters={
            "order_id": ToolParameter(name="order_id", type="string", required=True),
            "reason": ToolParameter(name="reason", type="string", required=True),
        },
        tags=["returns", "refunds"]
    ),
    branch_label="delivered_branch",
    condition=TransitionCondition(
        description="Order status is delivered",
        condition_type="output_contains",
        expression="delivered"
    ),
    system_prompt_template="You are a customer support agent for {store_name}. Tone: {tone}.",
    prompt_variables={"store_name": "ShopEase", "tone": "helpful and friendly"}
)

# 4. Prompt Injection
print(return_node.get_system_prompt(tone="empathetic"))
# Output: "You are a customer support agent for ShopEase. Tone: empathetic."

# 5. Dynamic Experience Learning
# Agent learns that when an item arrives defective, it should issue an instant refund
tree.record_agent_experience(
    parent_id=return_node.id,
    next_tool=ToolDefinition(
        name="issue_instant_refund",
        description="Issues an immediate refund without requiring a return shipment.",
        parameters={"order_id": ToolParameter(name="order_id", type="string", required=True)},
        tags=["refund", "instant"]
    ),
    trigger_condition_description="Item arrived damaged or defective",
    condition_type="output_contains",
    condition_expression="defective",
    experience_note="Automatically issue instant refund for defective items under $50."
)

# 6. Intelligent Next-Tool Suggestion
next_tool = tree.suggest_next_tool(return_node.id, previous_output="Customer report: item arrived defective and broken.")
print(next_tool.tool.name)  # Output: "issue_instant_refund"

# 7. RAG Semantic Tool Retrieval (Natural Language Intent)
best_match = tree.retrieve_best_tool("Where is my package right now?")
print(best_match.tool.name)  # Output: "track_shipment"

# 8. Token-Tiered Views & MCP Export
print(tree.export_tool_views(level=1))  # Level 1 compact signatures (~15 tokens/tool)
```

---

## Core API Reference

| Method | Description |
| :--- | :--- |
| `Dendron(name, root_tool, ...)` | Initializes a tree with root tool and discovery instructions. (Alias: `DendronTree`) |
| `tree.add_node(parent_id, tool, ...)` | Attaches a child tool node (`DendronNode`) to a parent. |
| `tree.record_agent_experience(...)` | Dynamically appends a learned tool path and re-indexes for RAG. |
| `tree.suggest_next_tool(parent_id, output)` | Evaluates condition rules against output to suggest the next tool. |
| `tree.export_tool_views(nodes, level)` | **Token Optimization**: Exports tools at Level 1 (~15 tokens), Level 2 (~50 tokens), or Level 3. |
| `tree.inspect_tool(name_or_id)` | Expands a specific tool to its full Level 3 MCP JSON Schema on demand. |
| `tree.get_actionable_tools(available_inputs, level)` | **Info-State Matching**: Returns tools runnable with inputs the LLM currently possesses. |
| `tree.get_reachable_tools(node_id, max_hops, level)` | Returns reachable tools within N hops with step distance and breadcrumbs. |
| `tree.search(query, required_inputs, tags, level)` | Multi-faceted fast search combining query, inputs, and tags. |
| `tree.find_by_input_param(param)` / `find_by_tag(tag)` | $O(1)$ inverted index lookups by parameter or tag. |
| `tree.retrieve_tools(query, top_k)` | **RAG**: Returns top-$k$ relevant tools using semantic/lexical search. |
| `tree.retrieve_best_tool(query)` | **RAG**: Convenience method returning the single best `DendronNode`. |
| `tree.set_embedding_function(fn)` | Plugs in optional dense embeddings (OpenAI, HuggingFace, etc.) for hybrid RAG. |
| `tree.search_bfs(query, predicate)` | Breadth-first traversal matching query or predicate. |
| `tree.search_dfs(query, predicate)` | Depth-first traversal along execution paths. |
| `tree.find_by_name(name)` / `find_by_id(id)` | $O(1)$ direct node lookup. |
| `tree.get_frequently_accessed_tools(limit)` | Returns Most Frequently Used (MFU) tools for cache-warming. |
| `tree.get_node_addition_guidelines()` | Structured guidelines for the LLM on when and how to dynamically add nodes. |
| `tree.save(path)` / `Dendron.load(path)` | Saves and loads the tree to/from a JSON file. |
| `node.to_view(level)` | Formats node at Level 1 (compact string), Level 2 (param summary), or Level 3 (MCP). |
| `node.get_system_prompt(**kwargs)` | Renders system prompt template with node and runtime variables. |
| `node.get_user_prompt(**kwargs)` | Renders user prompt template with node and runtime variables. |
| `MCPAdapter.to_mcp_tools_list(tree)` | Exports tree to standard Model Context Protocol `tools/list` format. |
| `MCPAdapter.from_mcp_tools_list(...)` | Builds a `Dendron` tree from an MCP tools list. |

---

## Testing & Examples

- **Run Unit Tests** (55 tests, zero external dependencies):
  ```bash
  python3 -m unittest discover tests
  ```
- **Run Standalone Examples**:
  ```bash
  # 1. Email Management Agent Demo
  python3 dendron/examples/email_agent_example.py

  # 2. DevOps Incident Response & SRE Demo
  python3 dendron/examples/devops_incident_agent.py

  # 3. Code Intelligence & CI/CD Review Demo
  python3 dendron/examples/code_intelligence_agent.py
  ```
- **In-Depth Guide**: See [DEVELOPER_GUIDE.md](file:///Users/sumanthmallya/Desktop/dendron/DEVELOPER_GUIDE.md) for architectural patterns, custom transition conditions, stateful agent designs, and progressive token optimization.

---

## Author

**Sumanth Mallya**
- GitHub: [@sumanth1989](https://github.com/sumanth1989)

---

## License

[MIT License](LICENSE)

