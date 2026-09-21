# Dendron 🌳🧠

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib%20Only)-green.svg?style=flat)](#installation)
[![MCP](https://img.shields.io/badge/MCP-Compliant-8A2BE2.svg?style=flat)](https://modelcontextprotocol.io)

An **adaptive, tree-based tool execution and dynamic discovery library** for AI agents.

> **Framework-Agnostic & Zero Dependencies**: Built purely on the Python standard library. Works with any LLM (OpenAI, Claude, Gemini, Apple Foundation Models, MLX, Ollama) and any agent framework (LangChain, CrewAI, AutoGen, or vanilla Python).

---

## Why Dendron?

Standard agent architectures dump 20–50 tools into every LLM prompt turn. This wastes context tokens, increases hallucinations, and loses the natural flow of execution.

**Dendron** structures tool use as an **adaptive execution tree**:
- **Context-Efficient**: The agent only sees relevant next-step tools along its current branch.
- **Dynamic Learning**: Agents discover and attach new execution paths at runtime as they encounter novel tasks.
- **Deterministic Transitions**: Evaluates tool outputs against rules (`output_contains`, `key_equals`, or custom predicates) to suggest next steps.
- **RAG Semantic Discovery**: When an agent doesn't know what tool to use, it queries the tree using natural language intent.

```
                      [ Root Tool Node ]
                      (get_all_emails)
                     /                \
        condition: "respond"       condition: "unread"
                   /                    \
     [ DendronNode: respond_to_email ]   [ DendronNode: extract_unread_emails ]
                                        /                        \
                           condition: "read"           condition: "vital"
                                      /                            \
                        [ DendronNode: read_email ]    [ DendronNode: extract_vital_info ]
                                      |
                               (Learned dynamically)
                                      |
                        [ DendronNode: archive_email ]
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

# 1. Define Root Tool (MCP-Compliant)
root_tool = ToolDefinition(
    name="get_all_emails",
    description="Fetches recent emails from the mailbox API.",
    parameters={
        "mailbox": ToolParameter(name="mailbox", type="string", default="INBOX"),
        "max_count": ToolParameter(name="max_count", type="integer", default=25),
    },
    tags=["email", "fetch", "root"]
)

# 2. Create the Tree with Discovery Instructions
tree = Dendron(
    name="EmailAgentTree",
    root_tool=root_tool,
    discovery_instructions="Start at 'get_all_emails'. Branch to 'extract_unread_emails' or 'respond_to_email'."
)

# 3. Attach Execution Branches with Prompt Templates
unread_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(name="extract_unread_emails", description="Filter unread messages", tags=["filter"]),
    branch_label="unread_branch"
)

respond_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(name="respond_to_email", description="Send an email reply", tags=["reply", "send"]),
    branch_label="respond_branch",
    system_prompt_template="You are replying for {user_name}. Tone: {tone}.",
    prompt_variables={"user_name": "Alice", "tone": "professional"}
)

# 4. Prompt Injection
print(respond_node.get_system_prompt(tone="concise"))
# Output: "You are replying for Alice. Tone: concise."

# 5. Dynamic Experience Learning
tree.record_agent_experience(
    parent_id=unread_node.id,
    next_tool=ToolDefinition(name="archive_email", description="Archive email"),
    trigger_condition_description="Email is a newsletter",
    condition_type="output_contains",
    condition_expression="newsletter",
    experience_note="Automatically archive newsletters after reading."
)

# 6. Intelligent Next-Tool Suggestion
next_tool = tree.suggest_next_tool(unread_node.id, previous_output="Received newsletter #42")
print(next_tool.tool.name)  # Output: "archive_email"

# 7. RAG Semantic Tool Retrieval (Natural Language Intent)
best_match = tree.retrieve_best_tool("I need to send a quick reply to a client")
print(best_match.tool.name)  # Output: "respond_to_email"

# 8. MCP Export
from dendron import MCPAdapter
mcp_tools = MCPAdapter.to_mcp_tools_list(tree)
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

## License

[MIT License](LICENSE)
