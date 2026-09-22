# Dendron 🌳🧠

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib%20Only)-green.svg?style=flat)](#installation)
[![MCP](https://img.shields.io/badge/MCP-Compliant-8A2BE2.svg?style=flat)](https://modelcontextprotocol.io)
[![Issues](https://img.shields.io/github/issues/sumanth1989/Dendron.svg?style=flat)](https://github.com/sumanth1989/Dendron/issues)

An **adaptive, tree-based tool execution and dynamic discovery library** for AI agents.

> **Framework-Agnostic & Zero Dependencies**: Built purely on the Python standard library. Works with any LLM (OpenAI, Claude, Gemini, Apple Foundation Models, MLX, Ollama) and any agent framework (LangChain, CrewAI, AutoGen, or vanilla Python).

---

## Why Dendron? (Explained Simply 🎒🌳)

### The Giant Backpack Problem
Imagine you are sitting at your desk, and someone dumps a giant 50-pound backpack containing **50 different tools** in front of you — a hammer, a blender, scuba goggles, a wrench, scissors, and a pencil.

Every time you just want to write your name, you have to dig through that giant pile. You get distracted, waste time, and might accidentally grab a wrench or hammer instead of a pencil!

In AI, that is how standard agent frameworks work today: they dump all 20 to 50 tools into the AI's prompt on every single turn. This creates critical problems:
1. **Wastes Brain Space (Tokens)**: The AI has to re-read descriptions of 50 tools over and over on every turn.
2. **Tool Selection Overload**: Searching through an unorganized pile of 50 tools slows down tool selection and makes it difficult to know which tool to pick next.

---

### How Dendron Fixes This (The "Choose-Your-Own-Adventure" Tree)

Instead of dumping a messy pile of 50 tools onto your desk, **Dendron** organizes tools into a structured, step-by-step execution tree:

1. **Step-by-Step (Only See What You Need Right Now)**:
   When you sit down to start your task, you only see the initial tool on your workbench — like starting with the `pencil` (`sketch_plan`) to draft your plan. You never have to rummage through blenders, hammers, or wrenches.
2. **The Next Tool Appears Automatically (Branches)**:
   - If your plan requires **building furniture**, the tree branches to hand you the `hammer` (`hammer_nails`) — and once nails are driven in, only the `wrench` appears to tighten bolts.
   - If your plan requires **making a smoothie**, the tree branches to hand you the `blender` (`blend_ingredients`) — and once ingredients are pureed, only the `pour_pitcher` appears.
   - You **never** see the blender when you are hammering nails, and you **never** see the hammer when you are blending smoothies!
3. **Add Specialized Tools On the Fly (Dynamic Learning)**:
   If you discover a recurring specialized task (like needing a `fine_strainer` for smoothies or a `wood_chisel` for carpentry), Dendron attaches that new tool directly onto that specific branch for future runs.
4. **Find Any Tool Instantly (RAG Search)**:
   If you ever need an unusual tool from your workshop storage, you can simply ask in plain English (*"Where is the torque wrench?"* or *"Where is the citrus juicer?"*) and Dendron retrieves the exact tool in milliseconds without cluttering your desk.

```
                     [ 1. Start: Pencil / Blueprint ]
                            (sketch_plan)
                            /           \
         Needs Carpentry?              Needs Kitchen Prep?
       (task: "woodwork")              (task: "culinary")
                /                               \
      [ 2a. Hammer Tool ]              [ 2b. Blender Tool ]
        (hammer_nails)                  (blend_ingredients)
               |                                |
     Bolts need tightening?            Ingredients blended?
      (status: "nailed")                 (status: "pureed")
               |                                |
      [ 3a. Wrench Tool ]             [ 3b. Pour Pitcher Tool ]
```

---

### Real-World Examples of the Tools Problem

#### 1. DevOps & SRE Incident Response (The 40-Tool Cloud Pile)
- **The Problem**: Standard agents dump 40+ tools into every turn: AWS CLI, Kubernetes kubectl, Datadog queries, Postgres metrics, Redis flush, PagerDuty alerts, and Slack notifications.
- **The Dendron Solution**: The agent starts with `fetch_alert_details`. If it is a Kubernetes CPU spike, only `inspect_pod_metrics` is offered. If metrics reveal an out-of-memory error, only `restart_pod` or `scale_deployment` is provided.

#### 2. Code Review & CI/CD Pipeline (The 30-Tool Developer Pile)
- **The Problem**: The agent is given git diff, commit, push, PR comments, ruff linter, pytest, docker build, and terraform apply all at once.
- **The Dendron Solution**: The agent starts with `fetch_pr_diff`. If Python files changed, `run_ruff_linter` appears. If tests fail, `extract_traceback` appears. The agent never sees deployment tools until linting and tests pass.

#### 3. Academic & Literature Research (The 25-Tool Research Pile)
- **The Problem**: Dumping arXiv search, PubMed search, PDF download, OCR text extraction, citation parser, BibTeX exporter, and summarizer tools into the model simultaneously.
- **The Dendron Solution**: The agent starts with `search_papers`. Once papers are selected, `download_pdf` appears; once downloaded, `extract_citations` appears; and once parsed, `generate_bibtex` appears.

---

## Installation

Dendron supports multiple installation methods depending on your environment and deployment workflow:

### 1. From PyPI (Standard Production)
```bash
# Core library (Zero external dependencies)
pip install dendron-ai

# With optional LangChain support
pip install "dendron-ai[langchain]"
```

### 2. Direct from GitHub (Latest Commits)
```bash
pip install git+https://github.com/sumanth1989/Dendron.git

# With LangChain support
pip install "dendron-ai[langchain] @ git+https://github.com/sumanth1989/Dendron.git"
```

### 3. From Local Source
```bash
# Standard local install
pip install .

# Or editable install for active development
pip install -e .
```

### 4. From Built Wheel (`.whl`)
```bash
# Build the distribution
python -m build

# Install from the generated wheel
pip install dist/dendron_ai-0.1.0-py3-none-any.whl
```

### 5. Using Poetry
```bash
# Add from PyPI
poetry add dendron-ai

# Or add directly from GitHub
poetry add git+https://github.com/sumanth1989/Dendron.git

# Or add from local path
poetry add /path/to/Dendron
```

### 6. Zero-Dependency Vendoring (No Package Manager Needed)
Because Dendron's core has **zero external dependencies** (built purely on the Python standard library), you can simply copy the `dendron/` directory directly into your project, AWS Lambda function, Cloud Run container, or embedded agent repository:
```text
my_project/
├── dendron/          <-- Zero external dependencies
└── agent.py          <-- from dendron import Dendron, ToolDefinition
```

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
| `tree.to_langchain_tools()` | Converts all tree nodes into LangChain `StructuredTool` instances. |
| `Dendron.from_langchain_tools(tools)` | Builds an executable Dendron tree directly from a list of LangChain tools. |
| `tree.fetch_tools_for_model(provider)` | Formats tools for `'openai'`, `'anthropic'`, `'gemini'`, `'mcp'`, or `'langchain'`. |
| `node.to_view(level)` | Formats node at Level 1 (compact string), Level 2 (param summary), or Level 3 (MCP). |
| `node.to_langchain()` | Converts a single Dendron node into a LangChain `StructuredTool`. |
| `node.get_system_prompt(**kwargs)` | Renders system prompt template with node and runtime variables. |
| `node.get_user_prompt(**kwargs)` | Renders user prompt template with node and runtime variables. |
| `MCPAdapter.to_mcp_tools_list(tree)` | Exports tree to standard Model Context Protocol `tools/list` format. |
| `MCPAdapter.from_mcp_tools_list(...)` | Builds a `Dendron` tree from an MCP tools list. |
| `LangChainAdapter.from_langchain_tool(...)` | Ingests a LangChain `@tool` or `StructuredTool` into a `ToolDefinition`. |

---

## Testing & Examples

- **Run Unit Tests** (123 tests, zero external dependencies):
  ```bash
  python3 -m unittest discover tests
  ```
- **Run Standalone Examples**:
  ```bash
  # 1. LangChain Interoperability Demo
  python3 dendron/examples/langchain_integration_example.py

  # 2. Email Management Agent Demo
  python3 dendron/examples/email_agent_example.py

  # 3. DevOps Incident Response & SRE Demo
  python3 dendron/examples/devops_incident_agent.py

  # 4. Code Intelligence & CI/CD Review Demo
  python3 dendron/examples/code_intelligence_agent.py
  ```
- **In-Depth Guide**: See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for architectural patterns, custom transition conditions, stateful agent designs, and progressive token optimization.

---

## LangChain Integration & Interoperability

Dendron provides seamless bidirectional interoperability with the **LangChain** ecosystem (`BaseTool`, `StructuredTool`, `@tool`, and `AgentExecutor`):

```python
from langchain_core.tools import tool
from dendron import Dendron, TransitionCondition

# 1. Define standard LangChain tools
@tool
def lookup_customer(customer_id: str) -> dict:
    """Looks up a customer profile and balance."""
    return {"customer_id": customer_id, "balance": -50.0}

@tool
def issue_refund(customer_id: str, amount: float) -> dict:
    """Issues a refund to customer."""
    return {"status": "refunded", "amount": amount}

# 2. Ingest LangChain tools directly into a Dendron execution tree
tree = Dendron.from_langchain_tools(
    tools=[lookup_customer, issue_refund],
    name="SupportTree",
    root_tool_name="lookup_customer"
)

# 3. Add transition conditions, prompt templates, or DAG routing
refund_node = tree.find_by_name("issue_refund")
refund_node.transition_condition = TransitionCondition(
    description="Refund if customer has negative balance",
    condition_type="custom",
    expression=lambda out: isinstance(out, dict) and out.get("balance", 0) < 0
)

# 4. Export back to LangChain StructuredTool instances for LangChain agents / ChatOpenAI
langchain_tools = tree.to_langchain_tools()

# Or convert a single node
lc_tool = tree.root.to_langchain()
result = lc_tool.invoke({"customer_id": "CUST-100"})
```

> **Full Runnable Demo**: See [`dendron/examples/langchain_integration_example.py`](dendron/examples/langchain_integration_example.py) for an end-to-end walkthrough featuring conditional routing, token views, and LangChain `.invoke()` execution.

---

## Author

**Sumanth Mallya**
- GitHub: [@sumanth1989](https://github.com/sumanth1989)

---

## License

[MIT License](LICENSE)

