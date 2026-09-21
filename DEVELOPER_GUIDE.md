# Developer Guide: Building Agents with `Dendron`

This guide explains how developers can design, extend, and test autonomous agent workflows using `Dendron`.

---

## 1. Architectural Philosophy

### Why a Tree Instead of a List?

Standard AI agent tool use (such as OpenAI Tools or Anthropic Tool Use) exposes a flat array of tools on every prompt turn:

```text
[ LLM Turn ] ---> Sees: [Tool1, Tool2, Tool3, ..., Tool40]
```

**Drawbacks:**
- **Context Pollution**: Sending 40 JSON schemas uses hundreds or thousands of prompt tokens on every turn.
- **Ordering Ambiguity**: The LLM must deduce the correct sequence of operations with no structural guidance.
- **No Dynamic Learning**: The agent cannot persist which tools work well together without retraining or rewriting code.

### The `Dendron` Approach:

```
[ Root Node ] ---> LLM sees ONLY immediate branches or searches on demand
      │
  [ Step 1 ] (Evaluates output against TransitionCondition)
      │
  [ Step 2 ] (Learns new path & attaches new DendronNode dynamically)
```

1. **Focused Tool Context**: The LLM only sees the relevant tools for its current execution state.
2. **Deterministic & Learned Transitions**: High-confidence transitions are handled automatically by `TransitionCondition` or suggested by `suggest_next_tool()`.
3. **Experience Preservation**: The agent records its own workflow paths at runtime with `record_agent_experience()`.

---

## 2. Setting Up an Execution Tree

### Step 1: Define Tools with `ToolDefinition` & `ToolParameter`

Each tool definition conforms to the Model Context Protocol (MCP):

```python
from dendron import ToolDefinition, ToolParameter

search_tool = ToolDefinition(
    name="web_search",
    description="Search the web for real-time information",
    parameters={
        "query": ToolParameter(name="query", type="string", description="Search query string", required=True),
        "max_results": ToolParameter(name="max_results", type="integer", description="Max hits", default=5),
    },
    tags=["web", "search"]
)
```

### Step 2: Initialize `Dendron` with Discovery Instructions

Discovery instructions guide the agent on how to traverse the tree when it needs to discover capabilities:

```python
from dendron import Dendron

tree = Dendron(
    name="ResearchAgentTree",
    root_tool=search_tool,
    discovery_instructions="""
    DISCOVERY INSTRUCTIONS:
    - Root tool is 'web_search'.
    - If search results contain academic papers, follow branch 'academic_parser'.
    - If search results contain news articles, follow branch 'news_summarizer'.
    """
)
```

### Step 3: Attach Execution Branches with Prompt Templates

```python
tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(name="academic_parser", description="Extract citations and abstracts"),
    branch_label="academic_parser",
    system_prompt_template="You are an academic researcher for {lab_name}. Format in {citation_style}.",
    prompt_variables={"lab_name": "AI Research Lab", "citation_style": "APA"}
)
```

---

## 3. Dynamic Experience Learning

When an agent encounters a new pattern during execution, it dynamically appends a new node:

```python
academic_node = tree.find_by_name("academic_parser")

# Dynamic addition based on agent experience
tree.record_agent_experience(
    parent_id=academic_node.id,
    next_tool=ToolDefinition(
        name="bibtex_exporter",
        description="Converts parsed citations to BibTeX entries",
        parameters={"citations": ToolParameter(name="citations", type="array")}
    ),
    trigger_condition_description="Output contains DOI or ISBN numbers",
    condition_type="output_contains",
    condition_expression="doi:",
    experience_note="Learned that academic papers with DOIs should immediately generate BibTeX."
)
```

### Next Tool Suggestion Logic

When calling `tree.suggest_next_tool(current_node_id, previous_output)`:
1. **First Pass (Specific Conditions)**: Evaluates child nodes that have explicit conditions (e.g., `output_contains`, `key_equals`, or custom lambda predicates).
2. **Second Pass (Unconditional Fallbacks)**: If no specific condition matches, it evaluates fallback children (`always` or unconditioned).

```python
# Evaluates DOIs and automatically routes to bibtex_exporter
next_step = tree.suggest_next_tool(
    current_node_id=academic_node.id,
    previous_output="Paper title: Attention Is All You Need. doi:10.1234/5678"
)
print(next_step.tool.name) # "bibtex_exporter"
```

---

## 4. Generic Agent Loop (Showcasing All Functionality)

Because `Dendron` is 100% model-agnostic and framework-independent, you can pair it with any LLM (OpenAI, Anthropic Claude, Google Gemini, Ollama, local models, or custom APIs) and any agent orchestrator.

Below is a complete, end-to-end example demonstrating how an autonomous agent uses `Dendron` across every stage of execution:

```python
import json
from typing import Any, Dict, List, Optional
from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    MCPAdapter,
)

# ----------------------------------------------------------------------
# 1. Initialize the Execution Tree
# ----------------------------------------------------------------------

root_tool = ToolDefinition(
    name="fetch_customer_data",
    description="Fetches customer account details and recent support tickets.",
    parameters={
        "customer_id": ToolParameter(name="customer_id", type="string", description="Customer ID"),
    },
    tags=["customer", "fetch", "root"]
)

tree = Dendron(
    name="CustomerSupportTree",
    root_tool=root_tool,
    discovery_instructions="""
    DISCOVERY INSTRUCTIONS:
    - Root tool is 'fetch_customer_data'.
    - If billing issue is detected, branch to 'refund_transaction'.
    - If technical issue is detected, branch to 'diagnose_system_error'.
    """
)

# Attach execution branches with prompt templates
refund_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(
        name="refund_transaction",
        description="Processes a payment refund for a customer transaction.",
        parameters={
            "transaction_id": ToolParameter(name="transaction_id", type="string"),
            "amount": ToolParameter(name="amount", type="number"),
        },
        tags=["billing", "refund", "payment"]
    ),
    branch_label="refund_branch",
    condition=TransitionCondition(
        description="Customer mentions billing, overcharge, or refund",
        condition_type="output_contains",
        expression="billing_issue"
    ),
    system_prompt_template="You are a senior billing agent for {company_name}. Tone: {tone}.",
    user_prompt_template="Process refund request for customer {customer_id}.",
    prompt_variables={"company_name": "CloudCorp", "tone": "empathetic and professional"}
)

diag_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(
        name="diagnose_system_error",
        description="Runs diagnostic scans on customer cloud services.",
        parameters={
            "service_name": ToolParameter(name="service_name", type="string"),
        },
        tags=["technical", "diagnostics", "logs"]
    ),
    branch_label="diagnostics_branch",
    condition=TransitionCondition(
        description="Customer reports outages, bugs, or errors",
        condition_type="output_contains",
        expression="tech_issue"
    )
)

# ----------------------------------------------------------------------
# 2. Generic Autonomous Agent
# ----------------------------------------------------------------------

class AutonomousSupportAgent:
    def __init__(self, tree: Dendron):
        self.tree = tree
        self.current_node: DendronNode = tree.root

    def call_llm(self, system_prompt: str, user_prompt: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulated LLM call. In production, connect this to:
        - OpenAI: client.chat.completions.create(model="gpt-4o", ...)
        - Anthropic: client.messages.create(model="claude-3-5-sonnet", ...)
        - Google Gemini: client.models.generate_content(...)
        - Ollama / Local: requests.post("http://localhost:11434/api/chat", ...)
        """
        # For demonstration, returns a structured decision
        return {
            "selected_tool": available_tools[0]["name"] if available_tools else None,
            "rationale": "Tool matches current execution stage."
        }

    def run_step(self, customer_id: str, runtime_output: str) -> None:
        print(f"\n--- Current Node: {self.current_node.tool.name} ---")

        # A. Prompt Injection: render system & user prompts
        sys_prompt = self.current_node.get_system_prompt()
        usr_prompt = self.current_node.get_user_prompt(customer_id=customer_id)
        print(f"System Prompt: {sys_prompt}")
        print(f"User Prompt:   {usr_prompt}")

        # B. Export immediate branch tools to MCP tools/list format
        immediate_children = [c.tool.to_mcp_dict() for c in self.current_node.children]
        decision = self.call_llm(sys_prompt, usr_prompt, immediate_children)
        print(f"LLM Decision:  {decision}")

        # C. Record execution history on the node
        self.current_node.record_execution(
            result=ToolResult(tool_name=self.current_node.tool.name, is_success=True, output=runtime_output),
            note=f"Successfully handled ticket for customer {customer_id}."
        )

        # D. Rule-based next-tool routing via transition conditions
        next_step = self.tree.suggest_next_tool(self.current_node.id, previous_output=runtime_output)
        if next_step:
            print(f"-> TransitionCondition suggested next tool: '{next_step.tool.name}'")
            self.current_node = next_step
        else:
            print("-> No automatic transition rule matched.")

    def learn_new_capability(self, parent_node_id: str, tool: ToolDefinition, condition_text: str, note: str) -> None:
        """Dynamically appends a new tool to the tree at runtime based on experience."""
        new_node = self.tree.record_agent_experience(
            parent_id=parent_node_id,
            next_tool=tool,
            trigger_condition_description=condition_text,
            condition_type="output_contains",
            condition_expression="escalate",
            experience_note=note
        )
        print(f"-> Agent dynamically learned and attached '{new_node.tool.name}'!")

    def discover_tool_by_intent(self, user_intent: str) -> Optional[DendronNode]:
        """RAG Semantic Discovery: query the tree using natural language."""
        print(f"\nSearching tree with intent: '{user_intent}'...")
        results = self.tree.retrieve_tools(user_intent, top_k=2)
        for r in results:
            print(f"  Match: {r.tool_name} (Score: {r.score:.2f}, Reasons: {', '.join(r.match_reasons)})")
        return self.tree.retrieve_best_tool(user_intent)


# ----------------------------------------------------------------------
# 3. Execution Walkthrough
# ----------------------------------------------------------------------

agent = AutonomousSupportAgent(tree)

# Step 1: Execute Root with billing output
agent.run_step(customer_id="cust_1001", runtime_output="Customer reported billing_issue: duplicate charge on invoice.")

# Step 2: Agent dynamically learns a new path (e.g. manager escalation)
agent.learn_new_capability(
    parent_node_id=refund_node.id,
    tool=ToolDefinition(
        name="escalate_to_finance_manager",
        description="Escalates high-value refunds over $1000 to finance supervisor.",
        parameters={"amount": ToolParameter(name="amount", type="number")},
        tags=["escalation", "finance", "manager"]
    ),
    condition_text="Refund exceeds limit or customer demands manager",
    note="Learned that refunds over $1000 require manager approval."
)

# Step 3: Verify dynamic transition works immediately
escalation_step = tree.suggest_next_tool(refund_node.id, previous_output="Need to escalate refund of $1500")
print(f"Suggestion after learning: '{escalation_step.tool.name}'")

# Step 4: RAG-based semantic discovery
matched_tool = agent.discover_tool_by_intent("I need to run hardware health checks and check system logs")
print(f"Best RAG Tool: '{matched_tool.tool.name}'")

# Step 5: Fast lookups and MFU caching
frequent_tools = tree.get_frequently_accessed_tools(limit=3)
print("\nTop Frequently Used Tools:")
for t in frequent_tools:
    print(f"  - {t.tool.name} (Access count: {t.access_count})")

# Step 6: Export entire tree to standard MCP format
mcp_catalog = MCPAdapter.to_mcp_tools_list(tree)
print(f"\nTotal MCP Tools Exported: {len(mcp_catalog['tools'])}")
```

---

## 5. MFU Caching & Fast Access

In high-throughput agent loops, searching the tree repeatedly can introduce overhead. `Dendron` provides:
1. **$O(1)$ Hash Registries**: `find_by_name()` and `find_by_id()` query dictionary registries instantly.
2. **Access Tracking**: Every lookup, search match, and prompt generation increments `access_count` and updates `last_accessed_at`.
3. **MFU Ranking**: `get_frequently_accessed_tools(limit=N)` returns the most frequently used tools, enabling cache-warming and prioritization in LLM context windows.

---

## 6. Model Context Protocol (MCP) Interoperability

`Dendron` can act as both an MCP client consumer and an MCP server exporter:

```python
from dendron import MCPAdapter

# Export all tools to MCP tools/list response:
mcp_response = MCPAdapter.to_mcp_tools_list(tree)

# The output matches MCP server specification:
# {
#   "tools": [
#     {
#       "name": "get_all_emails",
#       "description": "...",
#       "inputSchema": {
#         "type": "object",
#         "properties": { ... },
#         "required": [ ... ]
#       }
#     }
#   ]
# }
```

---

## 7. Token Optimization: Progressive Disclosure (Levels 1, 2, 3)

In complex agents with dozens of tools, loading full JSON Schemas wastes thousands of context tokens on every turn. `Dendron` provides **progressive disclosure**:

```python
# Level 1: Ultra-compact signature (~10-20 tokens/tool)
level_1 = tree.export_tool_views(level=1)
# Example: "refund_transaction(transaction_id, amount) - Processes a payment refund. #billing,refund"

# Level 2: Clean parameter dictionary without JSON Schema boilerplate (~50 tokens/tool)
level_2 = node.to_view(level=2)

# Level 3: Full Model Context Protocol schema for execution (~200+ tokens/tool)
# Expanded ONLY for the specific chosen tool on demand
level_3 = tree.inspect_tool("refund_transaction")
```

The LLM inspects Level 1 summaries to select a tool candidate, and only calls `tree.inspect_tool(name)` to fetch the Level 3 inputSchema right before calling it.

---

## 8. Information-State Matching & Reachability Horizons

### Actionable Tools Matching
When the LLM possesses specific information (e.g. from user input or prior outputs), it can query which tools are immediately runnable:

```python
# Returns tools whose required parameters are satisfied by the provided keys
actionable = tree.get_actionable_tools(
    available_inputs=["transaction_id", "amount"],
    detail_level=1
)
```

### Reachability Horizons
Agents can inspect reachable tools within $N$ steps from their current execution position:

```python
# Returns reachable tools up to 2 hops away with step distance and breadcrumb paths
reachable = tree.get_reachable_tools(
    current_node_id=tree.root.id,
    max_hops=2,
    detail_level=1
)
for r in reachable:
    print(f"[{r['hops']} hops] {r['name']} via {' -> '.join(r['path'])}")
```

---

## 9. Dynamic Node Addition Guidelines

Every node in Dendron is a standard `DendronNode`. To instruct the LLM on **when** and **how** to add nodes dynamically:

```python
guidelines = tree.get_node_addition_guidelines()
# Inject guidelines into LLM system prompt:
# 1. New Capability: newly discovered external tool/API
# 2. Niche / Specialized Call: recurring fixed parameters or edge-case constraints
# 3. Learned Transition: reliable output-to-next-step sequence
```

---

## 10. File Persistence (Save & Load)

Dendron trees can be saved to disk and loaded across agent sessions:

```python
# Save tree to file
tree.save("agent_playbook.json")

# Load tree in a new session
loaded_tree = Dendron.load("agent_playbook.json")
```

