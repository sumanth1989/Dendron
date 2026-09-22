# LangChain Integration

Dendron provides seamless bidirectional interoperability with the **LangChain** ecosystem. You can:

1. Ingest existing LangChain tools (`@tool`, `StructuredTool`, `BaseTool`) directly into Dendron execution trees.
2. Structure flat LangChain tools into hierarchical execution workflows with conditional transitions, DAG routing, and token-tiered views.
3. Export Dendron trees, subtrees, or individual nodes back into LangChain `StructuredTool` instances compatible with `ChatOpenAI.bind_tools()`, Anthropic, Google Gemini, or LangChain agents (`create_tool_calling_agent`, `AgentExecutor`).
4. Run in environments without `langchain-core` using Dendron's built-in duck-typed `DendronLangChainTool` bridge.

---

## Installation

To enable first-class LangChain support with Pydantic v2 schema generation, install Dendron with the `langchain` extra:

```bash
pip install "dendron-ai[langchain]"
```

---

## 1. Converting LangChain Tools to Dendron

You can convert any LangChain `@tool` or `StructuredTool` into a Dendron `ToolDefinition` using `LangChainAdapter.from_langchain_tool()` or `ToolDefinition.from_langchain()`:

```python
from langchain_core.tools import tool
from dendron import ToolDefinition, LangChainAdapter

@tool
def lookup_customer(customer_id: str) -> dict:
    """Looks up a customer profile and current balance."""
    return {"customer_id": customer_id, "tier": "enterprise", "balance": -45.0}

# Convert to Dendron ToolDefinition
tool_def = ToolDefinition.from_langchain(lookup_customer)

# Execute tool through Dendron
result = tool_def.execute(customer_id="CUST-100")
print(result.output_data)
# {'customer_id': 'CUST-100', 'tier': 'enterprise', 'balance': -45.0}
```

### Ingesting Multiple Tools into a Dendron Tree

Use `Dendron.from_langchain_tools()` to assemble a list of LangChain tools into an executable tree in one call:

```python
from dendron import Dendron

tree = Dendron.from_langchain_tools(
    tools=[lookup_customer, apply_refund, escalate_ticket],
    name="SupportTree",
    root_tool_name="lookup_customer",
    discovery_instructions="Run lookup_customer first, then conditionally route to refund or escalation."
)
```

---

## 2. Adding Conditional Transitions & Prompts

Once LangChain tools are held in Dendron, you can enrich them with execution intelligence:

```python
from dendron import TransitionCondition

refund_node = tree.find_by_name("apply_refund")
refund_node.transition_condition = TransitionCondition(
    description="Issue refund only if customer balance is negative or disputed",
    condition_type="custom",
    expression=lambda out: isinstance(out, dict) and out.get("balance", 0) < 0
)
refund_node.system_prompt_template = (
    "You are a billing specialist. Confirm that the refund amount does not exceed the dispute."
)
```

Now, after `lookup_customer` executes, Dendron automatically evaluates the output:

```python
output = tree.root.execute(customer_id="CUST-100").output_data
next_tool = tree.suggest_next_tool(tree.root.id, output)
print(next_tool.tool.name)  # "apply_refund"
```

---

## 3. Exporting Dendron Trees to LangChain Tools

You can convert Dendron nodes or trees back into LangChain `StructuredTool` instances for use with LangChain agents or LLMs:

```python
# Convert all nodes in the tree
langchain_tools = tree.to_langchain_tools()

# Or use the unified model fetcher
langchain_tools = tree.fetch_tools_for_model("langchain")

# Or convert a single node
lc_tool = tree.root.to_langchain()
```

### Using with LangChain Chat Models

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tree.to_langchain_tools())

response = llm_with_tools.invoke("Look up profile for customer CUST-100")
```

### Direct Tool Invocation

Each converted tool implements LangChain's `.invoke()` interface:

```python
result = lc_tool.invoke({"customer_id": "CUST-100"})
print(result)
```

---

## 4. Progressive Disclosure & Token Optimization

Standard LangChain schemas consume significant context window space. With Dendron, you can inject ultra-compact Level 1 tool signatures into the system prompt, saving up to 80% of tool definition tokens:

```python
# Level 1 compact view (~15 tokens per tool)
prompt_tools = tree.export_tool_views(level=1)
print(prompt_tools)
# lookup_customer(customer_id) - Looks up a customer profile and current balance.
```

---

## 5. Zero-Dependency Fallback (`DendronLangChainTool`)

If `langchain-core` is not installed, Dendron automatically returns duck-typed `DendronLangChainTool` instances that provide:

- `.invoke(input_dict)`
- `.run(tool_input, **kwargs)`
- Direct callable `__call__(*args, **kwargs)`
- `.args` dictionary matching LangChain's schema interface

This guarantees that applications using Dendron do not crash if LangChain is absent from production container environments.
