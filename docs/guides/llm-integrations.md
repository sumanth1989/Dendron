# LLM Integrations

Dendron is model-agnostic and framework-independent. It works with any LLM provider.

---

## OpenAI

```python
import openai
from dendron import Dendron

client = openai.OpenAI()

# 1. Export tools formatted for OpenAI function calling
tools = tree.to_openai_tools(level=1)  # Or level=3 for full JSON schemas

# 2. Call OpenAI Chat Completions
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an order support assistant."},
        {"role": "user", "content": "Look up order ORD-12345"}
    ],
    tools=tools
)
```

---

## Anthropic Claude

```python
import anthropic

client = anthropic.Anthropic()

# Export tools formatted for Anthropic tool use
tools = tree.to_anthropic_tools(level=1)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Check package status for ORD-12345"}],
    tools=tools
)
```

---

## Google Gemini

Dendron natively exports tool definitions to Google Gemini `FunctionDeclaration` specifications with OpenAPI uppercase types (`STRING`, `INTEGER`, `NUMBER`, `BOOLEAN`, `ARRAY`, `OBJECT`):

```python
from google import genai
from google.genai import types
from dendron import Dendron

client = genai.Client()

# 1. Export tools formatted for Google Gemini function calling
gemini_tools = tree.to_gemini_tools(level=1)
# Or use the universal model fetcher:
# gemini_tools = tree.fetch_tools_for_model("gemini")

# 2. Call Google Gemini with function declarations
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Look up order ORD-12345 and check status",
    config=types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=gemini_tools)]
    )
)
```

---

## Autonomous Action Planning

Dendron includes an `AutonomousActionPlanner` that formulates multi-step action plans to fulfill goals and can execute or fetch tools dynamically:

```python
from dendron import AutonomousActionPlanner

planner = AutonomousActionPlanner(tree)

# 1. Create a plan for a user objective
plan = planner.plan(goal="Customer reported damaged item for order ORD-1001 and wants a refund")

# 2. Fetch tools formatted for Gemini for the planned steps
tools_for_gemini = plan.fetch_tools_for_model("gemini")

# 3. Autonomously plan and execute the entire multi-step sequence
results = tree.plan_and_execute(
    goal="Issue refund for damaged item",
    input_context={"order_id": "ORD-1001", "amount": 49.99}
)
```

---

## Local & Edge Models (Ollama, MLX, Apple Foundation Models)

On edge devices, feeding 1,500 tokens of schemas causes significant latency. Use Level 1 compact views:

```python
# Level 1 compact signatures in system prompt (~95 tokens total)
system_prompt = f"""
Available Tools:
{chr(10).join(tree.export_tool_views(level=1))}

To call a tool, respond with JSON: {{"tool": "<name>", "arguments": {{...}}}}
"""
```
