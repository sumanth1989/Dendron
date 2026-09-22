# Progressive Token-Tiered Views

One of Dendron's most powerful capabilities is **Progressive Token-Tiered Views**. Instead of always sending full, verbose JSON schemas to the LLM, Dendron provides three levels of detail.

---

## The Three Detail Levels

### Level 1: Compact Signatures (~10–20 tokens/tool)
An ultra-compact, human-readable function signature showing required arguments and description:

```python
views = tree.export_tool_views(level=1)
print(views)
```
Output:
```text
[
  "lookup_order(order_id) - Retrieves customer order details and fulfillment status. #orders,lookup",
  "track_shipment(order_id, [courier]) - Fetches live courier tracking and ETA. #shipping"
]
```

### Level 2: Parameter Summary (~50 tokens/tool)
A structured dictionary listing parameter names, types, descriptions, and constraints without JSON Schema boilerplate:

```python
views = tree.export_tool_views(level=2)
```
Output:
```json
[
  {
    "id": "node-uuid-1",
    "name": "lookup_order",
    "description": "Retrieves customer order details.",
    "parameters": {
      "order_id": {
        "type": "string",
        "required": true,
        "description": "Customer order ID"
      }
    }
  }
]
```

### Level 3: Full MCP JSON Schema (~200+ tokens/tool)
The complete Model Context Protocol (MCP) `inputSchema` dictionary containing full JSON Schema validation metadata:

```python
full_schema = tree.inspect_tool("lookup_order")
```
Output:
```json
{
  "name": "lookup_order",
  "description": "Retrieves customer order details.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "description": "Customer order ID"
      }
    },
    "required": ["order_id"]
  }
}
```

---

## Recommended Two-Stage LLM Pattern

1. **Stage 1 (Selection)**: Supply Level 1 compact signatures in the system prompt. The LLM selects the tool name based on the compact signature (~15 tokens).
2. **Stage 2 (Inspection & Calling)**: If the LLM needs parameter schemas, it inspects that specific tool with `tree.inspect_tool(name)` to retrieve the full schema before execution.

This reduces token usage by **up to 93%**!
