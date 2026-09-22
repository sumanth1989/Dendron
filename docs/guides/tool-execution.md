# Tool Execution Engine

Dendron allows tools to bind Python functions directly via `handler`. This enables end-to-end execution, argument validation, and automated next-tool routing.

---

## Binding Handlers to `ToolDefinition`

You can bind any synchronous Python callable (function, lambda, or class method) to `ToolDefinition.handler`:

```python
from dendron import ToolDefinition, ToolParameter, ToolResult

def refund_order(order_id: str, amount: float) -> dict:
    # Business logic here
    return {"order_id": order_id, "refunded_amount": amount, "status": "processed"}

refund_tool = ToolDefinition(
    name="refund_order",
    description="Processes an order refund.",
    parameters={
        "order_id": ToolParameter(name="order_id", type="string", required=True),
        "amount": ToolParameter(name="amount", type="number", required=True),
    },
    handler=refund_order
)
```

---

## Executing with `tree.execute(...)`

Calling `tree.execute(node_id_or_name, **kwargs)` performs the following steps automatically:
1. **Placeholder Validation**: Checks argument values for unresolved placeholders (e.g. `[TODO]`, `[Insert Date]`).
2. **Execution**: Invokes the bound handler with `kwargs`.
3. **Execution History**: Records the outcome (`ToolResult`), increments access count, and updates success/failure counters.
4. **Next-Tool Suggestion**: Automatically evaluates downstream transition rules and returns the suggested next node!

```python
result, next_node = tree.execute("refund_order", order_id="ORD-101", amount=49.99)

if result.is_success:
    print("Success:", result.output_data)
    if next_node:
        print("Suggested next action:", next_node.tool.name)
else:
    print("Execution failed:", result.error_message)
```
