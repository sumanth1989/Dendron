# 5-Minute Quickstart

Learn how to build, traverse, and execute a tool tree in 5 minutes.

## 1. Define Tools and Handlers

Each tool in Dendron is defined with `ToolDefinition` and optional `ToolParameter` specifications:

```python
from dendron import Dendron, ToolDefinition, ToolParameter, TransitionCondition

# 1. Root Tool: Look up an order
lookup_tool = ToolDefinition(
    name="lookup_order",
    description="Retrieves customer order details and fulfillment status by order ID.",
    parameters={
        "order_id": ToolParameter(name="order_id", type="string", description="Order ID e.g. ORD-12345", required=True),
    },
    handler=lambda order_id: {"order_id": order_id, "status": "delivered", "item": "Wireless Headphones"},
    tags=["orders", "lookup", "root"]
)
```

## 2. Initialize the Tree

```python
tree = Dendron(
    name="OrderSupportTree",
    root_tool=lookup_tool,
    discovery_instructions="Start at 'lookup_order'. If delivered, branch to 'process_return'. If in transit, branch to 'track_shipment'."
)
```

## 3. Attach Execution Branches

```python
# Branch A: When package is in transit
tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(
        name="track_shipment",
        description="Fetches live courier tracking and ETA.",
        parameters={"order_id": ToolParameter(name="order_id", type="string")},
        handler=lambda order_id: {"courier": "FedEx", "eta": "Tomorrow by 5 PM"}
    ),
    branch_label="in_transit_branch",
    condition=TransitionCondition(
        description="Order status is in_transit",
        condition_type="output_contains",
        expression="in_transit"
    )
)

# Branch B: When package is delivered
return_node = tree.add_node(
    parent_id=tree.root.id,
    tool=ToolDefinition(
        name="process_return",
        description="Initiates a return for a delivered order.",
        parameters={
            "order_id": ToolParameter(name="order_id", type="string", required=True),
            "reason": ToolParameter(name="reason", type="string", required=True),
        },
        handler=lambda order_id, reason: {"return_label_id": "RET-9921", "status": "approved"}
    ),
    branch_label="delivered_branch",
    condition=TransitionCondition(
        description="Order status is delivered",
        condition_type="output_contains",
        expression="delivered"
    )
)
```

## 4. Execute Tools and Auto-Suggest Next Steps

```python
# Execute the root tool
result, next_tool = tree.execute("lookup_order", order_id="ORD-1001")

print("Execution Result:", result.output_data)
# Output: {'order_id': 'ORD-1001', 'status': 'delivered', 'item': 'Wireless Headphones'}

print("Suggested Next Tool:", next_tool.tool.name)
# Output: 'process_return' (automatically matched because status is delivered!)
```

## 5. Visualize the Tree

```python
print(tree.visualize(format="ascii"))
```

Output:
```text
└── lookup_order
    ├── track_shipment [in_transit_branch]
    └── process_return [delivered_branch]
```
