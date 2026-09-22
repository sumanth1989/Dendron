# Dynamic Experience Learning

A static tool tree can only handle predicted workflows. Real-world agents encounter unexpected situations. **Dendron allows agents to dynamically learn new tool execution paths at runtime.**

---

## Recording Learned Paths

When an agent discovers that `next_tool` should follow `parent_id` when a condition is met, it records the path using `record_agent_experience`:

```python
# Agent learns that defective items under $50 should trigger an instant refund
tree.record_agent_experience(
    parent_id=return_node.id,
    next_tool=ToolDefinition(
        name="issue_instant_refund",
        description="Issues an instant refund without requiring a return shipment.",
        parameters={"order_id": ToolParameter(name="order_id", type="string")},
        tags=["refund", "instant"]
    ),
    trigger_condition_description="Customer item was damaged or defective",
    condition_type="output_contains",
    condition_expression="defective",
    experience_note="Automatically issue instant refund for defective items."
)
```

### What Happens Behind the Scenes:
1. The new tool node is attached to `return_node`.
2. The node is immediately indexed in the $O(1)$ fast lookup cache.
3. The node's search document and experience notes are indexed in the RAG engine.
4. Subsequent calls to `suggest_next_tool` will automatically evaluate this path!

---

## Negative Experience & Behavioral Suppression

When a user rejects or dismisses a tool suggestion, repeated suggestions degrade the user experience.

Dendron provides **Behavioral Suppression**:

```python
# User dismisses Twitter sharing 3 times
tree.record_negative_experience("share_on_twitter", reason="User prefers LinkedIn")
tree.record_negative_experience("share_on_twitter")
tree.record_negative_experience("share_on_twitter")

# Automatically suppressed from future next-tool suggestions!
next_tool = tree.suggest_next_tool(parent_id, output="article published")
# "share_on_twitter" will not be suggested.
```
