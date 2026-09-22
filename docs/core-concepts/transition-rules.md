# Transition Rules & Conditions

In Dendron, branches connect tools together. A branch can be unconditional or governed by a `TransitionCondition`.

---

## Condition Types

`TransitionCondition` supports four condition types:

### 1. `always` (Default / Fallback)
Unconditionally allows transition from parent to child.

```python
TransitionCondition(
    description="Always proceed to notification",
    condition_type="always"
)
```

### 2. `output_contains` (Substring Match)
Evaluates whether a string or keyword appears in the previous tool's output:

```python
TransitionCondition(
    description="Previous output indicated defective product",
    condition_type="output_contains",
    expression="defective"
)
```

### 3. `key_equals` (Dictionary Key Matching)
Evaluates whether a specific key in a dictionary output matches an expected value (syntax: `"key:value"`):

```python
TransitionCondition(
    description="Fulfillment status is delivered",
    condition_type="key_equals",
    expression="status:delivered"
)
```

### 4. Custom Lambda / Callable
Pass any Python function that accepts the previous output and returns a boolean:

```python
TransitionCondition(
    description="Order value exceeds $500",
    condition_type="custom",
    expression=lambda out: isinstance(out, dict) and out.get("total_amount", 0) > 500
)
```

---

## Next-Tool Evaluation Priority

When calling `tree.suggest_next_tool(current_node_id, previous_output)`:
1. **First Pass (Specific Conditions)**: Evaluates child nodes that have explicit conditions (`output_contains`, `key_equals`, `custom`).
2. **Second Pass (DAG Specific Conditions)**: Evaluates cross-branch DAG transitions with specific conditions.
3. **Third Pass (Unconditional Fallbacks)**: Evaluates unconditional children (`always`).
4. **Fourth Pass (Unconditional DAG Transitions)**: Evaluates unconditional cross-branch transitions.
5. **Suppression Filter**: Any tool that has received $\ge 3$ negative feedback dismissals via `record_negative_experience()` is automatically excluded!
