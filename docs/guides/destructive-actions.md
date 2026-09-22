# Guarding Destructive Actions

Autonomous agents operating on user environments (executing terminal commands, modifying files, writing databases) can cause irreversible damage if destructive actions are executed without confirmation.

---

## Security Attributes on `ToolDefinition`

Each tool in Dendron carries native security flags:

```python
from dendron import ToolDefinition, ToolParameter

deploy_tool = ToolDefinition(
    name="deploy_to_production",
    description="Deploys the build artifact to production Kubernetes cluster",
    parameters={"cluster_id": ToolParameter(name="cluster_id", type="string")},
    is_destructive=True,
    security_level="destructive",       # "safe" | "guarded" | "destructive"
    requires_confirmation=True          # Demands Human-in-the-Loop approval
)
```

---

## Human-in-the-Loop Approval Pattern

Before executing a tool, inspect its security tier:

```python
def safe_execute(tree, tool_name: str, **kwargs):
    node = tree.find_by_name(tool_name)
    if not node:
        raise KeyError(f"Tool {tool_name} not found")

    if node.tool.requires_confirmation:
        # Prompt user or system admin for explicit approval
        user_confirmed = input(f"WARNING: '{tool_name}' is destructive. Confirm execution? (y/n): ")
        if user_confirmed.lower() != "y":
            print("Execution cancelled by user.")
            return None

    return tree.execute(tool_name, **kwargs)
```

---

## Template Placeholder Validation

When an LLM attempts to execute an action without all information, it frequently inserts placeholder strings such as `[Insert Date]`, `[TODO]`, or `<TODO>`.

`validate_arguments(args)` automatically identifies unresolved placeholders:

```python
valid, placeholders = deploy_tool.validate_arguments({"target": "[Insert Cluster]"})

if not valid:
    print("Action blocked! Unresolved placeholders detected:", placeholders)
    # Re-prompt the LLM for missing values
```
