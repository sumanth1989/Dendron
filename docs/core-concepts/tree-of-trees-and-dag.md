# Tree-of-Trees & DAG Workflows

Complex agent systems require more than a single monolithic tree. Dendron supports **Hierarchical Tree-of-Trees** and **Directed Acyclic Graph (DAG)** workflow convergence.

---

## Hierarchical Tree-of-Trees (`mount_subtree`)

In multi-domain desktop assistants or orchestrator agents (e.g. SnapTab), agents manage distinct domains:
- **DevOps Tree**: Git actions, CI/CD builds, deployments.
- **Communication Tree**: Slack alerts, email responses, calendar events.
- **Knowledge Tree**: Notion updates, documentation lookups.

Instead of keeping one giant tree, build modular sub-trees and mount them to the orchestrator:

```python
# 1. Master orchestrator tree
orchestrator = Dendron("MasterAgent", root_tool=triage_tool)

# 2. Domain-specific subtrees
devops_tree = Dendron("DevOpsTree", root_tool=git_status_tool)
devops_tree.add_node(devops_tree.root.id, tool=git_commit_tool)

# 3. Mount onto orchestrator
orchestrator.mount_subtree(
    parent_id=orchestrator.root.id,
    subtree=devops_tree,
    branch_label="devops_workflow"
)

# devops_tree's nodes are now indexed and searchable in orchestrator!
```

---

## Directed Acyclic Graph (DAG) Convergence (`add_transition`)

In real-world workflows, multiple execution branches converge to the same subsequent action. For example:
- `process_return` $\rightarrow$ `send_confirmation_email`
- `cancel_order` $\rightarrow$ `send_confirmation_email`

Instead of duplicating the `send_confirmation_email` node, connect branches with `add_transition`:

```python
tree.add_transition(
    source_id_or_name="cancel_order",
    target_id_or_name="send_confirmation_email",
    condition=TransitionCondition(
        description="When cancellation succeeds",
        condition_type="output_contains",
        expression="cancelled"
    )
)
```

Now, calling `tree.suggest_next_tool("cancel_order", output="Order successfully cancelled")` evaluates both child nodes and cross-branch DAG transitions!
