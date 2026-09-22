"""
DevOps Incident Response Agent Example demonstrating Dendron.
Showcases:
1. Root tool with discovery instructions for SRE alerts.
2. Progressive token-tiered views (Level 1, 2, 3) to minimize prompt overhead.
3. Information-state matching: finding actionable tools given currently known inputs (e.g. cluster_id, pod_name).
4. Reachability horizons: inspecting reachable tools with hop count and breadcrumbs.
5. Dynamic node addition on the fly when learning a niche incident remediation.
6. RAG semantic search and file persistence.
"""

import os
import sys

# Ensure library root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
    MCPAdapter,
)


def create_devops_incident_tree() -> Dendron:
    """Constructs the initial DevOps Incident Response Tree."""
    
    # 1. Root Tool: get_active_alerts
    root_tool = ToolDefinition(
        name="get_active_alerts",
        description="Queries monitoring service (PagerDuty/Datadog) for triggered incident alerts.",
        parameters={
            "environment": ToolParameter(name="environment", type="string", description="Environment e.g. production, staging", default="production"),
            "severity": ToolParameter(name="severity", type="string", enum=["CRITICAL", "HIGH", "MEDIUM"], default="CRITICAL"),
        },
        tags=["monitoring", "alerts", "sre", "root"]
    )

    discovery_instructions = (
        "INCIDENT TRIAGE PLAYBOOK:\n"
        "1. Start at 'get_active_alerts' (root) to identify firing incidents.\n"
        "2. If Kubernetes or container errors are detected, branch to 'k8s_cluster_diagnostics'.\n"
        "3. If database saturation is detected, branch to 'database_metrics'.\n"
        "4. If network latency or DNS failures are reported, branch to 'network_traceroute'."
    )

    tree = Dendron(
        name="DevOpsIncidentTree",
        root_tool=root_tool,
        discovery_instructions=discovery_instructions,
        system_prompt_template="You are an automated Site Reliability Engineer (SRE). On-call shift: {shift_id}.",
        user_prompt_template="Investigate incident alert for service {service_name}.",
        prompt_variables={"shift_id": "NA-WEST-01", "service_name": "payments-api"}
    )

    # 2. Branch 1: k8s_cluster_diagnostics
    k8s_tool = ToolDefinition(
        name="k8s_cluster_diagnostics",
        description="Inspects pod health, deployment status, and container restart counts in Kubernetes.",
        parameters={
            "cluster_id": ToolParameter(name="cluster_id", type="string", description="Kubernetes cluster ID", required=True),
            "namespace": ToolParameter(name="namespace", type="string", description="K8s namespace", default="production"),
        },
        tags=["k8s", "containers", "diagnostics"]
    )
    k8s_node = tree.add_node(
        parent_id=tree.root.id,
        tool=k8s_tool,
        branch_label="k8s_branch",
        condition=TransitionCondition(
            description="Alert mentions pod crash, OOM, or deployment failure",
            condition_type="output_contains",
            expression="k8s_error"
        ),
        system_prompt_template="Analyze Kubernetes health for cluster {cluster_id}. Check OOM and CrashLoopBackOff.",
        prompt_variables={"cluster_id": "k8s-prod-us-west"}
    )

    # Sub-tool under k8s: restart_failing_pod
    restart_pod_tool = ToolDefinition(
        name="restart_failing_pod",
        description="Terminates and restarts an unhealthy pod to trigger container recreation.",
        parameters={
            "cluster_id": ToolParameter(name="cluster_id", type="string", required=True),
            "namespace": ToolParameter(name="namespace", type="string", required=True),
            "pod_name": ToolParameter(name="pod_name", type="string", required=True),
        },
        tags=["k8s", "remediation", "action"]
    )
    tree.add_node(
        parent_id=k8s_node.id,
        tool=restart_pod_tool,
        branch_label="pod_restart",
        condition=TransitionCondition(
            description="Pod is stuck in CrashLoopBackOff or Error state",
            condition_type="output_contains",
            expression="CrashLoopBackOff"
        )
    )

    # 3. Branch 2: database_metrics
    db_tool = ToolDefinition(
        name="database_metrics",
        description="Checks database query latencies, active connection count, and lock contention.",
        parameters={
            "db_cluster": ToolParameter(name="db_cluster", type="string", description="Database cluster identifier", required=True),
            "window_minutes": ToolParameter(name="window_minutes", type="integer", default=15),
        },
        tags=["database", "sql", "metrics"]
    )
    tree.add_node(
        parent_id=tree.root.id,
        tool=db_tool,
        branch_label="database_branch",
        condition=TransitionCondition(
            description="Alert mentions slow queries, 504 gateway timeout, or database pool saturation",
            condition_type="output_contains",
            expression="db_latency"
        )
    )

    return tree


def run_devops_agent_demo():
    print("=" * 75)
    print("=== DENDRON: DevOps Incident Response Agent Demonstration ===")
    print("=" * 75)

    # 1. Initialize Tree
    tree = create_devops_incident_tree()
    print(f"\n[1] Initialized Tree: '{tree.name}' with {len(tree._registry_by_id)} nodes.")

    # 2. Level 1 Token-Tiered View (Ultra-Compact for LLM Context Window)
    print("\n[2] Exporting Level 1 Compact Views (~15 tokens per tool vs ~200 for full schemas):")
    compact_views = tree.export_tool_views(level=1)
    for view in compact_views:
        print(f"    - {view}")

    # 3. Reachability Horizons from Root
    print("\n[3] Reachability Horizon from Root (max_hops=2):")
    reachable = tree.get_reachable_tools(tree.root.id, max_hops=2, detail_level=1)
    for r in reachable:
        print(f"    - [{r['hops']} hop(s)] {r['name']} | Path: {' -> '.join(r['path'])}")
        if r['condition']:
            print(f"      Condition: {r['condition']}")

    # 4. Information-State Matching: What tools can the LLM run with current information?
    # Suppose the LLM already knows 'cluster_id', 'namespace', and 'pod_name' from an incoming webhook
    known_info = ["cluster_id", "namespace", "pod_name"]
    print(f"\n[4] LLM Information State: possesses {known_info}")
    actionable = tree.get_actionable_tools(available_inputs=known_info, detail_level=1)
    print("    Actionable tools (can run immediately without missing parameters):")
    for a in actionable:
        print(f"    [Actionable] {a}")

    # 5. On-Demand Inspection (Level 3)
    # LLM decides to execute 'restart_failing_pod', expanding only that tool's full schema
    print("\n[5] On-Demand Inspection for 'restart_failing_pod' (Level 3 MCP Schema):")
    mcp_schema = tree.inspect_tool("restart_failing_pod")
    print(f"    Tool Name: {mcp_schema['name']}")
    print(f"    Required Inputs: {mcp_schema['inputSchema']['required']}")

    # 6. Dynamic Node Addition (Learning Niche Remediations on the Fly)
    print("\n[6] Dynamic Experience Learning:")
    print("    Incident response revealed recurring Redis cache exhaustion causing OOM.")
    print("    Agent uses guidelines to attach a specialized 'flush_redis_cache' tool dynamically.")
    
    k8s_node = tree.find_by_name("k8s_cluster_diagnostics")
    new_node = tree.record_agent_experience(
        parent_id=k8s_node.id,
        next_tool=ToolDefinition(
            name="flush_redis_cache",
            description="Flushes ephemeral redis cache keys to release memory during OOM spikes.",
            parameters={
                "cluster_id": ToolParameter(name="cluster_id", type="string", required=True),
                "cache_cluster": ToolParameter(name="cache_cluster", type="string", required=True),
            },
            tags=["redis", "cache", "oom", "remediation"]
        ),
        trigger_condition_description="Output indicates Redis memory saturation or OOM kill",
        condition_type="output_contains",
        condition_expression="redis_oom",
        experience_note="Flushing ephemeral cache keys resolves 90% of Redis OOM spikes without restart."
    )
    print(f"    Successfully attached new node: {new_node.tool.name}")

    # 7. Verification: Suggest Next Tool with learned condition
    suggested = tree.suggest_next_tool(k8s_node.id, previous_output="Pod terminated with redis_oom error")
    print(f"    Evaluated output with 'redis_oom' -> Next suggested tool: '{suggested.tool.name}'")

    # 8. RAG Semantic Intent Retrieval
    print("\n[7] RAG Natural Language Discovery:")
    query = "database queries are taking too long and timing out"
    best_tool = tree.retrieve_best_tool(query)
    print(f"    Query: '{query}'")
    print(f"    Best Tool: {best_tool.tool.name} ({best_tool.tool.description})")

    # 9. Inverted Index Lookups
    print("\n[8] Inverted Index Parameter Lookups:")
    db_param_nodes = tree.find_by_input_param("db_cluster")
    print(f"    Tools accepting 'db_cluster': {[n.tool.name for n in db_param_nodes]}")

    print("\n" + "=" * 75)
    print("=== Demonstration Complete ===")
    print("=" * 75)


if __name__ == "__main__":
    run_devops_agent_demo()
