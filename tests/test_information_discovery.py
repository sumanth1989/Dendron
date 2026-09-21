"""
Tests for information-state discovery, inverted indices, reachability horizons, and multi-faceted search.
"""

import unittest
from dendron import Dendron, ToolDefinition, ToolParameter, TransitionCondition


class TestInformationDiscovery(unittest.TestCase):
    def setUp(self):
        # 1. Root: list_clusters
        self.root_tool = ToolDefinition(
            name="list_clusters",
            description="Lists all Kubernetes clusters in the organization.",
            parameters={
                "region": ToolParameter(name="region", type="string", description="Cloud region", required=False, default="us-east-1")
            },
            tags=["k8s", "infra", "cloud"]
        )
        self.tree = Dendron(name="DevOpsTree", root_tool=self.root_tool)

        # 2. Level 1 child: get_pod_logs
        self.pod_tool = ToolDefinition(
            name="get_pod_logs",
            description="Fetches stdout logs from pods in a cluster.",
            parameters={
                "cluster_id": ToolParameter(name="cluster_id", type="string", required=True),
                "pod_name": ToolParameter(name="pod_name", type="string", required=True),
                "lines": ToolParameter(name="lines", type="integer", required=False, default=100)
            },
            tags=["k8s", "logs", "diagnostics"]
        )
        self.pod_node = self.tree.add_node(
            parent_id=self.tree.root.id,
            tool=self.pod_tool,
            branch_label="pod_logs",
            condition=TransitionCondition(description="When cluster is identified", condition_type="output_contains", expression="cluster_id")
        )

        # 3. Level 2 child: restart_pod
        self.restart_tool = ToolDefinition(
            name="restart_pod",
            description="Restarts a failing pod by deleting it.",
            parameters={
                "cluster_id": ToolParameter(name="cluster_id", type="string", required=True),
                "pod_name": ToolParameter(name="pod_name", type="string", required=True),
                "force": ToolParameter(name="force", type="boolean", required=False, default=False)
            },
            tags=["k8s", "remediation", "action"]
        )
        self.restart_node = self.tree.add_node(
            parent_id=self.pod_node.id,
            tool=self.restart_tool,
            branch_label="restart",
            condition=TransitionCondition(description="When CrashLoopBackOff detected in logs", condition_type="output_contains", expression="CrashLoopBackOff")
        )

        # 4. Another branch from root: check_dns
        self.dns_tool = ToolDefinition(
            name="check_dns",
            description="Performs DNS lookup health checks.",
            parameters={
                "domain": ToolParameter(name="domain", type="string", required=True)
            },
            tags=["network", "dns"]
        )
        self.dns_node = self.tree.add_node(parent_id=self.tree.root.id, tool=self.dns_tool, branch_label="dns")

    def test_find_by_input_param(self):
        nodes_with_cluster = self.tree.find_by_input_param("cluster_id")
        self.assertEqual(len(nodes_with_cluster), 2)
        names = {n.tool.name for n in nodes_with_cluster}
        self.assertEqual(names, {"get_pod_logs", "restart_pod"})

        nodes_with_domain = self.tree.find_by_input_param("domain")
        self.assertEqual(len(nodes_with_domain), 1)
        self.assertEqual(nodes_with_domain[0].tool.name, "check_dns")

        nodes_none = self.tree.find_by_input_param("nonexistent_param")
        self.assertEqual(len(nodes_none), 0)

    def test_find_by_input_params(self):
        # match_all=True: must accept both cluster_id and pod_name
        both = self.tree.find_by_input_params(["cluster_id", "pod_name"], match_all=True)
        self.assertEqual(len(both), 2)

        # match_all=True: cluster_id and domain (no tool accepts both)
        none = self.tree.find_by_input_params(["cluster_id", "domain"], match_all=True)
        self.assertEqual(len(none), 0)

        # match_all=False: accepts either cluster_id or domain
        either = self.tree.find_by_input_params(["cluster_id", "domain"], match_all=False)
        self.assertEqual(len(either), 3)

    def test_find_by_tag(self):
        k8s_tools = self.tree.find_by_tag("k8s")
        self.assertEqual(len(k8s_tools), 3)

        dns_tools = self.tree.find_by_tag("dns")
        self.assertEqual(len(dns_tools), 1)
        self.assertEqual(dns_tools[0].tool.name, "check_dns")

    def test_get_actionable_tools(self):
        # Case A: LLM has no inputs -> only list_clusters (0 required inputs)
        actionable_empty = self.tree.get_actionable_tools(available_inputs=[], detail_level=1)
        self.assertEqual(len(actionable_empty), 1)
        self.assertIn("list_clusters", actionable_empty[0])

        # Case B: LLM has domain -> list_clusters and check_dns are actionable
        actionable_domain = self.tree.get_actionable_tools(available_inputs=["domain"], detail_level=1)
        self.assertEqual(len(actionable_domain), 2)

        # Case C: LLM has cluster_id and pod_name -> list_clusters, get_pod_logs, restart_pod are all actionable
        actionable_cluster = self.tree.get_actionable_tools(available_inputs=["cluster_id", "pod_name"], detail_level=2)
        names = {a["name"] for a in actionable_cluster}
        self.assertTrue({"get_pod_logs", "restart_pod"}.issubset(names))

    def test_get_reachable_tools(self):
        # From root: 1 hop reaches get_pod_logs and check_dns; 2 hops reaches restart_pod
        reachable_1_hop = self.tree.get_reachable_tools(self.tree.root.id, max_hops=1, detail_level=1)
        self.assertEqual(len(reachable_1_hop), 2)
        names_1 = {r["name"] for r in reachable_1_hop}
        self.assertEqual(names_1, {"get_pod_logs", "check_dns"})

        reachable_2_hops = self.tree.get_reachable_tools(self.tree.root.id, max_hops=2, detail_level=1)
        self.assertEqual(len(reachable_2_hops), 3)
        restart_entry = [r for r in reachable_2_hops if r["name"] == "restart_pod"][0]
        self.assertEqual(restart_entry["hops"], 2)
        self.assertEqual(restart_entry["path"], ["list_clusters", "get_pod_logs", "restart_pod"])
        self.assertEqual(restart_entry["condition"], "When CrashLoopBackOff detected in logs")

    def test_multi_faceted_search(self):
        # Search by query and tags
        results = self.tree.search(query="restart failing pod", tags=["remediation"], detail_level=1)
        self.assertTrue(len(results) > 0)
        self.assertIn("restart_pod", results[0])

        # Search filtered by required inputs
        results_with_domain = self.tree.search(required_inputs=["domain"], detail_level=1)
        names = [r for r in results_with_domain if "check_dns" in r]
        self.assertTrue(len(names) > 0)


if __name__ == "__main__":
    unittest.main()
