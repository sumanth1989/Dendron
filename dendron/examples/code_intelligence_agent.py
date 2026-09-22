"""
Code Intelligence & Automated CI/CD Review Agent Example demonstrating Dendron.
Showcases:
1. Multi-branch code review workflow (linting, test suite, dependency security).
2. Progressive disclosure: Level 1 compact view vs Level 2 parameter view.
3. Information-state parameter filtering (finding tools runnable with PR metadata).
4. Multi-faceted fast search (query + tags + required inputs).
5. Dynamic node addition for niche vulnerability remediation.
6. Saving and loading the playbook from disk.
"""

import os
import sys
import tempfile

# Ensure library root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
)


def create_code_review_tree() -> Dendron:
    """Constructs the initial Code Intelligence Tree."""
    
    # 1. Root Tool: inspect_pull_request
    root_tool = ToolDefinition(
        name="inspect_pull_request",
        description="Fetches pull request diff, author, and changed files from GitHub/GitLab.",
        parameters={
            "repo": ToolParameter(name="repo", type="string", description="Repository e.g. org/project", required=True),
            "pr_number": ToolParameter(name="pr_number", type="integer", description="Pull request number", required=True),
        },
        tags=["git", "pr", "github", "root"]
    )

    discovery_instructions = (
        "CODE REVIEW WORKFLOW:\n"
        "1. Start at 'inspect_pull_request' (root) to parse the changeset.\n"
        "2. If Python/JS source code is modified, branch to 'run_ast_linting'.\n"
        "3. If dependencies (requirements.txt, package.json) are modified, branch to 'scan_dependency_vulnerabilities'.\n"
        "4. If business logic or tests are modified, branch to 'run_test_suite'."
    )

    tree = Dendron(
        name="CodeReviewTree",
        root_tool=root_tool,
        discovery_instructions=discovery_instructions,
        system_prompt_template="You are an automated Code Reviewer for repository {repo}.",
        user_prompt_template="Review Pull Request #{pr_number}.",
        prompt_variables={"repo": "acme/payment-service", "pr_number": 342}
    )

    # 2. Branch 1: run_ast_linting
    lint_tool = ToolDefinition(
        name="run_ast_linting",
        description="Executes static analysis and style checkers (flake8, ruff, eslint).",
        parameters={
            "repo": ToolParameter(name="repo", type="string", required=True),
            "changed_files": ToolParameter(name="changed_files", type="array", required=True),
            "strict_mode": ToolParameter(name="strict_mode", type="boolean", default=False),
        },
        tags=["lint", "ast", "code_quality"]
    )
    tree.add_node(
        parent_id=tree.root.id,
        tool=lint_tool,
        branch_label="lint_branch",
        condition=TransitionCondition(
            description="Changeset includes source code files (.py, .ts, .js)",
            condition_type="output_contains",
            expression="source_code"
        )
    )

    # 3. Branch 2: scan_dependency_vulnerabilities
    sec_tool = ToolDefinition(
        name="scan_dependency_vulnerabilities",
        description="Scans project dependencies against known CVE databases.",
        parameters={
            "repo": ToolParameter(name="repo", type="string", required=True),
            "manifest_file": ToolParameter(name="manifest_file", type="string", default="requirements.txt"),
        },
        tags=["security", "cve", "dependencies"]
    )
    sec_node = tree.add_node(
        parent_id=tree.root.id,
        tool=sec_tool,
        branch_label="security_branch",
        condition=TransitionCondition(
            description="Changeset modifies dependency manifest files",
            condition_type="output_contains",
            expression="manifest_modified"
        )
    )

    # Sub-tool under security: patch_dependency_version
    patch_tool = ToolDefinition(
        name="patch_dependency_version",
        description="Updates an insecure package version in manifest to a secure patched release.",
        parameters={
            "repo": ToolParameter(name="repo", type="string", required=True),
            "package_name": ToolParameter(name="package_name", type="string", required=True),
            "target_version": ToolParameter(name="target_version", type="string", required=True),
        },
        tags=["security", "patch", "action"]
    )
    tree.add_node(
        parent_id=sec_node.id,
        tool=patch_tool,
        branch_label="patch_branch",
        condition=TransitionCondition(
            description="Scan reports high or critical CVE vulnerability",
            condition_type="output_contains",
            expression="CVE_CRITICAL"
        )
    )

    return tree


def run_code_intelligence_demo():
    print("=" * 75)
    print("=== DENDRON: Code Intelligence & CI/CD Review Agent Demonstration ===")
    print("=" * 75)

    # 1. Initialize Tree
    tree = create_code_review_tree()
    print(f"\n[1] Initialized Tree: '{tree.name}' with {len(tree._registry_by_id)} nodes.")

    # 2. Compare Token-Tiered Views (Level 1 vs Level 2)
    print("\n[2] Level 1 (Compact Signature - ~15 tokens) vs Level 2 (Parameter Summary - ~50 tokens):")
    root_node = tree.root
    print(f"    Level 1 String: {root_node.to_view(level=1)}")
    print(f"    Level 2 Dict:   {root_node.to_view(level=2)}")

    # 3. Parameter-Driven Discovery:
    # Suppose the LLM already knows 'repo', 'package_name', and 'target_version'
    known_keys = ["repo", "package_name", "target_version"]
    print(f"\n[3] Finding Actionable Tools given information: {known_keys}")
    actionable = tree.get_actionable_tools(available_inputs=known_keys, detail_level=1)
    for tool_view in actionable:
        print(f"    [Actionable] {tool_view}")

    # 4. Multi-Faceted Fast Search:
    # Query: "security vulnerability" + tag: "security"
    print("\n[4] Multi-Faceted Fast Search (query='cve vulnerability', tag='security'):")
    search_results = tree.search(query="cve vulnerability", tags=["security"], detail_level=1)
    for res in search_results:
        print(f"    [Match] {res}")

    # 5. Dynamic Node Addition (Specialized Niche Security Advisory)
    sec_node = tree.find_by_name("scan_dependency_vulnerabilities")
    print("\n[5] Dynamic Node Addition:")
    print("    When a critical CVE is flagged, the agent learns to generate a specialized security advisory.")
    new_advisory_node = tree.record_agent_experience(
        parent_id=sec_node.id,
        next_tool=ToolDefinition(
            name="generate_security_advisory",
            description="Publishes a GitHub Security Advisory (GHSA) for critical CVE exposures.",
            parameters={
                "repo": ToolParameter(name="repo", type="string", required=True),
                "cve_id": ToolParameter(name="cve_id", type="string", required=True),
                "severity": ToolParameter(name="severity", type="string", default="CRITICAL"),
            },
            tags=["security", "advisory", "ghsa"]
        ),
        trigger_condition_description="CVE severity is CVSS 9.0+ or zero-day",
        condition_type="output_contains",
        condition_expression="CVSS_CRITICAL",
        experience_note="Learned to immediately draft a GHSA when CVSS >= 9.0."
    )
    print(f"    Attached new node: '{new_advisory_node.tool.name}' under '{sec_node.tool.name}'")

    # Verify next-tool suggestion with new condition
    suggested = tree.suggest_next_tool(sec_node.id, previous_output="Found vulnerability in openssl: CVSS_CRITICAL 9.8")
    print(f"    Evaluated output with 'CVSS_CRITICAL' -> Next tool: '{suggested.tool.name}'")

    # 6. Tree File Persistence (Save and Load)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        print(f"\n[6] Testing File Persistence: saving tree to '{temp_path}'...")
        tree.save(temp_path)
        print(f"    Saved successfully ({os.path.getsize(temp_path)} bytes).")

        loaded_tree = Dendron.load(temp_path)
        print(f"    Loaded tree: '{loaded_tree.name}' with {len(loaded_tree._registry_by_id)} nodes.")
        assert len(loaded_tree._registry_by_id) == len(tree._registry_by_id)
        print("    Verification: Loaded node count matches original!")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n" + "=" * 75)
    print("=== Demonstration Complete ===")
    print("=" * 75)


if __name__ == "__main__":
    run_code_intelligence_demo()
