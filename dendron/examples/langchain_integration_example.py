"""
LangChain + Dendron Integration Example.
Showcases:
1. Converting modern LangChain `@tool` decorated functions into Dendron ToolDefinitions.
2. Ingesting multiple LangChain tools into an executable Dendron tree via `Dendron.from_langchain_tools()`.
3. Enriching the tree with branch transitions, prompt templates, and experience notes.
4. Exporting Dendron nodes/trees back to LangChain StructuredTool instances (`tree.to_langchain_tools()`).
5. Running LangChain `.invoke()` executions against Dendron-managed tools.
6. Using progressive disclosure (Level 1 compact view) for token-efficient agent prompt generation.
7. Fallback duck-typed execution when langchain-core is not present.
"""

import os
import sys
import json

# Ensure library root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dendron import (
    Dendron,
    DendronNode,
    ToolDefinition,
    ToolParameter,
    TransitionCondition,
    LangChainAdapter,
    DendronLangChainTool,
)

# Optional LangChain import
try:
    from langchain_core.tools import tool, StructuredTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("[INFO] langchain_core is not installed. Using Dendron's built-in duck-typed LangChain emulation.")


# =====================================================================
# 1. Define Standard LangChain Tools
# =====================================================================

if HAS_LANGCHAIN:
    @tool
    def lookup_customer(customer_id: str) -> dict:
        """Looks up a customer profile, subscription status, and credit balance by customer ID."""
        profiles = {
            "CUST-100": {"name": "Alice Corp", "tier": "enterprise", "balance": 450.0, "status": "active"},
            "CUST-200": {"name": "Bob Ltd", "tier": "starter", "balance": -50.0, "status": "delinquent"},
        }
        profile = profiles.get(customer_id, {"name": "Unknown", "tier": "free", "balance": 0.0, "status": "inactive"})
        return {"customer_id": customer_id, **profile}

    @tool
    def apply_billing_adjustment(customer_id: str, credit_amount: float, reason: str) -> dict:
        """Applies a credit adjustment or refund to a customer's account balance."""
        return {
            "status": "applied",
            "customer_id": customer_id,
            "adjusted_amount": credit_amount,
            "reason": reason,
            "transaction_id": "TX-994821"
        }

    @tool
    def escalate_to_support(customer_id: str, urgency: str, notes: str) -> dict:
        """Escalates an incident ticket to Tier-3 customer success and support engineering."""
        return {
            "ticket_id": "INC-7731",
            "customer_id": customer_id,
            "urgency": urgency,
            "assigned_team": "tier3-support",
            "status": "queued"
        }
else:
    # Emulate LangChain tool callables for demonstration if langchain-core is missing
    def lookup_customer(customer_id: str) -> dict:
        """Looks up a customer profile, subscription status, and credit balance by customer ID."""
        return {"customer_id": customer_id, "name": "Alice Corp", "tier": "enterprise", "balance": 450.0}
    lookup_customer.name = "lookup_customer"
    lookup_customer.description = "Looks up a customer profile, subscription status, and credit balance by customer ID."
    lookup_customer.args = {"customer_id": {"type": "string", "description": "Customer ID"}}

    def apply_billing_adjustment(customer_id: str, credit_amount: float, reason: str) -> dict:
        """Applies a credit adjustment or refund to a customer's account balance."""
        return {"status": "applied", "customer_id": customer_id, "adjusted_amount": credit_amount}
    apply_billing_adjustment.name = "apply_billing_adjustment"
    apply_billing_adjustment.description = "Applies a credit adjustment or refund to a customer's account balance."
    apply_billing_adjustment.args = {
        "customer_id": {"type": "string"},
        "credit_amount": {"type": "number"},
        "reason": {"type": "string"}
    }

    def escalate_to_support(customer_id: str, urgency: str, notes: str) -> dict:
        """Escalates an incident ticket to Tier-3 customer success and support engineering."""
        return {"ticket_id": "INC-7731", "customer_id": customer_id, "urgency": urgency}
    escalate_to_support.name = "escalate_to_support"
    escalate_to_support.description = "Escalates an incident ticket to Tier-3 customer success and support engineering."
    escalate_to_support.args = {
        "customer_id": {"type": "string"},
        "urgency": {"type": "string"},
        "notes": {"type": "string"}
    }


def main():
    print("=" * 70)
    print("Dendron + LangChain Seamless Interoperability Example")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Ingest LangChain Tools into a Dendron Tree
    # -----------------------------------------------------------------
    print("\n--- Step 1: Converting LangChain Tools into Dendron ---")
    raw_tools = [lookup_customer, apply_billing_adjustment, escalate_to_support]

    tree = Dendron.from_langchain_tools(
        tools=raw_tools,
        name="CustomerSupportTree",
        root_tool_name="lookup_customer",
        discovery_instructions=(
            "1. Run lookup_customer first to determine account tier and balance.\n"
            "2. If balance < 0 or billing dispute, route to apply_billing_adjustment.\n"
            "3. If enterprise escalation needed, route to escalate_to_support."
        )
    )

    print(f"Tree Created: {tree.name} with {len(tree)} tool nodes.")
    print(f"Root Node Tool: {tree.root.tool.name}")
    for child in tree.root.children:
        print(f" - Child Node: {child.tool.name} (branch: '{child.branch_label}')")

    # -----------------------------------------------------------------
    # Step 2: Enrich with Dendron Transitions and Prompt Contexts
    # -----------------------------------------------------------------
    print("\n--- Step 2: Adding Conditional DAG Transitions & Prompts ---")
    adjustment_node = tree.find_by_name("apply_billing_adjustment")
    escalate_node = tree.find_by_name("escalate_to_support")

    # Set transition condition on adjustment node
    if adjustment_node:
        adjustment_node.transition_condition = TransitionCondition(
            description="Execute billing adjustment if account is delinquent or balance is negative",
            condition_type="custom",
            expression=lambda out: isinstance(out, dict) and out.get("balance", 0) < 0
        )
        adjustment_node.system_prompt_template = (
            "You are a billing operations assistant. Verify adjustment reason before submitting."
        )

    # Re-link escalate node with conditional transition
    if escalate_node:
        escalate_node.transition_condition = TransitionCondition(
            description="Escalate to Tier 3 if customer is enterprise",
            condition_type="key_equals",
            expression="tier:enterprise"
        )

    print("Transitions and prompt templates successfully configured.")

    # -----------------------------------------------------------------
    # Step 3: Execute Root Tool through Dendron & Evaluate Transitions
    # -----------------------------------------------------------------
    print("\n--- Step 3: Executing Root Tool and Automatic Route Suggestion ---")
    exec_result = tree.root.execute(customer_id="CUST-200")
    print("Execution Status:", exec_result.status)
    print("Output Data:", exec_result.output_data)

    # Next tool recommendation based on previous output
    next_node = tree.suggest_next_tool(tree.root.id, exec_result.output_data)
    if next_node:
        print(f"Recommended Next Tool: '{next_node.tool.name}' (Reason: {next_node.transition_condition.description})")

    # -----------------------------------------------------------------
    # Step 4: Export Dendron Tree back to LangChain Structured Tools
    # -----------------------------------------------------------------
    print("\n--- Step 4: Exporting Dendron Tree back to LangChain Tools ---")
    langchain_tools = tree.to_langchain_tools()
    print(f"Exported {len(langchain_tools)} tools for LangChain agent consumption:")

    for lc_t in langchain_tools:
        args_repr = getattr(lc_t, "args", {})
        if hasattr(lc_t, "args_schema") and lc_t.args_schema:
            args_repr = list(lc_t.args_schema.model_json_schema().get("properties", {}).keys())
        print(f" * LangChain Tool: '{lc_t.name}' | Args: {args_repr}")

    # -----------------------------------------------------------------
    # Step 5: Execute LangChain Tool via .invoke()
    # -----------------------------------------------------------------
    print("\n--- Step 5: Executing via LangChain .invoke() ---")
    target_lc_tool = next(t for t in langchain_tools if t.name == "lookup_customer")
    customer_response = target_lc_tool.invoke({"customer_id": "CUST-100"})
    print("LangChain Tool .invoke() returned:")
    print(json.dumps(customer_response, indent=2))

    # -----------------------------------------------------------------
    # Step 6: Progressive Disclosure (Token Savings for LLM Prompts)
    # -----------------------------------------------------------------
    print("\n--- Step 6: Progressive Disclosure Views for LLM Prompts ---")
    print("Level 1 (Ultra-compact tool signature - approx 15 tokens):")
    print(tree.root.to_compact_summary())

    print("\nLevel 2 (Parameter Summary - approx 50 tokens):")
    print(json.dumps(tree.root.to_parameter_summary(), indent=2))

    # -----------------------------------------------------------------
    # Step 7: Model-Specific Export via fetch_tools_for_model
    # -----------------------------------------------------------------
    print("\n--- Step 7: Fetch Tools for 'langchain' Model Provider ---")
    provider_tools = tree.fetch_tools_for_model("langchain")
    print(f"Successfully retrieved {len(provider_tools)} tools via tree.fetch_tools_for_model('langchain').")

    print("\n[SUCCESS] LangChain integration completed flawlessly.")


if __name__ == "__main__":
    main()
