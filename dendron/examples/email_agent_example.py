"""
Email Agent Example demonstrating the Dendron library.
Matches the user's requested specification:
- Root: `get_all_emails` API call with discovery instructions.
- Left node: `respond_to_email` tool with system & user prompt templates.
- Right node: `extract_unread_emails` tool.
- Sub-tree under right node: `read_email` and `extract_vital_information`.
- Agent dynamic learning: records experience and adds `archive_email` on the fly.
- Traversal via BFS and DFS.
- Fast MFU retrieval for most frequently accessed tools.
"""

import os
import sys

# Ensure library root is in sys.path when running example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dendron import (
    Dendron,
    DendronTree,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    TransitionCondition,
)


def create_email_tool_tree() -> Dendron:
    """Constructs the initial Email Management Dendron tree."""
    
    # 1. Root Tool: get_all_emails
    root_tool = ToolDefinition(
        name="get_all_emails",
        description="Fetches recent emails from the mailbox API.",
        parameters={
            "mailbox": ToolParameter(name="mailbox", type="string", description="Mailbox name e.g. INBOX", default="INBOX"),
            "max_count": ToolParameter(name="max_count", type="integer", description="Maximum number of emails to retrieve", default=25),
        },
        tags=["email", "fetch", "root"]
    )

    discovery_instructions = (
        "BEST WAY TO NAVIGATE SUB-TOOLS:\n"
        "1. Start at 'get_all_emails' (root) to obtain the latest message list.\n"
        "2. If unread emails exist, take the RIGHT branch to 'extract_unread_emails'.\n"
        "   - From 'extract_unread_emails', navigate to 'read_email' to fetch message bodies.\n"
        "   - From 'read_email', navigate to 'extract_vital_information' to parse deadlines & action items.\n"
        "3. If an immediate reply is needed for a specific email, take the LEFT branch to 'respond_to_email'."
    )

    tree = Dendron(
        name="EmailAgentTree",
        root_tool=root_tool,
        discovery_instructions=discovery_instructions,
        system_prompt_template="You are an autonomous Email Assistant. Today is {current_date}.",
        user_prompt_template="Process emails for user {user_email}.",
        prompt_variables={"current_date": "2026-09-21", "user_email": "user@example.com"}
    )

    # 2. Left Node: respond_to_email
    respond_tool = ToolDefinition(
        name="respond_to_email",
        description="Drafts and sends a reply to a specific email message.",
        parameters={
            "email_id": ToolParameter(name="email_id", type="string", description="ID of the email to reply to"),
            "reply_body": ToolParameter(name="reply_body", type="string", description="Text content of the reply"),
            "tone": ToolParameter(name="tone", type="string", description="Response tone e.g. professional, casual", default="professional"),
        },
        tags=["email", "reply", "send"]
    )

    tree.add_node(
        parent_id=tree.root.id,
        tool=respond_tool,
        branch_label="left",
        condition=TransitionCondition(
            description="Execute when user or message requests an immediate response",
            condition_type="output_contains",
            expression="requires_reply"
        ),
        system_prompt_template="You are an executive email assistant. Tone: {tone}. Be concise and polite.",
        user_prompt_template="Draft response to {sender} regarding subject: '{subject}'. Key points to address: {key_points}.",
        prompt_variables={"tone": "professional", "sender": "Unknown", "subject": "", "key_points": ""}
    )

    # 3. Right Node: extract_unread_emails
    extract_unread_tool = ToolDefinition(
        name="extract_unread_emails",
        description="Filters the email list to extract only unread messages.",
        parameters={
            "min_priority": ToolParameter(name="min_priority", type="string", description="Minimum priority filter", default="normal"),
        },
        tags=["email", "filter", "unread"]
    )

    unread_node = tree.add_node(
        parent_id=tree.root.id,
        tool=extract_unread_tool,
        branch_label="right",
        condition=TransitionCondition(
            description="Transition when unread emails are present in the mailbox",
            condition_type="key_equals",
            expression="has_unread:true"
        ),
        system_prompt_template="Focus on unread emails. Filter by min_priority: {min_priority}.",
        user_prompt_template="Extract all unread emails from mailbox {mailbox}.",
        prompt_variables={"min_priority": "normal", "mailbox": "INBOX"}
    )

    # 4. Sub-tree under Right Node: read_email
    read_email_tool = ToolDefinition(
        name="read_email",
        description="Retrieves the full body and metadata of an individual email.",
        parameters={
            "email_id": ToolParameter(name="email_id", type="string", description="Unique email identifier"),
        },
        tags=["email", "read", "content"]
    )

    read_node = tree.add_node(
        parent_id=unread_node.id,
        tool=read_email_tool,
        branch_label="read_email",
        condition=TransitionCondition(
            description="Read specific email content after unread extraction",
            condition_type="always"
        ),
        system_prompt_template="Inspect the email body, headers, and attachments for safety and authenticity.",
        user_prompt_template="Read full email content for email ID: {email_id}.",
        prompt_variables={"email_id": ""}
    )

    # 5. Sub-tree under read_email: extract_vital_information
    vital_info_tool = ToolDefinition(
        name="extract_vital_information",
        description="Extracts deadlines, action items, dates, and contact info from email body.",
        parameters={
            "email_body": ToolParameter(name="email_body", type="string", description="Raw or markdown email body"),
            "categories": ToolParameter(name="categories", type="array", description="List of categories e.g. ['deadline', 'action_item']", default=["deadline", "action_item"]),
        },
        tags=["email", "nlp", "extract", "vital"]
    )

    tree.add_node(
        parent_id=read_node.id,
        tool=vital_info_tool,
        branch_label="extract_vital_info",
        condition=TransitionCondition(
            description="Extract vital information when email body has text",
            condition_type="always"
        ),
        system_prompt_template="You are a data extraction specialist. Identify actionable commitments, dates, and contacts.",
        user_prompt_template="Extract vital info from email: '{email_subject}'.",
        prompt_variables={"email_subject": ""}
    )

    return tree


def run_demonstration():
    print("=" * 70)
    print("🚀 DENDRON: Email Agent Demonstration")
    print("=" * 70)

    # 1. Initialize tree
    tree = create_email_tool_tree()
    print(f"\nCreated tree: {tree.name} (Total nodes: {len(tree.search_bfs())})")
    print(f"\nRoot Discovery Instructions:\n{tree.discovery_instructions}")

    # 2. Simulate agent execution at root
    root_node = tree.root
    print(f"\n[Step 1] Executing Root Tool: {root_node.tool.name}")
    root_result = ToolResult(
        tool_name=root_node.tool.name,
        input_args={"mailbox": "INBOX", "max_count": 10},
        output_data={"total_emails": 10, "has_unread": "true", "unread_count": 3},
        status="success"
    )
    root_node.record_execution(root_result)

    # 3. Predict next tool based on previous output
    next_node = tree.suggest_next_tool(root_node.id, root_result.output_data)
    print(f"[Step 2] Agent evaluated output: Next recommended tool is '{next_node.tool.name}' (Branch: {next_node.branch_label})")

    # 4. Retrieve prompt variables for the next tool
    next_node.set_prompt_variable("mailbox", "INBOX")
    rendered_sys = next_node.get_system_prompt(min_priority="high")
    rendered_user = next_node.get_user_prompt(mailbox="INBOX/Work")
    print(f"  -> Injected System Prompt: {rendered_sys}")
    print(f"  -> Injected User Prompt:   {rendered_user}")

    # 5. Agent Experience: Dynamically adding a new tool on the fly!
    print("\n[Step 3] Dynamic Tree Building (Agent Experience):")
    read_node = tree.find_by_name("read_email")
    print(f"Agent reads an email and learns that it should archive promotional messages.")
    
    archive_tool = ToolDefinition(
        name="archive_email",
        description="Moves the processed email into the archive folder.",
        parameters={
            "email_id": ToolParameter(name="email_id", type="string", description="ID of email to archive"),
        },
        tags=["email", "archive", "cleanup"]
    )

    new_node = tree.record_agent_experience(
        parent_id=read_node.id,
        next_tool=archive_tool,
        trigger_condition_description="Archive when email is marked as processed or promotional",
        condition_type="output_contains",
        condition_expression="archive_ready",
        branch_label="archive",
        system_prompt="Archive email immediately after processing.",
        user_prompt="Archive email ID: {email_id}",
        experience_note="Learned that emails from newsletters should be archived automatically after vital extraction."
    )
    print(f"  -> Successfully attached '{new_node.tool.name}' as child of '{read_node.tool.name}'!")
    print(f"  -> Stored Experience Note: {new_node.experience_notes[0]}")

    # 6. BFS & DFS Search
    print("\n[Step 4] Searching Tools (BFS vs DFS):")
    print("BFS Search for 'extract':")
    for n in tree.search_bfs("extract"):
        print(f"  - [BFS] {n.tool.name} (Branch: {n.branch_label})")

    print("DFS Search for all nodes:")
    for n in tree.search_dfs():
        print(f"  - [DFS] {n.tool.name} (Branch: {n.branch_label})")

    # 7. Frequently Accessed Tools (MFU)
    # Simulate multiple accesses to respond_to_email and extract_unread_emails
    for _ in range(5):
        tree.find_by_name("respond_to_email")
    for _ in range(8):
        tree.find_by_name("extract_unread_emails")

    print("\n[Step 5] Most Frequently Accessed Tools (MFU Cache):")
    mfu_tools = tree.get_frequently_accessed_tools(limit=3)
    for rank, node in enumerate(mfu_tools, 1):
        print(f"  #{rank}: {node.tool.name} (Accessed {node.access_count} times)")

    # 8. RAG Semantic Tool Retrieval
    print("\n[Step 6] RAG Semantic Tool Retrieval:")
    query_1 = "I need to send a quick reply to an important client"
    print(f"Query: '{query_1}'")
    matches = tree.retrieve_tools(query_1, top_k=2)
    for m in matches:
        print(f"  -> Match: {m.tool_name} (Score: {m.score:.2f}, Reasons: {', '.join(m.match_reasons)})")

    query_2 = "parse calendar meetings and upcoming deadlines from message"
    best_tool = tree.retrieve_best_tool(query_2)
    print(f"Query: '{query_2}'")
    print(f"  -> Best Tool: {best_tool.tool.name if best_tool else 'None'}")

    # 9. MCP Export
    print("\n[Step 7] MCP Server Compatibility:")
    mcp_tools = tree.export_all_mcp_tools()
    print(f"Exported {len(mcp_tools)} tools in standard MCP schema format.")
    print(f"Sample MCP Tool Schema for '{mcp_tools[0]['name']}':\n{mcp_tools[0]}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_demonstration()
