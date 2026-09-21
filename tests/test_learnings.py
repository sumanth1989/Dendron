import unittest
from dendron import Dendron, ToolDefinition, ToolParameter, TransitionCondition
from dendron.models import CompositeToolDefinition


class TestSnapTabLearnings(unittest.TestCase):
    """
    Tests enhancements learned from SnapTab production desktop agent:
    - Destructive action flags & security levels
    - Placeholder validation in tool arguments
    - Composite/macro tool definitions
    - Negative experience learning & suppression
    """

    def test_destructive_flags_and_security_levels(self):
        safe_tool = ToolDefinition(
            name="read_file",
            description="Reads file contents",
            is_destructive=False,
            security_level="safe"
        )
        self.assertFalse(safe_tool.is_destructive)
        self.assertEqual(safe_tool.security_level, "safe")

        destructive_tool = ToolDefinition(
            name="delete_database",
            description="Drops database",
            is_destructive=True,
            security_level="destructive",
            requires_confirmation=True
        )
        self.assertTrue(destructive_tool.is_destructive)
        self.assertEqual(destructive_tool.security_level, "destructive")
        self.assertTrue(destructive_tool.requires_confirmation)

    def test_argument_placeholder_validation(self):
        email_tool = ToolDefinition(
            name="send_email",
            description="Sends email",
            parameters={
                "recipient": ToolParameter(name="recipient", type="string", required=True),
                "body": ToolParameter(name="body", type="string", required=True),
            }
        )
        # Valid arguments
        valid, placeholders = email_tool.validate_arguments({
            "recipient": "alice@example.com",
            "body": "Hi Alice, let's meet tomorrow at 10am."
        })
        self.assertTrue(valid)
        self.assertEqual(len(placeholders), 0)

        # Arguments with template placeholders
        invalid, placeholders = email_tool.validate_arguments({
            "recipient": "alice@example.com",
            "body": "Hi Alice, let's meet on [Insert Date] to discuss [TODO]."
        })
        self.assertFalse(invalid)
        self.assertEqual(len(placeholders), 1)
        self.assertIn("[Insert Date]", placeholders[0])

    def test_composite_tool_definition(self):
        step1 = ToolDefinition(name="git_add", description="Stage files")
        step2 = ToolDefinition(name="git_commit", description="Commit changes")
        step3 = ToolDefinition(name="git_push", description="Push to remote")

        deploy_macro = CompositeToolDefinition(
            name="git_ship",
            description="Stage, commit, and push in one pipeline",
            sub_tools=[step1, step2, step3],
            execution_mode="sequential"
        )
        self.assertEqual(len(deploy_macro.sub_tools), 3)
        self.assertEqual(deploy_macro.sub_tools[0].name, "git_add")
        self.assertEqual(deploy_macro.execution_mode, "sequential")

    def test_negative_experience_suppression(self):
        root = ToolDefinition(name="read_article", description="Read article")
        tree = Dendron(name="ArticleTree", root_tool=root)

        # Add child tools
        share_node = tree.add_node(
            parent_id=tree.root.id,
            tool=ToolDefinition(name="share_on_twitter", description="Tweet link"),
            branch_label="share_branch",
            condition=TransitionCondition(description="User wants to share", expression="share")
        )
        bookmark_node = tree.add_node(
            parent_id=tree.root.id,
            tool=ToolDefinition(name="save_bookmark", description="Bookmark article"),
            branch_label="bookmark_branch",
            condition=TransitionCondition(description="User wants to save", expression="save")
        )

        # Initially, "share" output suggests share_on_twitter
        suggested = tree.suggest_next_tool(tree.root.id, previous_output="User says: share this")
        self.assertIsNotNone(suggested)
        self.assertEqual(suggested.tool.name, "share_on_twitter")

        # Record 3 negative dismissals for share_on_twitter
        tree.record_negative_experience("share_on_twitter", reason="User repeatedly dismissed Twitter sharing")
        tree.record_negative_experience("share_on_twitter", reason="User repeatedly dismissed Twitter sharing")
        tree.record_negative_experience("share_on_twitter", reason="User repeatedly dismissed Twitter sharing")

        # Now, share_on_twitter is suppressed!
        suggested_after_suppression = tree.suggest_next_tool(tree.root.id, previous_output="User says: share this")
        self.assertIsNone(suggested_after_suppression)


if __name__ == "__main__":
    unittest.main()
