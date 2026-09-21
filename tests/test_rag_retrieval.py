"""
Tests for RAG-based tool retrieval in Dendron.
"""

import unittest
from dendron.models import ToolDefinition, ToolParameter
from dendron.tree import Dendron
from dendron.retriever import DendronRetriever, RAGToolRetriever, RAGSearchResult


class TestRAGToolRetrieval(unittest.TestCase):
    def setUp(self):
        # 1. Root Tool
        self.root_tool = ToolDefinition(
            name="get_all_emails",
            description="Fetches recent emails from the user's inbox",
            parameters={"limit": ToolParameter(name="limit", type="integer", default=20)},
            tags=["email", "inbox", "fetch"]
        )
        self.tree = Dendron(
            name="EmailRAGTree",
            root_tool=self.root_tool,
            discovery_instructions="Start at get_all_emails and navigate."
        )

        # 2. Branch 1: respond_to_email
        self.respond_tool = ToolDefinition(
            name="respond_to_email",
            description="Compose and send a response reply to an email thread",
            parameters={
                "email_id": ToolParameter(name="email_id", type="string", description="ID of email"),
                "body": ToolParameter(name="body", type="string", description="Response text")
            },
            tags=["email", "reply", "compose"]
        )
        self.respond_node = self.tree.add_node(
            self.tree.root.id,
            self.respond_tool,
            branch_label="respond_branch"
        )

        # 3. Branch 2: extract_unread_emails
        self.unread_tool = ToolDefinition(
            name="extract_unread_emails",
            description="Filters and extracts all unread emails from the inbox",
            tags=["email", "filter", "unread"]
        )
        self.unread_node = self.tree.add_node(
            self.tree.root.id,
            self.unread_tool,
            branch_label="unread_branch"
        )

        # 4. Sub-branch: extract_vital_information
        self.vital_tool = ToolDefinition(
            name="extract_vital_information",
            description="Parses deadlines, calendar meeting invites, and key action items from email body",
            parameters={
                "body": ToolParameter(name="body", type="string", description="Raw email content")
            },
            tags=["nlp", "calendar", "deadlines", "extract"]
        )
        self.vital_node = self.tree.add_node(
            self.unread_node.id,
            self.vital_tool,
            branch_label="vital_branch"
        )

    def test_retrieve_by_semantic_intent(self):
        # Query: "I need to send a quick reply to someone"
        results = self.tree.retrieve_tools("I need to send a quick reply to someone", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].tool_name, "respond_to_email")
        self.assertGreater(results[0].score, 0.0)

    def test_retrieve_by_task_description(self):
        # Query: "find meeting schedules and deadlines from text"
        best_tool = self.tree.retrieve_best_tool("find meeting schedules and deadlines from text")
        self.assertIsNotNone(best_tool)
        self.assertEqual(best_tool.tool.name, "extract_vital_information")

    def test_retrieve_by_parameter_name(self):
        # Query mentioning parameter "email_id"
        results = self.tree.retrieve_tools("requires email_id to proceed")
        names = [r.tool_name for r in results]
        self.assertIn("respond_to_email", names)

    def test_dynamic_learning_indexed_immediately(self):
        # Dynamically record agent experience
        archive_tool = ToolDefinition(
            name="archive_email",
            description="Archive or delete spam and unwanted newsletters",
            tags=["cleanup", "spam", "archive"]
        )
        self.tree.record_agent_experience(
            parent_id=self.unread_node.id,
            next_tool=archive_tool,
            trigger_condition_description="Email is spam or promotional",
            experience_note="Learned to cleanup newsletters and spam"
        )

        # RAG should immediately find the newly learned tool!
        results = self.tree.retrieve_tools("clean up spam and newsletter messages", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_name, "archive_email")

    def test_access_count_increments_on_rag_retrieval(self):
        initial_access = self.respond_node.access_count
        self.tree.retrieve_tools("send an email response")
        self.assertGreater(self.respond_node.access_count, initial_access)

    def test_custom_dense_embedding_function(self):
        # Mock embedding function returning simple deterministic vectors
        def mock_embed(texts):
            vectors = []
            for t in texts:
                t_low = t.lower()
                # 3-dim vector: [has_reply, has_unread, has_extract]
                v = [
                    1.0 if "reply" in t_low or "respond" in t_low else 0.0,
                    1.0 if "unread" in t_low else 0.0,
                    1.0 if "extract" in t_low or "deadline" in t_low else 0.0,
                ]
                vectors.append(v)
            return vectors

        self.tree.set_embedding_function(mock_embed)
        results = self.tree.retrieve_tools("reply to client", top_k=1)
        self.assertEqual(results[0].tool_name, "respond_to_email")

    def test_retriever_alias(self):
        self.assertIs(RAGToolRetriever, DendronRetriever)


if __name__ == "__main__":
    unittest.main()
