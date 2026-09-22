"""
Comprehensive unit tests for dendron.retriever.DendronRetriever.
Tests lexical tokenization, TF-IDF vectorization, cosine similarity,
indexing, retrieval bonuses, and dense vector blending.
"""

import unittest
from dendron.models import ToolDefinition, ToolParameter
from dendron.node import DendronNode
from dendron.retriever import DendronRetriever, RAGSearchResult


class TestRetrieverComprehensive(unittest.TestCase):

    def setUp(self):
        self.retriever = DendronRetriever()

    def test_tokenize(self):
        text = "Hello, world! This is a test."
        tokens = DendronRetriever._tokenize(text)
        # Stopwords 'this', 'is', 'a' should be removed
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)

        # 3-grams for words >= 4 chars
        self.assertIn("ngram:hel", tokens)
        self.assertIn("ngram:wor", tokens)

    def test_cosine_similarity(self):
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 0.0]
        self.assertAlmostEqual(DendronRetriever._cosine_similarity(vec_a, vec_b), 1.0)

        vec_c = [0.0, 1.0]
        self.assertAlmostEqual(DendronRetriever._cosine_similarity(vec_a, vec_c), 0.0)

        # Zero vector handling
        vec_zero = [0.0, 0.0]
        self.assertEqual(DendronRetriever._cosine_similarity(vec_a, vec_zero), 0.0)

    def test_indexing_and_removal(self):
        tool = ToolDefinition(name="billing_info", description="Fetches customer billing invoices")
        node = DendronNode(tool=tool)

        self.retriever.index_node(node)
        self.assertEqual(self.retriever._total_docs, 1)
        self.assertIn(node.id, self.retriever._nodes)

        # Retrieve should find it
        results = self.retriever.retrieve(query="billing invoices")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_name, "billing_info")

        # Remove node
        self.retriever.remove_node(node.id)
        self.assertEqual(self.retriever._total_docs, 0)
        self.assertNotIn(node.id, self.retriever._nodes)

        results_after = self.retriever.retrieve(query="billing invoices")
        self.assertEqual(len(results_after), 0)

    def test_retrieval_bonuses(self):
        tool = ToolDefinition(
            name="send_slack_message",
            description="Sends message to channel",
            parameters={"channel": ToolParameter(name="channel", type="string")},
            tags=["chat", "slack"]
        )
        node = DendronNode(tool=tool)
        node.add_experience_note("Preferred tool for urgent team alerts")
        self.retriever.index_node(node)

        # Direct name match
        res_name = self.retriever.retrieve("send_slack_message")
        self.assertGreater(res_name[0].score, 0.25)
        self.assertTrue(any("Tool name" in reason for reason in res_name[0].match_reasons))

        # Tag match
        res_tag = self.retriever.retrieve("chat notification")
        self.assertTrue(any("Matched tag 'chat'" in reason for reason in res_tag[0].match_reasons))

        # Experience note match
        res_exp = self.retriever.retrieve("urgent team alerts")
        self.assertTrue(any("Matched learned experience note" in reason for reason in res_exp[0].match_reasons))

    def test_dense_embedding_integration(self):
        # Mock embedding function returning 2D vector
        def mock_embedding_fn(texts):
            embeddings = []
            for t in texts:
                if "billing" in t.lower():
                    embeddings.append([1.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0])
            return embeddings

        retriever = DendronRetriever(embedding_fn=mock_embedding_fn)
        tool = ToolDefinition(name="billing_service", description="Handles invoices")
        node = DendronNode(tool=tool)
        retriever.index_node(node)

        results = retriever.retrieve("billing query")
        self.assertEqual(len(results), 1)
        self.assertTrue(any("Dense semantic similarity" in reason for reason in results[0].match_reasons))

    def test_retrieve_best(self):
        tool1 = ToolDefinition(name="read_file", description="Reads text file")
        tool2 = ToolDefinition(name="write_file", description="Writes text file")
        self.retriever.index_node(DendronNode(tool=tool1))
        self.retriever.index_node(DendronNode(tool=tool2))

        best = self.retriever.retrieve_best("I want to read a file from disk")
        self.assertIsNotNone(best)
        self.assertEqual(best.tool.name, "read_file")


if __name__ == "__main__":
    unittest.main()
