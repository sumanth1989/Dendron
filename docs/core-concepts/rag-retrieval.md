# RAG Tool Retrieval

When an agent needs to discover tools dynamically based on user intent, it can perform semantic and lexical retrieval across the tool execution tree using Dendron's built-in RAG engine.

---

## Zero-Dependency Hybrid Retrieval

Dendron includes a self-contained vector space engine built into the standard library:
- **Sublinear TF-IDF**: Evaluates term importance across tool names, descriptions, tags, parameters, and experience notes.
- **Subword Character 3-Grams**: Generates character n-grams for words $\ge 4$ characters to gracefully handle typos, stems, and fuzzy queries.
- **Exact-Match Bonuses**: Awards score boosts when tool names, parameter names, or tags match the query directly.
- **Cosine Similarity**: Ranks matches with normalized relevance scores between 0.0 and 1.0.

---

## Querying Tools

```python
# Retrieve top 3 tools matching natural language intent
results = tree.retrieve_tools("Where is my package right now?", top_k=3)

for r in results:
    print(f"Tool: {r.tool_name} | Score: {r.score:.3f}")
    print(f"Match reasons: {r.match_reasons}")
```

Or retrieve the single best tool directly:

```python
best_node = tree.retrieve_best_tool("Cancel this subscription")
print("Best Tool:", best_node.tool.name)
```

---

## Pluggable Dense Embeddings

For large-scale enterprise deployments, you can plug in any dense vector embedding provider (OpenAI, Hugging Face, Cohere, Ollama):

```python
def my_openai_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in response.data]

# Re-indexes the entire tree with hybrid scoring (60% dense, 40% lexical)
tree.set_embedding_function(my_openai_embeddings)
```
