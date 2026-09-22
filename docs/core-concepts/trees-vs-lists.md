# Trees vs. Flat Lists

Standard LLM tool calling forces agents to operate on flat arrays of tool schemas. This page examines why this architecture degrades as agent capability grows, and how Dendron's tree structure solves it.

---

## The Flat List Approach

In standard tool use (OpenAI, Anthropic, LangChain):

```text
[ LLM Turn 1 ] ---> Sees: [Tool1, Tool2, Tool3, ..., Tool50] (1,500+ tokens)
[ LLM Turn 2 ] ---> Sees: [Tool1, Tool2, Tool3, ..., Tool50] (1,500+ tokens)
[ LLM Turn 3 ] ---> Sees: [Tool1, Tool2, Tool3, ..., Tool50] (1,500+ tokens)
```

### Critical Drawbacks:
1. **Context Bloat**: Feeding 50 JSON schemas costs 1,500–3,000 tokens on *every* turn. For a 10-turn conversation, you spend 15,000–30,000 tokens just repeating tool descriptions.
2. **Order Ambiguity**: The model has no structural knowledge that `deploy_code` requires `run_tests` first, or that `refund_payment` requires `verify_purchase`.
3. **Hallucination Risk**: When 50 tools are visible, LLMs frequently pick tools that sound plausible but are inappropriate for the current state.
4. **Edge / Local LLM Bottlenecks**: On smaller models (e.g. 2B–8B parameter models running on Apple Silicon via MLX), 1,500 prompt tokens introduces a 1–2 second prefill latency penalty on every turn!

---

## The Dendron Tree Approach

Dendron organizes tools into a navigable execution graph:

```text
               [ Root Tool: lookup_order ]
                     /                \
        (status: "in_transit")    (status: "delivered")
                   /                    \
       [ track_shipment ]         [ process_return ]
                                        |
                              (condition: "damaged")
                                        |
                             [ issue_instant_refund ]
```

### Advantages:
1. **Scattered Context Eliminated**: On any turn, the LLM only sees the immediate next viable actions (typically 1–3 tools), dropping prompt overhead by up to **93%**.
2. **Deterministic & Guided Transitions**: High-confidence paths are evaluated automatically via `TransitionCondition`.
3. **On-Demand Expansion**: If the LLM needs to jump elsewhere, it can query the tree via RAG semantic search or fast parameter/tag filters.
4. **Dynamic Evolution**: The agent learns from real-world turns and records new branches at runtime.
