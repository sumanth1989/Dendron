# Model Context Protocol (MCP)

Dendron is built with native compatibility for Anthropic's **Model Context Protocol (MCP)** tool specification.

---

## Exporting Dendron to MCP

If you are running an MCP server (e.g. using `mcp` Python SDK), you can export the entire tree or subset into the standard MCP `tools/list` format:

```python
from dendron import MCPAdapter

# Format entire tree as standard MCP tools/list response
mcp_response = MCPAdapter.to_mcp_tools_list(tree)

print(mcp_response)
# {
#   "tools": [
#     {
#       "name": "lookup_order",
#       "description": "Retrieves customer order details.",
#       "inputSchema": { "type": "object", "properties": {...}, "required": [...] }
#     }, ...
#   ]
# }
```

---

## Importing MCP Tools into Dendron

If you have an existing MCP server, you can ingest its `tools/list` output into an executable Dendron tree:

```python
mcp_tools_data = [
    {
        "name": "fetch_logs",
        "description": "Fetches service logs",
        "inputSchema": {"type": "object", "properties": {"service": {"type": "string"}}}
    },
    {
        "name": "restart_pod",
        "description": "Restarts a Kubernetes pod",
        "inputSchema": {"type": "object", "properties": {"pod_name": {"type": "string"}}}
    }
]

# Ingest and create a tree with 'fetch_logs' as root
tree = MCPAdapter.from_mcp_tools_list(
    name="DevOpsMCPTree",
    tools_data=mcp_tools_data,
    root_tool_name="fetch_logs"
)
```
