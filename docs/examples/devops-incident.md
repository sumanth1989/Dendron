# DevOps & SRE Incident Agent Example

This example demonstrates an autonomous Site Reliability Engineering (SRE) agent triaging production incidents, analyzing metrics, and executing rollback or restarts.

See the complete runnable script at [`dendron/examples/devops_incident_agent.py`](file:///Users/sumanthmallya/Desktop/dendron/dendron/examples/devops_incident_agent.py).

## Workflow

```mermaid
graph TD
    A["fetch_incident_alert"] -->|alert_type: 'cpu_spike'| B["query_metrics"]
    A -->|alert_type: 'crashloop'| C["inspect_pod_logs"]
    B -->|cpu > 90%| D["scale_deployment"]
    C -->|OOMKilled| E["restart_service"]
```

## Running the Demo

```bash
python3 dendron/examples/devops_incident_agent.py
```
