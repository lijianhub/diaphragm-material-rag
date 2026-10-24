# Experiment 10: Agentic RAG with Human Approval

`python experiments/10_agentic_rag/run.py` runs the agent offline with a scripted planner in place of the model. It shows the approval flow: the agent proposes a nonconformance report, nothing is filed until a person approves, and a second approval is refused.

For the real model, the tools, the guardrails and the measurements, see [docs/13-agentic-rag.md](../../docs/13-agentic-rag.md) and `scripts/agent.py`.
