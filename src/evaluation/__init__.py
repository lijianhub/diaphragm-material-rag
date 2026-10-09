from .dataset import EvalExample, load_dataset
from .evaluator import EvalReport, QuestionResult, evaluate_retrieval, match_evidence
from .metrics import hit_at_k, recall_at_k, reciprocal_rank

__all__ = [
    "EvalExample",
    "EvalReport",
    "QuestionResult",
    "evaluate_retrieval",
    "hit_at_k",
    "load_dataset",
    "match_evidence",
    "recall_at_k",
    "reciprocal_rank",
]
