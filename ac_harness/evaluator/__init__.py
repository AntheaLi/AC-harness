# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
from . import copy_task, lm_eval_adapter, mqar, needle, perplexity

EVALUATORS = {
    "perplexity": perplexity,
    "needle": needle,
    "mqar": mqar,
    "copy_task": copy_task,
    "lm_eval": lm_eval_adapter,
}

__all__ = ["EVALUATORS", "perplexity", "needle", "mqar", "copy_task", "lm_eval_adapter"]
