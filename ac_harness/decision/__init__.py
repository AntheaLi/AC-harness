# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
from .state import build_decision_state, KEY_METRIC_BUCKETS
from .frontier import compute_frontier, FrontierReport
from .disagreement import detect_disagreements
from .report import generate_report

__all__ = [
    "build_decision_state",
    "KEY_METRIC_BUCKETS",
    "compute_frontier",
    "FrontierReport",
    "detect_disagreements",
    "generate_report",
]
