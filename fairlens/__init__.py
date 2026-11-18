"""fairlens: bias and fairness auditing for binary decision systems."""

__version__ = "0.1.0"

from .audit import Audit  # noqa: E402
from .datasets import load_compas, make_lending_data  # noqa: E402
from .disparity import choose_reference_groups, disparities, fairness  # noqa: E402
from .metrics import binarize, confusion_counts, group_metrics  # noqa: E402
from .mitigation import apply_thresholds, group_thresholds, reweighing  # noqa: E402

__all__ = ["Audit", "load_compas", "make_lending_data", "group_metrics", "disparities", "fairness",
           "choose_reference_groups", "binarize", "confusion_counts", "reweighing", "group_thresholds",
           "apply_thresholds"]
