"""Package init with compatibility patches applied early for all submodules."""

import warnings

import numpy as np

# RLlib < 3.0 still references the deprecated np.bool alias during environment
# checks. Python workers created by Ray import human_aware_rl modules, so we
# register the alias here to keep compatibility with modern NumPy builds.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
    warnings.filterwarnings(
        "ignore",
        message=r"In the future `np.bool` will be defined as the corresponding NumPy scalar",
        category=FutureWarning,
    )
