"""HPO search space for community detection algorithms.

Defines hyperparameter search spaces per algorithm and utilities to generate
candidate AlgoConfig objects for budget-aware racing (e.g. Successive Halving).

Each algorithm exposes a grid of values for its key parameters. The cartesian
product of these grids forms the candidate set that the racer explores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List

# Hyperparameter search spaces per algorithm.
# Values are ordered from conservative (cheap/safe) to exploratory.
SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "louvain": {
        "gamma": [0.5, 1.0, 1.5, 2.0],
        "omega": [0.1, 0.5, 1.0, 2.0],
    },
    "leiden": {
        "gamma": [0.5, 1.0, 1.5, 2.0],
        "omega": [0.1, 0.5, 1.0, 2.0],
        "n_iterations": [2, 5],
    },
    "label_propagation": {
        "max_iter": [50, 100, 200],
    },
    "sbm": {
        "K_range": [[2, 3, 4, 5], [2, 3, 4, 5, 6, 7, 8]],
    },
    "dc_sbm": {
        "K_range": [[2, 3, 4, 5], [2, 3, 4, 5, 6, 7, 8]],
    },
    "infomap": {},
}


@dataclass
class AlgoConfig:
    """A single (algorithm, hyperparameters) pair — the atomic unit raced by SHA.

    Attributes:
        algo_name: Base algorithm name (e.g. "louvain")
        hyperparams: Hyperparameter values (e.g. {"gamma": 1.5, "omega": 0.5})
    """

    algo_name: str
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    @property
    def algo_id(self) -> str:
        """Unique string ID encoding both algorithm and hyperparameters."""
        if not self.hyperparams:
            return f"{self.algo_name}:default"
        parts = ",".join(f"{k}={v}" for k, v in sorted(self.hyperparams.items()))
        return f"{self.algo_name}:{parts}"

    def __repr__(self) -> str:
        return f"AlgoConfig({self.algo_id})"

    def __hash__(self) -> int:
        return hash(self.algo_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgoConfig):
            return False
        return self.algo_id == other.algo_id


def generate_configs(
    algo_name: str,
    max_configs: int = 8,
    fast: bool = True,
) -> List[AlgoConfig]:
    """Generate candidate configurations for a single algorithm.

    Always includes a default (empty hyperparams) config first, then
    expands the grid up to max_configs total.

    Args:
        algo_name: Algorithm name (must be a key in SEARCH_SPACES or unknown)
        max_configs: Maximum number of configs to return
        fast: If True, use every-other-value subgrid (halves the grid)

    Returns:
        List of AlgoConfig objects

    Examples:
        >>> configs = generate_configs("leiden", max_configs=4, fast=True)
        >>> [c.algo_id for c in configs]
        ['leiden:gamma=0.5,n_iterations=2,omega=0.1', ...]
    """
    space = SEARCH_SPACES.get(algo_name, {})

    if not space:
        return [AlgoConfig(algo_name=algo_name)]

    if fast:
        space = {
            param: (values[::2] or values[:1])
            for param, values in space.items()
        }

    param_names = list(space.keys())
    param_values = list(space.values())

    configs: List[AlgoConfig] = []
    for combo in product(*param_values):
        hyperparams = dict(zip(param_names, combo))
        configs.append(AlgoConfig(algo_name=algo_name, hyperparams=hyperparams))
        if len(configs) >= max_configs:
            break

    return configs


def generate_all_configs(
    algo_names: List[str],
    max_configs_per_algo: int = 4,
    fast: bool = True,
) -> List[AlgoConfig]:
    """Generate configs for all specified algorithms.

    Args:
        algo_names: Algorithm names to expand
        max_configs_per_algo: Max configs per algorithm
        fast: If True, use reduced grid

    Returns:
        Flat list of AlgoConfig across all algorithms

    Examples:
        >>> configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=2)
        >>> len(configs) <= 4
        True
    """
    all_configs: List[AlgoConfig] = []
    for algo_name in algo_names:
        configs = generate_configs(
            algo_name, max_configs=max_configs_per_algo, fast=fast
        )
        all_configs.extend(configs)
    return all_configs