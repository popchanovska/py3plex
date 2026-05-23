"""Meta-learner warm-start for community detection algorithm selection.

Maps graph regime features (computed by _compute_graph_regime) to a prior
ranking over AlgoConfig candidates, enabling warm-start exploration in
Successive Halving racing instead of a cold random ordering.

Scoring heuristics are grounded in known algorithm–network relationships:

  High degree heterogeneity (power-law)
      → generative block models (SBM/DC-SBM) fit better than modularity methods

  Strong interlayer coupling
      → high-omega configurations for Louvain/Leiden are preferred

  Low density (sparse)
      → modularity methods (Louvain/Leiden) handle sparsity well

  High density
      → SBM/DC-SBM are more appropriate (modularity resolution limit)

  High layer-density variance
      → non-default gamma (resolution != 1.0) helps differentiate layers

No training data is required: the scores are deterministic rule-based
functions of the observed regime features. This gives a meaningful starting
order without the cold-start overhead of random exploration.
"""

from __future__ import annotations

from typing import Dict, List

from py3plex.algorithms.community_detection.hpo import AlgoConfig


def warmstart_scores(
    configs: List[AlgoConfig],
    regime: Dict[str, float],
) -> Dict[str, float]:
    """Compute warm-start prior scores for a list of algorithm configs.

    Args:
        configs: Candidate algorithm configurations
        regime: Graph regime features from _compute_graph_regime()
                Expected keys: degree_heterogeneity, coupling_strength,
                mean_density, layer_density_variance, mean_degree

    Returns:
        Dict mapping algo_id -> prior score (higher = more promising)
    """
    return {cfg.algo_id: _score_config(cfg, regime) for cfg in configs}


def rank_by_warmstart(
    configs: List[AlgoConfig],
    regime: Dict[str, float],
) -> List[AlgoConfig]:
    """Return configs sorted by warm-start prior score (best first).

    Falls back to the original order when regime is empty (no features
    available), preserving whatever ordering was passed in.

    Args:
        configs: Candidate algorithm configurations
        regime: Graph regime features from _compute_graph_regime()

    Returns:
        Configs sorted descending by prior score
    """
    if not regime:
        return list(configs)

    scores = warmstart_scores(configs, regime)
    return sorted(configs, key=lambda c: scores.get(c.algo_id, 0.0), reverse=True)


def _score_config(config: AlgoConfig, regime: Dict[str, float]) -> float:
    """Score a single AlgoConfig given the graph regime.

    Score is the sum of an algorithm-level term and a hyperparameter-level
    term. Both are heuristic and additive, so the ranking is robust to
    any individual component being slightly off.

    Args:
        config: Algorithm configuration to score
        regime: Graph regime features

    Returns:
        Scalar prior score (higher = more promising)
    """
    degree_het = regime.get("degree_heterogeneity", 1.0)
    coupling = regime.get("coupling_strength", 0.0)
    mean_density = regime.get("mean_density", 0.1)
    layer_density_var = regime.get("layer_density_variance", 0.0)

    algo = config.algo_name
    hp = config.hyperparams
    score = 0.0

    # ── Algorithm-level priors ──────────────────────────────────────────────

    # SBM / DC-SBM fit power-law (high heterogeneity) networks better.
    # Score grows with heterogeneity up to a cap at 1.5× the base term.
    if algo in ("sbm", "dc_sbm"):
        score += 1.5 * min(degree_het / 2.0, 1.5)

    # DC-SBM has an additional edge for very heavy-tailed degree distributions
    # because it explicitly models degree variation inside blocks.
    if algo == "dc_sbm" and degree_het > 2.0:
        score += 0.5

    # Louvain / Leiden are strong for moderate heterogeneity and sparse nets.
    if algo in ("louvain", "leiden") and 0.5 <= degree_het <= 2.0:
        score += 1.0

    # Sparse networks favour modularity methods (null model is well-calibrated).
    if algo in ("louvain", "leiden") and mean_density < 0.05:
        score += 0.5

    # Dense networks favour block models (modularity resolution limit hurts here).
    if algo in ("sbm", "dc_sbm") and mean_density > 0.2:
        score += 0.5

    # ── Hyperparameter-level priors ─────────────────────────────────────────

    # omega should match the observed coupling strength.
    # Penalise configs where omega is far from 2× coupling (empirical rule).
    if "omega" in hp:
        ideal_omega = max(0.1, coupling * 2.0)
        score -= 0.3 * abs(hp["omega"] - ideal_omega)

    # gamma ~ 1 + mean_density gives finer resolution in denser networks.
    if "gamma" in hp:
        ideal_gamma = 1.0 + mean_density
        score -= 0.2 * abs(hp["gamma"] - ideal_gamma)

    # More Leiden iterations help converge in strongly-coupled networks.
    if "n_iterations" in hp and coupling > 0.3:
        score += 0.2 * (hp["n_iterations"] / 5.0)

    # Non-default gamma adds value when layer densities are heterogeneous
    # (different layers need different resolution).
    if layer_density_var > 0.1 and "gamma" in hp and hp["gamma"] != 1.0:
        score += 0.3

    return score