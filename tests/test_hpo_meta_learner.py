"""Tests for HPO search space and meta-learner warm-start.

Covers:
- AlgoConfig: creation, algo_id, equality, hashing
- generate_configs: grid expansion, fast mode, max_configs cap
- generate_all_configs: multi-algorithm expansion
- warmstart_scores: score types and range
- rank_by_warmstart: ordering, fallback on empty regime
- Score heuristics: algorithm-level and hyperparameter-level priors
- runner integration: hyperparams forwarded through run_community_algorithm
- successive_halving integration: configs param populates hyperparams map
- autocommunity_executor integration: HPO configs passed to _run_candidate_algorithms
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from py3plex.core import multinet
from py3plex.algorithms.community_detection.hpo import (
    AlgoConfig,
    SEARCH_SPACES,
    generate_configs,
    generate_all_configs,
)
from py3plex.algorithms.community_detection.meta_learner import (
    warmstart_scores,
    rank_by_warmstart,
    _score_config,
)
from py3plex.algorithms.community_detection.budget import BudgetSpec
from py3plex.algorithms.community_detection.runner import run_community_algorithm
from py3plex.algorithms.community_detection.successive_halving import (
    SuccessiveHalvingRacer,
    SuccessiveHalvingConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def two_layer_network():
    """Two-layer network with two clear communities per layer."""
    net = multinet.multi_layer_network(directed=False)
    net.add_edges([
        ["A", "L1", "B", "L1", 1], ["A", "L1", "C", "L1", 1], ["B", "L1", "C", "L1", 1],
        ["D", "L1", "E", "L1", 1], ["D", "L1", "F", "L1", 1], ["E", "L1", "F", "L1", 1],
        ["C", "L1", "D", "L1", 1],
        ["A", "L2", "B", "L2", 1], ["D", "L2", "E", "L2", 1],
        ["B", "L2", "D", "L2", 1],
    ], input_type="list")
    return net


@pytest.fixture
def sparse_regime():
    """Regime features for a sparse, moderate-heterogeneity network."""
    return {
        "degree_heterogeneity": 0.8,
        "coupling_strength": 0.2,
        "mean_density": 0.03,
        "layer_density_variance": 0.05,
        "mean_degree": 3.0,
    }


@pytest.fixture
def dense_het_regime():
    """Regime features for a dense, high-heterogeneity (power-law) network."""
    return {
        "degree_heterogeneity": 3.5,
        "coupling_strength": 0.6,
        "mean_density": 0.35,
        "layer_density_variance": 0.2,
        "mean_degree": 12.0,
    }


# ── AlgoConfig ────────────────────────────────────────────────────────────────

class TestAlgoConfig:
    def test_algo_id_default(self):
        cfg = AlgoConfig(algo_name="louvain")
        assert cfg.algo_id == "louvain:default"

    def test_algo_id_with_hyperparams(self):
        cfg = AlgoConfig(algo_name="leiden", hyperparams={"gamma": 1.5, "omega": 0.5})
        assert "leiden:" in cfg.algo_id
        assert "gamma=1.5" in cfg.algo_id
        assert "omega=0.5" in cfg.algo_id

    def test_algo_id_params_sorted(self):
        """Parameters should appear in sorted order for a stable ID."""
        cfg1 = AlgoConfig(algo_name="leiden", hyperparams={"omega": 0.5, "gamma": 1.5})
        cfg2 = AlgoConfig(algo_name="leiden", hyperparams={"gamma": 1.5, "omega": 0.5})
        assert cfg1.algo_id == cfg2.algo_id

    def test_equality(self):
        cfg1 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        cfg2 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        assert cfg1 == cfg2

    def test_inequality_different_algo(self):
        cfg1 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        cfg2 = AlgoConfig(algo_name="leiden", hyperparams={"gamma": 1.0})
        assert cfg1 != cfg2

    def test_inequality_different_params(self):
        cfg1 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        cfg2 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 2.0})
        assert cfg1 != cfg2

    def test_hashable(self):
        """AlgoConfig must be usable as a dict key / set member."""
        cfg1 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        cfg2 = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0})
        s = {cfg1, cfg2}
        assert len(s) == 1

    def test_repr(self):
        cfg = AlgoConfig(algo_name="louvain")
        assert "louvain" in repr(cfg)


# ── generate_configs ──────────────────────────────────────────────────────────

class TestGenerateConfigs:
    def test_unknown_algorithm_returns_default(self):
        configs = generate_configs("unknown_algo")
        assert len(configs) == 1
        assert configs[0].algo_name == "unknown_algo"
        assert configs[0].hyperparams == {}

    def test_known_algorithm_returns_configs(self):
        configs = generate_configs("louvain", fast=False)
        assert len(configs) > 1
        assert all(c.algo_name == "louvain" for c in configs)

    def test_fast_mode_fewer_configs(self):
        slow = generate_configs("louvain", fast=False)
        fast = generate_configs("louvain", fast=True)
        assert len(fast) <= len(slow)

    def test_max_configs_respected(self):
        configs = generate_configs("louvain", max_configs=2, fast=False)
        assert len(configs) <= 2

    def test_all_configs_have_hyperparams(self):
        configs = generate_configs("leiden", fast=True)
        for cfg in configs:
            assert isinstance(cfg.hyperparams, dict)
            assert len(cfg.hyperparams) > 0

    def test_hyperparams_within_search_space(self):
        space = SEARCH_SPACES["louvain"]
        configs = generate_configs("louvain", fast=False)
        for cfg in configs:
            for key, val in cfg.hyperparams.items():
                assert val in space[key], f"{key}={val} not in search space"

    def test_infomap_returns_default_only(self):
        """infomap has no tunable params → single default config."""
        configs = generate_configs("infomap")
        assert len(configs) == 1
        assert configs[0].hyperparams == {}

    def test_configs_have_unique_ids(self):
        configs = generate_configs("leiden", fast=False)
        ids = [c.algo_id for c in configs]
        assert len(ids) == len(set(ids))


# ── generate_all_configs ──────────────────────────────────────────────────────

class TestGenerateAllConfigs:
    def test_all_algos_represented(self):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=3)
        algo_names = {c.algo_name for c in configs}
        assert "louvain" in algo_names
        assert "leiden" in algo_names

    def test_total_count_bounded(self):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=3)
        louvain_count = sum(1 for c in configs if c.algo_name == "louvain")
        leiden_count = sum(1 for c in configs if c.algo_name == "leiden")
        assert louvain_count <= 3
        assert leiden_count <= 3

    def test_empty_algo_list(self):
        configs = generate_all_configs([])
        assert configs == []

    def test_single_algo(self):
        configs = generate_all_configs(["louvain"], max_configs_per_algo=2)
        assert all(c.algo_name == "louvain" for c in configs)

    def test_returns_list_of_algo_configs(self):
        configs = generate_all_configs(["louvain"])
        assert all(isinstance(c, AlgoConfig) for c in configs)


# ── warmstart_scores ──────────────────────────────────────────────────────────

class TestWarmstartScores:
    def test_returns_dict(self, sparse_regime):
        configs = generate_all_configs(["louvain"], max_configs_per_algo=2)
        scores = warmstart_scores(configs, sparse_regime)
        assert isinstance(scores, dict)

    def test_all_configs_scored(self, sparse_regime):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=2)
        scores = warmstart_scores(configs, sparse_regime)
        assert len(scores) == len(configs)
        for cfg in configs:
            assert cfg.algo_id in scores

    def test_scores_are_finite(self, sparse_regime):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=4)
        scores = warmstart_scores(configs, sparse_regime)
        assert all(np.isfinite(v) for v in scores.values())

    def test_empty_regime_still_scores(self):
        configs = generate_all_configs(["louvain"], max_configs_per_algo=2)
        scores = warmstart_scores(configs, {})
        assert len(scores) == len(configs)


# ── rank_by_warmstart ─────────────────────────────────────────────────────────

class TestRankByWarmstart:
    def test_same_configs_returned(self, sparse_regime):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=3)
        ranked = rank_by_warmstart(configs, sparse_regime)
        assert set(c.algo_id for c in ranked) == set(c.algo_id for c in configs)

    def test_same_length(self, sparse_regime):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=3)
        ranked = rank_by_warmstart(configs, sparse_regime)
        assert len(ranked) == len(configs)

    def test_empty_regime_returns_original_order(self):
        configs = generate_all_configs(["louvain"], max_configs_per_algo=3)
        ranked = rank_by_warmstart(configs, {})
        assert [c.algo_id for c in ranked] == [c.algo_id for c in configs]

    def test_ordering_is_descending(self, sparse_regime):
        """Scores of ranked configs must be non-increasing."""
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=4)
        ranked = rank_by_warmstart(configs, sparse_regime)
        scores = warmstart_scores(configs, sparse_regime)
        vals = [scores[c.algo_id] for c in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_empty_configs(self, sparse_regime):
        ranked = rank_by_warmstart([], sparse_regime)
        assert ranked == []


# ── Score heuristics ──────────────────────────────────────────────────────────

class TestScoreHeuristics:
    """Verify that the heuristic rules push scores in the expected direction."""

    def test_sbm_preferred_for_high_heterogeneity(self):
        high_het = {"degree_heterogeneity": 4.0, "coupling_strength": 0.1,
                    "mean_density": 0.1, "layer_density_variance": 0.0}
        low_het = {"degree_heterogeneity": 0.3, "coupling_strength": 0.1,
                   "mean_density": 0.1, "layer_density_variance": 0.0}
        sbm_cfg = AlgoConfig(algo_name="sbm", hyperparams={})
        score_high = _score_config(sbm_cfg, high_het)
        score_low = _score_config(sbm_cfg, low_het)
        assert score_high > score_low

    def test_louvain_preferred_for_low_heterogeneity(self):
        low_het = {"degree_heterogeneity": 0.6, "coupling_strength": 0.1,
                   "mean_density": 0.05, "layer_density_variance": 0.0}
        high_het = {"degree_heterogeneity": 4.0, "coupling_strength": 0.1,
                    "mean_density": 0.05, "layer_density_variance": 0.0}
        louvain_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0, "omega": 0.2})
        score_low = _score_config(louvain_cfg, low_het)
        score_high = _score_config(louvain_cfg, high_het)
        assert score_low > score_high

    def test_dc_sbm_bonus_for_very_high_heterogeneity(self):
        very_high = {"degree_heterogeneity": 3.0, "coupling_strength": 0.1,
                     "mean_density": 0.1, "layer_density_variance": 0.0}
        moderate = {"degree_heterogeneity": 1.5, "coupling_strength": 0.1,
                    "mean_density": 0.1, "layer_density_variance": 0.0}
        dc_sbm_cfg = AlgoConfig(algo_name="dc_sbm", hyperparams={})
        score_very_high = _score_config(dc_sbm_cfg, very_high)
        score_moderate = _score_config(dc_sbm_cfg, moderate)
        assert score_very_high > score_moderate

    def test_omega_penalty_for_mismatch(self):
        regime = {"degree_heterogeneity": 1.0, "coupling_strength": 0.0,
                  "mean_density": 0.1, "layer_density_variance": 0.0}
        # ideal_omega = max(0.1, 0.0 * 2.0) = 0.1
        close_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0, "omega": 0.1})
        far_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0, "omega": 2.0})
        assert _score_config(close_cfg, regime) > _score_config(far_cfg, regime)

    def test_gamma_penalty_for_mismatch(self):
        regime = {"degree_heterogeneity": 1.0, "coupling_strength": 0.1,
                  "mean_density": 0.5, "layer_density_variance": 0.0}
        # ideal_gamma = 1 + 0.5 = 1.5
        close_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.5, "omega": 0.1})
        far_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 0.5, "omega": 0.1})
        assert _score_config(close_cfg, regime) > _score_config(far_cfg, regime)

    def test_layer_density_variance_bonus(self):
        high_var = {"degree_heterogeneity": 1.0, "coupling_strength": 0.1,
                    "mean_density": 0.1, "layer_density_variance": 0.3}
        low_var = {"degree_heterogeneity": 1.0, "coupling_strength": 0.1,
                   "mean_density": 0.1, "layer_density_variance": 0.0}
        nonde_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.5, "omega": 0.2})
        assert _score_config(nonde_cfg, high_var) > _score_config(nonde_cfg, low_var)

    def test_sparse_network_favours_modularity(self, sparse_regime):
        louvain_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0, "omega": 0.4})
        sbm_cfg = AlgoConfig(algo_name="sbm", hyperparams={})
        # For sparse (density=0.03, het=0.8), louvain should score higher than sbm
        assert _score_config(louvain_cfg, sparse_regime) > _score_config(sbm_cfg, sparse_regime)

    def test_dense_het_network_favours_sbm(self, dense_het_regime):
        sbm_cfg = AlgoConfig(algo_name="sbm", hyperparams={})
        louvain_cfg = AlgoConfig(algo_name="louvain", hyperparams={"gamma": 1.0, "omega": 0.1})
        assert _score_config(sbm_cfg, dense_het_regime) > _score_config(louvain_cfg, dense_het_regime)


# ── runner integration ────────────────────────────────────────────────────────

class TestRunnerHyperparams:
    """Verify hyperparams are forwarded through run_community_algorithm."""

    def test_louvain_accepts_hyperparams(self, two_layer_network):
        budget = BudgetSpec(max_iter=10)
        hp = {"gamma": 1.5, "omega": 0.5}
        result = run_community_algorithm(
            "louvain", two_layer_network, budget, seed=42, hyperparams=hp
        )
        assert result.partition
        assert result.meta.get("gamma") == 1.5
        assert result.meta.get("omega") == 0.5

    def test_leiden_accepts_hyperparams(self, two_layer_network):
        budget = BudgetSpec(max_iter=5)
        hp = {"gamma": 0.5, "omega": 0.1, "n_iterations": 2}
        result = run_community_algorithm(
            "leiden", two_layer_network, budget, seed=42, hyperparams=hp
        )
        assert result.partition
        assert result.meta.get("gamma") == 0.5
        assert result.meta.get("omega") == 0.1

    def test_none_hyperparams_uses_defaults(self, two_layer_network):
        budget = BudgetSpec(max_iter=10)
        result = run_community_algorithm(
            "louvain", two_layer_network, budget, seed=42, hyperparams=None
        )
        assert result.partition
        # defaults: gamma=1.0, omega=1.0
        assert result.meta.get("gamma") == 1.0
        assert result.meta.get("omega") == 1.0

    def test_algo_id_with_colon_still_routes_correctly(self, two_layer_network):
        """algo_id like 'louvain:gamma=1.5,omega=0.5' must route to louvain."""
        budget = BudgetSpec(max_iter=10)
        result = run_community_algorithm(
            "louvain:gamma=1.5,omega=0.5",
            two_layer_network,
            budget,
            seed=42,
            hyperparams={"gamma": 1.5, "omega": 0.5},
        )
        assert result.partition

    def test_different_gamma_changes_partition(self, two_layer_network):
        """Different gamma values should generally produce different community counts."""
        budget = BudgetSpec(max_iter=20)
        result_low = run_community_algorithm(
            "louvain", two_layer_network, budget, seed=42, hyperparams={"gamma": 0.5, "omega": 1.0}
        )
        result_high = run_community_algorithm(
            "louvain", two_layer_network, budget, seed=42, hyperparams={"gamma": 2.0, "omega": 1.0}
        )
        # Higher gamma → finer partition (more communities expected)
        n_low = len(set(result_low.partition.values()))
        n_high = len(set(result_high.partition.values()))
        assert n_high >= n_low


# ── successive_halving integration ────────────────────────────────────────────

class TestSuccessiveHalvingWithConfigs:
    """Verify the SHA racer correctly uses AlgoConfig objects."""

    def test_race_with_configs_completes(self, two_layer_network):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=2)
        config = SuccessiveHalvingConfig(
            eta=2,
            budget0=BudgetSpec(max_iter=5, uq_samples=None),
            rounds=1,
        )
        racer = SuccessiveHalvingRacer(config, seed=42)
        history = racer.race(
            network=two_layer_network,
            algorithm_ids=[],
            metric_names=["modularity"],
            configs=configs,
        )
        assert history.winner_algo_id is not None

    def test_hyperparams_map_populated(self, two_layer_network):
        configs = generate_all_configs(["louvain"], max_configs_per_algo=2)
        config = SuccessiveHalvingConfig(
            eta=2,
            budget0=BudgetSpec(max_iter=5),
            rounds=1,
        )
        racer = SuccessiveHalvingRacer(config, seed=42)
        racer.race(
            network=two_layer_network,
            algorithm_ids=[],
            metric_names=["modularity"],
            configs=configs,
        )
        # After race, _hyperparams_map should be set with config algo_ids
        assert hasattr(racer, "_hyperparams_map")
        for cfg in configs:
            assert cfg.algo_id in racer._hyperparams_map
            assert racer._hyperparams_map[cfg.algo_id] == cfg.hyperparams

    def test_winner_is_valid_algo_id(self, two_layer_network):
        configs = generate_all_configs(["louvain", "leiden"], max_configs_per_algo=2)
        algo_ids = {c.algo_id for c in configs}
        config = SuccessiveHalvingConfig(
            eta=2,
            budget0=BudgetSpec(max_iter=5),
            rounds=1,
        )
        racer = SuccessiveHalvingRacer(config, seed=42)
        history = racer.race(
            network=two_layer_network,
            algorithm_ids=[],
            metric_names=["modularity"],
            configs=configs,
        )
        assert history.winner_algo_id in algo_ids

    def test_fallback_to_algorithm_ids_without_configs(self, two_layer_network):
        """When configs=None, the racer must still work with plain algorithm_ids."""
        config = SuccessiveHalvingConfig(
            eta=2,
            budget0=BudgetSpec(max_iter=5),
            rounds=1,
        )
        racer = SuccessiveHalvingRacer(config, seed=42)
        history = racer.race(
            network=two_layer_network,
            algorithm_ids=["louvain", "leiden"],
            metric_names=["modularity"],
            configs=None,
        )
        assert history.winner_algo_id in ("louvain", "leiden")


# ── autocommunity_executor integration ───────────────────────────────────────

class TestExecutorHPOIntegration:
    """Verify HPO + warm-start are wired into execute_autocommunity."""

    def test_generate_all_configs_called(self, two_layer_network):
        from py3plex.algorithms.community_detection.autocommunity_executor import (
            execute_autocommunity,
        )

        with patch(
            "py3plex.algorithms.community_detection.autocommunity_executor.generate_all_configs",
            wraps=generate_all_configs,
        ) as mock_gen:
            try:
                execute_autocommunity(
                    network=two_layer_network,
                    candidate_algorithms=["louvain", "leiden"],
                    metric_names=["modularity"],
                    uq_config=None,
                    null_config=None,
                    use_pareto=True,
                    seed=42,
                    custom_metrics=[],
                    custom_candidates=[],
                )
            except Exception:
                pass  # Result correctness tested elsewhere; we only check the call
            assert mock_gen.called

    def test_rank_by_warmstart_called(self, two_layer_network):
        from py3plex.algorithms.community_detection.autocommunity_executor import (
            execute_autocommunity,
        )

        with patch(
            "py3plex.algorithms.community_detection.autocommunity_executor.rank_by_warmstart",
            wraps=rank_by_warmstart,
        ) as mock_rank:
            try:
                execute_autocommunity(
                    network=two_layer_network,
                    candidate_algorithms=["louvain", "leiden"],
                    metric_names=["modularity"],
                    uq_config=None,
                    null_config=None,
                    use_pareto=True,
                    seed=42,
                    custom_metrics=[],
                    custom_candidates=[],
                )
            except Exception:
                pass
            assert mock_rank.called

    def test_full_pipeline_returns_result(self, two_layer_network):
        from py3plex.algorithms.community_detection.autocommunity_executor import (
            execute_autocommunity,
        )
        from py3plex.algorithms.community_detection.autocommunity import AutoCommunityResult

        result = execute_autocommunity(
            network=two_layer_network,
            candidate_algorithms=["louvain", "leiden"],
            metric_names=["modularity", "coverage"],
            uq_config=None,
            null_config=None,
            use_pareto=True,
            seed=42,
            custom_metrics=[],
            custom_candidates=[],
        )
        assert isinstance(result, AutoCommunityResult)
        assert result.consensus_partition
        assert result.selected in result.algorithms_tested or result.selected == "consensus"

    def test_partition_covers_all_nodes(self, two_layer_network):
        from py3plex.algorithms.community_detection.autocommunity_executor import (
            execute_autocommunity,
        )

        result = execute_autocommunity(
            network=two_layer_network,
            candidate_algorithms=["louvain"],
            metric_names=["modularity"],
            uq_config=None,
            null_config=None,
            use_pareto=True,
            seed=42,
            custom_metrics=[],
            custom_candidates=[],
        )
        expected_nodes = set(two_layer_network.get_nodes())
        partition_nodes = set(result.consensus_partition.keys())
        assert partition_nodes == expected_nodes

    def test_community_ids_are_integers(self, two_layer_network):
        from py3plex.algorithms.community_detection.autocommunity_executor import (
            execute_autocommunity,
        )

        result = execute_autocommunity(
            network=two_layer_network,
            candidate_algorithms=["louvain"],
            metric_names=["modularity"],
            uq_config=None,
            null_config=None,
            use_pareto=True,
            seed=42,
            custom_metrics=[],
            custom_candidates=[],
        )
        assert all(isinstance(v, int) for v in result.consensus_partition.values())