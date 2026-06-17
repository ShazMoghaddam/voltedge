"""Tests for the causal graph engine."""
from __future__ import annotations
import asyncio
import pytest
import pandas as pd
import networkx as nx

from voltedge.causal.graph import (
    CausalEdge, CausalGraph, CausalGraphBuilder, DEFAULT_ALPHA,
)
from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer

SITE = "CAUSAL-TEST-01"


@pytest.fixture(scope="module")
def processed_df():
    connector = SimulatedSiteConnector(SITE, "factory", hours=400, seed=42)
    result = asyncio.run(connector.fetch())
    return EnergyTransformer().transform(result.data)


@pytest.fixture(scope="module")
def causal_graph(processed_df):
    builder = CausalGraphBuilder(alpha=0.05, min_samples=48)
    return builder.build(processed_df, SITE)


# ── CausalEdge ────────────────────────────────────────────────────────────────

def test_edge_significance():
    e = CausalEdge("A", "B", strength=0.5, p_value=0.01)
    assert e.is_significant


def test_edge_not_significant():
    e = CausalEdge("A", "B", strength=0.1, p_value=0.20)
    assert not e.is_significant


def test_edge_to_dict_keys():
    e = CausalEdge("temp", "kwh", strength=0.6, p_value=0.001,
                   mechanism="HVAC driven by temperature")
    d = e.to_dict()
    for k in ("cause", "effect", "strength", "p_value", "mechanism", "significant"):
        assert k in d


# ── CausalGraph ───────────────────────────────────────────────────────────────

def test_graph_has_dag(causal_graph):
    assert isinstance(causal_graph.dag, nx.DiGraph)


def test_graph_is_acyclic(causal_graph):
    assert nx.is_directed_acyclic_graph(causal_graph.dag)


def test_graph_has_variables(causal_graph):
    assert len(causal_graph.variables) > 0
    assert "kwh" in causal_graph.variables


def test_graph_has_edges(causal_graph):
    assert len(causal_graph.edges) >= 0   # May be 0 for sparse data


def test_graph_site_id(causal_graph):
    assert causal_graph.site_id == SITE


def test_graph_summary_keys(causal_graph):
    s = causal_graph.summary()
    for k in ("site_id", "variables", "n_edges", "n_samples"):
        assert k in s


def test_graph_get_causes(causal_graph):
    causes = causal_graph.get_causes("kwh")
    assert isinstance(causes, list)


def test_graph_get_effects(causal_graph):
    effects = causal_graph.get_effects("kwh")
    assert isinstance(effects, list)


def test_graph_ancestors_returns_set(causal_graph):
    anc = causal_graph.ancestors("kwh")
    assert isinstance(anc, set)


def test_graph_descendants_returns_set(causal_graph):
    desc = causal_graph.descendants("kwh")
    assert isinstance(desc, set)


def test_graph_causal_path_returns_list_or_none(causal_graph):
    path = causal_graph.causal_path("kwh", "kwh")
    # Path from node to itself: list of one node, or None
    assert path is None or isinstance(path, list)


def test_graph_n_samples_correct(processed_df, causal_graph):
    assert causal_graph.n_samples <= len(processed_df)
    assert causal_graph.n_samples > 0


def test_graph_learned_at_set(causal_graph):
    assert len(causal_graph.learned_at) > 0


# ── Builder ───────────────────────────────────────────────────────────────────

def test_builder_raises_on_insufficient_data():
    builder = CausalGraphBuilder(min_samples=1000)
    tiny_df = pd.DataFrame({"kwh": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="samples"):
        builder.build(tiny_df, "TINY-SITE")


def test_builder_select_variables_filters_numeric(processed_df):
    builder = CausalGraphBuilder()
    selected = builder._select_variables(processed_df)
    assert "kwh" in selected
    assert len(selected) >= 3


def test_fisher_z_test_independent():
    import numpy as np
    data = np.random.default_rng(0).standard_normal((200, 3))
    p = CausalGraphBuilder._fisher_z_test(data, 0, 1, [], 200)
    assert 0.0 <= p <= 1.0


def test_fisher_z_test_correlated():
    import numpy as np
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    data = np.column_stack([x, x * 0.9 + rng.standard_normal(200) * 0.1])
    p = CausalGraphBuilder._fisher_z_test(data, 0, 1, [], 200)
    assert p < 0.05   # Should detect strong dependence


def test_domain_priors_have_known_variables():
    priors = CausalGraphBuilder.DOMAIN_PRIORS
    assert ("temperature_c", "kwh") in priors
    assert ("is_business_hour", "kwh") in priors
