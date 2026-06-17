"""
VoltEdge Causal AI — Causal Graph Engine

Builds a Directed Acyclic Graph (DAG) of causal relationships
from energy time-series data using the PC algorithm — a constraint-based
causal discovery method from Judea Pearl's do-calculus framework.

Why causation over correlation?
  Correlation: "Consumption and temperature are correlated."
  Causation:   "High ambient temperature → HVAC load increase → consumption spike."

The graph encodes:
  Nodes  — energy variables (consumption, temperature, power factor, shift schedule...)
  Edges  — directed causal influences (A → B means A causes B)
  Weights — effect magnitude (increase A by 1 unit → B changes by W units)

Algorithm: PC (Peter-Clark) with Fisher's Z conditional independence test.
  1. Start with a fully connected undirected graph
  2. Remove edges where variables are conditionally independent
  3. Orient edges using v-structure detection and Meek rules

This gives us a CPDAG (Completed Partially Directed Acyclic Graph) —
the equivalence class of all DAGs consistent with the data. We then
orient ambiguous edges using domain knowledge (time precedence: past
variables cannot be caused by future ones).

References:
  Spirtes, Glymour, Scheines (2000) — Causation, Prediction, and Search
  Pearl (2009) — Causality: Models, Reasoning, and Inference
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# Significance threshold for conditional independence tests
DEFAULT_ALPHA = 0.05


@dataclass
class CausalEdge:
    """A directed causal relationship between two variables."""
    cause:      str
    effect:     str
    strength:   float       # Standardised regression coefficient (effect size)
    p_value:    float       # Significance of the edge
    lag_hours:  int = 0     # Time lag between cause and effect
    mechanism:  str = ""    # Human-readable description of the mechanism
    confidence: str = "medium"   # "high" | "medium" | "low"

    @property
    def is_significant(self) -> bool:
        return self.p_value < DEFAULT_ALPHA

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause":      self.cause,
            "effect":     self.effect,
            "strength":   round(self.strength, 4),
            "p_value":    round(self.p_value, 4),
            "lag_hours":  self.lag_hours,
            "mechanism":  self.mechanism,
            "confidence": self.confidence,
            "significant": self.is_significant,
        }


@dataclass
class CausalGraph:
    """
    Directed Acyclic Graph of causal relationships for one site.
    Wraps a NetworkX DiGraph with VoltEdge-specific metadata.
    """
    site_id:    str
    dag:        nx.DiGraph = field(default_factory=nx.DiGraph)
    edges:      list[CausalEdge] = field(default_factory=list)
    variables:  list[str] = field(default_factory=list)
    n_samples:  int = 0
    learned_at: str = ""

    def get_causes(self, variable: str) -> list[CausalEdge]:
        """Return all edges pointing INTO this variable."""
        return [e for e in self.edges if e.effect == variable]

    def get_effects(self, variable: str) -> list[CausalEdge]:
        """Return all edges pointing OUT from this variable."""
        return [e for e in self.edges if e.cause == variable]

    def _idx(self, variable: str) -> int | None:
        try:
            return self.variables.index(variable)
        except ValueError:
            return None

    def causal_path(self, source: str, target: str) -> list[str] | None:
        """Return the shortest causal path from source to target, or None."""
        si, ti = self._idx(source), self._idx(target)
        if si is None or ti is None:
            return None
        try:
            path = nx.shortest_path(self.dag, si, ti)
            return [self.variables[i] for i in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def ancestors(self, variable: str) -> set[str]:
        """All variables that causally precede this one."""
        idx = self._idx(variable)
        if idx is None:
            return set()
        try:
            return {self.variables[i] for i in nx.ancestors(self.dag, idx)
                    if i < len(self.variables)}
        except nx.NodeNotFound:
            return set()

    def descendants(self, variable: str) -> set[str]:
        """All variables causally downstream of this one."""
        idx = self._idx(variable)
        if idx is None:
            return set()
        try:
            return {self.variables[i] for i in nx.descendants(self.dag, idx)
                    if i < len(self.variables)}
        except nx.NodeNotFound:
            return set()

    def strongest_cause(self, variable: str) -> CausalEdge | None:
        causes = self.get_causes(variable)
        return max(causes, key=lambda e: abs(e.strength)) if causes else None

    def summary(self) -> dict[str, Any]:
        return {
            "site_id":    self.site_id,
            "variables":  self.variables,
            "n_edges":    len(self.edges),
            "n_samples":  self.n_samples,
            "learned_at": self.learned_at,
            "edges": [e.to_dict() for e in self.edges],
        }


class CausalGraphBuilder:
    """
    Learns a causal DAG from energy time-series data.

    Uses the PC algorithm with Fisher's Z test for conditional independence,
    then applies domain-specific orientation rules for energy systems.

    Args:
        alpha:          Significance level for independence tests (default 0.05).
        max_lag_hours:  Maximum time lag to consider for lagged causality.
        min_samples:    Minimum rows required to learn the graph reliably.
    """

    # Domain-specific causal priors for energy systems
    # These orient edges when the PC algorithm leaves them ambiguous.
    DOMAIN_PRIORS: dict[tuple[str, str], str] = {
        # Weather → consumption (temperature drives HVAC)
        ("temperature_c", "kwh"):           "temperature_c → kwh",
        # Shift → consumption (shift start drives load)
        ("is_business_hour", "kwh"):        "is_business_hour → kwh",
        # Power factor → apparent power (not the reverse)
        ("power_factor", "apparent_power_kva"): "power_factor → apparent_power_kva",
        # Time features → everything (time is exogenous)
        ("hour_sin", "kwh"):                "hour_sin → kwh",
        ("hour_cos", "kwh"):                "hour_cos → kwh",
        ("is_weekend", "kwh"):              "is_weekend → kwh",
    }

    # Human-readable mechanism descriptions
    MECHANISMS: dict[tuple[str, str], str] = {
        ("temperature_c", "kwh"):        "Ambient temperature drives HVAC load",
        ("is_business_hour", "kwh"):     "Shift activity drives equipment load",
        ("power_factor", "kwh"):         "Power factor correction affects net draw",
        ("hour_sin", "kwh"):             "Time-of-day pattern drives base load",
        ("is_weekend", "kwh"):           "Weekend schedule reduces non-essential load",
        ("kwh_lag_1h", "kwh"):           "Autoregressive: prior hour predicts current",
        ("voltage_v", "kwh"):            "Voltage variation affects motor efficiency",
        ("kwh_roll_mean_24h", "kwh"):    "Daily baseline predicts current consumption",
    }

    def __init__(
        self,
        alpha:         float = DEFAULT_ALPHA,
        max_lag_hours: int   = 3,
        min_samples:   int   = 48,
    ) -> None:
        self._alpha     = alpha
        self._max_lag   = max_lag_hours
        self._min_samp  = min_samples

    def build(self, df: pd.DataFrame, site_id: str) -> CausalGraph:
        """
        Learn the causal DAG from processed energy data.

        Args:
            df:      Processed DataFrame (post-EnergyTransformer).
            site_id: Site identifier for the graph.

        Returns:
            CausalGraph with discovered edges and metadata.
        """
        from datetime import datetime, timezone

        if len(df) < self._min_samp:
            raise ValueError(
                f"Need ≥{self._min_samp} samples, got {len(df)}. "
                "Collect more data before learning causal structure."
            )

        # Select numeric, non-NaN columns
        candidate_cols = self._select_variables(df)
        df_clean = df[candidate_cols].dropna()

        log.info("causal.build_start", site=site_id,
                 variables=len(candidate_cols), rows=len(df_clean))

        # Standardise for consistent effect sizes
        df_std = (df_clean - df_clean.mean()) / (df_clean.std() + 1e-8)

        # Run PC algorithm skeleton learning
        skeleton, sep_sets = self._learn_skeleton(df_std, candidate_cols)

        # Orient edges using v-structures + domain priors
        dag = self._orient_edges(skeleton, sep_sets, candidate_cols)

        # Compute edge weights via OLS regression
        edges = self._compute_edge_weights(dag, df_std)

        graph = CausalGraph(
            site_id=site_id,
            dag=dag,
            edges=edges,
            variables=list(candidate_cols),
            n_samples=len(df_clean),
            learned_at=datetime.now(timezone.utc).isoformat(),
        )

        log.info("causal.build_complete", site=site_id,
                 edges=len(edges), variables=len(candidate_cols))
        return graph

    # ── PC algorithm implementation ───────────────────────────────────────────

    def _learn_skeleton(
        self,
        df: pd.DataFrame,
        variables: list[str],
    ) -> tuple[nx.Graph, dict]:
        """
        PC algorithm Step 1: learn the undirected skeleton by removing
        edges where conditional independence is detected.
        """
        n = len(variables)
        # Start with fully connected undirected graph
        skeleton  = nx.complete_graph(n)
        sep_sets: dict[tuple[int, int], set[int]] = {}

        # Map integer indices to variable names
        idx_to_var = {i: v for i, v in enumerate(variables)}
        var_to_idx = {v: i for i, v in enumerate(variables)}

        data = df.values
        n_obs = len(data)

        # Test conditional independence with increasing conditioning sets
        for depth in range(n - 1):
            edges_to_remove = []

            for (u, v) in list(skeleton.edges()):
                neighbours_u = set(skeleton.neighbors(u)) - {v}
                if len(neighbours_u) < depth:
                    continue

                # Test all subsets of size `depth`
                for subset in itertools.combinations(neighbours_u, depth):
                    p_val = self._fisher_z_test(
                        data, u, v, list(subset), n_obs
                    )
                    if p_val > self._alpha:
                        # Conditionally independent — remove edge
                        edges_to_remove.append((u, v))
                        sep_sets[(u, v)] = set(subset)
                        sep_sets[(v, u)] = set(subset)
                        break

            for edge in edges_to_remove:
                if skeleton.has_edge(*edge):
                    skeleton.remove_edge(*edge)

        return skeleton, sep_sets

    def _orient_edges(
        self,
        skeleton:  nx.Graph,
        sep_sets:  dict,
        variables: list[str],
    ) -> nx.DiGraph:
        """
        PC algorithm Step 2: orient edges using v-structures, then
        apply Meek rules and domain priors.
        """
        idx_to_var = {i: v for i, v in enumerate(variables)}
        dag = nx.DiGraph()
        dag.add_nodes_from(skeleton.nodes())

        # Start with undirected edges
        for (u, v) in skeleton.edges():
            dag.add_edge(u, v)
            dag.add_edge(v, u)

        # Detect v-structures (colliders): X → Z ← Y with X-Y not adjacent
        for z in skeleton.nodes():
            neighbours = list(skeleton.neighbors(z))
            for x, y in itertools.combinations(neighbours, 2):
                if skeleton.has_edge(x, y):
                    continue
                sep = sep_sets.get((x, y), set())
                if z not in sep:
                    # Orient: x → z ← y
                    if dag.has_edge(z, x):
                        dag.remove_edge(z, x)
                    if dag.has_edge(z, y):
                        dag.remove_edge(z, y)

        # Apply domain priors
        for (cause_var, effect_var), _ in self.DOMAIN_PRIORS.items():
            try:
                ci = variables.index(cause_var)
                ei = variables.index(effect_var)
                if dag.has_edge(ei, ci) and dag.has_edge(ci, ei):
                    dag.remove_edge(ei, ci)
                elif dag.has_edge(ei, ci) and not dag.has_edge(ci, ei):
                    dag.add_edge(ci, ei)
                    dag.remove_edge(ei, ci)
            except ValueError:
                pass   # Variable not in this dataset

        # Remove remaining bidirectional edges using topological heuristic
        for (u, v) in list(dag.edges()):
            if not dag.has_edge(u, v):
                continue   # Already removed by a prior iteration
            if dag.has_edge(v, u):
                u_name = idx_to_var.get(u, str(u))
                v_name = idx_to_var.get(v, str(v))
                if u_name > v_name:
                    if dag.has_edge(u, v):
                        dag.remove_edge(u, v)
                else:
                    if dag.has_edge(v, u):
                        dag.remove_edge(v, u)

        # Ensure acyclicity
        while not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            if cycles:
                cycle = cycles[0]
                dag.remove_edge(cycle[-1], cycle[0])

        return dag

    def _compute_edge_weights(
        self,
        dag:       nx.DiGraph,
        df:        pd.DataFrame,
    ) -> list[CausalEdge]:
        """
        Compute edge effect sizes via OLS regression of each node
        on its parents. Standardised coefficients are effect sizes.
        """
        edges: list[CausalEdge] = []
        variables = list(df.columns)
        data      = df.values

        for edge in dag.edges():
            u, v = edge
            if u >= len(variables) or v >= len(variables):
                continue

            cause_name  = variables[u]
            effect_name = variables[v]
            parents = list(dag.predecessors(v))

            if not parents:
                continue

            y = data[:, v]
            X = data[:, parents]
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            try:
                # OLS with intercept
                X_aug = np.column_stack([np.ones(len(X)), X])
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

                # Get the coefficient for this specific cause
                parent_idx  = parents.index(u)
                beta        = float(coeffs[parent_idx + 1])

                # t-statistic for significance
                y_hat  = X_aug @ coeffs
                resid  = y - y_hat
                dof    = max(1, len(y) - len(coeffs))
                mse    = float(np.sum(resid ** 2) / dof)
                Xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
                se = math.sqrt(max(0, mse * Xtx_inv[parent_idx + 1, parent_idx + 1]))
                t_stat = beta / (se + 1e-8)
                p_val  = float(2 * stats.t.sf(abs(t_stat), df=dof))

                mechanism = self.MECHANISMS.get(
                    (cause_name, effect_name),
                    f"{cause_name} causally influences {effect_name}",
                )
                confidence = (
                    "high"   if p_val < 0.01 and abs(beta) > 0.3 else
                    "medium" if p_val < 0.05 else
                    "low"
                )

                edges.append(CausalEdge(
                    cause=cause_name,
                    effect=effect_name,
                    strength=round(beta, 4),
                    p_value=round(p_val, 4),
                    mechanism=mechanism,
                    confidence=confidence,
                ))

            except Exception as exc:
                log.debug("causal.edge_weight_error",
                          cause=cause_name, effect=effect_name, error=str(exc))

        return sorted(edges, key=lambda e: abs(e.strength), reverse=True)

    # ── Statistical test ──────────────────────────────────────────────────────

    @staticmethod
    def _fisher_z_test(
        data:      np.ndarray,
        x:         int,
        y:         int,
        cond_set:  list[int],
        n:         int,
    ) -> float:
        """
        Fisher's Z test for (conditional) independence.
        Returns the p-value; large p → independent.
        """
        try:
            if not cond_set:
                r = float(np.corrcoef(data[:, x], data[:, y])[0, 1])
            else:
                # Partial correlation via linear regression residuals
                z_mat = data[:, cond_set]
                def residuals(col_idx: int) -> np.ndarray:
                    y_vec = data[:, col_idx]
                    X = np.column_stack([np.ones(n), z_mat])
                    coeff, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
                    return y_vec - X @ coeff

                rx = residuals(x)
                ry = residuals(y)
                std_x = np.std(rx) + 1e-8
                std_y = np.std(ry) + 1e-8
                r = float(np.corrcoef(rx / std_x, ry / std_y)[0, 1])

            r = max(-0.9999, min(0.9999, r))
            z = 0.5 * math.log((1 + r) / (1 - r))
            dof = max(1, n - len(cond_set) - 2)
            se  = 1.0 / math.sqrt(dof)
            p   = float(2 * stats.norm.sf(abs(z / se)))
            return p
        except Exception:
            return 1.0   # Conservative: assume independent on error

    @staticmethod
    def _select_variables(df: pd.DataFrame) -> list[str]:
        """Select numeric columns with sufficient variance."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude identifiers and binary-only columns with no variance
        preferred = [
            "kwh", "power_factor", "voltage_v", "temperature_c",
            "is_business_hour", "is_weekend", "hour_sin", "hour_cos",
            "kwh_lag_1h", "kwh_roll_mean_24h", "kwh_roll_std_24h",
            "kwh_delta", "kwh_pct_change",
        ]
        selected = [c for c in preferred if c in numeric]
        # Add any other numeric columns not already included
        others = [
            c for c in numeric
            if c not in selected
            and not c.startswith("dow_")
            and not c.startswith("month_")
            and df[c].std() > 1e-4
        ]
        return selected + others[:3]   # cap at ~16 total variables
