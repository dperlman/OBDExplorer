"""PCA precomputation for explorer2 HTML (same math as OBDexplorer3 / OBD2Dprojection)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binom
from sklearn.decomposition import PCA


def precompute_pca(
    n_vals: list[int],
    p_num: int,
    p_min: float,
    p_max: float,
    fit_on_sorted: bool,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    """Fit PCA on p in ``[p_min, p_max]``; project full ``p`` in ``[0, 1]``. Return row list + global ranges."""
    p_vals_fit = np.linspace(p_min, p_max, p_num)
    p_vals_full = np.linspace(0.0, 1.0, p_num)
    out: list[dict[str, Any]] = []
    r12 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    r23 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    for n in n_vals:
        k_vals = np.arange(n + 1, dtype=float)
        unsorted_pmf_fit = binom.pmf(k_vals[:, np.newaxis], n, p_vals_fit[np.newaxis, :])
        ordered_pmf_fit = np.sort(unsorted_pmf_fit, axis=0)
        X_unsorted_fit = unsorted_pmf_fit.T
        X_ordered_fit = ordered_pmf_fit.T
        X_fit = X_ordered_fit if fit_on_sorted else X_unsorted_fit
        n_comp = min(4, n + 1)
        reducer = PCA(n_components=n_comp, random_state=0)
        reducer.fit(X_fit)
        unsorted_pmf_full = binom.pmf(k_vals[:, np.newaxis], n, p_vals_full[np.newaxis, :])
        ordered_pmf_full = np.sort(unsorted_pmf_full, axis=0)
        X_unsorted_full = unsorted_pmf_full.T
        X_ordered_full = ordered_pmf_full.T
        embed_unsorted = reducer.transform(X_unsorted_full)
        embed_ordered = reducer.transform(X_ordered_full)
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        if n_comp < 4:
            pad_u = np.zeros((embed_unsorted.shape[0], 4 - n_comp))
            pad_o = np.zeros((embed_ordered.shape[0], 4 - n_comp))
            pad_v = np.zeros((vertex_embed.shape[0], 4 - n_comp))
            embed_unsorted = np.hstack([embed_unsorted, pad_u])
            embed_ordered = np.hstack([embed_ordered, pad_o])
            vertex_embed = np.hstack([vertex_embed, pad_v])
        curve_embed = np.vstack([embed_unsorted, embed_ordered])
        r12["xMin"] = min(r12["xMin"], float(curve_embed[:, 0].min()))
        r12["xMax"] = max(r12["xMax"], float(curve_embed[:, 0].max()))
        r12["yMin"] = min(r12["yMin"], float(curve_embed[:, 1].min()))
        r12["yMax"] = max(r12["yMax"], float(curve_embed[:, 1].max()))
        r23["xMin"] = min(r23["xMin"], float(curve_embed[:, 1].min()))
        r23["xMax"] = max(r23["xMax"], float(curve_embed[:, 1].max()))
        range_12_x = [float(curve_embed[:, 0].min()), float(curve_embed[:, 0].max())]
        range_12_y = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_x = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_y = [float(curve_embed[:, 2].min()), float(curve_embed[:, 2].max())]
        p_list = [round(float(p), 6) for p in p_vals_full]
        out.append(
            {
                "n": n,
                "p": p_list,
                "unsorted_pc1": embed_unsorted[:, 0].tolist(),
                "unsorted_pc2": embed_unsorted[:, 1].tolist(),
                "unsorted_pc3": embed_unsorted[:, 2].tolist(),
                "unsorted_pc4": embed_unsorted[:, 3].tolist(),
                "ordered_pc1": embed_ordered[:, 0].tolist(),
                "ordered_pc2": embed_ordered[:, 1].tolist(),
                "ordered_pc3": embed_ordered[:, 2].tolist(),
                "ordered_pc4": embed_ordered[:, 3].tolist(),
                "vertex_pc1": vertex_embed[:, 0].tolist(),
                "vertex_pc2": vertex_embed[:, 1].tolist(),
                "vertex_pc3": vertex_embed[:, 2].tolist(),
                "vertex_pc4": vertex_embed[:, 3].tolist(),
                "range_12_x": range_12_x,
                "range_12_y": range_12_y,
                "range_23_x": range_23_x,
                "range_23_y": range_23_y,
            }
        )
    return out, r12, r23


def precompute_tie_pca(
    n_vals: list[int],
    p_num: int,
    fit_on_sorted: bool,
    last_tie_by_n: dict[int, float],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    """Fit PCA on p in ``[0.5, last_tie(n)]`` per n; project full p in ``[0, 1]``."""
    p_vals_full = np.linspace(0.0, 1.0, p_num)
    out: list[dict[str, Any]] = []
    r12 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    r23 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    for n in n_vals:
        p_tie = float(last_tie_by_n[n])
        p_vals_fit = np.linspace(0.5, p_tie, p_num)
        k_vals = np.arange(n + 1, dtype=float)
        unsorted_pmf_fit = binom.pmf(k_vals[:, np.newaxis], n, p_vals_fit[np.newaxis, :])
        ordered_pmf_fit = np.sort(unsorted_pmf_fit, axis=0)
        X_unsorted_fit = unsorted_pmf_fit.T
        X_ordered_fit = ordered_pmf_fit.T
        X_fit = X_ordered_fit if fit_on_sorted else X_unsorted_fit
        n_comp = min(4, n + 1)
        reducer = PCA(n_components=n_comp, random_state=0)
        reducer.fit(X_fit)
        unsorted_pmf_full = binom.pmf(k_vals[:, np.newaxis], n, p_vals_full[np.newaxis, :])
        ordered_pmf_full = np.sort(unsorted_pmf_full, axis=0)
        X_unsorted_full = unsorted_pmf_full.T
        X_ordered_full = ordered_pmf_full.T
        embed_unsorted = reducer.transform(X_unsorted_full)
        embed_ordered = reducer.transform(X_ordered_full)
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        if n_comp < 4:
            embed_unsorted = np.hstack(
                [embed_unsorted, np.zeros((embed_unsorted.shape[0], 4 - n_comp))]
            )
            embed_ordered = np.hstack(
                [embed_ordered, np.zeros((embed_ordered.shape[0], 4 - n_comp))]
            )
            vertex_embed = np.hstack(
                [vertex_embed, np.zeros((vertex_embed.shape[0], 4 - n_comp))]
            )
        curve_embed = np.vstack([embed_unsorted, embed_ordered])
        r12["xMin"] = min(r12["xMin"], float(curve_embed[:, 0].min()))
        r12["xMax"] = max(r12["xMax"], float(curve_embed[:, 0].max()))
        r12["yMin"] = min(r12["yMin"], float(curve_embed[:, 1].min()))
        r12["yMax"] = max(r12["yMax"], float(curve_embed[:, 1].max()))
        r23["xMin"] = min(r23["xMin"], float(curve_embed[:, 1].min()))
        r23["xMax"] = max(r23["xMax"], float(curve_embed[:, 1].max()))
        range_12_x = [float(curve_embed[:, 0].min()), float(curve_embed[:, 0].max())]
        range_12_y = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_x = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_y = [float(curve_embed[:, 2].min()), float(curve_embed[:, 2].max())]
        p_list = [round(float(p), 6) for p in p_vals_full]
        out.append(
            {
                "n": n,
                "p": p_list,
                "unsorted_pc1": embed_unsorted[:, 0].tolist(),
                "unsorted_pc2": embed_unsorted[:, 1].tolist(),
                "unsorted_pc3": embed_unsorted[:, 2].tolist(),
                "unsorted_pc4": embed_unsorted[:, 3].tolist(),
                "ordered_pc1": embed_ordered[:, 0].tolist(),
                "ordered_pc2": embed_ordered[:, 1].tolist(),
                "ordered_pc3": embed_ordered[:, 2].tolist(),
                "ordered_pc4": embed_ordered[:, 3].tolist(),
                "vertex_pc1": vertex_embed[:, 0].tolist(),
                "vertex_pc2": vertex_embed[:, 1].tolist(),
                "vertex_pc3": vertex_embed[:, 2].tolist(),
                "vertex_pc4": vertex_embed[:, 3].tolist(),
                "range_12_x": range_12_x,
                "range_12_y": range_12_y,
                "range_23_x": range_23_x,
                "range_23_y": range_23_y,
            }
        )
    return out, r12, r23
