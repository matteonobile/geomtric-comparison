# -*- coding: utf-8 -*-
"""
app_comparison.py
=================
Executive 3D + 2D Risk Topography Comparator (1 Year Ago vs. Today).

Visual encoding:
- (X, Y)        : Cluster coordinates from PCA / MDS (with optional Procrustes alignment)
- Z & Terrain   : Active Weight (Mountains = Overweight, Valleys = Underweight)
- Footprint σ   : Intra-Cluster Stress (clipped & log-scaled, safe baseline for 'Not Held')
- Peak Markers  : Stress Traffic-Light Halo (<25% Green, 25–50% Amber, >50% Red, Not Held Slate)
- Rings & Strip : Concentric Benchmark Factor-Distance Rings + Core / Mid / Outer breakdown
- Tension Arcs  : "3rd Bet" bridges linking major overweight peaks straddling the Benchmark
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3D Risk Topography Comparator", layout="wide")


# ---------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------
CAMERA_PRESETS = {
    "3D Isometric (Balanced)": dict(eye=dict(x=1.45, y=-1.55, z=1.05)),
    "Low Horizon (Dramatic Peaks)": dict(eye=dict(x=1.75, y=-1.75, z=0.55)),
    "Top-Down 3D (Factor Rays)": dict(eye=dict(x=0.01, y=-0.01, z=2.35)),
}

ACTIVE_COLORSCALE = [
    [0.00, "#7f0000"],
    [0.32, "#d7301f"],
    [0.46, "#fc8d59"],
    [0.50, "#eaeef3"],
    [0.54, "#a1d99b"],
    [0.68, "#31a354"],
    [1.00, "#00441b"],
]


def plotly_export_config(filename: str) -> dict:
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "scale": 3,  # Crisp 3x retina export for slides
        },
    }


# ---------------------------------------------------------
# DATA LOADING & PARSING HELPERS
# ---------------------------------------------------------
def load_cluster_file(uploaded_file, key_prefix: str) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    buf = io.BytesIO(uploaded_file.getvalue())

    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith(".parquet"):
        return pd.read_parquet(buf)
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(buf)
        default_idx = 0
        for i, s in enumerate(xl.sheet_names):
            if s.startswith("L1_") or "cluster" in s.lower():
                default_idx = i
                break
        sheet = st.sidebar.selectbox(
            f"Excel sheet ({key_prefix})",
            xl.sheet_names,
            index=default_idx,
            key=f"sheet_{key_prefix}",
        )
        return xl.parse(sheet)
    return None


def parse_stress_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    is_not_held = np.zeros(len(series), dtype=bool)
    vals = np.zeros(len(series), dtype=float)

    for idx, v in enumerate(series):
        if pd.isna(v):
            is_not_held[idx] = True
            vals[idx] = 0.0
        elif isinstance(v, str):
            s = v.strip()
            if s.lower() in {"not held", "n/a", "na", "none", ""}:
                is_not_held[idx] = True
                vals[idx] = 0.0
            else:
                try:
                    vals[idx] = float(s.replace("%", "").strip())
                except ValueError:
                    is_not_held[idx] = True
                    vals[idx] = 0.0
        else:
            num = float(v)
            vals[idx] = num * 100.0 if 0.0 < abs(num) <= 1.0 else num

    return np.clip(vals, 0.0, None), is_not_held


def prepare_dataframe(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    out = df.copy()
    required = ["Node_ID", "X", "Y", weight_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Node_Label" not in out.columns:
        out["Node_Label"] = out["Node_ID"]

    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out["Active_Pct"] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0) * 100.0

    prefix = "Local" if weight_col.startswith("Local") else "Global"
    b_col = f"{prefix}_Bench_Weight"
    p_col = f"{prefix}_Port_Weight"

    out["Bench_W"] = (
        pd.to_numeric(out[b_col], errors="coerce").fillna(0.0)
        if b_col in out.columns
        else np.nan
    )
    out["Port_W"] = (
        pd.to_numeric(out[p_col], errors="coerce").fillna(0.0)
        if p_col in out.columns
        else np.nan
    )

    if "Child_Stress_%" in out.columns:
        stress_vals, not_held = parse_stress_series(out["Child_Stress_%"])
    else:
        stress_vals = np.zeros(len(out))
        not_held = np.ones(len(out), dtype=bool)

    out["Stress_Pct"] = stress_vals
    out["Is_Not_Held"] = not_held
    return out.dropna(subset=["X", "Y"]).reset_index(drop=True)


def align_procrustes(df_old: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Rigidly aligns df_old (X, Y) onto df_new (X, Y) via orthogonal Procrustes
    to remove arbitrary PCA/MDS sign flips or global rotations.
    """
    merged = pd.merge(
        df_old[["Node_ID", "X", "Y"]],
        df_new[["Node_ID", "X", "Y"]],
        on="Node_ID",
        suffixes=("_old", "_new"),
    )
    if len(merged) < 3:
        return df_old.copy(), np.nan

    P = merged[["X_old", "Y_old"]].to_numpy(dtype=float)
    Q = merged[["X_new", "Y_new"]].to_numpy(dtype=float)

    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    P_c, Q_c = P - p_mean, Q - q_mean

    U, _, Vt = np.linalg.svd(P_c.T @ Q_c)
    R = U @ Vt

    rmsd = float(np.sqrt(np.mean(np.sum((P_c @ R + q_mean - Q) ** 2, axis=1))))

    aligned_df = df_old.copy()
    coords = aligned_df[["X", "Y"]].to_numpy(dtype=float)
    aligned_coords = (coords - p_mean) @ R + q_mean
    aligned_df["X"] = aligned_coords[:, 0]
    aligned_df["Y"] = aligned_coords[:, 1]
    return aligned_df, rmsd


def compute_anchors(df: pd.DataFrame) -> dict[str, tuple[float, float]] | None:
    """
    Uses exact exported Bench_X/Bench_Y/Port_X/Port_Y if present (Option B),
    otherwise computes 2D weight-weighted barycenters (Option A).
    """
    exact_cols = {"Bench_X", "Bench_Y", "Port_X", "Port_Y"}
    if exact_cols.issubset(df.columns):
        return {
            "bench": (float(df["Bench_X"].iloc[0]), float(df["Bench_Y"].iloc[0])),
            "port": (float(df["Port_X"].iloc[0]), float(df["Port_Y"].iloc[0])),
        }

    if df["Bench_W"].isna().all() or df["Port_W"].isna().all():
        return None

    bw = df["Bench_W"].to_numpy(dtype=float)
    pw = df["Port_W"].to_numpy(dtype=float)
    xs = df["X"].to_numpy(dtype=float)
    ys = df["Y"].to_numpy(dtype=float)

    b_sum, p_sum = float(np.sum(bw)), float(np.sum(pw))
    if b_sum <= 1e-12 or p_sum <= 1e-12:
        return None

    bw_n, pw_n = bw / b_sum, pw / p_sum
    return {
        "bench": (float(np.sum(bw_n * xs)), float(np.sum(bw_n * ys))),
        "port": (float(np.sum(pw_n * xs)), float(np.sum(pw_n * ys))),
    }


# ---------------------------------------------------------
# GEOMETRIC TENSION, RINGS & "3RD BET" STRADDLE BRIDGES
# ---------------------------------------------------------
def compute_geometric_metrics(
    df: pd.DataFrame, anchors: dict[str, tuple[float, float]] | None
) -> dict[str, float]:
    active_abs = df["Active_Pct"].abs().to_numpy(dtype=float)
    active_share = 0.5 * float(np.sum(active_abs))
    total_abs = float(np.sum(active_abs))

    if anchors is None or total_abs <= 1e-12:
        return {
            "active_share": active_share,
            "net_tilt": np.nan,
            "tension_rms": np.nan,
            "active_work": np.nan,
        }

    bx, by = anchors["bench"]
    px, py = anchors["port"]
    net_tilt = float(np.hypot(px - bx, py - by))

    dists = np.hypot(df["X"].to_numpy(float) - bx, df["Y"].to_numpy(float) - by)
    w_norm = active_abs / total_abs
    tension_rms = float(np.sqrt(np.sum(w_norm * (dists ** 2))))
    # Sum(|a_i| * d_i) with active weights in decimals -> mirrors W1 transport work
    active_work = float(np.sum((active_abs / 100.0) * dists))

    return {
        "active_share": active_share,
        "net_tilt": net_tilt,
        "tension_rms": tension_rms,
        "active_work": active_work,
    }


def compute_ring_breakdown(
    df: pd.DataFrame,
    anchors: dict[str, tuple[float, float]] | None,
    r_core: float,
    r_mid: float,
) -> dict[str, float] | None:
    if anchors is None:
        return None
    bx, by = anchors["bench"]
    dists = np.hypot(df["X"].to_numpy(float) - bx, df["Y"].to_numpy(float) - by)
    abs_a = df["Active_Pct"].abs().to_numpy(float)
    tot = float(np.sum(abs_a))
    if tot <= 1e-12:
        return {"Core": 0.0, "Mid": 0.0, "Outer Tail": 0.0}

    core_pct = 100.0 * float(np.sum(abs_a[dists <= r_core])) / tot
    mid_pct = 100.0 * float(np.sum(abs_a[(dists > r_core) & (dists <= r_mid)])) / tot
    outer_pct = 100.0 * float(np.sum(abs_a[dists > r_mid])) / tot
    return {"Core": core_pct, "Mid": mid_pct, "Outer Tail": outer_pct}


def find_straddle_bridges(
    df: pd.DataFrame,
    node_z: np.ndarray,
    anchors: dict[str, tuple[float, float]] | None,
    min_ow_pct: float = 1.5,
    max_bridges: int = 3,
    cos_threshold: float = -0.15,
) -> list[dict]:
    """
    Finds the highest-tension pairs of Overweight clusters sitting on opposite flanks
    of the Benchmark (angle across Benchmark > ~98.6 deg) -> the implicit "3rd Bet".
    """
    if anchors is None or max_bridges <= 0:
        return []

    bx, by = anchors["bench"]
    ow_idx = np.where(df["Active_Pct"].to_numpy(float) >= min_ow_pct)[0]
    if len(ow_idx) < 2:
        return []

    bridges = []
    xs = df["X"].to_numpy(float)
    ys = df["Y"].to_numpy(float)
    acts = df["Active_Pct"].to_numpy(float)
    labels = df["Node_Label"].astype(str).to_numpy()

    for idx_a in range(len(ow_idx)):
        i = ow_idx[idx_a]
        ui_x, ui_y = xs[i] - bx, ys[i] - by
        ni = float(np.hypot(ui_x, ui_y))
        if ni < 1e-6:
            continue
        for idx_b in range(idx_a + 1, len(ow_idx)):
            j = ow_idx[idx_b]
            uj_x, uj_y = xs[j] - bx, ys[j] - by
            nj = float(np.hypot(uj_x, uj_y))
            if nj < 1e-6:
                continue

            cos_th = float((ui_x * uj_x + ui_y * uj_y) / (ni * nj))
            if cos_th < cos_threshold:
                sep = float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))
                score = float(np.sqrt(acts[i] * acts[j]) * sep * (1.0 - cos_th) * 0.5)
                bridges.append(
                    {
                        "label_i": labels[i],
                        "label_j": labels[j],
                        "act_i": float(acts[i]),
                        "act_j": float(acts[j]),
                        "xi": float(xs[i]),
                        "yi": float(ys[i]),
                        "zi": float(node_z[i]),
                        "xj": float(xs[j]),
                        "yj": float(ys[j]),
                        "zj": float(node_z[j]),
                        "sep": sep,
                        "angle_deg": angle_deg,
                        "score": score,
                    }
                )

    bridges.sort(key=lambda b: b["score"], reverse=True)
    return bridges[:max_bridges]


# ---------------------------------------------------------
# SURFACE EVALUATION & LABEL DE-COLLISION
# ---------------------------------------------------------
def compute_sigmas(
    stress_pct: np.ndarray,
    span: float,
    sigma_min_frac: float,
    sigma_max_frac: float,
    stress_cap: float,
    use_log: bool,
    log_k: float,
) -> np.ndarray:
    s_clipped = np.clip(stress_pct, 0.0, stress_cap)
    if stress_cap <= 0:
        u = np.zeros_like(s_clipped)
    elif use_log:
        u = np.log1p(log_k * s_clipped) / np.log1p(log_k * stress_cap)
    else:
        u = s_clipped / stress_cap

    return (sigma_min_frac + (sigma_max_frac - sigma_min_frac) * u) * span


def eval_z_at_points(
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    sigmas: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    blend_mode: str,
) -> np.ndarray:
    out = np.zeros(len(pts_x), dtype=float)
    pos_mask = ws >= 0
    neg_mask = ws < 0
    for j, (xj, yj) in enumerate(zip(pts_x, pts_y)):
        d2 = (xj - xs) ** 2 + (yj - ys) ** 2
        k_vals = np.exp(-d2 / (2.0 * sigmas ** 2))
        if blend_mode == "Additive Sum":
            out[j] = float(np.sum(ws * k_vals))
        else:
            zp = float(np.max(ws[pos_mask] * k_vals[pos_mask])) if np.any(pos_mask) else 0.0
            zn = float(np.max(np.abs(ws[neg_mask]) * k_vals[neg_mask])) if np.any(neg_mask) else 0.0
            out[j] = zp - zn
    return out


def evaluate_surface(
    df: pd.DataFrame,
    gx: np.ndarray,
    gy: np.ndarray,
    span: float,
    sigma_min_frac: float,
    sigma_max_frac: float,
    stress_cap: float,
    use_log: bool,
    log_k: float,
    blend_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = df["X"].to_numpy(dtype=float)
    ys = df["Y"].to_numpy(dtype=float)
    ws = df["Active_Pct"].to_numpy(dtype=float)
    stresses = df["Stress_Pct"].to_numpy(dtype=float)

    sigmas = compute_sigmas(
        stresses, span, sigma_min_frac, sigma_max_frac, stress_cap, use_log, log_k
    )

    Z_pos = np.zeros_like(gx, dtype=float)
    Z_neg = np.zeros_like(gx, dtype=float)
    Z_sum = np.zeros_like(gx, dtype=float)
    stress_num = np.zeros_like(gx, dtype=float)
    stress_den = np.zeros_like(gx, dtype=float)

    for xi, yi, wi, si, sig in zip(xs, ys, ws, stresses, sigmas):
        dist2 = (gx - xi) ** 2 + (gy - yi) ** 2
        kernel = np.exp(-dist2 / (2.0 * sig ** 2))
        Z_sum += wi * kernel
        if wi >= 0:
            Z_pos = np.maximum(Z_pos, wi * kernel)
        else:
            Z_neg = np.maximum(Z_neg, abs(wi) * kernel)

        w_mag = abs(wi) + 0.25
        stress_num += (si * w_mag) * kernel
        stress_den += w_mag * kernel

    Z = Z_sum if blend_mode == "Additive Sum" else (Z_pos - Z_neg)
    StressField = np.divide(
        stress_num, stress_den, out=np.zeros_like(stress_num), where=stress_den > 1e-6
    )
    node_z = eval_z_at_points(xs, ys, ws, sigmas, xs, ys, blend_mode)
    return Z, StressField, node_z, sigmas


def select_decollided_labels(
    df: pd.DataFrame,
    node_z: np.ndarray,
    max_labels: int,
    min_dist: float,
    min_active_pct: float,
) -> pd.DataFrame:
    """
    Greedily picks the largest |Active_Pct| clusters while enforcing a minimum 2D
    separation so peak labels don't pile on top of each other.
    """
    if max_labels <= 0:
        return df.iloc[0:0].copy()

    cand = df.assign(_abs=df["Active_Pct"].abs(), _nz=node_z)
    cand = cand[cand["_abs"] >= min_active_pct].sort_values("_abs", ascending=False)

    chosen_rows = []
    placed_xy = []
    for row in cand.itertuples():
        if len(chosen_rows) >= max_labels:
            break
        if all(np.hypot(row.X - px, row.Y - py) >= min_dist for px, py in placed_xy):
            chosen_rows.append(row.Index)
            placed_xy.append((row.X, row.Y))

    return cand.loc[chosen_rows]


def stress_marker_color(stress_pct: float, is_not_held: bool, active_pct: float, use_stress_halo: bool) -> str:
    if not use_stress_halo:
        return "#00441b" if active_pct >= 0 else "#67000d"
    if is_not_held:
        return "#64748b"  # Slate: Not Held
    if stress_pct >= 50.0:
        return "#dc2626"  # Red: High intra-cluster stress
    if stress_pct >= 25.0:
        return "#f59e0b"  # Amber: Moderate stress
    return "#16a34a"      # Green: Low stress


def build_hover_texts(df: pd.DataFrame, sigmas: np.ndarray) -> list[str]:
    texts = []
    for row, sig in zip(df.itertuples(), sigmas):
        stress_str = "Not Held (0%)" if row.Is_Not_Held else f"{row.Stress_Pct:.1f}%"
        bw_str = f"{row.Bench_W:.2%}" if pd.notna(row.Bench_W) else "—"
        pw_str = f"{row.Port_W:.2%}" if pd.notna(row.Port_W) else "—"
        texts.append(
            f"<b>{row.Node_Label}</b><br>"
            f"Active Weight: {row.Active_Pct:+.2f}%<br>"
            f"Bench / Port: {bw_str} / {pw_str}<br>"
            f"Child Stress: {stress_str}<br>"
            f"Footprint σ: {sig:.3f}"
        )
    return texts


def clean_axis_dtick(span: float) -> float:
    if span > 0.75:
        return 0.2
    if span > 0.35:
        return 0.1
    return 0.05


# ---------------------------------------------------------
# 3D LANDSCAPE FIGURE
# ---------------------------------------------------------
def make_3d_landscape_figure(
    title: str,
    df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    Z: np.ndarray,
    StressField: np.ndarray,
    node_z: np.ndarray,
    sigmas: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_lim: float,
    color_mode: str,
    stress_cap: float,
    show_sea_level: bool,
    show_labels: bool,
    top_n_labels: int,
    min_marker_active: float,
    use_stress_halo: bool,
    anchors: dict[str, tuple[float, float]] | None,
    show_rings: bool,
    ring_radii: tuple[float, float, float],
    bridges: list[dict],
    blend_mode: str,
    camera_cfg: dict,
) -> go.Figure:
    fig = go.Figure()
    span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    dtick_xy = clean_axis_dtick(span)

    if color_mode == "Over / Underweight (Active %)":
        surface_color = Z
        colorscale = ACTIVE_COLORSCALE
        cmin, cmax = -z_lim, z_lim
        cb_title = "Active Wt (%)"
    else:
        surface_color = np.clip(StressField, 0.0, stress_cap)
        colorscale = "YlOrRd"
        cmin, cmax = 0.0, stress_cap
        cb_title = "Stress (%)"

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=Z,
            surfacecolor=surface_color,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=cb_title, len=0.58, thickness=13, x=1.02),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    color="rgba(40, 40, 40, 0.24)",
                    width=1,
                    project_z=False,
                )
            ),
            lighting=dict(ambient=0.68, diffuse=0.72, specular=0.20, roughness=0.58),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if show_sea_level:
        fig.add_trace(
            go.Surface(
                x=[x_range[0], x_range[1]],
                y=[y_range[0], y_range[1]],
                z=np.zeros((2, 2)),
                colorscale=[[0, "rgba(120, 140, 160, 0.22)"], [1, "rgba(120, 140, 160, 0.22)"]],
                showscale=False,
                opacity=0.22,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Concentric Benchmark Factor-Distance Rings on the Z ≈ 0 floor
    if show_rings and anchors is not None:
        bx, by = anchors["bench"]
        theta = np.linspace(0, 2 * np.pi, 96)
        ring_styles = [
            (ring_radii[0], "Core", "rgba(37, 99, 235, 0.45)", "dash"),
            (ring_radii[1], "Mid", "rgba(100, 116, 139, 0.40)", "dot"),
            (ring_radii[2], "Outer", "rgba(148, 163, 184, 0.35)", "dot"),
        ]
        for r_val, r_lbl, r_col, r_dash in ring_styles:
            rx = bx + r_val * np.cos(theta)
            ry = by + r_val * np.sin(theta)
            mask = (
                (rx >= x_range[0]) & (rx <= x_range[1]) &
                (ry >= y_range[0]) & (ry <= y_range[1])
            )
            rx_m = np.where(mask, rx, np.nan)
            ry_m = np.where(mask, ry, np.nan)
            rz_m = np.full_like(rx_m, 0.006 * z_lim)
            fig.add_trace(
                go.Scatter3d(
                    x=rx_m,
                    y=ry_m,
                    z=rz_m,
                    mode="lines",
                    line=dict(color=r_col, width=2.2, dash=r_dash),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Subtle ring tag at -45 deg if inside bounds
            tx, ty = bx + r_val * 0.707, by - r_val * 0.707
            if x_range[0] < tx < x_range[1] and y_range[0] < ty < y_range[1]:
                fig.add_trace(
                    go.Scatter3d(
                        x=[tx],
                        y=[ty],
                        z=[0.015 * z_lim],
                        mode="text",
                        text=[r_lbl],
                        textfont=dict(size=9, color="rgba(71, 85, 105, 0.75)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Material cluster markers (filtered to avoid tiny stray dots on mountain flanks)
    hover_all = build_hover_texts(df, sigmas)
    mat_mask = df["Active_Pct"].abs().to_numpy(float) >= min_marker_active
    df_mat = df.loc[mat_mask]
    nz_mat = node_z[mat_mask]
    hover_mat = [h for h, keep in zip(hover_all, mat_mask) if keep]

    if len(df_mat) > 0:
        m_colors = [
            stress_marker_color(r.Stress_Pct, r.Is_Not_Held, r.Active_Pct, use_stress_halo)
            for r in df_mat.itertuples()
        ]
        m_sizes = np.clip(4.0 + 0.35 * df_mat["Active_Pct"].abs().to_numpy(float), 4.2, 8.2)
        z_off = np.where(nz_mat >= 0, 0.018 * z_lim, -0.018 * z_lim)

        fig.add_trace(
            go.Scatter3d(
                x=df_mat["X"],
                y=df_mat["Y"],
                z=nz_mat + z_off,
                mode="markers",
                marker=dict(size=m_sizes, color=m_colors, line=dict(color="white", width=1.2)),
                text=hover_mat,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    # De-collided peak labels
    if show_labels and top_n_labels > 0:
        lbl_df = select_decollided_labels(
            df, node_z, top_n_labels, min_dist=0.12 * span, min_active_pct=min_marker_active
        )
        if len(lbl_df) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=lbl_df["X"],
                    y=lbl_df["Y"],
                    z=lbl_df["_nz"] + np.where(lbl_df["_nz"] >= 0, 0.09 * z_lim, -0.10 * z_lim),
                    mode="text",
                    text=[lbl[:22] + ("…" if len(lbl) > 22 else "") for lbl in lbl_df["Node_Label"]],
                    textfont=dict(size=10, color="#111827"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # "3rd Bet" Cross-Benchmark Tension Bridges
    for b_idx, br in enumerate(bridges):
        t = np.linspace(0.0, 1.0, 25)
        ax = br["xi"] + t * (br["xj"] - br["xi"])
        ay = br["yi"] + t * (br["yj"] - br["yi"])
        arch_lift = 0.14 * z_lim * 4.0 * t * (1.0 - t)
        az = br["zi"] + t * (br["zj"] - br["zi"]) + arch_lift

        fig.add_trace(
            go.Scatter3d(
                x=ax,
                y=ay,
                z=az,
                mode="lines",
                line=dict(color="#d97706", width=3.5, dash="dash"),
                hoverinfo="skip",
                name="3rd Bet (Correlation Bridge)" if b_idx == 0 else None,
                showlegend=(b_idx == 0),
            )
        )
        # Midpoint tooltip marker
        mid = len(t) // 2
        fig.add_trace(
            go.Scatter3d(
                x=[ax[mid]],
                y=[ay[mid]],
                z=[az[mid]],
                mode="markers",
                marker=dict(size=4, color="#d97706", symbol="diamond-open"),
                hovertemplate=(
                    f"<b>⚠️ 3rd Bet (Cross-Benchmark Tension)</b><br>"
                    f"{br['label_i'][:28]} ({br['act_i']:+.1f}%)<br>"
                    f"↔ {br['label_j'][:28]} ({br['act_j']:+.1f}%)<br>"
                    f"Separation d_ij: {br['sep']:.3f} | Angle: {br['angle_deg']:.0f}°<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Benchmark & Portfolio Beacons (with anti-overlap text layout)
    if anchors is not None:
        bx, by = anchors["bench"]
        px, py = anchors["port"]
        xs = df["X"].to_numpy(dtype=float)
        ys = df["Y"].to_numpy(dtype=float)
        ws = df["Active_Pct"].to_numpy(dtype=float)

        surf_zs = eval_z_at_points(
            xs, ys, ws, sigmas, np.array([bx, px]), np.array([by, py]), blend_mode
        )
        z_surf_b, z_surf_p = float(surf_zs[0]), float(surf_zs[1])
        z_beacon_b = z_lim * 0.82
        z_beacon_p = z_lim * 0.88 if np.hypot(px - bx, py - by) < 0.06 * span else z_beacon_b
        tilt_dist = float(np.hypot(px - bx, py - by))

        fig.add_trace(
            go.Scatter3d(
                x=[bx, bx, None, px, px],
                y=[by, by, None, py, py],
                z=[z_surf_b, z_beacon_b, None, z_surf_p, z_beacon_p],
                mode="lines",
                line=dict(color="#4b5563", width=2.2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if tilt_dist > 1e-4:
            fig.add_trace(
                go.Scatter3d(
                    x=[bx, px],
                    y=[by, py],
                    z=[z_beacon_b, z_beacon_p],
                    mode="lines",
                    line=dict(color="#111827", width=3.5, dash="dash"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        tight_overlap = tilt_dist < 0.06 * span
        fig.add_trace(
            go.Scatter3d(
                x=[bx],
                y=[by],
                z=[z_beacon_b],
                mode="markers" if tight_overlap else "markers+text",
                marker=dict(
                    size=8, symbol="diamond", color="#2563eb",
                    line=dict(color="white", width=1.5),
                ),
                text=["Benchmark"],
                textposition="bottom left",
                textfont=dict(size=10, color="#1e40af"),
                hovertemplate=(
                    f"<b>Benchmark</b><br>"
                    f"X: {bx:+.4f}, Y: {by:+.4f}<br>"
                    f"Underlying Terrain Z: {z_surf_b:+.2f}%<extra></extra>"
                ),
                name="Benchmark",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[px],
                y=[py],
                z=[z_beacon_p],
                mode="markers+text",
                marker=dict(
                    size=7, symbol="square", color="#7c3aed",
                    line=dict(color="white", width=1.5),
                ),
                text=["Port ≈ Bench" if tight_overlap else "Portfolio"],
                textposition="top right",
                textfont=dict(size=10, color="#5b21b6"),
                hovertemplate=(
                    f"<b>Portfolio</b><br>"
                    f"X: {px:+.4f}, Y: {py:+.4f}<br>"
                    f"Net Tilt ||P - B||: {tilt_dist:.4f}<br>"
                    f"Underlying Terrain Z: {z_surf_p:+.2f}%<extra></extra>"
                ),
                name="Portfolio",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16)),
        height=590,
        margin=dict(l=8, r=18, t=42, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.96, xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.8)", font=dict(size=11),
        ),
        scene=dict(
            xaxis=dict(title="MDS X", range=list(x_range), dtick=dtick_xy, tickformat=".1f"),
            yaxis=dict(title="MDS Y", range=list(y_range), dtick=dtick_xy, tickformat=".1f"),
            zaxis=dict(title="Active Wt (%)", range=[-z_lim, z_lim]),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.62),
            camera=camera_cfg,
        ),
    )
    return fig


# ---------------------------------------------------------
# 2D METRIC-TRUE TOP-DOWN TOPOGRAPHIC MAP
# ---------------------------------------------------------
def make_2d_topomap_figure(
    title: str,
    df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    Z: np.ndarray,
    StressField: np.ndarray,
    node_z: np.ndarray,
    sigmas: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_lim: float,
    color_mode: str,
    stress_cap: float,
    show_labels: bool,
    top_n_labels: int,
    min_marker_active: float,
    use_stress_halo: bool,
    anchors: dict[str, tuple[float, float]] | None,
    show_rings: bool,
    ring_radii: tuple[float, float, float],
    bridges: list[dict],
) -> go.Figure:
    fig = go.Figure()
    span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])

    if color_mode == "Over / Underweight (Active %)":
        z_plot = Z
        colorscale = ACTIVE_COLORSCALE
        zmin, zmax = -z_lim, z_lim
        cb_title = "Active Wt (%)"
    else:
        z_plot = np.clip(StressField, 0.0, stress_cap)
        colorscale = "YlOrRd"
        zmin, zmax = 0.0, stress_cap
        cb_title = "Stress (%)"

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_plot,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(color="rgba(50,50,50,0.20)", width=0.8),
            colorbar=dict(title=cb_title, len=0.72, thickness=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Metric-true concentric Benchmark distance rings
    if show_rings and anchors is not None:
        bx, by = anchors["bench"]
        theta = np.linspace(0, 2 * np.pi, 128)
        for r_val, r_lbl, r_col, r_dash in [
            (ring_radii[0], "Core", "rgba(37, 99, 235, 0.60)", "dash"),
            (ring_radii[1], "Mid", "rgba(71, 85, 105, 0.45)", "dot"),
            (ring_radii[2], "Outer", "rgba(100, 116, 139, 0.35)", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=bx + r_val * np.cos(theta),
                    y=by + r_val * np.sin(theta),
                    mode="lines",
                    line=dict(color=r_col, width=1.6, dash=r_dash),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            tx, ty = bx + r_val * 0.707, by - r_val * 0.707
            if x_range[0] < tx < x_range[1] and y_range[0] < ty < y_range[1]:
                fig.add_annotation(
                    x=tx, y=ty, text=r_lbl, showarrow=False,
                    font=dict(size=9, color="#475569"),
                    bgcolor="rgba(255,255,255,0.65)",
                )

    # "3rd Bet" Tension Bridges
    for b_idx, br in enumerate(bridges):
        fig.add_trace(
            go.Scatter(
                x=[br["xi"], br["xj"]],
                y=[br["yi"], br["yj"]],
                mode="lines",
                line=dict(color="#d97706", width=2.8, dash="dash"),
                hoverinfo="skip",
                name="3rd Bet Bridge" if b_idx == 0 else None,
                showlegend=(b_idx == 0),
            )
        )
        mx, my = 0.5 * (br["xi"] + br["xj"]), 0.5 * (br["yi"] + br["yj"])
        fig.add_trace(
            go.Scatter(
                x=[mx],
                y=[my],
                mode="markers",
                marker=dict(size=7, color="#d97706", symbol="diamond-open", line=dict(width=2)),
                hovertemplate=(
                    f"<b>⚠️ 3rd Bet Bridge</b><br>"
                    f"{br['label_i'][:28]} ({br['act_i']:+.1f}%)<br>"
                    f"↔ {br['label_j'][:28]} ({br['act_j']:+.1f}%)<br>"
                    f"Separation d_ij: {br['sep']:.3f} | Angle: {br['angle_deg']:.0f}°<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Material cluster markers
    hover_all = build_hover_texts(df, sigmas)
    mat_mask = df["Active_Pct"].abs().to_numpy(float) >= min_marker_active
    df_mat = df.loc[mat_mask]
    hover_mat = [h for h, keep in zip(hover_all, mat_mask) if keep]

    if len(df_mat) > 0:
        m_colors = [
            stress_marker_color(r.Stress_Pct, r.Is_Not_Held, r.Active_Pct, use_stress_halo)
            for r in df_mat.itertuples()
        ]
        m_sizes = np.clip(6.5 + 0.75 * df_mat["Active_Pct"].abs().to_numpy(float), 7.0, 15.0)
        fig.add_trace(
            go.Scatter(
                x=df_mat["X"],
                y=df_mat["Y"],
                mode="markers",
                marker=dict(size=m_sizes, color=m_colors, line=dict(color="white", width=1.4)),
                text=hover_mat,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    # De-collided labels
    if show_labels and top_n_labels > 0:
        lbl_df = select_decollided_labels(
            df, node_z, top_n_labels, min_dist=0.12 * span, min_active_pct=min_marker_active
        )
        for row in lbl_df.itertuples():
            short = row.Node_Label[:22] + ("…" if len(row.Node_Label) > 22 else "")
            fig.add_annotation(
                x=row.X,
                y=row.Y,
                text=f"{short} ({row.Active_Pct:+.1f}%)",
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.8,
                arrowcolor="rgba(55,65,81,0.5)",
                ax=0,
                ay=-20 if row.Active_Pct >= 0 else 20,
                font=dict(size=9.5, color="#111827"),
                bgcolor="rgba(255,255,255,0.78)",
                borderpad=2,
            )

    # Benchmark & Portfolio Markers + Tilt Arrow
    if anchors is not None:
        bx, by = anchors["bench"]
        px, py = anchors["port"]
        tilt_dist = float(np.hypot(px - bx, py - by))

        if tilt_dist > 1e-4:
            fig.add_trace(
                go.Scatter(
                    x=[bx, px],
                    y=[by, py],
                    mode="lines",
                    line=dict(color="#111827", width=2.4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[bx],
                y=[by],
                mode="markers",
                marker=dict(
                    size=13, symbol="diamond", color="#2563eb",
                    line=dict(color="white", width=1.8),
                ),
                name="Benchmark",
                hovertemplate=f"<b>Benchmark</b><br>X: {bx:+.4f}, Y: {by:+.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers",
                marker=dict(
                    size=11, symbol="square", color="#7c3aed",
                    line=dict(color="white", width=1.8),
                ),
                name="Portfolio",
                hovertemplate=(
                    f"<b>Portfolio</b><br>X: {px:+.4f}, Y: {py:+.4f}<br>"
                    f"Net Tilt ||P - B||: {tilt_dist:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15)),
        height=490,
        margin=dict(l=45, r=15, t=42, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=0.99,
            font=dict(size=10),
        ),
        xaxis=dict(title="MDS X", range=list(x_range), zeroline=False, showgrid=False),
        yaxis=dict(
            title="MDS Y", range=list(y_range), zeroline=False, showgrid=False,
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#eaeef3",
    )
    return fig


# ---------------------------------------------------------
# CORE / MID / OUTER ACTIVE-RISK STRIP
# ---------------------------------------------------------
def make_ring_strip_figure(breakdown: dict[str, float], label: str) -> go.Figure:
    fig = go.Figure()
    colors = {"Core": "#2563eb", "Mid": "#64748b", "Outer Tail": "#dc2626"}
    for zone in ["Core", "Mid", "Outer Tail"]:
        val = breakdown[zone]
        fig.add_trace(
            go.Bar(
                y=[label],
                x=[val],
                name=zone,
                orientation="h",
                marker=dict(color=colors[zone]),
                text=f"{zone}: {val:.0f}%" if val >= 14 else (f"{val:.0f}%" if val >= 6 else ""),
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=11),
                hovertemplate=f"<b>{zone}</b>: {val:.1f}% of Active Share<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        height=68,
        margin=dict(l=6, r=6, t=22, b=4),
        title=dict(
            text="Active Share by Distance from Benchmark (Core vs. Mid vs. Outer Tail)",
            font=dict(size=11.5, color="#475569"),
            x=0.0,
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_synthetic_1y_ago(df_today: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df_old = df_today.copy()
    df_old["X"] = df_old["X"] + rng.normal(0, 0.015, len(df_old))
    df_old["Y"] = df_old["Y"] + rng.normal(0, 0.015, len(df_old))

    prefix = "Local" if "Local_Active_Weight" in df_old.columns else "Global"
    w_col, p_col, b_col = f"{prefix}_Active_Weight", f"{prefix}_Port_Weight", f"{prefix}_Bench_Weight"
    delta_w = rng.normal(0, 0.018, len(df_old))

    if p_col in df_old.columns and b_col in df_old.columns:
        p_new = np.clip(df_old[p_col].to_numpy(float) + delta_w, 0.0, None)
        if p_new.sum() > 0:
            p_new = p_new / p_new.sum()
        df_old[p_col] = p_new
        df_old[w_col] = df_old[p_col] - df_old[b_col]
    else:
        df_old[w_col] = df_old[w_col] + delta_w

    if "Child_Stress_%" in df_old.columns:
        s_vals, not_held = parse_stress_series(df_old["Child_Stress_%"])
        s_new = np.clip(s_vals + rng.normal(0, 12.0, len(s_vals)), 2.0, 95.0)
        df_old["Child_Stress_%"] = [
            "Not Held" if nh else f"{sv:.1f}%" for sv, nh in zip(s_new, not_held)
        ]
    return df_old


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("1. Upload Cluster Reports")
file_old = st.sidebar.file_uploader(
    "Snapshot A: 1 Year Ago (.csv, .xlsx, .parquet)",
    type=["csv", "xlsx", "parquet"],
    key="upload_old",
)
file_new = st.sidebar.file_uploader(
    "Snapshot B: Today (.csv, .xlsx, .parquet)",
    type=["csv", "xlsx", "parquet"],
    key="upload_new",
)

demo_mode = False
if file_old is None and file_new is not None:
    demo_mode = st.sidebar.checkbox("Simulate '1 Year Ago' from Today (Demo)", value=False)

st.sidebar.header("2. Layout & Presentation")
view_mode = st.sidebar.radio(
    "Chart Views",
    ["3D Terrain + 2D Top-Down Map", "3D Terrain Only", "2D Top-Down Map Only"],
    index=0,
)
camera_name = st.sidebar.selectbox(
    "3D Synced Camera Preset",
    list(CAMERA_PRESETS.keys()),
    index=0,
    disabled=(view_mode == "2D Top-Down Map Only"),
)
slide_mode = st.sidebar.checkbox(
    "Slide / Presentation Mode (Minimal Chrome)",
    value=False,
    help="Hides helper captions and secondary tables for clean screenshots.",
)

st.sidebar.header("3. Surface & Stress Geometry")
weight_col = st.sidebar.selectbox(
    "Elevation Metric (Z-axis)", ["Local_Active_Weight", "Global_Active_Weight"], index=0
)
blend_mode = st.sidebar.radio(
    "Overlapping Peak Behavior", ["Peak-Preserving Envelope", "Additive Sum"], index=0
)
stress_cap = st.sidebar.slider("Stress Clip Ceiling (%)", 20.0, 150.0, 80.0, 5.0)
use_log_stress = st.sidebar.checkbox("Log-scale Stress Width", value=True)
log_k = st.sidebar.slider(
    "Log Curvature (k)", 0.02, 0.50, 0.12, 0.02, disabled=not use_log_stress
)
sigma_min_frac = st.sidebar.slider("Min Footprint σ_min (% span)", 0.5, 5.0, 1.8, 0.25) / 100.0
sigma_max_frac = st.sidebar.slider("Max Footprint σ_max (% span)", 1.5, 10.0, 4.2, 0.25) / 100.0
if sigma_max_frac < sigma_min_frac:
    sigma_max_frac = sigma_min_frac

st.sidebar.header("4. Risk Overlays & Decluttering")
do_procrustes = st.sidebar.checkbox("Align 1Y Ago to Today (Procrustes)", value=True)
show_beacons = st.sidebar.checkbox("Show Benchmark & Portfolio Beacons", value=True)
center_on_bench = st.sidebar.checkbox(
    "Center Origin (0, 0) on Benchmark", value=False, disabled=not show_beacons
)
show_rings = st.sidebar.checkbox(
    "Show Concentric Benchmark Rings + Strip", value=True, disabled=not show_beacons
)
core_ring_frac = st.sidebar.slider(
    "Core Ring Radius (% of max distance)", 15, 50, 32, 2, disabled=not (show_beacons and show_rings)
) / 100.0
show_bridges = st.sidebar.checkbox(
    "Show '3rd Bet' Tension Bridges",
    value=True,
    disabled=not show_beacons,
    help="Connects major overweight peaks straddling opposite sides of the Benchmark.",
)
min_bridge_ow = st.sidebar.slider(
    "Min Overweight for Bridge (%)", 0.5, 5.0, 1.5, 0.25, disabled=not (show_beacons and show_bridges)
)
max_bridges = st.sidebar.slider(
    "Max Tension Bridges", 1, 5, 3, 1, disabled=not (show_beacons and show_bridges)
)
use_stress_halo = st.sidebar.checkbox(
    "Peak Markers = Stress Traffic Light",
    value=True,
    help="Green <25%, Amber 25–50%, Red >50%, Slate = Not Held.",
)
min_marker_active = st.sidebar.slider(
    "Hide Cluster Markers Below |Active Wt| (%)", 0.0, 3.0, 0.5, 0.1
)
color_mode = st.sidebar.radio(
    "Surface Color Encoding", ["Over / Underweight (Active %)", "Local Stress Intensity (%)"]
)
show_sea_level = st.sidebar.checkbox("Show Sea-Level Plane (Z = 0%)", value=False)
show_labels = st.sidebar.checkbox("Show De-Collided Cluster Labels", value=True)
top_n_labels = st.sidebar.slider("Max Labels", 0, 15, 7, disabled=not show_labels)
grid_res = st.sidebar.select_slider("Grid Resolution", options=[70, 90, 110, 130, 160], value=110)


# ---------------------------------------------------------
# HEADER & FILE LOADING
# ---------------------------------------------------------
if not slide_mode:
    st.title("🏔️ Portfolio Risk Topography: 1 Year Ago vs. Today")
    st.caption(
        "Mountains = Overweight · Valleys = Underweight · Width = Intra-Cluster Stress · "
        "Markers = Stress Traffic Light (🟢 <25% · 🟠 25–50% · 🔴 >50% · ⚪ Not Held)"
    )
else:
    st.subheader("Portfolio Risk Topography — 1 Year Ago vs. Today")

raw_new = load_cluster_file(file_new, "Today")
raw_old = load_cluster_file(file_old, "1Y_Ago")

if raw_new is not None and demo_mode and raw_old is None:
    raw_old = make_synthetic_1y_ago(raw_new)

if raw_old is None or raw_new is None:
    st.info(
        "👈 Upload both **1 Year Ago** and **Today** `clusters risk` files in the sidebar "
        "(or upload Today's file and enable the demo toggle to preview)."
    )
    st.stop()

df_old = prepare_dataframe(raw_old, weight_col)
df_new = prepare_dataframe(raw_new, weight_col)

rmsd = np.nan
if do_procrustes:
    df_old, rmsd = align_procrustes(df_old, df_new)

anchors_old = compute_anchors(df_old) if show_beacons else None
anchors_new = compute_anchors(df_new) if show_beacons else None

if show_beacons and center_on_bench:
    if anchors_old is not None:
        df_old["X"] -= anchors_old["bench"][0]
        df_old["Y"] -= anchors_old["bench"][1]
        anchors_old = compute_anchors(df_old)
    if anchors_new is not None:
        df_new["X"] -= anchors_new["bench"][0]
        df_new["Y"] -= anchors_new["bench"][1]
        anchors_new = compute_anchors(df_new)

# Shared X, Y bounding box across BOTH periods
all_x = np.concatenate([df_old["X"].to_numpy(), df_new["X"].to_numpy()])
all_y = np.concatenate([df_old["Y"].to_numpy(), df_new["Y"].to_numpy()])
raw_span = max(float(all_x.max() - all_x.min()), float(all_y.max() - all_y.min()), 1e-3)
pad = 2.6 * sigma_max_frac * raw_span

x_range = (float(all_x.min() - pad), float(all_x.max() + pad))
y_range = (float(all_y.min() - pad), float(all_y.max() + pad))
span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])

x_grid = np.linspace(x_range[0], x_range[1], grid_res)
y_grid = np.linspace(y_range[0], y_range[1], grid_res)
gx, gy = np.meshgrid(x_grid, y_grid)

Z_old, S_old, nz_old, sig_old = evaluate_surface(
    df_old, gx, gy, span, sigma_min_frac, sigma_max_frac, stress_cap, use_log_stress, log_k, blend_mode
)
Z_new, S_new, nz_new, sig_new = evaluate_surface(
    df_new, gx, gy, span, sigma_min_frac, sigma_max_frac, stress_cap, use_log_stress, log_k, blend_mode
)
z_lim = float(max(np.abs(Z_old).max(), np.abs(Z_new).max(), 1.0)) * 1.15

# Shared concentric Benchmark ring radii across BOTH snapshots
max_r_bench = 0.25
if anchors_old is not None and anchors_new is not None:
    d_o = np.hypot(df_old["X"] - anchors_old["bench"][0], df_old["Y"] - anchors_old["bench"][1]).max()
    d_n = np.hypot(df_new["X"] - anchors_new["bench"][0], df_new["Y"] - anchors_new["bench"][1]).max()
    max_r_bench = max(float(d_o), float(d_n), 0.1)

r_core = core_ring_frac * max_r_bench
r_mid = min(2.0 * r_core, 0.72 * max_r_bench)
r_outer = min(3.0 * r_core, 0.98 * max_r_bench)
ring_radii = (r_core, r_mid, r_outer)

# Geometric metrics & straddle bridges
m_old = compute_geometric_metrics(df_old, anchors_old)
m_new = compute_geometric_metrics(df_new, anchors_new)

bridges_old = (
    find_straddle_bridges(df_old, nz_old, anchors_old, min_bridge_ow, max_bridges)
    if (show_beacons and show_bridges)
    else []
)
bridges_new = (
    find_straddle_bridges(df_new, nz_new, anchors_new, min_bridge_ow, max_bridges)
    if (show_beacons and show_bridges)
    else []
)

# ---------------------------------------------------------
# TOP KPI BAR (HONEST TWO-MOMENT DIAGNOSIS)
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "Active Share (1Y → Now)",
    f"{m_new['active_share']:.1f}%",
    delta=f"{m_new['active_share'] - m_old['active_share']:+.1f}%",
    delta_color="off",
)

if pd.notna(m_new["net_tilt"]) and pd.notna(m_old["net_tilt"]):
    k2.metric(
        "Net Tilt ||P − B|| (1st Moment)",
        f"{m_new['net_tilt']:.4f}",
        delta=f"{m_new['net_tilt'] - m_old['net_tilt']:+.4f}",
        delta_color="inverse",
        help="Directional center-of-mass drift. Can approach 0 even under a barbell straddle.",
    )
    k3.metric(
        "Geometric Tension (2nd Moment)",
        f"{m_new['tension_rms']:.4f}",
        delta=f"{m_new['tension_rms'] - m_old['tension_rms']:+.4f}",
        delta_color="inverse",
        help="Active-weighted RMS distance from Benchmark. Exposes barbell / counter-bet tension.",
    )
    k4.metric(
        "Active Work Σ|aᵢ|·dᵢ",
        f"{m_new['active_work']:.4f}",
        delta=f"{m_new['active_work'] - m_old['active_work']:+.4f}",
        delta_color="inverse",
        help="Total active transport work around Benchmark (mirrors W1 distance).",
    )
else:
    k2.metric("Net Tilt ||P − B||", "N/A")
    k3.metric("Geometric Tension", "N/A")
    k4.metric("Active Work", "N/A")

held_old = df_old.loc[~df_old["Is_Not_Held"], "Stress_Pct"]
held_new = df_new.loc[~df_new["Is_Not_Held"], "Stress_Pct"]
avg_s_old = float(held_old.mean()) if len(held_old) else 0.0
avg_s_new = float(held_new.mean()) if len(held_new) else 0.0
k5.metric(
    "Mean Held Stress",
    f"{avg_s_new:.1f}%",
    delta=f"{avg_s_new - avg_s_old:+.1f}%",
    delta_color="inverse",
)

if not slide_mode and pd.notna(rmsd):
    st.caption(
        f"Procrustes Alignment RMSD: `{rmsd:.4f}` · "
        f"Shared Rings: Core ≤ `{r_core:.2f}` | Mid ≤ `{r_mid:.2f}` | Outer > `{r_mid:.2f}` · "
        f"Tip: click the 📷 icon on any chart for a 3× high-res PNG slide export."
    )

# ---------------------------------------------------------
# 3D TERRAIN ROW
# ---------------------------------------------------------
if view_mode in ("3D Terrain + 2D Top-Down Map", "3D Terrain Only"):
    cam_cfg = CAMERA_PRESETS[camera_name]
    c3d_l, c3d_r = st.columns(2)

    with c3d_l:
        fig_3d_old = make_3d_landscape_figure(
            "📅 1 Year Ago",
            df_old, x_grid, y_grid, Z_old, S_old, nz_old, sig_old,
            x_range, y_range, z_lim, color_mode, stress_cap,
            show_sea_level, show_labels, top_n_labels, min_marker_active,
            use_stress_halo, anchors_old, show_rings, ring_radii, bridges_old,
            blend_mode, cam_cfg,
        )
        st.plotly_chart(fig_3d_old, use_container_width=True, config=plotly_export_config("Risk_3D_1Y_Ago"))

    with c3d_r:
        fig_3d_new = make_3d_landscape_figure(
            "📍 Today",
            df_new, x_grid, y_grid, Z_new, S_new, nz_new, sig_new,
            x_range, y_range, z_lim, color_mode, stress_cap,
            show_sea_level, show_labels, top_n_labels, min_marker_active,
            use_stress_halo, anchors_new, show_rings, ring_radii, bridges_new,
            blend_mode, cam_cfg,
        )
        st.plotly_chart(fig_3d_new, use_container_width=True, config=plotly_export_config("Risk_3D_Today"))

# ---------------------------------------------------------
# CORE / MID / OUTER ACTIVE-RISK STRIP
# ---------------------------------------------------------
if show_beacons and show_rings:
    brk_old = compute_ring_breakdown(df_old, anchors_old, r_core, r_mid)
    brk_new = compute_ring_breakdown(df_new, anchors_new, r_core, r_mid)
    if brk_old is not None and brk_new is not None:
        sb_l, sb_r = st.columns(2)
        with sb_l:
            st.plotly_chart(
                make_ring_strip_figure(brk_old, "1Y Ago"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with sb_r:
            st.plotly_chart(
                make_ring_strip_figure(brk_new, "Today"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

# ---------------------------------------------------------
# 2D METRIC-TRUE TOP-DOWN FACTOR MAP ROW
# ---------------------------------------------------------
if view_mode in ("3D Terrain + 2D Top-Down Map", "2D Top-Down Map Only"):
    c2d_l, c2d_r = st.columns(2)

    with c2d_l:
        fig_2d_old = make_2d_topomap_figure(
            "🗺️ 1 Year Ago — Metric-True Top-Down Factor Map",
            df_old, x_grid, y_grid, Z_old, S_old, nz_old, sig_old,
            x_range, y_range, z_lim, color_mode, stress_cap,
            show_labels, top_n_labels, min_marker_active,
            use_stress_halo, anchors_old, show_rings, ring_radii, bridges_old,
        )
        st.plotly_chart(fig_2d_old, use_container_width=True, config=plotly_export_config("Risk_2D_1Y_Ago"))

    with c2d_r:
        fig_2d_new = make_2d_topomap_figure(
            "🗺️ Today — Metric-True Top-Down Factor Map",
            df_new, x_grid, y_grid, Z_new, S_new, nz_new, sig_new,
            x_range, y_range, z_lim, color_mode, stress_cap,
            show_labels, top_n_labels, min_marker_active,
            use_stress_halo, anchors_new, show_rings, ring_radii, bridges_new,
        )
        st.plotly_chart(fig_2d_new, use_container_width=True, config=plotly_export_config("Risk_2D_Today"))

# ---------------------------------------------------------
# SECONDARY TABLES (HIDDEN IN SLIDE MODE)
# ---------------------------------------------------------
if not slide_mode:
    if show_beacons and show_bridges and (bridges_old or bridges_new):
        st.markdown("---")
        st.subheader("⚠️ Active '3rd Bet' Cross-Benchmark Tension Pairs")
        bt_l, bt_r = st.columns(2)
        for col_obj, b_title, b_list in [
            (bt_l, "1 Year Ago — Top Straddling Overweights", bridges_old),
            (bt_r, "Today — Top Straddling Overweights", bridges_new),
        ]:
            with col_obj:
                st.markdown(f"**{b_title}**")
                if b_list:
                    b_df = pd.DataFrame(
                        [
                            {
                                "Leg A": f"{b['label_i']} ({b['act_i']:+.1f}%)",
                                "Leg B": f"{b['label_j']} ({b['act_j']:+.1f}%)",
                                "Separation d_ij": b["sep"],
                                "Angle (°)": b["angle_deg"],
                                "Tension Score": b["score"],
                            }
                            for b in b_list
                        ]
                    )
                    st.dataframe(
                        b_df.style.format(
                            {"Separation d_ij": "{:.3f}", "Angle (°)": "{:.0f}°", "Tension Score": "{:.3f}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No material opposite-flank overweight pairs detected.")

    st.markdown("---")
    st.subheader("🔄 Cluster Migration & Stress Shift (Today vs. 1 Year Ago)")

    comp = pd.merge(
        df_old[["Node_ID", "Node_Label", "Active_Pct", "Stress_Pct", "X", "Y"]],
        df_new[["Node_ID", "Node_Label", "Active_Pct", "Stress_Pct", "X", "Y"]],
        on="Node_ID",
        how="outer",
        suffixes=("_1Y", "_Now"),
    )
    comp["Cluster"] = comp["Node_Label_Now"].fillna(comp["Node_Label_1Y"])
    comp["Δ Active Wt (%)"] = comp["Active_Pct_Now"].fillna(0.0) - comp["Active_Pct_1Y"].fillna(0.0)
    comp["Δ Stress (pp)"] = comp["Stress_Pct_Now"].fillna(0.0) - comp["Stress_Pct_1Y"].fillna(0.0)
    comp["Map Shift (ΔXY)"] = np.sqrt(
        (comp["X_Now"] - comp["X_1Y"]) ** 2 + (comp["Y_Now"] - comp["Y_1Y"]) ** 2
    )

    display_tbl = comp[
        [
            "Cluster",
            "Active_Pct_1Y",
            "Active_Pct_Now",
            "Δ Active Wt (%)",
            "Stress_Pct_1Y",
            "Stress_Pct_Now",
            "Δ Stress (pp)",
            "Map Shift (ΔXY)",
        ]
    ].rename(
        columns={
            "Active_Pct_1Y": "Active Wt 1Y (%)",
            "Active_Pct_Now": "Active Wt Now (%)",
            "Stress_Pct_1Y": "Stress 1Y (%)",
            "Stress_Pct_Now": "Stress Now (%)",
        }
    ).sort_values("Δ Active Wt (%)", key=lambda s: s.abs(), ascending=False)

    st.dataframe(
        display_tbl.set_index("Cluster").style.format(
            {
                "Active Wt 1Y (%)": "{:+.2f}%",
                "Active Wt Now (%)": "{:+.2f}%",
                "Δ Active Wt (%)": "{:+.2f}%",
                "Stress 1Y (%)": "{:.1f}%",
                "Stress Now (%)": "{:.1f}%",
                "Δ Stress (pp)": "{:+.1f}",
                "Map Shift (ΔXY)": "{:.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        height=380,
    )