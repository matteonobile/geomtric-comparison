# -*- coding: utf-8 -*-
"""
app_comparison.py
=================
Side-by-side 3D Topographic Risk Landscape comparing Portfolio Exposure
(1 Year Ago vs. Today) from `clusters risk` exports.

Visual encoding:
- (X, Y) : Cluster coordinates from PCA / MDS (with optional Procrustes alignment)
- Z      : Active Weight (Mountains = Overweight, Valleys = Underweight, Z=0 = Neutral)
- Width  : Intra-cluster Stress (clipped & log-scaled, with safe baseline for 'Not Held')
- Beacons: Benchmark & Portfolio 2D weight-weighted barycenters + active tilt vector
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="3D Risk Topography Comparator", layout="wide")
st.title("🏔️ Portfolio Risk Topography: 1 Year Ago vs. Today")
st.caption(
    "Mountains = Overweight · Valleys = Underweight · "
    "Footprint Width = Intra-Cluster Stress (clipped & log-scaled) · Locked X-Y-Z Bounds"
)


# ---------------------------------------------------------
# DATA LOADING & PARSING HELPERS
# ---------------------------------------------------------
def load_cluster_file(uploaded_file, key_prefix: str) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    buf = io.BytesIO(raw_bytes)

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
    """
    Parses Child_Stress_% into (numeric_pct_0_to_100, is_not_held_bool).
    Handles '10.2%', 'Not Held', NaN, or already-numeric columns.
    """
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
                s_clean = s.replace("%", "").strip()
                try:
                    vals[idx] = float(s_clean)
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

    # Resolve matching Benchmark and Portfolio weight columns
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
    out = out.dropna(subset=["X", "Y"]).reset_index(drop=True)
    return out


def align_procrustes(df_old: pd.DataFrame, df_new: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Rigidly aligns df_old (X, Y) onto df_new (X, Y) using shared Node_IDs
    to eliminate arbitrary PCA/MDS rotations or sign flips.
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

    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_c = P - p_mean
    Q_c = Q - q_mean

    U, _, Vt = np.linalg.svd(P_c.T @ Q_c)
    R = U @ Vt

    P_aligned = P_c @ R + q_mean
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))

    aligned_df = df_old.copy()
    coords = aligned_df[["X", "Y"]].to_numpy(dtype=float)
    coords_aligned = (coords - p_mean) @ R + q_mean
    aligned_df["X"] = coords_aligned[:, 0]
    aligned_df["Y"] = coords_aligned[:, 1]
    return aligned_df, rmsd


def compute_barycenters(df: pd.DataFrame) -> dict[str, tuple[float, float]] | None:
    """
    Computes Option A linear 2D barycenters for Benchmark and Portfolio
    from cluster weights and (X, Y) coordinates.
    (Can later be swapped for exact exported MDS points if added to CSV.)
    """
    if df["Bench_W"].isna().all() or df["Port_W"].isna().all():
        return None

    bw = df["Bench_W"].to_numpy(dtype=float)
    pw = df["Port_W"].to_numpy(dtype=float)
    xs = df["X"].to_numpy(dtype=float)
    ys = df["Y"].to_numpy(dtype=float)

    b_sum = float(np.sum(bw))
    p_sum = float(np.sum(pw))
    if b_sum <= 1e-12 or p_sum <= 1e-12:
        return None

    bw_norm = bw / b_sum
    pw_norm = pw / p_sum

    bx, by = float(np.sum(bw_norm * xs)), float(np.sum(bw_norm * ys))
    px, py = float(np.sum(pw_norm * xs)), float(np.sum(pw_norm * ys))
    return {"bench": (bx, by), "port": (px, py)}


# ---------------------------------------------------------
# SURFACE EVALUATION
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

    sigma_min = sigma_min_frac * span
    sigma_max = sigma_max_frac * span
    return sigma_min + (sigma_max - sigma_min) * u


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


def make_landscape_figure(
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
    barycenters: dict[str, tuple[float, float]] | None,
    blend_mode: str,
) -> go.Figure:
    fig = go.Figure()

    if color_mode == "Over / Underweight (Active %)":
        surface_color = Z
        colorscale = [
            [0.00, "#7f0000"],
            [0.32, "#d7301f"],
            [0.46, "#fc8d59"],
            [0.50, "#eaeef3"],
            [0.54, "#a1d99b"],
            [0.68, "#31a354"],
            [1.00, "#00441b"],
        ]
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
            colorbar=dict(title=cb_title, len=0.6, thickness=14, x=1.02),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=False,
                    color="rgba(40, 40, 40, 0.28)",
                    width=1,
                    project_z=False,
                )
            ),
            lighting=dict(ambient=0.65, diffuse=0.75, specular=0.22, roughness=0.55),
            hoverinfo="skip",
            name="Surface",
            showlegend=False,
        )
    )

    if show_sea_level:
        fig.add_trace(
            go.Surface(
                x=[x_range[0], x_range[1]],
                y=[y_range[0], y_range[1]],
                z=np.zeros((2, 2)),
                colorscale=[[0, "rgba(120, 140, 160, 0.25)"], [1, "rgba(120, 140, 160, 0.25)"]],
                showscale=False,
                opacity=0.25,
                hoverinfo="skip",
                name="Neutral (0%)",
                showlegend=False,
            )
        )

    # Cluster peak/valley markers + hover info
    hover_texts = []
    for row, sig in zip(df.itertuples(), sigmas):
        stress_str = "Not Held (0%)" if row.Is_Not_Held else f"{row.Stress_Pct:.1f}%"
        bw_str = f"{row.Bench_W:.2%}" if pd.notna(row.Bench_W) else "—"
        pw_str = f"{row.Port_W:.2%}" if pd.notna(row.Port_W) else "—"
        hover_texts.append(
            f"<b>{row.Node_Label}</b><br>"
            f"Active Weight: {row.Active_Pct:+.2f}%<br>"
            f"Bench / Port: {bw_str} / {pw_str}<br>"
            f"Child Stress: {stress_str}<br>"
            f"Footprint σ: {sig:.3f}"
        )

    marker_colors = ["#00441b" if v >= 0 else "#67000d" for v in df["Active_Pct"]]
    z_offset = np.where(node_z >= 0, 0.015 * z_lim, -0.015 * z_lim)

    fig.add_trace(
        go.Scatter3d(
            x=df["X"],
            y=df["Y"],
            z=node_z + z_offset,
            mode="markers",
            marker=dict(size=4.2, color=marker_colors, line=dict(color="white", width=1)),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    if show_labels and top_n_labels > 0:
        top_df = df.assign(_abs=df["Active_Pct"].abs(), _nz=node_z).nlargest(top_n_labels, "_abs")
        fig.add_trace(
            go.Scatter3d(
                x=top_df["X"],
                y=top_df["Y"],
                z=top_df["_nz"] + np.where(top_df["_nz"] >= 0, 0.08 * z_lim, -0.09 * z_lim),
                mode="text",
                text=[lbl[:24] + ("…" if len(lbl) > 24 else "") for lbl in top_df["Node_Label"]],
                textfont=dict(size=10, color="#111111"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Benchmark & Portfolio Beacons (Option A Barycenters)
    if barycenters is not None:
        bx, by = barycenters["bench"]
        px, py = barycenters["port"]
        xs = df["X"].to_numpy(dtype=float)
        ys = df["Y"].to_numpy(dtype=float)
        ws = df["Active_Pct"].to_numpy(dtype=float)

        surf_zs = eval_z_at_points(
            xs, ys, ws, sigmas, np.array([bx, px]), np.array([by, py]), blend_mode
        )
        z_surf_b, z_surf_p = float(surf_zs[0]), float(surf_zs[1])
        z_beacon = z_lim * 0.85
        tilt_dist = float(np.hypot(px - bx, py - by))

        # Drop-lines from beacons down to the terrain directly beneath them
        fig.add_trace(
            go.Scatter3d(
                x=[bx, bx, None, px, px],
                y=[by, by, None, py, py],
                z=[z_surf_b, z_beacon, None, z_surf_p, z_beacon],
                mode="lines",
                line=dict(color="#4b5563", width=2.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Active tilt vector connecting Benchmark -> Portfolio at beacon height
        fig.add_trace(
            go.Scatter3d(
                x=[bx, px],
                y=[by, py],
                z=[z_beacon, z_beacon],
                mode="lines",
                line=dict(color="#111827", width=4, dash="dash"),
                hoverinfo="skip",
                name="Active Tilt",
                showlegend=False,
            )
        )

        # Benchmark beacon
        fig.add_trace(
            go.Scatter3d(
                x=[bx],
                y=[by],
                z=[z_beacon],
                mode="markers+text",
                marker=dict(
                    size=8, symbol="diamond", color="#2563eb",
                    line=dict(color="white", width=1.5),
                ),
                text=["Benchmark"],
                textposition="top center",
                textfont=dict(size=11, color="#1e40af"),
                hovertemplate=(
                    f"<b>Benchmark (2D Barycenter)</b><br>"
                    f"X: {bx:+.4f}, Y: {by:+.4f}<br>"
                    f"Underlying Terrain Z: {z_surf_b:+.2f}%<extra></extra>"
                ),
                name="Benchmark",
            )
        )

        # Portfolio beacon
        fig.add_trace(
            go.Scatter3d(
                x=[px],
                y=[py],
                z=[z_beacon],
                mode="markers+text",
                marker=dict(
                    size=7, symbol="square", color="#7c3aed",
                    line=dict(color="white", width=1.5),
                ),
                text=["Portfolio"],
                textposition="top center",
                textfont=dict(size=11, color="#5b21b6"),
                hovertemplate=(
                    f"<b>Portfolio (2D Barycenter)</b><br>"
                    f"X: {px:+.4f}, Y: {py:+.4f}<br>"
                    f"Net Tilt ||P - B||: {tilt_dist:.4f}<br>"
                    f"Underlying Terrain Z: {z_surf_p:+.2f}%<extra></extra>"
                ),
                name="Portfolio",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16)),
        height=640,
        margin=dict(l=10, r=20, t=45, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.96, xanchor="right", x=0.98,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        scene=dict(
            xaxis=dict(title="PCA / MDS X", range=list(x_range)),
            yaxis=dict(title="PCA / MDS Y", range=list(y_range)),
            zaxis=dict(title="Active Weight (%)", range=[-z_lim, z_lim]),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.65),
            camera=dict(eye=dict(x=1.45, y=-1.55, z=1.05)),
        ),
    )
    return fig


def make_synthetic_1y_ago(df_today: pd.DataFrame) -> pd.DataFrame:
    """Creates a realistic synthetic '1 Year Ago' snapshot for quick testing."""
    rng = np.random.default_rng(42)
    df_old = df_today.copy()
    df_old["X"] = df_old["X"] + rng.normal(0, 0.012, len(df_old))
    df_old["Y"] = df_old["Y"] + rng.normal(0, 0.012, len(df_old))

    prefix = "Local" if "Local_Active_Weight" in df_old.columns else "Global"
    w_col = f"{prefix}_Active_Weight"
    p_col = f"{prefix}_Port_Weight"
    b_col = f"{prefix}_Bench_Weight"

    delta_w = rng.normal(0, 0.015, len(df_old))
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
    demo_mode = st.sidebar.checkbox(
        "Simulate '1 Year Ago' from Today (Demo)", value=False
    )

st.sidebar.header("2. Surface & Stress Geometry")
weight_col = st.sidebar.selectbox(
    "Elevation Metric (Z-axis)",
    ["Local_Active_Weight", "Global_Active_Weight"],
    index=0,
)
blend_mode = st.sidebar.radio(
    "Overlapping Peak Behavior",
    ["Peak-Preserving Envelope", "Additive Sum"],
    index=0,
    help="Envelope preserves each cluster's true Active Weight height when clusters sit close together.",
)
stress_cap = st.sidebar.slider(
    "Stress Clip Ceiling (%)", min_value=20.0, max_value=150.0, value=80.0, step=5.0
)
use_log_stress = st.sidebar.checkbox("Log-scale Stress Width", value=True)
log_k = st.sidebar.slider(
    "Log Curvature (k)", min_value=0.02, max_value=0.50, value=0.12, step=0.02,
    disabled=not use_log_stress,
    help="Higher k compresses high-stress differences and highlights low-to-mid stress expansion.",
)
sigma_min_frac = st.sidebar.slider(
    "Min Footprint Width σ_min (% of map span)",
    min_value=0.5, max_value=5.0, value=1.8, step=0.25,
    help="Baseline footprint for 0% stress / 'Not Held' clusters.",
) / 100.0
sigma_max_frac = st.sidebar.slider(
    "Max Footprint Width σ_max (% of map span)",
    min_value=1.5, max_value=10.0, value=4.2, step=0.25,
    help="Footprint width at the Stress Clip Ceiling.",
) / 100.0

if sigma_max_frac < sigma_min_frac:
    sigma_max_frac = sigma_min_frac

st.sidebar.header("3. Visual & Alignment Options")
do_procrustes = st.sidebar.checkbox(
    "Align 1Y Ago (X, Y) to Today (Procrustes)",
    value=True,
    help="Removes arbitrary PCA/MDS axis sign-flips or rotations between two separate runs.",
)
show_beacons = st.sidebar.checkbox(
    "Show Benchmark & Portfolio Beacons",
    value=True,
    help="Plots the 2D weight-weighted centers of mass and active tilt vector.",
)
center_on_bench = st.sidebar.checkbox(
    "Center Origin (0, 0) on Benchmark",
    value=False,
    disabled=not show_beacons,
    help="Shifts each map's coordinates so the Benchmark sits at (0, 0) and the Portfolio moves relative to it.",
)
color_mode = st.sidebar.radio(
    "Surface Color Encoding",
    ["Over / Underweight (Active %)", "Local Stress Intensity (%)"],
)
show_sea_level = st.sidebar.checkbox("Show Sea-Level Plane (Z = 0%)", value=False)
show_labels = st.sidebar.checkbox("Show Top Cluster Labels", value=True)
top_n_labels = st.sidebar.slider("Number of Labels", 0, 15, 6, disabled=not show_labels)
grid_res = st.sidebar.select_slider(
    "Grid Resolution", options=[70, 90, 110, 130, 160], value=110
)


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
raw_new = load_cluster_file(file_new, "Today")
raw_old = load_cluster_file(file_old, "1Y_Ago")

if raw_new is not None and demo_mode and raw_old is None:
    raw_old = make_synthetic_1y_ago(raw_new)

if raw_old is None or raw_new is None:
    st.info(
        "👈 Upload both **1 Year Ago** and **Today** `clusters risk` files in the sidebar "
        "(or upload Today's file and enable the demo toggle to preview the layout)."
    )
    st.stop()

df_old = prepare_dataframe(raw_old, weight_col)
df_new = prepare_dataframe(raw_new, weight_col)

rmsd = np.nan
if do_procrustes:
    df_old, rmsd = align_procrustes(df_old, df_new)

bary_old = compute_barycenters(df_old) if show_beacons else None
bary_new = compute_barycenters(df_new) if show_beacons else None

if show_beacons and center_on_bench:
    if bary_old is not None:
        bx_o, by_o = bary_old["bench"]
        df_old["X"] -= bx_o
        df_old["Y"] -= by_o
        bary_old = compute_barycenters(df_old)
    if bary_new is not None:
        bx_n, by_n = bary_new["bench"]
        df_new["X"] -= bx_n
        df_new["Y"] -= by_n
        bary_new = compute_barycenters(df_new)

# Shared X, Y bounding box across BOTH periods (padded by 2.6 * max sigma)
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

# Shared symmetric Z limit so Z=0 (sea level) is identically placed in both plots
z_lim = float(max(np.abs(Z_old).max(), np.abs(Z_new).max(), 1.0)) * 1.15

# KPI Summary Row
k1, k2, k3, k4, k5 = st.columns(5)
as_old = 0.5 * float(df_old["Active_Pct"].abs().sum())
as_new = 0.5 * float(df_new["Active_Pct"].abs().sum())
k1.metric("Active Share (1Y → Now)", f"{as_new:.1f}%", delta=f"{as_new - as_old:+.1f}%")

held_old = df_old.loc[~df_old["Is_Not_Held"], "Stress_Pct"]
held_new = df_new.loc[~df_new["Is_Not_Held"], "Stress_Pct"]
avg_s_old = float(held_old.mean()) if len(held_old) else 0.0
avg_s_new = float(held_new.mean()) if len(held_new) else 0.0
k2.metric(
    "Mean Held Stress",
    f"{avg_s_new:.1f}%",
    delta=f"{avg_s_new - avg_s_old:+.1f}%",
    delta_color="inverse",
)

if bary_old is not None and bary_new is not None:
    tilt_old = float(np.hypot(
        bary_old["port"][0] - bary_old["bench"][0],
        bary_old["port"][1] - bary_old["bench"][1],
    ))
    tilt_new = float(np.hypot(
        bary_new["port"][0] - bary_new["bench"][0],
        bary_new["port"][1] - bary_new["bench"][1],
    ))
    k3.metric("Net Factor Tilt ||P - B||", f"{tilt_new:.4f}", delta=f"{tilt_new - tilt_old:+.4f}", delta_color="off")
else:
    k3.metric("Net Factor Tilt ||P - B||", "N/A")

k4.metric("Clusters (1Y / Now)", f"{len(df_old)} / {len(df_new)}")
k5.metric(
    "Procrustes RMSD",
    f"{rmsd:.4f}" if pd.notna(rmsd) else "Off / N/A",
    help="Residual coordinate shift after removing global PCA rotation/reflection.",
)

# Side-by-side 3D landscapes
col_left, col_right = st.columns(2)

with col_left:
    fig_old = make_landscape_figure(
        "📅 1 Year Ago",
        df_old, x_grid, y_grid, Z_old, S_old, nz_old, sig_old,
        x_range, y_range, z_lim, color_mode, stress_cap,
        show_sea_level, show_labels, top_n_labels, bary_old, blend_mode,
    )
    st.plotly_chart(fig_old, use_container_width=True)

with col_right:
    fig_new = make_landscape_figure(
        "📍 Today",
        df_new, x_grid, y_grid, Z_new, S_new, nz_new, sig_new,
        x_range, y_range, z_lim, color_mode, stress_cap,
        show_sea_level, show_labels, top_n_labels, bary_new, blend_mode,
    )
    st.plotly_chart(fig_new, use_container_width=True)

# ---------------------------------------------------------
# CLUSTER MIGRATION & DELTA TABLE
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔄 Cluster Migration & Stress Shift (Today vs. 1 Year Ago)")

comp = pd.merge(
    df_old[["Node_ID", "Node_Label", "Active_Pct", "Stress_Pct", "Is_Not_Held", "X", "Y"]],
    df_new[["Node_ID", "Node_Label", "Active_Pct", "Stress_Pct", "Is_Not_Held", "X", "Y"]],
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