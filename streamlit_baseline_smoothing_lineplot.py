# streamlit_baseline_smoothing_lineplot.py
# Baseline por retas entre as bases dos vales (FTIR). Aceita .txt/.csv/.dpt (2 colunas).
# Ordena X em crescente; detecção robusta com derivadas + find_peaks.

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, peak_prominences

st.set_page_config(page_title="Baseline (retas entre bases) • FTIR/XY", layout="wide")
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# ------------------ IO ------------------
def _safe_float(s: str) -> float:
    return float(s.replace(",", "."))

def _parse_xy_text(text: str) -> pd.DataFrame:
    xs, ys = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        nums = NUM_RE.findall(line)
        if len(nums) >= 2:
            try:
                xs.append(_safe_float(nums[0])); ys.append(_safe_float(nums[1]))
            except Exception:
                continue
    if len(xs) < 3:
        raise ValueError("Não encontrei pares X,Y suficientes.")
    df = pd.DataFrame({"X": xs, "Y": ys})
    if not np.all(np.diff(df["X"].values) >= 0):
        df = df.sort_values("X").reset_index(drop=True)
    return df

def load_xy(uploaded) -> pd.DataFrame:
    raw = uploaded.read()
    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore"); break
        except Exception: 
            pass
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python",
                         comment="#", header=None, names=["X","Y"])
        if df[["X","Y"]].dtypes.eq(object).any():
            raise ValueError
        if not np.all(np.diff(df["X"].values) >= 0):
            df = df.sort_values("X").reset_index(drop=True)
        return df
    except Exception:
        return _parse_xy_text(text)

# ------------------ Baseline logic ------------------
def infer_orientation(y: np.ndarray) -> str:
    med = np.median(y)
    return "Picos para cima" if (np.max(y)-med) >= (med-np.min(y)) else "Picos para baixo"

def detect_valleys(x, y, *,
                   use_deriv=True, w=21, p=3,
                   dist_min=150, prom_min=0.02, width_min=10):
    """Retorna (peaks_idx, left_bases, right_bases, y_norm) onde os 'peaks' são os VALes."""
    yr = np.ptp(y) if np.ptp(y)>0 else 1.0
    y_norm = (y - np.min(y)) / yr
    inv = 1.0 - y_norm  # vales -> picos

    if use_deriv:
        dx = float(np.mean(np.diff(x))) if len(x)>1 else 1.0
        dy = savgol_filter(y_norm, window_length=max(w, 5), polyorder=min(p, 5),
                           deriv=1, delta=dx, mode="interp")
        d2y = savgol_filter(y_norm, window_length=max(w, 5), polyorder=min(p, 5),
                            deriv=2, delta=dx, mode="interp")
        zc = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0]      # - -> +  (vale)
        half = max(2, w//2)
        peaks_idx = []
        for i in zc:
            if d2y[i] <= 0:  # requer curvatura positiva
                continue
            a = max(0, i-half); b = min(len(y_norm)-1, i+1+half)
            j = a + int(np.argmin(y_norm[a:b+1]))
            peaks_idx.append(j)
        peaks_idx = np.array(sorted(set(peaks_idx)), dtype=int)
        if len(peaks_idx) == 0:
            peaks_idx, _ = find_peaks(inv, distance=dist_min, width=width_min, prominence=prom_min)
    else:
        peaks_idx, _ = find_peaks(inv, distance=dist_min, width=width_min, prominence=prom_min)

    prom_vals, lb, rb = peak_prominences(inv, peaks_idx)
    m = prom_vals >= prom_min
    return peaks_idx[m], lb[m], rb[m], y_norm

def make_segments(x, y_display, left_bases, right_bases):
    segs = []
    for L, R in zip(left_bases, right_bases):
        L=int(L); R=int(R)
        if R <= L: 
            continue
        segs.append((float(x[L]), float(y_display[L]), float(x[R]), float(y_display[R]), L, R))
    return segs

def subtract_in_windows(x, y, segments, orientation):
    yc = y.copy()
    for (x0,y0,x1,y1,L,R) in segments:
        m = (y1-y0)/(x1-x0) if x1!=x0 else 0.0
        xr = x[L:R+1]
        yline = y0 + m*(xr-x0)
        if orientation == "Picos para baixo":
            yc[L:R+1] = yline - y[L:R+1]
        else:
            yc[L:R+1] = y[L:R+1] - yline
    return yc

def build_global_baseline(x, y, segments):
    base = np.zeros_like(y, dtype=float)
    for (x0,y0,x1,y1,L,R) in segments:
        m = (y1-y0)/(x1-x0) if x1!=x0 else 0.0
        xr = x[L:R+1]
        base[L:R+1] = y0 + m*(xr-x0)
    return base

# ------------------ UI ------------------
st.title("Subtração de linha-base por retas entre as bases dos picos (vales)")

up = st.file_uploader("Carregue .txt/.csv/.dpt (2 colunas X,Y)", type=["txt","csv","dpt"])
c1, c2, c3 = st.columns(3)

with c1:
    smooth_on = st.checkbox("Suavizar antes (Savitzky-Golay)", True)
    w = st.slider("Janela SG (ímpar)", 5, 201, 21, step=2)
    p = st.slider("Ordem SG", 1, 5, 3)

with c2:
    orientation = st.selectbox("Orientação dos picos", ["Picos para baixo", "Picos para cima", "Detectar automaticamente"], index=0)
    use_deriv = st.checkbox("Localizar picos por derivadas", True)

with c3:
    prom_min = st.slider("Proeminência mínima (rel.)", 0.0, 0.5, 0.02, 0.005)
    dist_min = st.slider("Distância mínima (pontos)", 1, 1000, 150, 1)
    width_min = st.slider("Largura mínima (pontos)", 1, 400, 10, 1)

show_lines = st.checkbox("Mostrar retas/bases", True)
show_bases = st.checkbox("Marcar bases e mínimos", False)
mode_correction = st.radio("Modo de correção", ["Somente janelas dos picos", "Baseline global por partes"], index=0)
show_corrected = st.checkbox("Mostrar gráfico corrigido", True)
enable_export = st.checkbox("Exportar dados corrigidos", False)

if up:
    df = load_xy(up)
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)

    y_proc = y.copy()
    if smooth_on and len(y_proc) >= w:
        y_proc = savgol_filter(y_proc, window_length=w, polyorder=p, mode="interp")

    orientation_eff = infer_orientation(y_proc) if orientation == "Detectar automaticamente" else orientation

    peaks_idx, lb, rb, y_norm = detect_valleys(
        x, y_proc, use_deriv=use_deriv, w=w, p=p,
        dist_min=dist_min, prom_min=prom_min, width_min=width_min
    )

    segments = make_segments(x, y_proc, lb, rb)

    if mode_correction == "Somente janelas dos picos":
        y_corr = subtract_in_windows(x, y_proc, segments, orientation_eff)
        baseline_plot = None
    else:
        baseline_plot = build_global_baseline(x, y_proc, segments)
        if orientation_eff == "Picos para baixo":
            y_corr = baseline_plot - y_proc
        else:
            y_corr = y_proc - baseline_plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_proc, mode="lines", name="Sinal (processado)",
                             line=dict(width=3, color="#E4572E")))
    if show_lines:
        for (x0,y0,x1,y1,_,_) in segments:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="white", width=6))
    if show_bases and len(segments)>0:
        xs_lb = x[lb.astype(int)]; ys_lb = y_proc[lb.astype(int)]
        xs_rb = x[rb.astype(int)]; ys_rb = y_proc[rb.astype(int)]
        xs_pk = x[peaks_idx];      ys_pk = y_proc[peaks_idx]
        fig.add_trace(go.Scatter(x=xs_lb, y=ys_lb, mode="markers", name="base esq.", marker=dict(symbol="triangle-left", size=10, color="white")))
        fig.add_trace(go.Scatter(x=xs_rb, y=ys_rb, mode="markers", name="base dir.", marker=dict(symbol="triangle-right", size=10, color="white")))
        fig.add_trace(go.Scatter(x=xs_pk, y=ys_pk, mode="markers", name="mínimo", marker=dict(symbol="x", size=10, color="cyan")))

    fig.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                      xaxis_title="X", yaxis_title="Y", height=520)

    tabs = st.tabs(["Original (com retas)", "Corrigido" if show_corrected else ""])
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    if show_corrected:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y_corr, mode="lines", name="Y corrigido",
                                  line=dict(width=3)))
        if baseline_plot is not None:
            fig2.add_trace(go.Scatter(x=x, y=baseline_plot, mode="lines", name="Baseline (por partes)", line=dict(width=2, dash="dash")))
        fig2.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                           xaxis_title="X", yaxis_title="Y corrigido", height=520)
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

    st.caption(f"Orientação: **{orientation_eff}** | picos detectados: **{len(segments)}** | derivadas: **{'on' if use_deriv else 'off'}**")

    if enable_export:
        out = pd.DataFrame({"X": x, "Y_original": y, "Y_processado": y_proc, "Y_corrigido": y_corr})
        st.download_button("Baixar .csv", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"{up.name.rsplit('.',1)[0]}_corrigido.csv", mime="text/csv")
else:
    st.info("Carregue um arquivo para começar. Dica: para FTIR, deixe 'Picos para baixo' selecionado.")













