# streamlit_baseline_smoothing_lineplot.py
# FTIR/XY: Baseline por retas (bases) OU automático (AsLS),
# com detecção de picos por derivadas, intensidade ou ambos.
# Aceita .txt/.csv/.dpt (2 colunas X,Y). Ordena X em crescente.

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy import sparse
from scipy.sparse.linalg import spsolve

st.set_page_config(page_title="Baseline FTIR • picos por derivadas/intensidade", layout="wide")
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# ---------- IO ----------
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
        except Exception: pass
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

# ---------- helpers ----------
def infer_orientation(y: np.ndarray) -> str:
    med = np.median(y)
    return "Picos para cima" if (np.max(y)-med) >= (med-np.min(y)) else "Picos para baixo"

def make_segments(x, y, left_bases, right_bases):
    segs = []
    for L, R in zip(left_bases, right_bases):
        L = int(L); R = int(R)
        if R <= L: 
            continue
        segs.append((float(x[L]), float(y[L]), float(x[R]), float(y[R]), L, R))
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

# --- Asymmetric Least Squares (automático) ---
def asls_baseline(y, lam=1e6, p=0.001, niter=10):
    """
    Eilers & Boelens (2005). Retorna baseline 'inferior'.
    Para picos para baixo, use: baseline_sup = -asls_baseline(-y, lam, p, niter)
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# ---------- detecção de picos ----------
def peaks_by_derivative(x, y_norm, w=21, p=3, dist_min=100, for_valleys=True):
    dx = float(np.mean(np.diff(x))) if len(x)>1 else 1.0
    dy = savgol_filter(y_norm, window_length=max(w,5), polyorder=min(p,5),
                       deriv=1, delta=dx, mode="interp")
    d2y = savgol_filter(y_norm, window_length=max(w,5), polyorder=min(p,5),
                        deriv=2, delta=dx, mode="interp")
    # vales: - -> + ; picos: + -> -
    if for_valleys:
        zc = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0]
        sign = +1
    else:
        zc = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0]
        sign = -1
    half = max(2, w//2)
    idx = []
    for i in zc:
        # exige curvatura do sinal correta
        if sign*d2y[i] <= 0: 
            continue
        a = max(0, i-half); b = min(len(y_norm)-1, i+1+half)
        if for_valleys:
            j = a + int(np.argmin(y_norm[a:b+1]))
        else:
            j = a + int(np.argmax(y_norm[a:b+1]))
        if len(idx)==0 or j-idx[-1] >= dist_min:
            idx.append(j)
    return np.array(idx, dtype=int)

def peaks_by_intensity(y_norm, dist_min=100, prom_min=0.02, width_min=10, height_min=None, for_valleys=True):
    if for_valleys:
        inv = 1.0 - y_norm
        pks, _ = find_peaks(inv, distance=dist_min, prominence=prom_min, width=width_min,
                            height=height_min if height_min is None else height_min)
        prom, lb, rb = peak_prominences(inv, pks)
    else:
        pks, _ = find_peaks(y_norm, distance=dist_min, prominence=prom_min, width=width_min,
                            height=height_min if height_min is None else height_min)
        prom, lb, rb = peak_prominences(y_norm, pks)
    return pks, prom, lb, rb

# ==================== UI ====================
st.title("Subtração de linha-base • Derivadas × Intensidade • Retas × Automático")

up = st.file_uploader("Carregue .txt/.csv/.dpt (2 colunas X,Y)", type=["txt","csv","dpt"])

c1, c2, c3 = st.columns(3)
with c1:
    smooth_on = st.checkbox("Suavizar (Savitzky-Golay)", True)
    w = st.slider("Janela SG (ímpar)", 5, 201, 21, step=2)
    poly = st.slider("Ordem SG", 1, 5, 3)

with c2:
    orientation = st.selectbox("Orientação dos picos", ["Picos para baixo","Picos para cima","Detectar automaticamente"], index=0)
    detector = st.selectbox("Detector de picos", ["Derivadas","Intensidade","Ambos"], index=2)

with c3:
    prom_min = st.slider("Proeminência mínima (rel.)", 0.0, 0.5, 0.02, 0.005)
    dist_min = st.slider("Distância mínima (pontos)", 1, 1000, 150, 1)
    width_min = st.slider("Largura mínima (pontos)", 1, 400, 10, 1)
    height_min = st.slider("Altura mínima (rel., modo Intensidade)", 0.0, 1.0, 0.0, 0.01)

st.markdown("**Modo de correção da linha-base**")
mode = st.radio("", ["Somente janelas dos picos","Baseline global por partes","Automático (AsLS)"], index=0)

with st.expander("Parâmetros do modo Automático (AsLS)"):
    lam = st.slider("λ (suavidade, log10)", 2, 8, 6, 1, help="Maior λ = baseline mais lisa")
    p_asls = st.slider("p (assimetria)", 0.000, 0.500, 0.001, 0.001, help="Menor p favorece baseline abaixo do sinal")
    niter = st.slider("Iterações", 1, 30, 10, 1)

show_lines = st.checkbox("Mostrar retas (bases)", True)
show_marks = st.checkbox("Marcar mínimos/máximos", False)
show_corrected = st.checkbox("Mostrar gráfico corrigido", True)
enable_export = st.checkbox("Exportar CSV do corrigido", False)

if not up:
    st.info("Carregue um arquivo para começar. Para FTIR, 'Picos para baixo' geralmente é o correto.")
    st.stop()

df = load_xy(up)
x = df["X"].to_numpy(dtype=float)
y = df["Y"].to_numpy(dtype=float)

y_proc = y.copy()
if smooth_on and len(y_proc) >= w:
    y_proc = savgol_filter(y_proc, window_length=w, polyorder=poly, mode="interp")

orientation_eff = infer_orientation(y_proc) if orientation == "Detectar automaticamente" else orientation

# normalização 0–1 para parâmetros relativos
yr = np.ptp(y_proc) if np.ptp(y_proc)>0 else 1.0
y_norm = (y_proc - np.min(y_proc)) / yr
for_valleys = (orientation_eff == "Picos para baixo")

# ---------- picos: derivadas × intensidade × ambos ----------
peaks_d = np.array([], dtype=int)
peaks_i = np.array([], dtype=int)
lb = rb = prom_vals = None

if detector in ("Derivadas","Ambos"):
    peaks_d = peaks_by_derivative(x, y_norm, w=w, p=poly, dist_min=dist_min, for_valleys=for_valleys)

if detector in ("Intensidade","Ambos"):
    peaks_i, prom_i, lb_i, rb_i = peaks_by_intensity(y_norm, dist_min=dist_min, prom_min=prom_min,
                                                     width_min=width_min,
                                                     height_min=height_min if height_min>0 else None,
                                                     for_valleys=for_valleys)

# união (e ordenação) dos índices de picos
if detector == "Derivadas":
    peaks_all = peaks_d
elif detector == "Intensidade":
    peaks_all = peaks_i
else:
    peaks_all = np.unique(np.concatenate([peaks_d, peaks_i])) if len(peaks_d)+len(peaks_i) else np.array([], dtype=int)

# bases sempre calculadas no mesmo "sinal" usado para detecção final
if for_valleys:
    base_signal = 1.0 - y_norm
else:
    base_signal = y_norm
if len(peaks_all):
    prom_vals, lb, rb = peak_prominences(base_signal, peaks_all)
else:
    prom_vals = np.array([]); lb = np.array([]); rb = np.array([])

segments = make_segments(x, y_proc, lb, rb)

# ---------- correção ----------
if mode == "Automático (AsLS)":
    lam_val = 10.0**lam
    if for_valleys:
        baseline = -asls_baseline(-y_proc, lam=lam_val, p=p_asls, niter=niter)  # envelope superior
        y_corr = baseline - y_proc
    else:
        baseline = asls_baseline(y_proc, lam=lam_val, p=p_asls, niter=niter)
        y_corr = y_proc - baseline
    baseline_plot = baseline
elif mode == "Baseline global por partes":
    baseline_plot = build_global_baseline(x, y_proc, segments)
    if for_valleys:
        y_corr = baseline_plot - y_proc
    else:
        y_corr = y_proc - baseline_plot
else:  # Somente janelas
    baseline_plot = None
    y_corr = subtract_in_windows(x, y_proc, segments, orientation_eff)

# ---------- plots ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_proc, mode="lines", name="Sinal (processado)",
                         line=dict(width=3, color="#E4572E")))
if show_lines and len(segments):
    for (x0,y0,x1,y1,_,_) in segments:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="white", width=6))
if show_marks and len(peaks_all):
    fig.add_trace(go.Scatter(x=x[peaks_all], y=y_proc[peaks_all], mode="markers",
                             name="picos/vales", marker=dict(size=9, color="cyan", symbol="x")))
fig.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                  xaxis_title="X", yaxis_title="Y", height=520)

tabs = st.tabs(["Original", "Corrigido" if show_corrected else ""])
with tabs[0]:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

if show_corrected:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y_corr, mode="lines", name="Y corrigido", line=dict(width=3)))
    if baseline_plot is not None:
        fig2.add_trace(go.Scatter(x=x, y=baseline_plot, mode="lines", name="Baseline", line=dict(width=2, dash="dash")))
    fig2.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                       xaxis_title="X", yaxis_title="Y corrigido", height=520)
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

st.caption(
    f"Orientação: **{orientation_eff}** | Detector: **{detector}** | "
    f"Picos/vales: **{len(segments)}** | Modo baseline: **{mode}**"
)

if enable_export:
    out = pd.DataFrame({"X": x, "Y_original": y, "Y_processado": y_proc, "Y_corrigido": y_corr})
    st.download_button("Baixar .csv", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"{up.name.rsplit('.',1)[0]}_corrigido.csv", mime="text/csv")














