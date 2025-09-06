# Streamlit — Baseline + Smoothing + Line Plot (Minimalist Tool)
# Author: ChatGPT (GPT-5 Thinking)
# Purpose:
#   A lightweight app that ONLY does: (1) baseline correction, (2) smoothing,
#   and (3) line plotting for one X column and multiple Y columns.
#   Supported baselines: None, AsLS (Eilers), Polynomial fit, Rolling-min.
#   Supported smoothing: None, Savitzky—Golay, Moving Average.

import io
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import plotly.io as pio

st.set_page_config(page_title="Baseline • Smooth • Line", page_icon="📈", layout="wide")

st.title("📈 Ajuste de Linha de Base • Suavização • Gráfico de Linha")
st.caption("Ferramenta enxuta: corrige baseline, suaviza e plota. Nada além disso.")

# ------------------------------- Helpers --------------------------------- #
@st.cache_data
def robust_read_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    bio = BytesIO(file_bytes)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0)
        try:
            return pd.read_csv(bio, sep=';')
        except Exception:
            bio.seek(0)
            try:
                return pd.read_csv(bio, decimal=',', sep=';')
            except Exception:
                bio.seek(0)
                return pd.read_csv(bio, decimal=',')

# Baselines
def asls_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    L = len(y)
    # Create second-order difference matrix
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L-2, L-2), format='csr')
    # Create the full difference matrix
    D_full = sparse.lil_matrix((L-2, L))
    for i in range(L-2):
        D_full[i, i] = 1
        D_full[i, i+1] = -2
        D_full[i, i+2] = 1
    D = D_full.tocsr()
    
    w = np.ones(L)
    z = y.copy()
    
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format='csr')
        A = W + lam * (D.T @ D)
        z = spsolve(A, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def poly_baseline(x: np.ndarray, y: np.ndarray, deg: int = 2) -> np.ndarray:
    deg = max(1, int(deg))
    coeff = np.polyfit(x, y, deg)
    return np.polyval(coeff, x)

def rolling_min_baseline(y: np.ndarray, window: int = 51) -> np.ndarray:
    w = max(5, int(window))
    return pd.Series(y).rolling(w, min_periods=1, center=True).min().to_numpy()

# Smoothing
def smooth_savgol(y: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    w = max(5, int(window))
    if w % 2 == 0:
        w += 1
    p = min(int(poly), w - 1)  # Ensure polynomial order is less than window length
    try:
        return savgol_filter(y, window_length=w, polyorder=p)
    except Exception:
        return y

def smooth_moving_average(y: np.ndarray, window: int = 11) -> np.ndarray:
    w = max(2, int(window))
    return pd.Series(y).rolling(w, min_periods=1, center=True).mean().to_numpy()

# ------------------------------- Sidebar --------------------------------- #
st.sidebar.header("Entrada de dados")
file = st.sidebar.file_uploader("CSV com 1 coluna X e várias Y", type=["csv", "txt"])
if not file:
    st.info("Envie um CSV. Ex.: `x,y1,y2,y3`. Se não houver X, o app usará o índice.")
    st.stop()

raw = robust_read_csv(file.getvalue())
cols = list(raw.columns)

st.sidebar.subheader("Mapeamento")
col_x = st.sidebar.selectbox("Coluna X (opcional)", ["<None>"] + cols, index=0)
ys = st.sidebar.multiselect("Colunas Y", [c for c in cols if c != col_x], default=[c for c in cols if c != col_x][:3])
if not ys:
    st.warning("Selecione ao menos uma Y.")
    st.stop()

# Pre-process
df = raw.copy()
if col_x == "<None>":
    x = np.arange(len(df))
else:
    x = pd.to_numeric(df[col_x], errors='coerce').to_numpy()

Y = {}
for c in ys:
    Y[c] = pd.to_numeric(df[c], errors='coerce').to_numpy()

st.sidebar.header("Baseline")
baseline_method = st.sidebar.selectbox("Método", ["Nenhum", "AsLS (Eilers)", "Polinomial", "Rolling Min"], index=1)
lam = st.sidebar.number_input("AsLS: λ (suavidade)", value=1e6, step=1e6, format="%.0f")
p_asls = st.sidebar.slider("AsLS: p (assimetria)", 0.001, 0.5, 0.01)
niter = st.sidebar.slider("AsLS: iterações", 5, 60, 15)
deg = st.sidebar.number_input("Polinomial: grau", value=2, min_value=1, max_value=8)
roll_w = st.sidebar.slider("Rolling Min: janela (pts)", 5, 501, 51, step=2)

apply_baseline = st.sidebar.selectbox("Aplicar baseline em", ["Sinal original", "Após suavização"], index=0)
subtract_mode = st.sidebar.selectbox("Correção final", ["Subtrair baseline (y - base)", "Dividir (y/base)"], index=0)

st.sidebar.header("Suavização")
smooth_method = st.sidebar.selectbox("Método", ["Nenhuma", "Savitzky—Golay", "Média móvel"], index=1)
sg_w = st.sidebar.slider("SG: janela (pts)", 5, 201, 21, step=2)
sg_p = st.sidebar.slider("SG: ordem", 2, 5, 3)
ma_w = st.sidebar.slider("Média móvel: janela (pts)", 3, 201, 11, step=2)

st.sidebar.header("Plotagem")
logx = st.sidebar.checkbox("Eixo X log", value=False)
logy = st.sidebar.checkbox("Eixo Y log", value=False)
show_orig = st.sidebar.checkbox("Mostrar série original", value=True)
show_base = st.sidebar.checkbox("Mostrar baseline", value=True)
show_proc = st.sidebar.checkbox("Mostrar série processada", value=True)

# ------------------------------- Processing ------------------------------ #
processed = {}
baselines = {}

for name, y in Y.items():
    y0 = y.copy()

    # Choose smoothing first or baseline first
    if apply_baseline == "Sinal original":
        y_for_base = y0
        # compute baseline
        if baseline_method == "AsLS (Eilers)":
            base = asls_baseline(y_for_base, lam=float(lam), p=float(p_asls), niter=int(niter))
        elif baseline_method == "Polinomial":
            base = poly_baseline(x, y_for_base, deg=int(deg))
        elif baseline_method == "Rolling Min":
            base = rolling_min_baseline(y_for_base, window=int(roll_w))
        else:
            base = np.zeros_like(y_for_base)

        # correct
        if subtract_mode.startswith("Subtrair"):
            y_corr = y0 - base
        else:
            denom = np.where(base == 0, 1.0, base)
            y_corr = y0 / denom

        # smoothing afterwards
        if smooth_method == "Savitzky—Golay":
            y_out = smooth_savgol(y_corr, window=int(sg_w), poly=int(sg_p))
        elif smooth_method == "Média móvel":
            y_out = smooth_moving_average(y_corr, window=int(ma_w))
        else:
            y_out = y_corr

    else:  # baseline after smoothing
        if smooth_method == "Savitzky—Golay":
            y_sm = smooth_savgol(y0, window=int(sg_w), poly=int(sg_p))
        elif smooth_method == "Média móvel":
            y_sm = smooth_moving_average(y0, window=int(ma_w))
        else:
            y_sm = y0

        if baseline_method == "AsLS (Eilers)":
            base = asls_baseline(y_sm, lam=float(lam), p=float(p_asls), niter=int(niter))
        elif baseline_method == "Polinomial":
            base = poly_baseline(x, y_sm, deg=int(deg))
        elif baseline_method == "Rolling Min":
            base = rolling_min_baseline(y_sm, window=int(roll_w))
        else:
            base = np.zeros_like(y_sm)

        if subtract_mode.startswith("Subtrair"):
            y_out = y_sm - base
        else:
            denom = np.where(base == 0, 1.0, base)
            y_out = y_sm / denom

    processed[name] = y_out
    baselines[name] = base

# ------------------------------- Plotting -------------------------------- #
fig = go.Figure()

for name in ys:
    if name not in processed:  # Skip if processing failed
        continue
        
    if show_orig:
        fig.add_trace(go.Scatter(x=x, y=Y[name], mode='lines', name=f"{name} (orig)", line=dict(width=1), opacity=0.45))
    if show_base and baseline_method != "Nenhum":
        fig.add_trace(go.Scatter(x=x, y=baselines[name], mode='lines', name=f"{name} (base)", line=dict(width=2, dash='dot')))
    if show_proc:
        fig.add_trace(go.Scatter(x=x, y=processed[name], mode='lines', name=f"{name} (proc)", line=dict(width=3)))

fig.update_layout(template='plotly_dark', height=620, legend_title='Séries')
fig.update_xaxes(title_text=col_x if col_x != "<None>" else "Índice")
fig.update_yaxes(title_text="Intensidade (u.a.)")
if logx:
    fig.update_xaxes(type='log')
if logy:
    fig.update_yaxes(type='log')

st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Export ---------------------------------- #
st.subheader("Exportar")
# Processed CSV
if processed:  # Only create export if there's processed data
    proc_df = pd.DataFrame({
        (col_x if col_x != "<None>" else "index"): x,
        **{f"{k}__processed": v for k, v in processed.items()},
        **{f"{k}__baseline": v for k, v in baselines.items()},
        **{f"{k}__original": Y[k] for k in ys if k in processed},
    })

    buf = io.StringIO(); proc_df.to_csv(buf, index=False)
    st.download_button("⬇️ Baixar CSV (processado)", buf.getvalue(), file_name="baseline_smooth_processed.csv", mime="text/csv")

    # HTML figure
    html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    st.download_button("⬇️ Baixar HTML interativo", data=html, file_name="baseline_smooth_plot.html")

# ------------------------------- Notes ----------------------------------- #
with st.expander("Notas & Boas Práticas"):
    st.markdown(
        """
        - **Pipeline**: escolha se o *baseline* é aplicado **antes** ou **depois** da suavização.
        - **AsLS (Eilers)**: ótimo para remover fundos curvos/assimétricos. Ajuste `λ` (mais alto = mais liso) e `p` (peso da assimetria).
        - **Polinomial**: bom para tendências globais (escolha o grau com parcimônia).
        - **Rolling Min**: aproxima o fundo pela envoltória inferior (útil p/ picos positivos).
        - **Savitzky—Golay**: janela **ímpar**, preserve formas de pico.
        - **Exportação**: CSV inclui colunas `__original`, `__baseline` e `__processed` para auditoria.
        """
    )
