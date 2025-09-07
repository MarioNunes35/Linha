# streamlit_baseline_smoothing_lineplot.py
# App: linha-base + suavização + detecção de pico (reta branca entre extremidades)
# Funciona com .txt, .csv e .dpt (2 colunas X,Y; vírgula ou ponto como decimal)

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, peak_prominences

st.set_page_config(page_title="Baseline & Smoothing (XY)", layout="wide")

# ----------------------------- UTIL ---------------------------------
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _infer_orientation(y: np.ndarray) -> str:
    # comparamos amplitudes relativas ao mediano
    med = np.median(y)
    up_amp = float(np.max(y) - med)
    down_amp = float(med - np.min(y))
    return "Picos para cima" if up_amp >= down_amp else "Picos para baixo"

def _safe_float(tok: str) -> float:
    return float(tok.replace(",", "."))

def _parse_xy_text(text: str) -> pd.DataFrame:
    """Lê qualquer texto com duas colunas numéricas (X,Y). Ignora cabeçalhos."""
    xs, ys = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        nums = NUM_RE.findall(line)
        if len(nums) >= 2:
            try:
                x = _safe_float(nums[0])
                y = _safe_float(nums[1])
                xs.append(x); ys.append(y)
            except Exception:
                continue
    if len(xs) < 3:
        raise ValueError("Não encontrei ao menos 3 pares X,Y no arquivo.")
    df = pd.DataFrame({"X": np.array(xs, dtype=float), "Y": np.array(ys, dtype=float)})
    # ordena por X se estiver fora de ordem
    if not np.all(np.diff(df["X"].values) >= 0):
        df = df.sort_values("X").reset_index(drop=True)
    return df

def load_xy(uploaded) -> pd.DataFrame:
    raw = uploaded.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    # tenta CSV rápido
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python",
                         comment="#", header=None, names=["X","Y"])
        # se deu certo mas tem coisas estranhas (strings), faz parser robusto
        if df[["X","Y"]].dtypes.eq(object).any():
            raise ValueError
        return df
    except Exception:
        return _parse_xy_text(text)

def piecewise_linear_lines(x, y, peaks, left_bases, right_bases):
    """Retorna lista de segmentos [(x0,y0,x1,y1, idxL, idxR), ...] conectando bases."""
    segs = []
    for lb, rb in zip(left_bases, right_bases):
        lb = int(lb); rb = int(rb)
        if rb <= lb:
            continue
        x0, y0 = float(x[lb]), float(y[lb])
        x1, y1 = float(x[rb]), float(y[rb])
        segs.append((x0, y0, x1, y1, lb, rb))
    return segs

def apply_baseline_over_peaks(x, y, segments, orientation="Picos para cima"):
    """Subtrai a linha-base apenas dentro das janelas de pico (entre as bases)."""
    yc = y.copy()
    for (x0, y0, x1, y1, lb, rb) in segments:
        # reta y_line(x) = y0 + m*(x - x0)
        m = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        xr = x[lb:rb+1]
        y_line = y0 + m * (xr - x0)
        if orientation == "Picos para cima":
            yc[lb:rb+1] = y[lb:rb+1] - y_line
        else:  # picos para baixo
            yc[lb:rb+1] = y_line - y[lb:rb+1]
    return yc

# ----------------------------- UI -----------------------------------
st.title("Ajuste de Linha-Base com Reta entre Extremidades do Pico")

uploaded = st.file_uploader("Carregue um arquivo (.txt, .csv, .dpt) com 2 colunas X,Y", type=["txt","csv","dpt"])
colA, colB, colC = st.columns([1,1,1])

with colA:
    smooth_on = st.checkbox("Suavizar (Savitzky-Golay)", value=True)
    if smooth_on:
        w = st.slider("Janela (ímpar)", min_value=5, max_value=201, value=31, step=2)
        p = st.slider("Ordem do polinômio", 1, 5, 3)

with colB:
    orientation = st.selectbox("Orientação dos picos", ["Detectar automaticamente", "Picos para cima", "Picos para baixo"])
    prom = st.slider("Proeminência mínima", 0.0, 0.2, 0.01, 0.005, help="Aumente para ignorar flutuações pequenas (normalizado pelo range Y)")
    dist = st.slider("Distância mínima entre picos (pontos)", 1, 1000, 50, 1)

with colC:
    show_segments = st.checkbox("Mostrar retas brancas", value=True)
    show_corrected = st.checkbox("Mostrar gráfico corrigido", value=True)
    export_corrected = st.checkbox("Habilitar exportação do corrigido", value=False)

if uploaded:
    df = load_xy(uploaded)
    x, y = df["X"].to_numpy(), df["Y"].to_numpy()

    # suavização opcional (não altera o arquivo original)
    y_proc = y.copy()
    if smooth_on and len(y_proc) >= w:
        y_proc = savgol_filter(y_proc, window_length=w, polyorder=p, mode="interp")

    # normalização para escolhas de proeminência estáveis
    yr = np.ptp(y_proc) if np.ptp(y_proc) > 0 else 1.0
    y_norm = (y_proc - np.min(y_proc)) / yr

    # orientação
    if orientation == "Detectar automaticamente":
        orientation_eff = _infer_orientation(y_proc)
    else:
        orientation_eff = orientation

    # detecção de picos (se “picos para baixo”, analisamos no sinal invertido)
    if orientation_eff == "Picos para cima":
        peaks, _ = find_peaks(y_norm, distance=dist, prominence=prom)
        prom_vals, left_bases, right_bases = peak_prominences(y_norm, peaks)
    else:
        inv = 1.0 - y_norm  # inverter preservando [0,1]
        peaks, _ = find_peaks(inv, distance=dist, prominence=prom)
        prom_vals, left_bases, right_bases = peak_prominences(inv, peaks)

    segments = piecewise_linear_lines(x, y_proc, peaks, left_bases, right_bases)
    y_corrected = apply_baseline_over_peaks(x, y_proc, segments, orientation=orientation_eff)

    # ------------------------- PLOTS -------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_proc, mode="lines",
                             name="Sinal (processado)", line=dict(width=3, color="#E4572E")))
    if show_segments:
        # desenha as “retas brancas” (entre as bases de cada pico)
        for (x0,y0,x1,y1,_,_) in segments:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="white", width=6))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=30, r=20, t=40, b=40),
        xaxis_title="X",
        yaxis_title="Y",
        height=520,
    )

    tabs = st.tabs(["Original (com retas)", "Corrigido" if show_corrected else ""])
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    if show_corrected:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y_corrected, mode="lines",
                                  name="Y corrigido", line=dict(width=3)))
        fig2.update_layout(template="plotly_dark", margin=dict(l=30, r=20, t=40, b=40),
                           xaxis_title="X", yaxis_title="Y corrigido", height=520)
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

    # ---------------------- EXPORTAÇÃO ----------------------
    if export_corrected:
        out = pd.DataFrame({"X": x, "Y_corr": y_corrected})
        st.download_button(
            "Baixar Y corrigido (.csv)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"{uploaded.name.rsplit('.',1)[0]}_corrigido.csv",
            mime="text/csv",
        )

    # infos úteis
    st.caption(f"Orientação em uso: **{orientation_eff}** | picos detectados: **{len(segments)}**")
else:
    st.info("Carregue um arquivo para começar. Aceita .txt, .csv e .dpt com duas colunas X,Y.")











