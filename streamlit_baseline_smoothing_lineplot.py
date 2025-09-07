# streamlit_baseline_smoothing_lineplot.py
# Linha-base + suavização + detecção por derivada + retas entre bases
# Aceita .txt/.csv/.dpt com 2 colunas X,Y (vírgula ou ponto decimal)

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, peak_prominences

st.set_page_config(page_title="Baseline & Peaks (derivative)", layout="wide")
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# ---------------- utils ----------------
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
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore"); break
        except Exception: pass
    # tenta csv rápido
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

def infer_orientation(y: np.ndarray) -> str:
    med = np.median(y)
    return "Picos para cima" if (np.max(y)-med) >= (med-np.min(y)) else "Picos para baixo"

def piecewise_segments(x, y, peaks_idx, y_norm, orientation):
    # bases via proeminência (no sinal invertido quando picos são para baixo)
    if orientation == "Picos para baixo":
        inv = 1.0 - y_norm
        prom, lb, rb = peak_prominences(inv, peaks_idx)
    else:
        prom, lb, rb = peak_prominences(y_norm, peaks_idx)
    segs = []
    for L,R in zip(lb, rb):
        L = int(L); R = int(R)
        if R <= L: 
            continue
        segs.append((float(x[L]), float(y[L]), float(x[R]), float(y[R]), L, R))
    return segs

def apply_baseline_in_windows(x, y, segments, orientation):
    yc = y.copy()
    for (x0,y0,x1,y1,L,R) in segments:
        m = (y1-y0)/(x1-x0) if x1 != x0 else 0.0
        xr = x[L:R+1]
        yline = y0 + m*(xr-x0)
        if orientation == "Picos para cima":
            yc[L:R+1] = y[L:R+1] - yline
        else:
            yc[L:R+1] = yline - y[L:R+1]
    return yc

# ---------------- UI ----------------
st.title("Ajuste de linha-base com detecção por derivada")

up = st.file_uploader("Carregue .txt/.csv/.dpt com 2 colunas X,Y", type=["txt","csv","dpt"])
c1, c2, c3 = st.columns(3)

with c1:
    smooth_on = st.checkbox("Suavizar (Savitzky-Golay)", True)
    win = st.slider("Janela (ímpar)", 5, 201, 31, step=2)
    poly = st.slider("Ordem do polinômio", 1, 5, 3)

with c2:
    orientation = st.selectbox("Orientação dos picos", ["Detectar automaticamente","Picos para cima","Picos para baixo"])
    prom_min = st.slider("Proeminência mínima (rel.)", 0.0, 0.5, 0.02, 0.005,
                         help="Aplicada após localizar os picos (0–0.5 do range normalizado).")
    dist_min = st.slider("Distância mínima entre picos (pontos)", 1, 1000, 50, 1)

with c3:
    use_deriv = st.checkbox("Detectar por derivada (1ª + 2ª)", True)
    curv_rel = st.slider("Curvatura mínima (|2ª deriv|, rel.)", 0.0, 0.5, 0.05, 0.01,
                         help="Filtra picos com curvatura fraca; relativo ao max |d²y/dx²|.")
    k_win = st.slider("Janela local para refinamento (±k)", 1, 50, 6, 1)
    show_corr = st.checkbox("Mostrar gráfico corrigido", True)

show_lines = st.checkbox("Mostrar retas brancas", True)

if not up:
    st.info("Carregue um arquivo para começar.")
    st.stop()

df = load_xy(up)
x = df["X"].to_numpy(dtype=float)
y = df["Y"].to_numpy(dtype=float)

# suavização base (opcional) para estabilidade das derivadas
y_proc = y.copy()
if smooth_on and len(y_proc) >= win:
    y_proc = savgol_filter(y_proc, window_length=win, polyorder=poly, mode="interp")

# orientação
orientation_eff = infer_orientation(y_proc) if orientation == "Detectar automaticamente" else orientation

# normalização para métricas relativas
yr = np.ptp(y_proc) if np.ptp(y_proc) > 0 else 1.0
y_norm = (y_proc - np.min(y_proc)) / yr

# ------------- detecção de picos -------------
if use_deriv:
    # 1ª e 2ª derivadas por Savitzky-Golay
    dx_mean = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dy = savgol_filter(y_proc, window_length=max(win,5), polyorder=min(poly,5),
                       deriv=1, delta=dx_mean, mode="interp")
    d2y = savgol_filter(y_proc, window_length=max(win,5), polyorder=min(poly,5),
                        deriv=2, delta=dx_mean, mode="interp")

    # zero-crossings da 1ª derivada + curvatura mínima
    if orientation_eff == "Picos para cima":
        zc = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0]  # + -> -
        curv_th = curv_rel * np.max(np.abs(d2y)) if np.max(np.abs(d2y))>0 else 0.0
        cand = [i for i in zc if d2y[i] < -curv_th]
        # refinar para o máximo real em ±k
        peaks_idx = []
        for i in cand:
            a = max(0, i-k_win); b = min(len(y_proc)-1, i+1+k_win)
            j = a + int(np.argmax(y_proc[a:b+1]))
            peaks_idx.append(j)
    else:  # picos para baixo
        zc = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0]  # - -> +
        curv_th = curv_rel * np.max(np.abs(d2y)) if np.max(np.abs(d2y))>0 else 0.0
        cand = [i for i in zc if d2y[i] > curv_th]
        peaks_idx = []
        for i in cand:
            a = max(0, i-k_win); b = min(len(y_proc)-1, i+1+k_win)
            j = a + int(np.argmin(y_proc[a:b+1]))
            peaks_idx.append(j)

    # remover picos muito próximos (distância mínima)
    peaks_idx = np.array(sorted(set(peaks_idx)), dtype=int)
    if len(peaks_idx) > 1 and dist_min > 1:
        keep = [peaks_idx[0]]
        for j in peaks_idx[1:]:
            if j - keep[-1] >= dist_min:
                keep.append(j)
            else:
                # mantém o mais proeminente dentro do bloco
                seg = slice(keep[-1], j+1)
                local = keep[-1] if orientation_eff=="Picos para cima" else keep[-1]
                keep[-1] = j  # simples: fica o último (mais à direita)
        peaks_idx = np.array(keep, dtype=int)

    # filtrar por proeminência (relativa)
    if orientation_eff == "Picos para baixo":
        prom_vals, lb, rb = peak_prominences(1.0 - y_norm, peaks_idx)
    else:
        prom_vals, lb, rb = peak_prominences(y_norm, peaks_idx)
    mask_prom = prom_vals >= prom_min
    peaks_idx, lb, rb = peaks_idx[mask_prom], lb[mask_prom], rb[mask_prom]
else:
    # fallback (apenas proeminência em y_norm)
    if orientation_eff == "Picos para baixo":
        inv = 1.0 - y_norm
        # usar threshold via proeminência -> capturamos todos e filtramos
        # (find_peaks já foi substituído por derivadas quando use_deriv=True)
        from scipy.signal import find_peaks
        pks, _ = find_peaks(inv, distance=dist_min, prominence=prom_min)
        prom_vals, lb, rb = peak_prominences(inv, pks)
        peaks_idx = pks
    else:
        from scipy.signal import find_peaks
        pks, _ = find_peaks(y_norm, distance=dist_min, prominence=prom_min)
        prom_vals, lb, rb = peak_prominences(y_norm, pks)
        peaks_idx = pks

# segmentos (retas entre as bases)
segments = []
for L,R in zip(lb, rb):
    L=int(L); R=int(R)
    if R>L:
        segments.append((x[L], y_proc[L], x[R], y_proc[R], L, R))

# correção
y_corr = apply_baseline_in_windows(x, y_proc, segments, orientation_eff)

# ---------------- plots ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_proc, mode="lines", name="Sinal (processado)",
                         line=dict(width=3, color="#E4572E")))
if show_lines:
    for (x0,y0,x1,y1,_,_) in segments:
        fig.add_shape(type="line", x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                      line=dict(color="white", width=6))
fig.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                  xaxis_title="X", yaxis_title="Y", height=520)

st.subheader("Original (com retas)")
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

if show_corr:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y_corr, mode="lines", name="Y corrigido",
                              line=dict(width=3)))
    fig2.update_layout(template="plotly_dark", margin=dict(l=30,r=20,t=40,b=40),
                       xaxis_title="X", yaxis_title="Y corrigido", height=520)
    st.subheader("Corrigido")
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

st.caption(f"Orientação: **{orientation_eff}** | Picos: **{len(segments)}** | Derivadas: **{'on' if use_deriv else 'off'}**")












