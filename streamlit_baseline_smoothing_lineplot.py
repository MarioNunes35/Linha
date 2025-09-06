# streamlit_baseline_smoothing_lineplot_auto.py
import io
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# SciPy para Savitzky-Golay e ALS (sparse)
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

# =========================
# Detectores automáticos (ponto/vírgula e milhar)
# =========================
PT_BR_PATTERN = r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?\d+,\d+$"
EN_US_PATTERN = r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d+\.\d+$"

def _score_pattern(series: pd.Series, pattern: str) -> int:
    return series.str.fullmatch(pattern).sum()

def _parse_with(series: pd.Series, decimal: str, thousands: str) -> pd.Series:
    s = series.str.replace("\xa0", " ", regex=False).str.strip()
    s = s.str.replace(r"[^0-9\-\+,\.eE ]", "", regex=True)   # mantém dígitos, sinais, ., ,, e/E, espaços
    s = s.str.replace(r"\s+", "", regex=True)
    if thousands:
        s = s.str.replace(thousands, "", regex=False)
    s = s.str.replace(decimal, ".", regex=False)
    # Se sobrou mais de um ".", mantém só o primeiro (decimal) e remove os demais
    s = s.apply(lambda x: x if x.count(".") <= 1 else x.replace(".", "", x.count(".")-1))
    return pd.to_numeric(s, errors="coerce")

def auto_numeric(series: pd.Series) -> Tuple[np.ndarray, Dict[str, object]]:
    raw = series.astype(str)

    pt_hits = _score_pattern(raw, PT_BR_PATTERN)
    en_hits = _score_pattern(raw, EN_US_PATTERN)

    if pt_hits > en_hits:
        candidates = [(",", "."), (".", ",")]  # preferir pt-BR
    elif en_hits > pt_hits:
        candidates = [(".", ","), (",", ".")]  # preferir en-US
    else:
        candidates = [(".", ","), (",", ".")]  # testar ambos

    best = None
    best_score = (-1, -1, -np.inf)  # (n_ok, n_frac_hint, -mediana_magnitude)
    for dec, thou in candidates:
        parsed = _parse_with(raw, decimal=dec, thousands=thou)
        n_ok = parsed.notna().sum()
        frac_hint = (raw.str.contains(re.escape(dec)) & ~raw.str.contains(re.escape(thou))).sum()
        mag = parsed.dropna().abs().median() if n_ok else np.inf
        score = (n_ok, frac_hint, -float(mag) if np.isfinite(mag) else -np.inf)
        if score > best_score:
            best = (parsed, dec, thou)
            best_score = score

    parsed, decimal_used, thousands_used = best
    info = {
        "decimal": decimal_used,
        "thousands": thousands_used,
        "non_numeric": int(parsed.isna().sum())
    }
    return parsed.to_numpy(), info

def convert_columns_auto(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    out = df.copy()
    meta: Dict[str, Dict[str, object]] = {}
    for c in cols:
        vals, info = auto_numeric(out[c].astype(str))
        out[c + "_num"] = vals
        meta[c] = info
    return out, meta

# =========================
# Suavização e baseline
# =========================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Média móvel centralizada, preservando NaNs nas bordas."""
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=max(1, window//2)).mean().to_numpy()

def safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay com correções de parâmetros para evitar erros."""
    n = np.count_nonzero(~np.isnan(y))
    if n < 3:
        return y.copy()
    # janela precisa ser ímpar e <= n
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, n if n % 2 == 1 else n-1))
    poly = max(1, min(poly, window - 1))
    # interpola NaNs antes de aplicar
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return y.copy()
    yi = np.interp(x, x[mask], y[mask])
    ys = savgol_filter(yi, window_length=window, polyorder=poly, mode="interp")
    # mantém NaNs originais
    ys[~mask] = np.nan
    return ys

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares baseline (Eilers & Boelens).
    lam: rigidez (10^4 a 10^7 típico), p: assimetria (0.001 a 0.1), niter: iterações.
    """
    # Trabalha só nos pontos válidos e reintroduz NaNs depois
    idx = np.arange(len(y))
    mask = ~np.isnan(y)
    yv = y[mask]
    L = yv.size
    if L < 3:
        return np.full_like(y, np.nan)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * yv)
        w = p * (yv > z) + (1 - p) * (yv <= z)

    z_full = np.full_like(y, np.nan, dtype=float)
    z_full[mask] = z
    return z_full

# =========================
# App
# =========================
st.set_page_config(page_title="Conversão + Suavização + Baseline", layout="wide")

st.title("Conversão automática (ponto/vírgula) + Suavização e Correção de Linha de Base")
st.caption("Carregue CSV/TXT, converta números automaticamente, suavize curvas e remova baseline (ALS).")

uploaded = st.file_uploader("Envie um arquivo .csv ou .txt", type=["csv", "txt"])

with st.expander("Preferências de leitura", expanded=False):
    read_hint = st.checkbox("Forçar leitura como texto (dtype=str)", value=True)
    st.caption("Recomendado: impede o pandas de 'adivinhar' números; a conversão é feita pelos detectores.")
    custom_sep = st.text_input("Separador (opcional, deixe vazio para auto-detecção)", value="")

# Leitura robusta
df = None
last_err = None
if uploaded:
    try_chain = []
    if custom_sep.strip():
        try_chain = [
            dict(sep=custom_sep.strip(), engine="python", encoding=None, dtype=str if read_hint else None),
            dict(sep=custom_sep.strip(), engine="python", encoding="utf-8", dtype=str if read_hint else None),
            dict(sep=custom_sep.strip(), engine="python", encoding="latin1", dtype=str if read_hint else None),
        ]
    else:
        try_chain = [
            dict(sep=None, engine="python", encoding=None, dtype=str if read_hint else None),
            dict(sep=None, engine="python", encoding="utf-8", dtype=str if read_hint else None),
            dict(sep=None, engine="python", encoding="latin1", dtype=str if read_hint else None),
        ]
    for kw in try_chain:
        try:
            df = pd.read_csv(uploaded, **{k: v for k, v in kw.items() if v is not None})
            break
        except Exception as e:
            last_err = e

    if df is None:
        st.error(f"Falha ao ler arquivo. Último erro: {last_err}")
        st.stop()

    st.subheader("Pré-visualização")
    st.dataframe(df.head(20), use_container_width=True)

    # Seleções
    cols = list(df.columns)
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        col_x = st.selectbox("Coluna X", options=cols)
    with c2:
        y_candidates = [c for c in cols if c != col_x]
        y_cols = st.multiselect("Colunas Y", options=y_candidates, default=y_candidates[:1])
    with c3:
        order_x = st.checkbox("Ordenar por X crescente", value=True)

    st.markdown("---")

    # Barra lateral: parâmetros de suavização e baseline
    st.sidebar.header("Processamento")
    smooth_on = st.sidebar.checkbox("Aplicar suavização", value=False)
    smooth_kind = st.sidebar.selectbox("Tipo de suavização", ["Média móvel", "Savitzky-Golay"], index=1)
    mov_window = st.sidebar.slider("Janela (média móvel)", 3, 201, 11, step=2)
    sg_window = st.sidebar.slider("Janela (Savitzky-Golay)", 3, 201, 21, step=2)
    sg_poly = st.sidebar.slider("Ordem do polinômio (Sav-Gol)", 1, 7, 3, step=1)

    baseline_on = st.sidebar.checkbox("Corrigir linha de base (ALS)", value=False)
    als_lambda = st.sidebar.slider("λ (rigidez) [log10]", 3, 8, 5, step=1,
                                   help="Internamente usa 10^valor. 5 → 1e5")
    als_p = st.sidebar.slider("p (assimetria)", 0.001, 0.100, 0.010, step=0.001)

    show_original = st.sidebar.checkbox("Mostrar curva original", value=True)
    show_baseline = st.sidebar.checkbox("Mostrar baseline estimada", value=False)
    show_processed = st.sidebar.checkbox("Mostrar curva processada", value=True)

    st.sidebar.header("Estilo do gráfico")
    line_width = st.sidebar.slider("Espessura da linha", 1, 8, 3)
    # Cores por série
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c",
                       "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                       "#bcbd22", "#17becf"]
    color_pickers: Dict[str, str] = {}
    for i, c in enumerate(y_cols):
        color_pickers[c] = st.sidebar.color_picker(f"Cor de '{c}'", value=default_palette[i % len(default_palette)])

    st.sidebar.header("Tamanhos de texto")
    title_size = st.sidebar.slider("Título (px)", 10, 36, 18)
    label_size = st.sidebar.slider("Rótulos dos eixos (px)", 8, 28, 14)
    tick_size = st.sidebar.slider("Ticks (px)", 6, 24, 12)
    legend_size = st.sidebar.slider("Legenda (px)", 8, 24, 12)

    if st.button("Converter e Processar", type="primary"):
        if not y_cols:
            st.warning("Selecione pelo menos uma coluna Y.")
            st.stop()

        # Converte X e Y selecionadas
        cols_to_convert = [col_x] + y_cols
        df_conv, meta = convert_columns_auto(df, cols_to_convert)

        # Diagnóstico
        st.markdown("### Detecção de formato por coluna")
        msgs = []
        for c, info in meta.items():
            msgs.append(
                f"**{c}** → decimal: '{info['decimal']}', milhar: '{info['thousands']}', "
                f"não-numéricos após limpeza: {info['non_numeric']}"
            )
        st.markdown("- " + "\n- ".join(msgs))

        # Data numérica
        x = df_conv[col_x + "_num"]
        data_num = pd.DataFrame({col_x + "_num": x})
        for c in y_cols:
            data_num[c + "_num"] = df_conv[c + "_num"]

        # drop linhas sem X; ordenar
        data_num = data_num.dropna(subset=[col_x + "_num"])
        if order_x:
            data_num = data_num.sort_values(by=col_x + "_num")

        # Eixos e range X opcional
        with st.expander("Qualidade pós-conversão / NaNs por coluna", expanded=False):
            st.write(data_num.isna().sum())

        # Processamento
        lam = 10 ** als_lambda
        results = {}  # c -> dict(original, baseline, processed)
        for c in y_cols:
            y = data_num[c + "_num"].to_numpy()
            y_orig = y.copy()

            # baseline (opcional)
            if baseline_on:
                z = baseline_als(y, lam=lam, p=als_p, niter=10)
            else:
                z = np.full_like(y, np.nan)

            # remove baseline
            y_detrended = y_orig - z if baseline_on else y_orig.copy()

            # suavização (opcional)
            if smooth_on:
                if smooth_kind == "Média móvel":
                    y_smooth = moving_average(y_detrended, mov_window)
                else:
                    y_smooth = safe_savgol(y_detrended, sg_window, sg_poly)
            else:
                y_smooth = y_detrended

            results[c] = dict(original=y_orig, baseline=z, processed=y_smooth)

        # Plot
        fig = go.Figure()
        X = data_num[col_x + "_num"].to_numpy()

        for c in y_cols:
            col = color_pickers.get(c, None)

            if show_original:
                fig.add_trace(go.Scatter(
                    x=X, y=results[c]["original"],
                    mode="lines",
                    name=f"{c} (orig.)",
                    line=dict(width=max(1, line_width-1), dash="dot", color=col),
                    connectgaps=False,
                ))

            if show_baseline and baseline_on:
                fig.add_trace(go.Scatter(
                    x=X, y=results[c]["baseline"],
                    mode="lines",
                    name=f"{c} (baseline)",
                    line=dict(width=max(1, line_width-1), dash="dash", color=col),
                    connectgaps=False,
                ))

            if show_processed:
                fig.add_trace(go.Scatter(
                    x=X, y=results[c]["processed"],
                    mode="lines",
                    name=f"{c} (proc.)",
                    line=dict(width=line_width, color=col),
                    connectgaps=False,
                ))

        x_label = col_x
        y_label = ", ".join(y_cols)
        fig.update_layout(
            title="Curvas – original, baseline e processada",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            legend_title_text="Séries",
            margin=dict(l=50, r=20, t=40, b=50),
            font=dict(size=12),
        )
        fig.update_layout(
            title_font_size=title_size,
            legend=dict(font=dict(size=legend_size)),
            xaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
            yaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Monta DataFrame processado para exportação
        out = pd.DataFrame({x_label: X})
        for c in y_cols:
            out[f"{c}_original"] = results[c]["original"]
            if baseline_on:
                out[f"{c}_baseline"] = results[c]["baseline"]
            out[f"{c}_processada"] = results[c]["processed"]

        # Botões de download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV processado",
            data=csv_bytes,
            file_name="series_processadas.csv",
            mime="text/csv",
        )

        # PNG em alta resolução com fallback amigável
        try:
            import plotly.io as pio
            png = pio.to_image(fig, format="png", width=2000, height=1200, scale=1)  # ~2K
            st.download_button(
                "Baixar PNG (2000×1200)",
                data=png,
                file_name="grafico.png",
                mime="image/png",
            )
        except Exception as e:
            st.info(
                "Exportação PNG indisponível neste ambiente. "
                "Ative o ícone de câmera na barra do gráfico ou instale `kaleido`."
            )
else:
    st.info("Envie um arquivo CSV/TXT. O app detecta automaticamente vírgula/ponto e separador de milhar por coluna.")


