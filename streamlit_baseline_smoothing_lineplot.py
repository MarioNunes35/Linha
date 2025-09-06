# streamlit_baseline_smoothing_lineplot_auto.py
import csv
import re
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =========================
# Detectores automáticos (ponto/vírgula e milhar)
# =========================
PT_BR_PATTERN = r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?\d+,\d+$"
EN_US_PATTERN = r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d+\.\d+$"

def _score_pattern(series: pd.Series, pattern: str) -> int:
    return series.astype(str).str.fullmatch(pattern).sum()

def _keep_first_dot(x: str) -> str:
    if x.count(".") <= 1:
        return x
    i = x.find(".")
    return x[:i+1] + x[i+1:].replace(".", "")

def _parse_with(series: pd.Series, decimal: str, thousands: str) -> pd.Series:
    s = series.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    s = s.str.replace(r"[^0-9\-\+,\.eE ]", "", regex=True)  # dígitos, sinais, ., ,, e/E, espaço
    s = s.str.replace(r"\s+", "", regex=True)               # remove espaços (evita "1 234,56")
    if thousands:
        s = s.str.replace(thousands, "", regex=False)
    s = s.str.replace(decimal, ".", regex=False)
    s = s.apply(_keep_first_dot)                             # mantêm só o primeiro ponto
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
    info = {"decimal": decimal_used, "thousands": thousands_used, "non_numeric": int(parsed.isna().sum())}
    return parsed.to_numpy(), info

def convert_columns_auto(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    out = df.copy()
    meta: Dict[str, Dict[str, object]] = {}
    for c in cols:
        vals, info = auto_numeric(out[c])
        out[c + "_num"] = vals
        meta[c] = info
    return out, meta


# =========================
# Suavização e baseline
# =========================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=max(1, window//2)).mean().to_numpy()

def safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = np.count_nonzero(~np.isnan(y))
    if n < 3:
        return y.copy()
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, n if n % 2 == 1 else n-1))
    poly = max(1, min(poly, window - 1))
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return y.copy()
    yi = np.interp(x, x[mask], y[mask])
    ys = savgol_filter(yi, window_length=window, polyorder=poly, mode="interp")
    ys[~mask] = np.nan
    return ys

def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
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
# Leitura “inteligente”: reconhece ; , | \t e **ESPAÇO**
# (com correção para casos como header "X;Y")
# =========================
def smart_read(uploaded_file, force_str: bool, custom_sep: str,
               allow_space_delim: bool, encodings: List[Optional[str]]):
    def rewind(f):
        try:
            f.seek(0)
        except Exception:
            pass

    dtype_opt = (str if force_str else None)

    # 1) Olha amostra para tentar adivinhar separador (header e csv.Sniffer)
    rewind(uploaded_file)
    raw = uploaded_file.read()
    rewind(uploaded_file)

    try:
        sample_text = raw[:65536].decode("utf-8")
    except UnicodeDecodeError:
        sample_text = raw[:65536].decode("latin1", errors="ignore")

    first_line = sample_text.splitlines()[0] if sample_text else ""
    sep_guess = None
    for cand in [";", ",", "|", "\t"] + ([" "] if allow_space_delim else []):
        if cand in first_line:
            sep_guess = cand
            break

    if sep_guess is None:
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=";,|\t " if allow_space_delim else ";,|\t")
            sep_guess = dialect.delimiter
        except Exception:
            sep_guess = None

    # 2) Ordem de tentativas
    attempts = []
    if custom_sep.strip():
        attempts.append(("custom", dict(sep=custom_sep.strip(), engine="python", dtype=dtype_opt)))
    if sep_guess:
        attempts.append((f"sniffer:{repr(sep_guess)}", dict(sep=sep_guess, engine="python", dtype=dtype_opt)))
    attempts.append(("sniff_pandas", dict(sep=None, engine="python", dtype=dtype_opt)))
    for s in [";", ",", "|", "\t"]:
        attempts.append((f"fixed:{s}", dict(sep=s, engine="python", dtype=dtype_opt)))
    if allow_space_delim:
        attempts.append(("space_ws", dict(delim_whitespace=True, engine="python", dtype=dtype_opt)))
        attempts.append(("space_re", dict(sep=r"\s+", engine="python", dtype=dtype_opt)))

    last_err, used, df = None, None, None
    for tag, kw in attempts:
        for enc in encodings:
            rewind(uploaded_file)
            try:
                df = pd.read_csv(uploaded_file, encoding=enc, **kw)
                used = (tag, enc or "auto")

                # Guarda de segurança:
                # se ainda veio UMA coluna e o header tem um separador claro (ex.: "X;Y"), relê forçando-o
                if df.shape[1] == 1 and not custom_sep.strip():
                    header = str(df.columns[0])
                    hint = next((d for d in [";", ",", "|", "\t", " "] if d in header), None)
                    if hint:
                        rewind(uploaded_file)
                        sep_kw = (r"\s+" if hint == " " else hint)
                        df = pd.read_csv(uploaded_file, sep=sep_kw, engine="python",
                                         encoding=enc, dtype=dtype_opt)
                        used = (f"header_hint:{repr(hint)}", enc or "auto")
                break
            except Exception as e:
                last_err, df = e, None
        if df is not None:
            break

    if df is None:
        raise RuntimeError(f"Falha ao ler arquivo. Último erro: {last_err}")

    meta = {
        "strategy": used[0],
        "encoding": used[1],
        "sep_guess": sep_guess,
        "header": first_line[:200]
    }
    return df, meta


# =========================
# App
# =========================
st.set_page_config(page_title="Conversão + Suavização + Baseline (auto separador)", layout="wide")
st.title("Conversão automática + Suavização + Linha de Base — com auto separador (; , | \\t e ESPAÇO)")
st.caption("Resolve casos como 'X;Y' em uma coluna, detecta decimal/milhar por coluna e processa séries.")

uploaded = st.file_uploader("Envie um arquivo .csv ou .txt", type=["csv", "txt"])

with st.expander("Preferências de leitura", expanded=False):
    force_str = st.checkbox("Forçar leitura como texto (dtype=str)", value=True)
    custom_sep = st.text_input("Separador (opcional; deixe vazio para auto)", value="")
    allow_space_delim = st.checkbox(
        "Permitir ESPAÇO como separador de COLUNAS",
        value=True,
        help="Ative para arquivos realmente separados por espaço. "
             "Se seus números usam espaço como milhar (ex.: '1 234,56'), desative."
    )

if uploaded:
    try:
        df, read_meta = smart_read(
            uploaded,
            force_str=force_str,
            custom_sep=custom_sep,
            allow_space_delim=allow_space_delim,
            encodings=[None, "utf-8", "latin1"]
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Info de leitura
    strategy_desc = {
        "custom": "separador definido pelo usuário",
        "sniff_pandas": "auto-detecção (pandas/csv.Sniffer)",
    }
    tag = read_meta["strategy"]
    desc = strategy_desc.get(tag, tag)
    st.info(f"Leitura: {desc} | encoding: {read_meta['encoding']} | header: {read_meta['header']}")

    st.subheader("Pré-visualização")
    st.dataframe(df.head(20), use_container_width=True)

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

    # Sidebar: processamento
    st.sidebar.header("Processamento")
    smooth_on = st.sidebar.checkbox("Aplicar suavização", value=False)
    smooth_kind = st.sidebar.selectbox("Tipo de suavização", ["Média móvel", "Savitzky-Golay"], index=1)
    mov_window = st.sidebar.slider("Janela (média móvel)", 3, 201, 11, step=2)
    sg_window = st.sidebar.slider("Janela (Savitzky-Golay)", 3, 201, 21, step=2)
    sg_poly = st.sidebar.slider("Ordem do polinômio (Sav-Gol)", 1, 7, 3, step=1)

    baseline_on = st.sidebar.checkbox("Corrigir linha de base (ALS)", value=False)
    als_lambda = st.sidebar.slider("λ (rigidez) [log10]", 3, 8, 5, step=1, help="Internamente usa 10^valor (5 → 1e5)")
    als_p = st.sidebar.slider("p (assimetria)", 0.001, 0.100, 0.010, step=0.001)

    show_original = st.sidebar.checkbox("Mostrar curva original", value=True)
    show_baseline = st.sidebar.checkbox("Mostrar baseline estimada", value=False)
    show_processed = st.sidebar.checkbox("Mostrar curva processada", value=True)

    st.sidebar.header("Estilo do gráfico")
    line_width = st.sidebar.slider("Espessura da linha", 1, 8, 3)
    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
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
        else:
            cols_to_convert = [col_x] + y_cols
            df_conv, meta = convert_columns_auto(df, cols_to_convert)

            st.markdown("### Detecção de formato por coluna")
            st.markdown("- " + "\n- ".join(
                f"**{c}** → decimal: '{info['decimal']}', milhar: '{info['thousands']}', "
                f"não-numéricos: {info['non_numeric']}"
                for c, info in meta.items()
            ))

            data_num = pd.DataFrame({col_x + "_num": df_conv[col_x + "_num"]})
            for c in y_cols:
                data_num[c + "_num"] = df_conv[c + "_num"]

            data_num = data_num.dropna(subset=[col_x + "_num"])
            if order_x:
                data_num = data_num.sort_values(by=col_x + "_num")

            with st.expander("NaNs por coluna (após conversão)", expanded=False):
                st.write(data_num.isna().sum())

            lam = 10 ** als_lambda
            results = {}
            for c in y_cols:
                y = data_num[c + "_num"].to_numpy()
                y_orig = y.copy()
                if baseline_on:
                    z = baseline_als(y, lam=lam, p=als_p, niter=10)
                else:
                    z = np.full_like(y, np.nan)
                y_detrended = y_orig - z if baseline_on else y_orig.copy()
                if smooth_on:
                    y_smooth = (moving_average(y_detrended, mov_window)
                                if smooth_kind == "Média móvel"
                                else safe_savgol(y_detrended, sg_window, sg_poly))
                else:
                    y_smooth = y_detrended
                results[c] = dict(original=y_orig, baseline=z, processed=y_smooth)

            X = data_num[col_x + "_num"].to_numpy()
            fig = go.Figure()
            for c in y_cols:
                col = color_pickers.get(c)
                if show_original:
                    fig.add_trace(go.Scatter(
                        x=X, y=results[c]["original"], mode="lines",
                        name=f"{c} (orig.)", line=dict(width=max(1, line_width-1), dash="dot", color=col),
                        connectgaps=False,
                    ))
                if show_baseline and baseline_on:
                    fig.add_trace(go.Scatter(
                        x=X, y=results[c]["baseline"], mode="lines",
                        name=f"{c} (baseline)", line=dict(width=max(1, line_width-1), dash="dash", color=col),
                        connectgaps=False,
                    ))
                if show_processed:
                    fig.add_trace(go.Scatter(
                        x=X, y=results[c]["processed"], mode="lines",
                        name=f"{c} (proc.)", line=dict(width=line_width, color=col),
                        connectgaps=False,
                    ))

            fig.update_layout(
                title="Curvas — original, baseline e processada",
                xaxis_title=col_x,
                yaxis_title=", ".join(y_cols),
                template="plotly_white",
                legend_title_text="Séries",
                margin=dict(l=50, r=20, t=40, b=50),
            )
            fig.update_layout(
                title_font_size=title_size,
                legend=dict(font=dict(size=legend_size)),
                xaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
                yaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
            )
            st.plotly_chart(fig, use_container_width=True)

            out = pd.DataFrame({col_x: X})
            for c in y_cols:
                out[f"{c}_original"] = results[c]["original"]
                if baseline_on:
                    out[f"{c}_baseline"] = results[c]["baseline"]
                out[f"{c}_processada"] = results[c]["processed"]

            st.download_button(
                "Baixar CSV processado",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="series_processadas.csv",
                mime="text/csv",
            )

            try:
                import plotly.io as pio
                png = pio.to_image(fig, format="png", width=2000, height=1200, scale=1)
                st.download_button("Baixar PNG (2000×1200)", data=png, file_name="grafico.png", mime="image/png")
            except Exception:
                st.info("PNG indisponível neste ambiente. Use o ícone de câmera do Plotly ou instale `kaleido`.")

else:
    st.info("Envie um arquivo CSV/TXT. O app detecta ; , | \\t e, se habilitado, **ESPAÇO** como separador.")




