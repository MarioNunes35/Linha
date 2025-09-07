# streamlit_baseline_smoothing_lineplot_auto_v3.py
# Conversão automática de ponto/vírgula e milhar, suporte a separadores ; , | \t e ESPAÇO,
# correção de X com formatos "quebrados", suavização e baseline ALS (com modo Auto).

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
# Utilitários numéricos
# =========================
PT_BR_PATTERN = r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?\d+,\d+$"
EN_US_PATTERN = r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d+\.\d+$"

def _score_pattern(series: pd.Series, pattern: str) -> int:
    return series.astype(str).str.fullmatch(pattern).sum()

def _keep_first_dot(s: str) -> str:
    if s.count(".") <= 1:
        return s
    i = s.find(".")
    return s[:i+1] + s[i+1:].replace(".", "")

def _parse_with(series: pd.Series, decimal: str, thousands: str) -> pd.Series:
    s = series.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    # mantém dígitos, sinais, ., ,, e/E, e espaços (removemos depois)
    s = s.str.replace(r"[^0-9\-\+,\.eE ]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)  # remove espaços (evita "1 234,56")
    if thousands:
        s = s.str.replace(thousands, "", regex=False)
    s = s.str.replace(decimal, ".", regex=False)
    s = s.apply(_keep_first_dot)
    return pd.to_numeric(s, errors="coerce")

def auto_numeric(series: pd.Series) -> Tuple[np.ndarray, Dict[str, object]]:
    """Detecta decimal/milhar automaticamente e retorna (valores, info)."""
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


# =========================
# Parser especial p/ X – reconstrói números “quebrados”
# =========================
def parse_x_smart(series: pd.Series, xmin: float = 300.0, xmax: float = 4500.0) -> np.ndarray:
    """
    Conserta X com formatos como "399.774.803", "39.934.632", "3.183.630"
    e também lê corretamente "3994,89" / "3994.89".
    Estratégia:
      - Mantém só dígitos e testa decimal após 3 e 4 dígitos (###.resto | ####.resto)
      - Fallbacks: "primeiro ponto como decimal" e "último ponto como decimal"
      - Escolhe o candidato mais próximo da faixa [xmin, xmax], preferindo valores >= 1000 quando há empate
    """
    def conv_first(s: str) -> float:
        s = s.replace(",", ".")
        if "." in s:
            i = s.find(".")
            return float(s[:i+1] + s[i+1:].replace(".", ""))
        return float(s)

    def conv_last(s: str) -> float:
        s = s.replace(",", ".")
        if "." in s:
            i = s.rfind(".")
            return float(s[:i].replace(".", "") + "." + s[i+1:])
        return float(s)

    def best_for_one(x: str) -> float:
        s = str(x)
        digits = re.sub(r"\D", "", s)  # só dígitos
        cands: List[Tuple[str, float]] = []

        # dec após 3 e 4 dígitos
        for k in (3, 4):
            if len(digits) > k:
                try:
                    cands.append((f"k{k}", float(digits[:k] + "." + digits[k:])))
                except Exception:
                    pass
        # fallbacks
        try:
            cands.append(("first", conv_first(s)))
        except Exception:
            pass
        try:
            cands.append(("last", conv_last(s)))
        except Exception:
            pass
        if not cands:
            return np.nan

        def dist(v: float) -> float:
            if xmin <= v <= xmax:
                return 0.0
            return min(abs(v - xmin), abs(v - xmax))

        # escolhe o(s) com menor distância à faixa; dentro da faixa prefere >= 1000
        dists = [(dist(v), tag, v) for tag, v in cands if np.isfinite(v)]
        dmin = min(d for d, _, _ in dists)
        near = [(tag, v) for d, tag, v in dists if abs(d - dmin) < 1e-12]
        inside = [(tag, v) for tag, v in near if xmin <= v <= xmax]
        if inside:
            ge = [(tag, v) for tag, v in inside if v >= 1000]
            return max(ge or inside, key=lambda kv: kv[1])[1]
        return max(near, key=lambda kv: kv[1])[1]

    return series.astype(str).apply(best_for_one).to_numpy()


# =========================
# Suavização e baseline
# =========================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=max(1, window // 2)).mean().to_numpy()

def safe_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = np.count_nonzero(~np.isnan(y))
    if n < 3:
        return y.copy()
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, n if n % 2 == 1 else n - 1))
    poly = max(1, min(poly, window - 1))
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return y.copy()
    yi = np.interp(x, x[mask], y[mask])
    ys = savgol_filter(yi, window_length=window, polyorder=poly, mode="interp")
    ys[~mask] = np.nan
    return ys

def baseline_als(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 20) -> np.ndarray:
    mask = ~np.isnan(y)
    yv = y[mask]
    L = yv.size
    if L < 3:
        return np.full_like(y, np.nan)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * yv)
        w = p * (yv > z) + (1 - p) * (yv <= z)
    z_full = np.full_like(y, np.nan, dtype=float)
    z_full[mask] = z
    return z_full

def auto_baseline(y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Busca simples de (λ, p) e retorna o melhor trio (λ, p, z)."""
    lam_list = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
    p_list = [0.005, 0.01, 0.02, 0.05]
    best, best_score = (1e6, 0.01, None), -1e9
    for lam in lam_list:
        for p in p_list:
            z = baseline_als(y, lam=lam, p=p, niter=15)
            r = y - z
            valid = ~np.isnan(r)
            if valid.sum() < 10:
                continue
            prop_above = np.mean(r[valid] >= 0)
            dz = np.diff(z[~np.isnan(z)])
            stdy = np.nanstd(y)
            smooth_penalty = (np.nanstd(dz) / stdy) if stdy else 0.0
            score = prop_above - 0.1 * smooth_penalty
            if score > best_score:
                best, best_score = (lam, p, z), score
    return best


# =========================
# Leitura “inteligente” com correção para header "X;Y"
# =========================
def smart_read(uploaded_file, force_str: bool, custom_sep: str,
               allow_space_delim: bool, encodings: List[Optional[str]]):
    def rewind(f):
        try:
            f.seek(0)
        except Exception:
            pass

    dtype_opt = (str if force_str else None)

    # Peek
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
            dialect = csv.Sniffer().sniff(
                sample_text,
                delimiters=";,|\t " if allow_space_delim else ";,|\t"
            )
            sep_guess = dialect.delimiter
        except Exception:
            sep_guess = None

    # Tentativas
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
                # Se ainda veio 1 coluna e o header indica separador (ex.: "X;Y"), reler forçando-o
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
        "header": first_line[:200]
    }
    return df, meta


# =========================
# App
# =========================
st.set_page_config(page_title="Conversão + Suavização + Baseline (v3)", layout="wide")
st.title("Conversão automática + Suavização + Linha de Base — v3")
st.caption("• Lê ; , | \\t e ESPAÇO • Corrige X (ex.: '399.774.803' → 3997.74803; '3994,89' → 3994.89) "
           "• Converte Y automaticamente • Suaviza e remove baseline (ALS)")

uploaded = st.file_uploader("Envie um arquivo .csv ou .txt", type=["csv", "txt"])

with st.expander("Preferências de leitura", expanded=False):
    force_str = st.checkbox("Forçar leitura como texto (dtype=str)", value=True)
    custom_sep = st.text_input("Separador (opcional; deixe vazio para auto)", value="")
    allow_space_delim = st.checkbox(
        "Permitir ESPAÇO como separador de colunas",
        value=True,
        help="Ative apenas se seu arquivo for realmente separado por espaço; "
             "se seus números usam espaço como milhar, desative."
    )

# Sidebar – processamento e estilo
st.sidebar.header("Processamento")
xmin, xmax = st.sidebar.slider("Faixa-alvo para X", 100.0, 20000.0, (300.0, 4500.0))
smooth_on = st.sidebar.checkbox("Aplicar suavização", value=False)
smooth_kind = st.sidebar.selectbox("Tipo de suavização", ["Média móvel", "Savitzky-Golay"], index=1)
mov_window = st.sidebar.slider("Janela (média móvel)", 3, 201, 11, step=2)
sg_window = st.sidebar.slider("Janela (Savitzky-Golay)", 3, 201, 21, step=2)
sg_poly = st.sidebar.slider("Ordem do polinômio (Sav-Gol)", 1, 7, 3, step=1)

baseline_on = st.sidebar.checkbox("Corrigir linha de base (ALS)", value=False)
baseline_auto = st.sidebar.checkbox("Auto (λ, p)", value=True, help="Testa alguns (λ, p) e escolhe o melhor.")
als_lambda_log = st.sidebar.slider("λ (log10)", 3, 8, 6, step=1)
als_p = st.sidebar.slider("p (assimetria)", 0.001, 0.100, 0.010, step=0.001)

st.sidebar.header("Estilo do gráfico")
line_width = st.sidebar.slider("Espessura da linha", 1, 8, 3)
title_size = st.sidebar.slider("Título (px)", 10, 36, 18)
label_size = st.sidebar.slider("Rótulos dos eixos (px)", 8, 28, 14)
tick_size = st.sidebar.slider("Ticks (px)", 6, 24, 12)
legend_size = st.sidebar.slider("Legenda (px)", 8, 24, 12)

if uploaded:
    try:
        df, read_meta = smart_read(
            uploaded,
            force_str=force_str,
            custom_sep=custom_sep,
            allow_space_delim=allow_space_delim,
            encodings=[None, "utf-8", "latin1"],
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Pré-visualização")
    st.caption(f"Leitura: {read_meta['strategy']} | encoding: {read_meta['encoding']}")
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

    if st.button("Converter e Processar", type="primary"):
        if not y_cols:
            st.warning("Selecione pelo menos uma coluna Y.")
            st.stop()

        # X com parser especial
        X = parse_x_smart(df[col_x], xmin=float(xmin), xmax=float(xmax))

        # Y com detecção automática de decimal/milhar
        data_num = pd.DataFrame({col_x: X})
        metaY: Dict[str, Dict[str, object]] = {}
        for c in y_cols:
            arr, info = auto_numeric(df[c])
            data_num[c] = arr
            metaY[c] = info

        # Diagnóstico
        st.markdown("### Detecção de formato (Y)")
        st.markdown("- " + "\n- ".join(
            f"**{c}** → decimal: '{info['decimal']}', milhar: '{info['thousands']}', "
            f"não-numéricos: {info['non_numeric']}"
            for c, info in metaY.items()
        ))

        data_num = data_num.dropna(subset=[col_x])
        if order_x:
            data_num = data_num.sort_values(by=col_x)

        with st.expander("NaNs por coluna (após conversão)", expanded=False):
            st.write(data_num.isna().sum())

        # Processamento
        results: Dict[str, Dict[str, np.ndarray]] = {}
        Xv = data_num[col_x].to_numpy()

        for c in y_cols:
            yv = data_num[c].to_numpy()
            y_orig = yv.copy()

            if baseline_on:
                if baseline_auto:
                    lam, p, z = auto_baseline(y_orig)
                else:
                    lam = 10 ** als_lambda_log
                    p = als_p
                    z = baseline_als(y_orig, lam=lam, p=p, niter=20)
            else:
                z = np.full_like(y_orig, np.nan)

            y_detr = y_orig - z if baseline_on else y_orig.copy()

            if smooth_on:
                y_sm = moving_average(y_detr, mov_window) if smooth_kind == "Média móvel" else safe_savgol(y_detr, sg_window, sg_poly)
            else:
                y_sm = y_detr

            results[c] = dict(original=y_orig, baseline=z, processed=y_sm)

        # Plot (tema escuro p/ combinar com seu print)
        fig = go.Figure()
        for c in y_cols:
            if True:  # original
                fig.add_trace(go.Scatter(
                    x=Xv, y=results[c]["original"], mode="lines",
                    name=f"{c} (orig.)",
                    line=dict(width=max(1, line_width - 1), dash="dot"),
                    connectgaps=False,
                ))
            if baseline_on:
                fig.add_trace(go.Scatter(
                    x=Xv, y=results[c]["baseline"], mode="lines",
                    name=f"{c} (baseline)",
                    line=dict(width=max(1, line_width - 1), dash="dash"),
                    connectgaps=False,
                ))
            fig.add_trace(go.Scatter(
                x=Xv, y=results[c]["processed"], mode="lines",
                name=f"{c} (proc.)", line=dict(width=line_width),
                connectgaps=False,
            ))

        fig.update_layout(
            title="Curvas — original, baseline e processada",
            xaxis_title=col_x,
            yaxis_title=", ".join(y_cols),
            template="plotly_dark",
            legend_title_text="Séries",
            margin=dict(l=50, r=20, t=40, b=50),
            title_font_size=title_size,
            legend=dict(font=dict(size=legend_size)),
            xaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
            yaxis=dict(title_font=dict(size=label_size), tickfont=dict(size=tick_size)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Exporta CSV processado
        out = pd.DataFrame({col_x: Xv})
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

        # PNG (kaleido)
        try:
            import plotly.io as pio
            png = pio.to_image(fig, format="png", width=2000, height=1200, scale=1)
            st.download_button("Baixar PNG (2000×1200)", data=png, file_name="grafico.png", mime="image/png")
        except Exception:
            st.info("PNG indisponível aqui. Use o ícone de câmera do Plotly ou instale `kaleido`.")

else:
    st.info("Envie um CSV/TXT. Se seu arquivo usar espaço como milhar, desative a opção de ESPAÇO como separador.")





