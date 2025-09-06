# app_auto_decimal.py
import re
import io
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# Helpers: detecção automática
# =========================
PT_BR_PATTERN = r"^-?\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^-?\d+,\d+$"
EN_US_PATTERN = r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d+\.\d+$"

def _score_pattern(series: pd.Series, pattern: str) -> int:
    return series.str.fullmatch(pattern).sum()

def _parse_with(series: pd.Series, decimal: str, thousands: str) -> pd.Series:
    s = series.str.replace("\xa0", " ", regex=False).str.strip()
    # mantém dígitos, sinais, ponto, vírgula, 'e/E' (notação científica) e espaços
    s = s.str.replace(r"[^0-9\-\+,\.eE ]", "", regex=True)
    # remove espaços (muitas vezes usados como milhar)
    s = s.str.replace(r"\s+", "", regex=True)
    # remove separador de milhar e normaliza decimal para ponto
    if thousands:
        s = s.str.replace(thousands, "", regex=False)
    s = s.str.replace(decimal, ".", regex=False)
    # se sobrou mais de um ponto, mantém só o primeiro (decimal) e remove o resto
    s = s.apply(lambda x: x if x.count(".") <= 1 else x.replace(".", "", x.count(".")-1))
    return pd.to_numeric(s, errors="coerce")

def auto_numeric(series: pd.Series) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Detecta decimal/milhar automaticamente para uma série textual e
    retorna (valores_float_numpy, info_dict).
    """
    raw = series.astype(str)

    # Heurística por padrão textual
    pt_hits = _score_pattern(raw, PT_BR_PATTERN)
    en_hits = _score_pattern(raw, EN_US_PATTERN)

    if pt_hits > en_hits:
        candidates = [(",", "."), (".", ",")]  # preferir pt-BR
    elif en_hits > pt_hits:
        candidates = [(".", ","), (",", ".")]  # preferir en-US
    else:
        candidates = [(".", ","), (",", ".")]  # testar ambos

    best = None
    best_score = (-1, -1, -np.inf)  # (n_ok, n_com_frac, -mediana_magnitude)
    for dec, thou in candidates:
        parsed = _parse_with(raw, decimal=dec, thousands=thou)
        n_ok = parsed.notna().sum()
        # presença do decimal escolhido (ajuda a diferenciar decimal real de milhar)
        has_frac_guess = (raw.str.contains(re.escape(dec)) & ~raw.str.contains(re.escape(thou))).sum()
        mag = parsed.dropna().abs().median() if n_ok else np.inf
        score = (n_ok, has_frac_guess, -float(mag) if np.isfinite(mag) else -np.inf)
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
    """
    Converte as colunas indicadas usando auto_numeric.
    Retorna um novo DataFrame com colunas *_num e metadados por coluna.
    """
    out = df.copy()
    meta: Dict[str, Dict[str, object]] = {}
    for c in cols:
        vals, info = auto_numeric(out[c].astype(str))
        out[c + "_num"] = vals
        meta[c] = info
    return out, meta


# =========================
# App
# =========================
st.set_page_config(page_title="Leitor com detecção automática de decimal/milhar", layout="wide")

st.title("Leitor (CSV/TXT) com detecção automática de ponto/vírgula")
st.caption("Lê o arquivo, detecta separador decimal/milhar por coluna, converte para número e plota.")

uploaded = st.file_uploader("Envie um arquivo .csv ou .txt", type=["csv", "txt"])

read_hint = st.checkbox("Forçar leitura como texto bruto (dtype=str)", value=False,
                        help="Se marcado, o pandas não tenta converter nada na leitura. "
                             "A conversão numérica será feita só depois, com o detector automático.")

if uploaded:
    # Tentativa robusta de leitura (detecta sep automaticamente). Fallback de encoding.
    try_orders = [
        dict(sep=None, engine="python", encoding=None, dtype=str if read_hint else None),
        dict(sep=None, engine="python", encoding="utf-8", dtype=str if read_hint else None),
        dict(sep=None, engine="python", encoding="latin1", dtype=str if read_hint else None),
    ]
    last_err = None
    df = None
    for kwargs in try_orders:
        try:
            df = pd.read_csv(uploaded, **{k: v for k, v in kwargs.items() if v is not None})
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        st.error(f"Falha ao ler o arquivo. Erro final: {last_err}")
        st.stop()

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    with col1:
        x_col = st.selectbox("Coluna X (texto ou numérica)", options=list(df.columns))
    with col2:
        # Sugere Y como todas as colunas exceto X
        y_candidates = [c for c in df.columns if c != x_col]
        y_cols = st.multiselect("Colunas Y (uma ou mais)", options=y_candidates, default=y_candidates[:1])
    with col3:
        order_x = st.checkbox("Ordenar por X crescente", value=True)
    st.divider()

    if st.button("Converter e Plotar", type="primary"):
        # Detecta/Converte X + Y
        cols_to_convert = [x_col] + y_cols
        df_conv, meta = convert_columns_auto(df, cols_to_convert)

        # Mensagens de diagnóstico
        diag = []
        for c, info in meta.items():
            diag.append(
                f"**{c}** → decimal: '{info['decimal']}', milhar: '{info['thousands']}', "
                f"não-numéricos após limpeza: {info['non_numeric']}"
            )
        st.markdown("### Detecção por coluna")
        st.markdown("- " + "\n- ".join(diag))

        # Monta DataFrame numérico para plot
        x_num = df_conv[x_col + "_num"]
        data_num = pd.DataFrame({x_col + "_num": x_num})
        for c in y_cols:
            data_num[c + "_num"] = df_conv[c + "_num"]

        # Remove linhas com NaN em X
        data_num = data_num.dropna(subset=[x_col + "_num"])
        # (opcional) remove linha NaN de Y também? Mantemos, mas cada traço ignora NaNs.
        if order_x:
            data_num = data_num.sort_values(by=x_col + "_num")

        # Contagens de NaN por coluna numérica
        with st.expander("Qualidade pós-conversão (NaNs por coluna)", expanded=False):
            st.write(data_num.isna().sum())

        # Configurações de plot
        st.markdown("### Configurações do gráfico")
        cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 1, 1])
        with cfg1:
            line_width = st.slider("Espessura da linha", 1, 8, 3)
        with cfg2:
            show_markers = st.checkbox("Mostrar marcadores", value=False)
        with cfg3:
            x_label = st.text_input("Rótulo eixo X", value=x_col)
        with cfg4:
            y_label = st.text_input("Rótulo eixo Y", value=", ".join(y_cols) if y_cols else "Y")

        fig = go.Figure()
        for c in y_cols:
            y_num = data_num[c + "_num"]
            fig.add_trace(
                go.Scatter(
                    x=data_num[x_col + "_num"],
                    y=y_num,
                    mode="lines+markers" if show_markers else "lines",
                    name=c,
                    line=dict(width=line_width),
                    connectgaps=False,
                )
            )

        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            legend_title_text="Séries",
            margin=dict(l=50, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Disponibiliza CSV limpo (com colunas *_num)
        cleaned = df_conv.copy()
        # Opcional: sobrescrever colunas originais com _num quando existir
        for c in cols_to_convert:
            cleaned[c] = cleaned[c + "_num"]
        # remove colunas auxiliares *_num duplicadas
        for c in cols_to_convert:
            aux = c + "_num"
            # deixe somente uma cópia (a original sobrescrita); apague a auxiliar
            if aux in cleaned.columns and aux != c:
                cleaned = cleaned.drop(columns=[aux])

        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV limpo",
            data=csv_bytes,
            file_name="dados_convertidos.csv",
            mime="text/csv",
        )

else:
    st.info("Envie um arquivo CSV/TXT para começar. Dica: muitos CSVs brasileiros usam ';' como separador; "
            "este app detecta automaticamente.")


# =========================
# Rodapé
# =========================
st.markdown("---")
st.caption(
    "Detecção automática tenta pt-BR e en-US, avalia taxa de conversão, presença de frações e magnitude típica "
    "para evitar números gigantes (ex.: 399.774.803). Compatível com notação científica."
)

