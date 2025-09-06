# Streamlit – Baseline + Smoothing + Line Plot (Minimalist Tool)
# Author: ChatGPT (GPT-5 Thinking) - Modified
# Purpose:
#   A lightweight app that ONLY does: (1) baseline correction, (2) smoothing,
#   and (3) line plotting for one X column and multiple Y columns.
#   Supported baselines: None, AsLS (Eilers), Polynomial fit, Rolling-min.
#   Supported smoothing: None, Savitzky–Golay, Moving Average.

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
    from io import BytesIO, StringIO
    bio = BytesIO(file_bytes)
    
    # First, try to decode the bytes to string to handle different encodings
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode('latin-1')
        except:
            text = file_bytes.decode('utf-8', errors='ignore')
    
    # Try different delimiters
    for sep in ['\t', ',', ';', ' ', '|']:
        try:
            df = pd.read_csv(StringIO(text), sep=sep)
            # Check if the dataframe has at least 2 columns (meaningful split)
            if len(df.columns) >= 2:
                # Clean column names - remove extra spaces and standardize
                df.columns = df.columns.str.strip()
                return df
        except Exception:
            continue
    
    # If all else fails, try with whitespace as separator
    try:
        df = pd.read_csv(StringIO(text), delim_whitespace=True)
        df.columns = df.columns.str.strip()
        return df
    except:
        # Final fallback
        return pd.read_csv(StringIO(text))

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
file = st.sidebar.file_uploader("CSV/TXT com 1 coluna X e várias Y", type=["csv", "txt"])
if not file:
    st.info("Envie um arquivo CSV ou TXT. Ex.: dados de FTIR com Wavenumber e Absorbance.")
    st.stop()

try:
    raw = robust_read_csv(file.getvalue())
    cols = list(raw.columns)
    
    # Validate that we have meaningful data
    if len(raw) == 0:
        st.error("O arquivo está vazio ou não pôde ser lido corretamente.")
        st.stop()
    
    # Check if we have at least numeric data
    numeric_cols = []
    for col in cols:
        try:
            # Try to convert to numeric to check if it's a valid data column
            test_numeric = pd.to_numeric(raw[col], errors='coerce')
            if test_numeric.notna().sum() > len(raw) * 0.5:  # At least 50% valid numbers
                numeric_cols.append(col)
        except:
            pass
    
    if len(numeric_cols) == 0:
        st.error("Nenhuma coluna numérica foi encontrada no arquivo. Verifique o formato dos dados.")
        st.info("Formato esperado: arquivo CSV ou TXT com colunas separadas por vírgula, ponto-e-vírgula, tab ou espaço.")
        st.stop()
    
    # Show data preview
    st.sidebar.write("**Preview dos dados:**")
    st.sidebar.dataframe(raw.head(5), height=150)
    
    # Show detected columns
    st.sidebar.write("**Colunas detectadas:**")
    for i, col in enumerate(cols):
        is_numeric = "✅ numérica" if col in numeric_cols else "❌ não-numérica"
        nan_count = raw[col].isna().sum()
        st.sidebar.write(f"{i+1}. `{col}` ({len(raw[col])} valores, {nan_count} NaN) {is_numeric}")
    
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.info("Dica: Certifique-se de que o arquivo está em formato CSV com colunas separadas.")
    st.info("Se os dados X e Y estão em uma única coluna (ex: 'X;Y'), tente separá-los em duas colunas antes de fazer upload.")
    st.stop()

st.sidebar.subheader("Mapeamento")

# Intelligent default selection for X column
default_x_index = 0
if len(cols) > 0:
    # Look for common X column names
    x_keywords = ['wavenumber', 'wavelength', 'frequency', 'x', 'time', 'position', 'angle']
    for i, col in enumerate(cols):
        if any(keyword in col.lower() for keyword in x_keywords):
            default_x_index = i + 1  # +1 because of "<None>" option
            break

col_x = st.sidebar.selectbox(
    "Coluna X (opcional)", 
    ["<None>"] + cols, 
    index=default_x_index,
    help="Selecione a coluna para o eixo X. Se '<None>', usará o índice."
)

# Y columns selection
available_y_cols = [c for c in cols if c != col_x]
if not available_y_cols:
    available_y_cols = cols  # If only one column, allow it to be used as Y

# Default Y selection - select all non-X columns by default (up to 5)
default_y = available_y_cols[:min(5, len(available_y_cols))]

ys = st.sidebar.multiselect(
    "Colunas Y", 
    available_y_cols, 
    default=default_y,
    help="Selecione as colunas para plotar no eixo Y"
)

if not ys:
    st.warning("⚠️ Selecione ao menos uma coluna Y para plotar.")
    st.stop()

# Pre-process
df = raw.copy()
if col_x == "<None>":
    x = np.arange(len(df))
    x_label = "Índice"
else:
    x = pd.to_numeric(df[col_x], errors='coerce').to_numpy()
    x_label = col_x

# Check for NaN values in X
if col_x != "<None>" and np.isnan(x).any():
    st.warning(f"⚠️ A coluna X '{col_x}' contém {np.isnan(x).sum()} valores não-numéricos que serão ignorados.")
    valid_mask = ~np.isnan(x)
    x = x[valid_mask]
    df = df[valid_mask].reset_index(drop=True)

Y = {}
for c in ys:
    y_values = pd.to_numeric(df[c], errors='coerce').to_numpy()
    if np.isnan(y_values).any():
        st.sidebar.warning(f"⚠️ Coluna '{c}' tem {np.isnan(y_values).sum()} valores NaN")
    Y[c] = y_values

st.sidebar.header("Baseline")
baseline_method = st.sidebar.selectbox("Método", ["Nenhum", "AsLS (Eilers)", "Polinomial", "Rolling Min"], index=1)

if baseline_method == "AsLS (Eilers)":
    lam = st.sidebar.number_input("λ (suavidade)", value=1e6, min_value=1e2, max_value=1e10, step=1e5, format="%.0e")
    p_asls = st.sidebar.slider("p (assimetria)", 0.001, 0.5, 0.01)
    niter = st.sidebar.slider("Iterações", 5, 60, 15)
elif baseline_method == "Polinomial":
    deg = st.sidebar.number_input("Grau do polinômio", value=2, min_value=1, max_value=8)
elif baseline_method == "Rolling Min":
    roll_w = st.sidebar.slider("Janela (pontos)", 5, 501, 51, step=2)

apply_baseline = st.sidebar.selectbox("Aplicar baseline em", ["Sinal original", "Após suavização"], index=0)
subtract_mode = st.sidebar.selectbox("Correção final", ["Subtrair baseline (y - base)", "Dividir (y/base)"], index=0)

st.sidebar.header("Suavização")
smooth_method = st.sidebar.selectbox("Método", ["Nenhuma", "Savitzky–Golay", "Média móvel"], index=1)

if smooth_method == "Savitzky–Golay":
    sg_w = st.sidebar.slider("Janela (pontos)", 5, 201, 21, step=2)
    sg_p = st.sidebar.slider("Ordem polinomial", 2, 5, 3)
elif smooth_method == "Média móvel":
    ma_w = st.sidebar.slider("Janela (pontos)", 3, 201, 11, step=2)

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
    
    # Skip processing if all values are NaN
    if np.isnan(y0).all():
        st.warning(f"⚠️ Coluna '{name}' contém apenas valores NaN. Pulando processamento.")
        continue

    # Replace NaN with interpolated values for processing
    if np.isnan(y0).any():
        mask = ~np.isnan(y0)
        if mask.sum() > 1:  # Need at least 2 points to interpolate
            y0 = np.interp(np.arange(len(y0)), np.arange(len(y0))[mask], y0[mask])

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
        if smooth_method == "Savitzky–Golay":
            y_out = smooth_savgol(y_corr, window=int(sg_w), poly=int(sg_p))
        elif smooth_method == "Média móvel":
            y_out = smooth_moving_average(y_corr, window=int(ma_w))
        else:
            y_out = y_corr

    else:  # baseline after smoothing
        if smooth_method == "Savitzky–Golay":
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
if not processed:
    st.error("Nenhum dado foi processado para plotagem!")
    st.stop()

# Create main plot
fig = go.Figure()

# Color palette for better visualization
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

plot_count = 0
for i, name in enumerate(ys):
    if name not in processed:  # Skip if processing failed
        continue
    
    color = colors[i % len(colors)]
    
    if show_orig and name in Y:
        fig.add_trace(go.Scatter(
            x=x, y=Y[name], 
            mode='lines', 
            name=f"{name} (original)", 
            line=dict(width=1, color=color), 
            opacity=0.3
        ))
        plot_count += 1
    
    if show_base and baseline_method != "Nenhum" and name in baselines:
        fig.add_trace(go.Scatter(
            x=x, y=baselines[name], 
            mode='lines', 
            name=f"{name} (baseline)", 
            line=dict(width=2, dash='dash', color=color),
            opacity=0.6
        ))
        plot_count += 1
    
    if show_proc and name in processed:
        fig.add_trace(go.Scatter(
            x=x, y=processed[name], 
            mode='lines', 
            name=f"{name} (processado)", 
            line=dict(width=2, color=color)
        ))
        plot_count += 1

if plot_count == 0:
    st.error("Nenhuma série foi adicionada ao gráfico. Verifique as configurações de plotagem.")
    st.stop()

# Update layout
fig.update_layout(
    template='plotly_white',
    height=620,
    title=f"Processamento de {len(ys)} série(s)",
    xaxis_title=x_label,
    yaxis_title="Intensidade",
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

if logx:
    fig.update_xaxes(type='log')
if logy:
    fig.update_yaxes(type='log')

st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Statistics --------------------------------- #
st.subheader("📊 Estatísticas")
col1, col2 = st.columns(2)

with col1:
    st.write("**Dados Originais**")
    stats_orig = pd.DataFrame({
        'Série': ys,
        'Mínimo': [Y[name].min() for name in ys if name in Y],
        'Máximo': [Y[name].max() for name in ys if name in Y],
        'Média': [Y[name].mean() for name in ys if name in Y],
        'Std': [Y[name].std() for name in ys if name in Y]
    })
    st.dataframe(stats_orig, use_container_width=True)

with col2:
    st.write("**Dados Processados**")
    if processed:
        stats_proc = pd.DataFrame({
            'Série': list(processed.keys()),
            'Mínimo': [processed[name].min() for name in processed],
            'Máximo': [processed[name].max() for name in processed],
            'Média': [processed[name].mean() for name in processed],
            'Std': [processed[name].std() for name in processed]
        })
        st.dataframe(stats_proc, use_container_width=True)

# ------------------------------- Export ---------------------------------- #
st.subheader("💾 Exportar")

col1, col2 = st.columns(2)

with col1:
    # Processed CSV
    if processed:
        proc_df = pd.DataFrame({
            x_label: x,
            **{f"{k}_original": Y[k] for k in ys if k in Y},
            **{f"{k}_baseline": baselines[k] for k in baselines},
            **{f"{k}_processado": processed[k] for k in processed}
        })
        
        buf = io.StringIO()
        proc_df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Baixar CSV (todos os dados)", 
            buf.getvalue(), 
            file_name="dados_processados.csv", 
            mime="text/csv"
        )

with col2:
    # HTML figure
    html = pio.to_html(fig, include_plotlyjs='cdn')
    st.download_button(
        "⬇️ Baixar Gráfico HTML Interativo", 
        data=html, 
        file_name="grafico_processado.html",
        mime="text/html"
    )

# ------------------------------- Notes ----------------------------------- #
with st.expander("📖 Notas & Instruções"):
    st.markdown(
        """
        ### Como usar:
        1. **Upload**: Faça upload de um arquivo CSV ou TXT com dados tabulares
        2. **Seleção de colunas**: Escolha a coluna X (opcional) e as colunas Y para processar
        3. **Baseline**: Configure o método de correção de linha de base
        4. **Suavização**: Aplique filtros para reduzir ruído
        5. **Visualização**: Ajuste as opções de exibição do gráfico
        6. **Export**: Baixe os dados processados ou o gráfico interativo
        
        ### Métodos disponíveis:
        
        **Correção de Baseline:**
        - **AsLS (Eilers)**: Asymmetric Least Squares - ótimo para espectros com picos
        - **Polinomial**: Ajuste polinomial global - bom para tendências suaves
        - **Rolling Min**: Mínimo móvel - útil para encontrar o envelope inferior
        
        **Suavização:**
        - **Savitzky-Golay**: Preserva características dos picos enquanto remove ruído
        - **Média Móvel**: Suavização simples por média de janela deslizante
        
        ### Dicas:
        - Para espectros FTIR/Raman: use AsLS com λ=1e6-1e8 e p=0.001-0.01
        - Janela Savitzky-Golay: maior = mais suave, menor = preserva mais detalhes
        - Pipeline: escolha se aplica baseline antes ou depois da suavização
        """
    )
