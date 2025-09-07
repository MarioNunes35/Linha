# streamlit_baseline_smoothing_lineplot_improved.py
# FTIR/XY: Correção de linha base com seleção MANUAL e automática
# Versão corrigida e melhorada

import io, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d, UnivariateSpline

st.set_page_config(page_title="FTIR Baseline • Manual + Auto", layout="wide")
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# Inicializar session state
if 'manual_points' not in st.session_state:
    st.session_state.manual_points = []
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# ---------- IO ----------
def _safe_float(s: str) -> float:
    """Converte string para float, lidando com vírgulas"""
    return float(s.replace(",", "."))

def _parse_xy_text(text: str) -> pd.DataFrame:
    """Parse de texto com pares X,Y"""
    xs, ys = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        nums = NUM_RE.findall(line)
        if len(nums) >= 2:
            try:
                xs.append(_safe_float(nums[0]))
                ys.append(_safe_float(nums[1]))
            except Exception:
                continue
    if len(xs) < 3:
        raise ValueError("Não encontrei pares X,Y suficientes.")
    df = pd.DataFrame({"X": xs, "Y": ys})
    if not np.all(np.diff(df["X"].values) >= 0):
        df = df.sort_values("X").reset_index(drop=True)
    return df

def load_xy(uploaded) -> pd.DataFrame:
    """Carrega arquivo de dados X,Y"""
    raw = uploaded.read()
    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc, errors="ignore")
            break
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

# ---------- Funções auxiliares ----------
def infer_orientation(y: np.ndarray) -> str:
    """Detecta se os picos são para cima ou para baixo"""
    med = np.median(y)
    return "Picos para cima" if (np.max(y)-med) >= (med-np.min(y)) else "Picos para baixo"

def find_local_minima(x, y, window_size=50):
    """Encontra mínimos locais com melhor robustez"""
    minima = []
    # Ajusta window_size se for muito grande
    window_size = min(window_size, len(y)//4)
    window_size = max(5, window_size)  # Mínimo de 5 pontos
    
    for i in range(window_size, len(y) - window_size):
        window = y[max(0, i-window_size):min(len(y), i+window_size+1)]
        if len(window) > 0 and y[i] == np.min(window):
            # Verifica se não está muito próximo do último mínimo adicionado
            if not minima or i - minima[-1] > window_size//2:
                minima.append(i)
    return np.array(minima)

def find_local_maxima(x, y, window_size=50):
    """Encontra máximos locais com melhor robustez"""
    maxima = []
    # Ajusta window_size se for muito grande
    window_size = min(window_size, len(y)//4)
    window_size = max(5, window_size)  # Mínimo de 5 pontos
    
    for i in range(window_size, len(y) - window_size):
        window = y[max(0, i-window_size):min(len(y), i+window_size+1)]
        if len(window) > 0 and y[i] == np.max(window):
            # Verifica se não está muito próximo do último máximo adicionado
            if not maxima or i - maxima[-1] > window_size//2:
                maxima.append(i)
    return np.array(maxima)

# --- Asymmetric Least Squares (automático) ---
def asls_baseline(y, lam=1e6, p=0.001, niter=10):
    """Baseline automático por AsLS"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * (D.T @ D)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def create_manual_baseline(x, y, manual_indices, method='linear', smoothness=0.01):
    """Cria baseline a partir de pontos selecionados manualmente"""
    if len(manual_indices) < 2:
        return np.zeros_like(y)
    
    # Ordena os índices e garante que são inteiros válidos
    manual_indices = sorted([int(idx) for idx in manual_indices if 0 <= idx < len(x)])
    
    if len(manual_indices) < 2:
        return np.zeros_like(y)
    
    x_base = x[manual_indices]
    y_base = y[manual_indices]
    
    try:
        if method == 'linear':
            # Interpolação linear entre pontos
            f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
            baseline = f(x)
        elif method == 'spline_smooth':
            # Spline muito suave (mais reta)
            if len(manual_indices) > 3:
                spline = UnivariateSpline(x_base, y_base, s=smoothness*np.ptp(y), k=min(3, len(manual_indices)-1))
                baseline = spline(x)
            else:
                f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
                baseline = f(x)
        elif method == 'spline_curved':
            # Spline que passa exatamente pelos pontos (mais curva)
            if len(manual_indices) > 3:
                spline = UnivariateSpline(x_base, y_base, s=0, k=min(3, len(manual_indices)-1))
                baseline = spline(x)
            elif len(manual_indices) > 2:
                f = interp1d(x_base, y_base, kind='quadratic', fill_value='extrapolate', bounds_error=False)
                baseline = f(x)
            else:
                f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
                baseline = f(x)
        elif method == 'cubic':
            if len(manual_indices) > 3:
                f = interp1d(x_base, y_base, kind='cubic', fill_value='extrapolate', bounds_error=False)
                baseline = f(x)
            else:
                f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
                baseline = f(x)
        else:
            # Fallback para linear
            f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
            baseline = f(x)
            
    except Exception as e:
        st.warning(f"Erro na criação da baseline: {e}. Usando interpolação linear.")
        f = interp1d(x_base, y_base, kind='linear', fill_value='extrapolate', bounds_error=False)
        baseline = f(x)
    
    return baseline

def adaptive_peak_detection(x, y_norm, sensitivity='medium'):
    """Detecção adaptativa de picos"""
    params = {
        'baixa': {'prominence': 0.05, 'distance': 200, 'width': 20},
        'medium': {'prominence': 0.02, 'distance': 100, 'width': 10},
        'alta': {'prominence': 0.01, 'distance': 50, 'width': 5},
        'muito_alta': {'prominence': 0.005, 'distance': 20, 'width': 3}
    }
    
    p = params.get(sensitivity, params['medium'])
    
    try:
        peaks, properties = find_peaks(y_norm, 
                                      prominence=p['prominence'],
                                      distance=p['distance'],
                                      width=p['width'])
    except:
        peaks = np.array([])
        properties = {}
    
    return peaks, properties

def suggest_baseline_points(x, y, orientation, n_points=10):
    """Sugere pontos automáticos para a linha base"""
    if orientation == "Picos para baixo":
        # Procura máximos locais (topos entre os vales)
        candidates = find_local_maxima(x, y, window_size=max(5, len(y)//20))
    else:
        # Procura mínimos locais (vales entre os picos)
        candidates = find_local_minima(x, y, window_size=max(5, len(y)//20))
    
    # Adiciona pontos nas extremidades
    edge_points = [0, len(y)-1]
    
    if len(candidates) > 0:
        # Seleciona pontos distribuídos
        if len(candidates) > n_points - 2:
            step = len(candidates) // (n_points - 2)
            selected = candidates[::step][:n_points-2]
        else:
            selected = candidates
        
        all_points = sorted(list(edge_points) + list(selected))
    else:
        # Distribui pontos uniformemente
        all_points = np.linspace(0, len(y)-1, n_points, dtype=int).tolist()
    
    return all_points

# ==================== INTERFACE ====================
st.title("🔬 Correção de Linha Base FTIR - Modo Manual + Automático")

# Sidebar com instruções
with st.sidebar:
    st.markdown("### 📋 Instruções de Uso")
    st.markdown("""
    **Modo Manual:**
    1. Carregue seu arquivo de dados
    2. Ative o 'Modo de seleção manual'
    3. Digite os valores X onde deseja colocar pontos da linha base
    4. Ou use 'Sugerir pontos automáticos'
    5. Ajuste fino com os controles
    
    **Modo Automático:**
    1. Escolha o método (AsLS ou Detecção de picos)
    2. Ajuste a sensibilidade
    3. Fine-tune com os parâmetros
    
    **Modo Híbrido:**
    Combine pontos manuais com correção automática
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Dicas")
    st.info("""
    - Para FTIR: geralmente use 'Picos para baixo'
    - Coloque pontos base entre os picos principais
    - Use spline para baseline mais suave
    - Exporte o resultado quando satisfeito
    """)

# Upload de arquivo
up = st.file_uploader("📁 Carregue arquivo .txt/.csv/.dpt (2 colunas X,Y)", type=["txt","csv","dpt"])

if not up:
    st.info("👆 Carregue um arquivo para começar")
    st.stop()

# Verifica se é um novo arquivo
if st.session_state.last_file != up.name:
    st.session_state.manual_points = []
    st.session_state.last_file = up.name

# Carrega dados
try:
    df = load_xy(up)
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
except Exception as e:
    st.error(f"Erro ao carregar arquivo: {e}")
    st.stop()

# Configurações principais
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Pré-processamento**")
    smooth_on = st.checkbox("Suavizar (Savitzky-Golay)", True)
    if smooth_on:
        w = st.slider("Janela SG", 5, 201, 21, step=2)
        poly = st.slider("Ordem polinomial", 1, 5, 3)
    else:
        w, poly = 21, 3

with col2:
    st.markdown("**🎯 Orientação e Método**")
    orientation = st.selectbox("Orientação dos picos", 
                              ["Picos para baixo", "Picos para cima", "Detectar automaticamente"], 
                              index=0)
    baseline_method = st.selectbox("Método de correção",
                                  ["Manual", "Automático (AsLS)", "Detecção de picos", "Híbrido"],
                                  index=0)

with col3:
    st.markdown("**⚙️ Configurações**")
    
    # Sempre define as variáveis para evitar erros
    sensitivity = 'medium'
    interp_method = 'linear'
    
    if baseline_method in ["Detecção de picos", "Híbrido"]:
        sensitivity = st.select_slider("Sensibilidade da detecção",
                                      options=['baixa', 'medium', 'alta', 'muito_alta'],
                                      value='medium')
    
    if baseline_method in ["Manual", "Híbrido"]:
        interp_method = st.selectbox("Tipo de linha base", 
                                    ["linear", "spline_smooth", "spline_curved", "cubic"],
                                    format_func=lambda x: {
                                        'linear': 'Linear (reta entre pontos)',
                                        'spline_smooth': 'Spline suave (mais reta)',
                                        'spline_curved': 'Spline curva (segue pontos)',
                                        'cubic': 'Cúbica'
                                    }.get(x, x))

# Processamento inicial
y_proc = y.copy()
if smooth_on and len(y_proc) >= w:
    y_proc = savgol_filter(y_proc, window_length=w, polyorder=poly, mode="interp")

orientation_eff = infer_orientation(y_proc) if orientation == "Detectar automaticamente" else orientation

# Variável de suavidade (sempre definida)
smoothness = 0.01

# SEÇÃO DE SELEÇÃO MANUAL
if baseline_method in ["Manual", "Híbrido"]:
    st.markdown("---")
    st.markdown("### 🖱️ Seleção Manual de Pontos Base")
    
    # Controle de suavidade para spline
    if interp_method == 'spline_smooth':
        smoothness = st.slider("Suavidade da spline (maior = mais reta)", 0.001, 0.1, 0.01, 0.001)
    
    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
    
    with col_m1:
        # Input para adicionar pontos por valor X
        x_input = st.text_input("Digite valores X (separados por vírgula):", 
                               placeholder="Ex: 500, 1000, 1500, 2000")
        if st.button("➕ Adicionar pontos", type="primary"):
            if x_input:
                try:
                    x_values = [float(v.strip()) for v in x_input.split(',')]
                    # Encontra índices mais próximos
                    for xv in x_values:
                        idx = np.argmin(np.abs(x - xv))
                        if idx not in st.session_state.manual_points:
                            st.session_state.manual_points.append(idx)
                    st.success(f"✅ {len(x_values)} pontos adicionados")
                    st.rerun()
                except:
                    st.error("❌ Formato inválido")
    
    with col_m2:
        # Sugestão automática de pontos
        n_suggest = st.number_input("Número de pontos a sugerir:", 5, 30, 10)
        if st.button("🎯 Sugerir pontos automáticos"):
            suggested = suggest_baseline_points(x, y_proc, orientation_eff, n_suggest)
            st.session_state.manual_points = suggested
            st.success(f"✅ {len(suggested)} pontos sugeridos")
            st.rerun()
    
    with col_m3:
        if st.button("🗑️ Limpar todos"):
            st.session_state.manual_points = []
            st.rerun()
    
    # Mostra pontos selecionados
    if st.session_state.manual_points:
        st.markdown(f"**Pontos selecionados:** {len(st.session_state.manual_points)}")
        
        # Valida índices
        valid_points = [p for p in st.session_state.manual_points if 0 <= p < len(x)]
        if len(valid_points) != len(st.session_state.manual_points):
            st.session_state.manual_points = valid_points
        
        # Tabela editável de pontos
        if valid_points:
            points_df = pd.DataFrame({
                'Índice': valid_points,
                'X': [x[i] for i in valid_points],
                'Y': [y_proc[i] for i in valid_points]
            })
            
            with st.expander("📝 Editar pontos selecionados"):
                edited_df = st.data_editor(points_df, num_rows="dynamic", use_container_width=True)
                if st.button("Atualizar pontos"):
                    st.session_state.manual_points = edited_df['Índice'].tolist()
                    st.rerun()

# Parâmetros adicionais para modo automático
if baseline_method == "Automático (AsLS)":
    with st.expander("⚙️ Parâmetros AsLS"):
        lam = st.slider("λ (suavidade, log10)", 2, 8, 6, 1)
        p_asls = st.slider("p (assimetria)", 0.000, 0.500, 0.001, 0.001)
        niter = st.slider("Iterações", 1, 30, 10, 1)
else:
    lam = 6
    p_asls = 0.001
    niter = 10

# Cálculo da linha base
baseline = None
y_corr = None

if baseline_method == "Manual":
    if st.session_state.manual_points and len(st.session_state.manual_points) >= 2:
        baseline = create_manual_baseline(x, y_proc, st.session_state.manual_points, interp_method, smoothness)
        if orientation_eff == "Picos para baixo":
            y_corr = y_proc - baseline
        else:
            y_corr = y_proc - baseline
    else:
        st.warning("⚠️ Selecione pelo menos 2 pontos para criar a linha base")

elif baseline_method == "Automático (AsLS)":
    lam_val = 10.0**lam
    
    if use_anchor_points:
        # Adiciona pontos de ancoragem para melhor controle
        st.session_state.manual_points = []  # Limpa pontos anteriores
        
        if anchor_method == "Distribuído":
            # Distribui pontos uniformemente
            anchor_indices = np.linspace(0, len(y)-1, n_anchors, dtype=int).tolist()
        elif anchor_method == "Vales":
            anchor_indices = suggest_baseline_points(x, y_proc, "Picos para cima", n_anchors)
        elif anchor_method == "Topos":
            anchor_indices = suggest_baseline_points(x, y_proc, "Picos para baixo", n_anchors)
        else:  # Adaptativo
            anchor_indices = suggest_baseline_points(x, y_proc, orientation_eff, n_anchors)
        
        st.session_state.manual_points = anchor_indices
    
    # Calcula baseline AsLS
    if orientation_eff == "Picos para baixo":
        baseline = -asls_baseline(-y_proc, lam=lam_val, p=p_asls, niter=niter)
    else:
        baseline = asls_baseline(y_proc, lam=lam_val, p=p_asls, niter=niter)
    
    # Se houver pontos âncora, faz uma correção adicional
    if use_anchor_points and st.session_state.manual_points:
        # Cria uma baseline híbrida que passa pelos pontos âncora
        anchor_baseline = create_manual_baseline(x, y_proc, st.session_state.manual_points, 'spline_smooth', 0.1)
        # Combina as duas baselines com peso
        weight = 0.3  # Peso para a baseline de âncoras
        baseline = (1 - weight) * baseline + weight * anchor_baseline
    
    y_corr = y_proc - baseline

elif baseline_method == "Detecção de picos":
    # Normaliza para detecção
    y_norm = (y_proc - np.min(y_proc)) / (np.ptp(y_proc) if np.ptp(y_proc) > 0 else 1)
    
    if orientation_eff == "Picos para baixo":
        peaks, props = adaptive_peak_detection(x, 1-y_norm, sensitivity)
    else:
        peaks, props = adaptive_peak_detection(x, y_norm, sensitivity)
    
    # Cria baseline passando pelos vales/topos entre picos
    if len(peaks) > 0:
        # Encontra pontos base entre picos
        base_points = [0]  # Começa na extremidade
        for i in range(len(peaks)-1):
            if peaks[i+1] < len(y_proc):  # Verifica limites
                region = y_proc[peaks[i]:peaks[i+1]]
                if len(region) > 0:
                    if orientation_eff == "Picos para baixo":
                        local_max = peaks[i] + np.argmax(region)
                        base_points.append(local_max)
                    else:
                        local_min = peaks[i] + np.argmin(region)
                        base_points.append(local_min)
        base_points.append(len(y)-1)  # Termina na extremidade
        
        if len(base_points) >= 2:
            baseline = create_manual_baseline(x, y_proc, base_points, 'spline_smooth', 0.01)
            if orientation_eff == "Picos para baixo":
                y_corr = y_proc - baseline
            else:
                y_corr = y_proc - baseline
    else:
        st.warning("⚠️ Nenhum pico detectado. Tente ajustar a sensibilidade.")

elif baseline_method == "Híbrido":
    if st.session_state.manual_points and len(st.session_state.manual_points) >= 2:
        baseline = create_manual_baseline(x, y_proc, st.session_state.manual_points, interp_method, smoothness)
        
        # Refina com AsLS
        if st.checkbox("Refinar com AsLS", value=False):
            if orientation_eff == "Picos para baixo":
                baseline_asls = -asls_baseline(-y_proc, lam=1e6, p=0.001, niter=5)
            else:
                baseline_asls = asls_baseline(y_proc, lam=1e6, p=0.001, niter=5)
            
            weight = st.slider("Peso do AsLS (0=manual, 1=auto)", 0.0, 1.0, 0.3)
            baseline = (1-weight) * baseline + weight * baseline_asls
        
        if orientation_eff == "Picos para baixo":
            y_corr = y_proc - baseline
        else:
            y_corr = y_proc - baseline
    else:
        st.warning("⚠️ Selecione pelo menos 2 pontos para criar a linha base")

# VISUALIZAÇÃO
st.markdown("---")
st.markdown("### 📈 Visualização")

# Opções de visualização
col_v1, col_v2, col_v3, col_v4 = st.columns(4)
with col_v1:
    show_original = st.checkbox("Mostrar original", True)
with col_v2:
    show_baseline = st.checkbox("Mostrar linha base", True)
with col_v3:
    show_points = st.checkbox("Mostrar pontos base", True)
with col_v4:
    show_corrected = st.checkbox("Mostrar corrigido", True)

# Gráfico principal
fig = go.Figure()

# Sinal original/processado
if show_original:
    fig.add_trace(go.Scatter(
        x=x, y=y_proc,
        mode='lines',
        name='Sinal processado',
        line=dict(width=2, color='#FF6B6B')
    ))

# Linha base
if show_baseline and baseline is not None:
    fig.add_trace(go.Scatter(
        x=x, y=baseline,
        mode='lines',
        name='Linha base',
        line=dict(width=2, color='#4ECDC4', dash='dash')
    ))

# Pontos de controle
if show_points and st.session_state.manual_points:
    valid_points = [p for p in st.session_state.manual_points if 0 <= p < len(x)]
    if valid_points:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in valid_points],
            y=[y_proc[i] for i in valid_points],
            mode='markers',
            name='Pontos base',
            marker=dict(size=10, color='#FFE66D', symbol='circle-open', line=dict(width=2))
        ))

fig.update_layout(
    template="plotly_dark",
    title=f"Espectro com Correção de Linha Base - {baseline_method}",
    xaxis_title="X",
    yaxis_title="Intensidade",
    height=500,
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})

# Gráfico corrigido
if show_corrected and y_corr is not None:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x, y=y_corr,
        mode='lines',
        name='Sinal corrigido',
        line=dict(width=2, color='#95E77E')
    ))
    
    # Linha zero de referência
    fig2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig2.update_layout(
        template="plotly_dark",
        title="Espectro Corrigido",
        xaxis_title="X",
        yaxis_title="Intensidade corrigida",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True, config={'displaylogo': False})

# Estatísticas
if y_corr is not None:
    st.markdown("---")
    st.markdown("### 📊 Estatísticas")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Orientação", orientation_eff)
    with col_s2:
        n_points = len(st.session_state.manual_points) if baseline_method in ["Manual", "Híbrido"] else "Auto"
        st.metric("Pontos base", n_points)
    with col_s3:
        st.metric("RMS original", f"{np.sqrt(np.mean(y_proc**2)):.4f}")
    with col_s4:
        st.metric("RMS corrigido", f"{np.sqrt(np.mean(y_corr**2)):.4f}")

# Exportação
if y_corr is not None:
    st.markdown("---")
    st.markdown("### 💾 Exportar Resultados")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        # Preparar dados para exportação
        export_df = pd.DataFrame({
            'X': x,
            'Y_original': y,
            'Y_processado': y_proc,
            'Y_baseline': baseline if baseline is not None else np.zeros_like(y),
            'Y_corrigido': y_corr
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar CSV completo",
            data=csv.encode('utf-8'),
            file_name=f"{up.name.rsplit('.', 1)[0]}_corrigido.csv",
            mime='text/csv'
        )
    
    with col_e2:
        # Exportar apenas X e Y corrigido
        simple_df = pd.DataFrame({'X': x, 'Y': y_corr})
        csv_simple = simple_df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar apenas X,Y corrigido",
            data=csv_simple.encode('utf-8'),
            file_name=f"{up.name.rsplit('.', 1)[0]}_XY_corrigido.csv",
            mime='text/csv'
        )

# Rodapé com informações
st.markdown("---")
st.caption("💡 **Dica:** Use o modo Manual para controle total sobre a linha base, ou Híbrido para combinar seleção manual com refinamento automático.")













