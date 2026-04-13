import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from utils import render_sidebar, get_name

st.set_page_config(page_title="Сетевой анализ", layout="wide", page_icon="🕸️")

df = render_sidebar()
if df is None: st.stop()

st.header("🕸️ Сетевая психометрия (Графы связей)")
st.markdown("Поиск корневых мотивов и смыслов. Личность представлена как нейросеть, где шкалы — это узлы, а корреляции — связи между ними.")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Инициализация сейфа для вкладки 9
if 'safe_cb_b_9' not in st.session_state: st.session_state.safe_cb_b_9 = False
if 'safe_cb_m_9' not in st.session_state: st.session_state.safe_cb_m_9 = False
if 'safe_cb_i_9' not in st.session_state: st.session_state.safe_cb_i_9 = False
if 'safe_net_vars' not in st.session_state: st.session_state.safe_net_vars = []
if 'safe_cross_method_9' not in st.session_state: st.session_state.safe_cross_method_9 = False

if 'cross_method_9' not in st.session_state: st.session_state.cross_method_9 = st.session_state.safe_cross_method_9
if 'cb_b_9' not in st.session_state: st.session_state.cb_b_9 = st.session_state.safe_cb_b_9
if 'cb_m_9' not in st.session_state: st.session_state.cb_m_9 = st.session_state.safe_cb_m_9
if 'cb_i_9' not in st.session_state: st.session_state.cb_i_9 = st.session_state.safe_cb_i_9
if 'net_vars' not in st.session_state: st.session_state.net_vars = st.session_state.safe_net_vars

# Колбэки
def toggle_b_9():
    cols = [c for c in num_cols if c.startswith('B_')]
    if st.session_state.cb_b_9: st.session_state.net_vars += [c for c in cols if c not in st.session_state.net_vars]
    else: st.session_state.net_vars = [c for c in st.session_state.net_vars if c not in cols]

def toggle_m_9():
    cols = [c for c in num_cols if c.startswith('M_')]
    if st.session_state.cb_m_9: st.session_state.net_vars += [c for c in cols if c not in st.session_state.net_vars]
    else: st.session_state.net_vars = [c for c in st.session_state.net_vars if c not in cols]

def toggle_i_9():
    cols = [c for c in num_cols if c.startswith('IPL_')]
    if st.session_state.cb_i_9: st.session_state.net_vars += [c for c in cols if c not in st.session_state.net_vars]
    else: st.session_state.net_vars = [c for c in st.session_state.net_vars if c not in cols]

c_net1, c_net2 = st.columns([1.5, 1]) 
with c_net1:
    st.markdown("**1. Выберите шкалы для построения сети:**")
    cb_n1, cb_n2, cb_n3 = st.columns(3)
    with cb_n1: st.checkbox("Шкалы Братуся (Сеть)", key='cb_b_9', on_change=toggle_b_9)
    with cb_n2: st.checkbox("Шкалы Мильмана (Сеть)", key='cb_m_9', on_change=toggle_m_9)
    with cb_n3: st.checkbox("Шкалы ИПЛ (Сеть)", key='cb_i_9', on_change=toggle_i_9)

    all_possible_vars = [c for c in num_cols if c.startswith('B_') or c.startswith('M_') or c.startswith('IPL_')]
    net_vars = st.multiselect("Включить в сеть (можно редактировать точечно):", all_possible_vars, key="net_vars", format_func=get_name)

with c_net2:
    st.markdown("**2. Настройки чувствительности сети:**")
    threshold = st.slider(
        "Отсекать слабые связи (Порог |r|):", 
        min_value=0.1, max_value=0.8, value=0.3, step=0.05,
        key="threshold_9", 
        help="Увеличьте порог, если граф похож на запутанный клубок ниток. Оставьте только самые сильные связи (>0.4)."
    )
    cross_method_only_9 = st.checkbox("🔀 Только межметодические связи", value=False, key="cross_method_9")

# КНОПКА ПОСТРОЕНИЯ (С сохранением в память)
if st.button("🕸️ Построить нейросеть личности", type="primary"):
    if len(net_vars) < 3:
        st.warning("⚠️ Для построения сети нужно выбрать минимум 3 шкалы.")
    else:
        with st.spinner("Рассчитываем топологию графа..."):
            df_net = df[net_vars].dropna()
            corr_matrix = df_net.corr()

            G = nx.Graph()
            for col in net_vars:
                G.add_node(col, name=get_name(col))

            for i in range(len(net_vars)):
                for j in range(i+1, len(net_vars)):
                    col1 = net_vars[i]
                    col2 = net_vars[j]
                    corr_val = corr_matrix.loc[col1, col2]
                    
                    if cross_method_only_9:
                        pref1 = col1.split('_')[0] if '_' in col1 else col1
                        pref2 = col2.split('_')[0] if '_' in col2 else col2
                        if pref1 == pref2:
                            continue
                    
                    if abs(corr_val) >= threshold:
                        G.add_edge(col1, col2, weight=abs(corr_val), corr=corr_val)

            isolated_nodes = list(nx.isolates(G))
            G.remove_nodes_from(isolated_nodes)

            if len(G.nodes) == 0:
                st.error("❌ При таком высоком пороге корреляции не найдено ни одной связи. Снизьте порог.")
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                degrees = dict(G.degree())
                
                edge_traces = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    weight = edge[2]['weight']
                    corr = edge[2]['corr']
                    
                    line_color = '#2ecc71' if corr > 0 else '#e74c3c'
                    line_width = weight * 5 
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        line=dict(width=line_width, color=line_color),
                        hoverinfo='none', mode='lines', opacity=0.6
                    )
                    edge_traces.append(edge_trace)

                node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_name = G.nodes[node]['name']
                    size = 20 + (degrees[node] * 5)
                    node_size.append(min(size, 60)) 
                    node_text.append(node_name)
                    node_hover.append(f"<b>{node_name}</b><br>Количество связей: {degrees[node]}")

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', text=node_text,
                    textposition="top center", hoverinfo='text', hovertext=node_hover,
                    marker=dict(showscale=False, color='#3498db', size=node_size, line_width=2, line_color='white'),
                    textfont=dict(size=11, color="black")
                )

                fig_net = go.Figure(data=edge_traces + [node_trace],
                     layout=go.Layout(
                        title=dict(text="Топология мотивов и смыслов", font=dict(size=16)),
                        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600, plot_bgcolor='white'
                    )
                )
                
                st.session_state['net_results'] = {
                    'fig_net': fig_net, 'G': G, 'degrees': degrees, 'isolated_nodes': isolated_nodes
                }

# ОТРИСОВКА ИНТЕРФЕЙСА
if 'net_results' in st.session_state:
    res_n = st.session_state['net_results']
    fig_net = res_n['fig_net']
    G = res_n['G']
    degrees = res_n['degrees']
    isolated_nodes = res_n['isolated_nodes']

    st.markdown("---")
    
    st.markdown("### 🗺️ Легенда графа")
    st.markdown("""
    * 🟢 **Зеленые линии** — прямая связь (шкалы растут и падают синхронно).
    * 🔴 **Красные линии** — обратная связь (конфликт мотивов: одна шкала растет, другая падает).
    * 🔵 **Размер кружка** — чем больше узел, тем больше у него связей (Центральный мотив).
    """)

    st.plotly_chart(fig_net, use_container_width=True, config={
        'displayModeBar': True,
        'toImageButtonOptions': {'format': 'png', 'filename': 'Network_Graph', 'scale': 3}
    })
    st.caption("👆 Наведите мышку в правый верхний угол графика и нажмите на иконку фотоаппарата, чтобы скачать граф как картинку (PNG).")

    html_bytes = fig_net.to_html(include_plotlyjs='cdn').encode('utf-8')
    st.download_button(
        label="💾 Скачать этот граф как интерактивный файл (HTML)",
        data=html_bytes, file_name="psychometric_network.html", mime="text/html",
        help="Вы скачаете файл, который можно открыть в любом браузере."
    )

    st.markdown("### 🏆 Аналитика графа")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        max_degree_node = max(degrees, key=degrees.get)
        st.info(f"**Центральный хаб (Смыслообразующий мотив):**\n\n🎯 {G.nodes[max_degree_node]['name']} (Связей: {degrees[max_degree_node]})")
    with res_col2:
        st.success(f"**Характеристики сети:**\n\n* Активных узлов: {len(G.nodes)}\n* Сильных связей: {len(G.edges)}\n* Изолированных шкал отсеяно: {len(isolated_nodes)}")

    st.markdown("---")
    st.markdown("### 🔍 Детальный анализ связей")
    st.markdown("Выберите любую шкалу из графа, чтобы посмотреть точный список всех её связей и их силу.")

    node_names = sorted([G.nodes[n]['name'] for n in G.nodes()])
    selected_node_name = st.selectbox("Шкала для детального анализа:", node_names)

    if selected_node_name:
        node_key = [n for n in G.nodes() if G.nodes[n]['name'] == selected_node_name][0]
        edges = G.edges(node_key, data=True)
        
        edge_data = []
        for u, v, data in edges:
            target_key = v if u == node_key else u
            target_name = G.nodes[target_key]['name']
            corr_val = data['corr']
            
            edge_data.append({
                'Связанная шкала': target_name,
                'Сила связи (Коэффициент r)': round(corr_val, 3),
                'Тип связи': '🟢 Прямая (Синхронная)' if corr_val > 0 else '🔴 Обратная (Конфликтная)',
                'Абсолютная сила': abs(corr_val) 
            })
            
        if edge_data:
            df_edges = pd.DataFrame(edge_data).sort_values(by='Абсолютная сила', ascending=False).drop(columns=['Абсолютная сила'])
            st.dataframe(df_edges, use_container_width=True)
        else:
            st.info("У этой шкалы нет сильных связей при текущем пороге.")

# Сохранение бэкапа
st.session_state.safe_cb_b_9 = st.session_state.cb_b_9
st.session_state.safe_cb_m_9 = st.session_state.cb_m_9
st.session_state.safe_cb_i_9 = st.session_state.cb_i_9
st.session_state.safe_net_vars = st.session_state.net_vars
st.session_state.safe_cross_method_9 = st.session_state.cross_method_9