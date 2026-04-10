import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import render_sidebar, get_name

st.set_page_config(page_title="Психометрика", layout="wide", page_icon="📐")

df = render_sidebar()
if df is None: st.stop()

st.header("📐 Психометрика (Надежность и Факторная структура)")
st.warning("⚠️ **Аналитический контекст:** Так как в систему загружены финальные баллы по шкалам, мы исследуем **макро-структуру** (связи шкал между собой и выделение вторичных факторов), а не классическую надежность отдельных вопросов.")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

if 'safe_alpha_sel' not in st.session_state: st.session_state.safe_alpha_sel = []
if 'alpha_sel' not in st.session_state: st.session_state.alpha_sel = st.session_state.safe_alpha_sel

if 'safe_fa_sel' not in st.session_state: st.session_state.safe_fa_sel = []
if 'fa_sel' not in st.session_state: st.session_state.fa_sel = st.session_state.safe_fa_sel

def add_to_state(state_key, prefix):
    current = st.session_state[state_key]
    new_items = [c for c in num_cols if c.startswith(prefix) and c not in current]
    st.session_state[state_key] = current + new_items

def clear_state(state_key):
    st.session_state[state_key] = []

subtab_alpha, subtab_fa = st.tabs(["Внутренняя согласованность (Кронбах)", "Факторная структура (Главные компоненты)"])

# ---------------------------------------------------------
# 1. АЛЬФА КРОНБАХА
# ---------------------------------------------------------
with subtab_alpha:
    st.subheader("Макро-согласованность шкал (Альфа Кронбаха)")
    st.markdown("Позволяет проверить, образуют ли выбранные шкалы единый теоретический конструкт.")
    
    st.write("**Быстрое добавление шкал:**")
    a_b1, a_b2, a_b3, a_b4 = st.columns(4)
    a_b1.button("➕ Братусь", on_click=add_to_state, args=('alpha_sel', 'B_'), key="btn_alpha_b")
    a_b2.button("➕ Мильман", on_click=add_to_state, args=('alpha_sel', 'M_'), key="btn_alpha_m")
    a_b3.button("➕ ИПЛ", on_click=add_to_state, args=('alpha_sel', 'IPL_'), key="btn_alpha_i")
    a_b4.button("❌ Очистить", on_click=clear_state, args=('alpha_sel',), key="btn_alpha_clear")

    alpha_cols = st.multiselect("Выберите шкалы для проверки согласованности:", num_cols, key="alpha_sel", format_func=get_name)
    
    if len(alpha_cols) >= 2:
        df_alpha = df[alpha_cols].dropna()
        if not df_alpha.empty:
            alpha, ci = pg.cronbach_alpha(data=df_alpha)
            if alpha >= 0.8: interpretation = "Высокая (шкалы измеряют один общий супер-фактор)"
            elif alpha >= 0.7: interpretation = "Приемлемая (хорошая согласованность)"
            elif alpha >= 0.6: interpretation = "Сомнительная (слабая связь между шкалами)"
            else: interpretation = "Низкая (шкалы измеряют принципиально разные вещи)"
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.metric("Альфа Кронбаха (α)", f"{alpha:.3f}")
                st.markdown(f"**Интерпретация:** {interpretation}")
                st.caption(f"95% Доверительный интервал: [{ci[0]:.3f}, {ci[1]:.3f}]")
            with col_a2:
                st.info("💡 **Как это понимать?** Если альфа высокая, значит респонденты отвечали на эти шкалы в едином ключе. Это позволяет объединить их в один комплексный индекс.")
        else:
            st.error("Недостаточно данных для расчета.")
    else:
        st.info("Выберите минимум 2 шкалы.")

# ---------------------------------------------------------
# 2. ФАКТОРНЫЙ АНАЛИЗ (PCA)
# ---------------------------------------------------------
with subtab_fa:
    st.subheader("Извлечение скрытых факторов (PCA)")
    st.markdown("Показывает, как исходные шкалы группируются в укрупненные, скрытые (латентные) факторы.")
    
    st.write("**Быстрое добавление шкал:**")
    f_b1, f_b2, f_b3, f_b4 = st.columns(4)
    f_b1.button("➕ Братусь", on_click=add_to_state, args=('fa_sel', 'B_'), key="btn_fa_b")
    f_b2.button("➕ Мильман", on_click=add_to_state, args=('fa_sel', 'M_'), key="btn_fa_m")
    f_b3.button("➕ ИПЛ", on_click=add_to_state, args=('fa_sel', 'IPL_'), key="btn_fa_i")
    f_b4.button("❌ Очистить", on_click=clear_state, args=('fa_sel',), key="btn_fa_clear")

    fa_cols = st.multiselect("Выберите шкалы для факторного анализа:", num_cols, key="fa_sel", format_func=get_name)
    
    if len(fa_cols) >= 3:
        df_fa = df[fa_cols].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_fa)
        
        pca_full = PCA()
        pca_full.fit(data_scaled)
        
        col_fa1, col_fa2 = st.columns([1, 2])
        with col_fa1:
            eigenvalues = pca_full.explained_variance_
            kaiser_factors = sum(eigenvalues > 1.0)
            st.success(f"**Оптимально факторов (по Кайзеру):** {kaiser_factors}")
            n_factors = st.number_input("Сколько факторов извлечь?", min_value=1, max_value=len(fa_cols), value=max(1, int(kaiser_factors)))
        
        with col_fa2:
            fig_scree = go.Figure(data=go.Scatter(x=list(range(1, len(fa_cols) + 1)), y=eigenvalues, mode='lines+markers', name='Собственные значения'))
            fig_scree.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Порог Кайзера (1.0)")
            fig_scree.update_layout(title="График 'Каменистой осыпи'", xaxis_title="Номер компоненты", yaxis_title="Собственное значение (Eigenvalue)", height=300)
            st.plotly_chart(fig_scree, use_container_width=True)

        pca_final = PCA(n_components=n_factors)
        pca_final.fit(data_scaled)
        loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)
        
        translated_fa_cols = [get_name(c) for c in fa_cols]
        factor_names = [f"Фактор {i+1} ({pca_final.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_factors)]
        
        fig_loadings = go.Figure(data=go.Heatmap(z=loadings, x=factor_names, y=translated_fa_cols, colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(loadings, 2), texttemplate="%{text}", hovertemplate="Шкала: %{y}<br>Фактор: %{x}<br>Нагрузка: %{z:.3f}<extra></extra>"))
        fig_loadings.update_layout(title="Матрица факторных нагрузок (чем ближе к 1 или -1, тем сильнее связь)", height=max(400, len(fa_cols) * 35))
        st.plotly_chart(fig_loadings, use_container_width=True)
        st.caption("🔍 **Как читать матрицу:** Смотрите на значения по модулю > 0.4. Они показывают, какие оригинальные шкалы 'вошли' в состав нового скрытого фактора.")
    else:
        st.info("Для факторного анализа требуется минимум 3 шкалы.")

st.session_state.safe_alpha_sel = st.session_state.alpha_sel
st.session_state.safe_fa_sel = st.session_state.fa_sel