import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import render_sidebar, get_name, run_clustering_analysis

st.set_page_config(page_title="Кластерный анализ", layout="wide", page_icon="🔬")

df = render_sidebar()
if df is None: st.stop()

st.header("🔬 Кластерный анализ (Иерархический и K-Means)")
st.markdown("Методы классификации наблюдений (респондентов) и переменных (шкал).")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

# --- ЛОГИКА КНОПОК БЫСТРОГО ВЫБОРА ---
if 'safe_hc_sel' not in st.session_state: st.session_state.safe_hc_sel = []
if 'hc_sel' not in st.session_state: st.session_state.hc_sel = st.session_state.safe_hc_sel

if 'safe_km_sel' not in st.session_state: st.session_state.safe_km_sel = []
if 'km_sel' not in st.session_state: st.session_state.km_sel = st.session_state.safe_km_sel

def add_to_state(state_key, prefix):
    current = st.session_state[state_key]
    new_items = [c for c in num_cols if c.startswith(prefix) and c not in current]
    st.session_state[state_key] = current + new_items

def clear_state(state_key):
    st.session_state[state_key] = []

subtab_hc, subtab_km = st.tabs(["🌳 Иерархическая кластеризация (SPSS)", "🎯 K-Means (Авто-выбор)"])

# ---------------------------------------------------------
# 1. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ
# ---------------------------------------------------------
with subtab_hc:
    st.subheader("Иерархический кластерный анализ и Дендрограммы")
    st.markdown("Позволяет визуально оценить естественные группировки (деревья) в данных.")

    st.write("**Быстрое добавление шкал:**")
    hc_b1, hc_b2, hc_b3, hc_b4 = st.columns(4)
    hc_b1.button("➕ Братусь", on_click=add_to_state, args=('hc_sel', 'B_'), key="hc_btn_b")
    hc_b2.button("➕ Мильман", on_click=add_to_state, args=('hc_sel', 'M_'), key="hc_btn_m")
    hc_b3.button("➕ ИПЛ", on_click=add_to_state, args=('hc_sel', 'IPL_'), key="hc_btn_i")
    hc_b4.button("❌ Очистить", on_click=clear_state, args=('hc_sel',), key="hc_btn_clear")

    hc_col1, hc_col2 = st.columns(2)
    with hc_col1:
        hc_features = st.multiselect("Выберите шкалы для кластеризации:", num_cols, key="hc_sel", format_func=get_name)
    with hc_col2:
        possible_labels = ["Номер строки (Index)"] + [c for c in df.columns if df[c].dtype == 'object']
        obs_label = st.selectbox("Подписывать респондентов по:", possible_labels)

    if len(hc_features) >= 2:
        df_hc = df.dropna(subset=hc_features).copy()
        X_hc = df_hc[hc_features]
        X_scaled = StandardScaler().fit_transform(X_hc)

        obs_names = df_hc.index.astype(str).tolist() if obs_label == "Номер строки (Index)" else df_hc[obs_label].astype(str).tolist()
        var_names = [get_name(c) for c in hc_features]

        st.markdown("---")
        hc_plot_type = st.radio("Выберите тип графика:", ["Кластеризация переменных (Шкал)", "Кластеризация наблюдений (Респондентов)", "Тепловая карта + Деревья (Clustergram)"], horizontal=True)

        if hc_plot_type == "Кластеризация переменных (Шкал)":
            st.markdown("**Дендрограмма переменных:** показывает, какие психологические шкалы ведут себя похоже.")
            Z_vars = linkage(X_scaled.T, method='ward')
            fig_v, ax_v = plt.subplots(figsize=(10, 6))
            dendrogram(Z_vars, labels=var_names, leaf_rotation=45, leaf_font_size=10, ax=ax_v)
            plt.title("Дендрограмма (Шкалы)")
            plt.tight_layout()
            st.pyplot(fig_v)

        elif hc_plot_type == "Кластеризация наблюдений (Респондентов)":
            st.markdown("**Дендрограмма наблюдений:** показывает, как респонденты объединяются в группы.")
            Z_obs = linkage(X_scaled, method='ward')
            fig_o, ax_o = plt.subplots(figsize=(12, 7))
            if len(obs_names) > 50:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o, truncate_mode='lastp', p=30, show_contracted=True)
                plt.title("Дендрограмма (Респонденты) - Показаны 30 верхних узловых групп")
            else:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o)
                plt.title("Дендрограмма (Респонденты)")
            plt.tight_layout()
            st.pyplot(fig_o)

        elif hc_plot_type == "Тепловая карта + Деревья (Clustergram)":
            st.markdown("**Clustergram:** объединяет обе дендрограммы и показывает выраженность признаков цветом.")
            df_cm = pd.DataFrame(X_scaled, index=obs_names, columns=var_names)
            fig_cm = sns.clustermap(df_cm, method='ward', cmap='coolwarm', figsize=(10, 10), yticklabels=True if len(obs_names) <= 50 else False, xticklabels=True)
            st.pyplot(fig_cm.figure)

# ---------------------------------------------------------
# 2. K-MEANS С АВТО-ВЫБОРОМ
# ---------------------------------------------------------
with subtab_km:
    st.subheader("K-Means кластеризация с авто-определением")

    st.write("**Быстрое добавление шкал:**")
    km_b1, km_b2, km_b3, km_b4 = st.columns(4)
    km_b1.button("➕ Братусь", on_click=add_to_state, args=('km_sel', 'B_'), key="km_btn_b")
    km_b2.button("➕ Мильман", on_click=add_to_state, args=('km_sel', 'M_'), key="km_btn_m")
    km_b3.button("➕ ИПЛ", on_click=add_to_state, args=('km_sel', 'IPL_'), key="km_btn_i")
    km_b4.button("❌ Очистить", on_click=clear_state, args=('km_sel',), key="km_btn_clear")

    km_features = st.multiselect("Признаки для кластеризации:", num_cols, key="km_sel", format_func=get_name)

    if len(km_features) >= 2:
        df_km = df.dropna(subset=km_features).copy()
        X_km = df_km[km_features]
        X_km_scaled = StandardScaler().fit_transform(X_km)

        max_k = min(10, len(X_km) - 1)
        sil_scores = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_km_scaled)
            sil_scores.append(silhouette_score(X_km_scaled, labels))

        best_k = K_range[np.argmax(sil_scores)]

        c_info1, c_info2 = st.columns([1, 2])
        with c_info1:
            st.success(f"**Оптимальное число кластеров:** {best_k}")
            st.caption("Рассчитано на основе максимума метрики Силуэта.")
            n_clusters = st.slider("Согласиться или выбрать вручную:", 2, max_k, int(best_k))
            run_btn = st.button("🚀 Запустить K-Means")

        with c_info2:
            fig_sil = go.Figure(data=go.Scatter(x=list(K_range), y=sil_scores, mode='lines+markers'))
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color="green", annotation_text="Оптимум")
            fig_sil.update_layout(title="Метрика Силуэта (выше = лучше)", height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig_sil, use_container_width=True)

        if run_btn:
            res_clustered = run_clustering_analysis(df_km, km_features, n_clusters)
            if res_clustered is not None:
                st.markdown("---")
                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.markdown("#### Карта кластеров (PCA)")
                    fig_pca = px.scatter(res_clustered, x='PC1', y='PC2', color='Cluster', title="Проекция групп на плоскость (2D)")
                    st.plotly_chart(fig_pca, use_container_width=True)

                with c_res2:
                    st.markdown("#### Психологический профиль (Радар)")
                    cluster_means = res_clustered.groupby('Cluster')[km_features].mean().reset_index()
                    radar_features = [get_name(f) for f in km_features]

                    fig_radar = go.Figure()
                    for i, row in cluster_means.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(r=row[km_features].values, theta=radar_features, fill='toself', name=f'Кластер {row["Cluster"]}'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), margin=dict(t=30, b=30))
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("#### Средние значения по группам")
                st.dataframe(cluster_means.rename(columns={f: get_name(f) for f in km_features}).style.highlight_max(axis=0, color='lightgreen'))

st.session_state.safe_hc_sel = st.session_state.hc_sel
st.session_state.safe_km_sel = st.session_state.km_sel