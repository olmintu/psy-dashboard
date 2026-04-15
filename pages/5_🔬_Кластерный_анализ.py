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
from utils import render_sidebar, get_name, run_clustering_analysis, smart_compare_groups

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
            
            # --- : Выравниваем текст по правому краю ---
            ax_v.set_xticklabels(ax_v.get_xticklabels(), rotation=90, ha='right', fontsize=8)
            
            plt.title("Дендрограмма (Шкалы)")
            plt.tight_layout()
            st.pyplot(fig_v)

        elif hc_plot_type == "Кластеризация наблюдений (Респондентов)":
            st.markdown("**Дендрограмма наблюдений:** показывает, как респонденты объединяются в группы.")
            Z_obs = linkage(X_scaled, method='ward')
            
            # ---  Галочка для показа полного дерева ---
            show_full_tree = False
            if len(obs_names) > 50:
                show_full_tree = st.checkbox("🌳 Показать полное дерево (все респонденты без сжатия)", value=False)
            
            fig_o, ax_o = plt.subplots(figsize=(12, 7))
            
            # Логика отрисовки в зависимости от галочки
            if len(obs_names) > 50 and not show_full_tree:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o, truncate_mode='lastp', p=30, show_contracted=True)
                plt.title("Дендрограмма (Респонденты) - Показаны 30 верхних узловых групп")
            else:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o)
                plt.title("Дендрограмма (Респонденты)")
                
            plt.tight_layout()
            st.pyplot(fig_o)

            # --- Продвинутая аналитика кластеров ---
            from scipy.cluster.hierarchy import fcluster
            import io
            
            st.markdown("---")
            st.subheader("📊 Характеристики выделенных групп")
            
            # 1. Настройка "разреза" дерева
            n_hc_clusters = st.slider("Количество групп (веток) для анализа:", min_value=2, max_value=15, value=3, step=1)
            cluster_labels = fcluster(Z_obs, t=n_hc_clusters, criterion='maxclust')
            
            # 2. Подготовка данных для анализа
            df_hc_res = pd.DataFrame({'Респондент': obs_names, 'Cluster_ID': cluster_labels})
            df_with_clusters = df_hc.copy()
            df_with_clusters['Cluster_Group'] = cluster_labels
            
            # 3. Расчет средних и статистической значимости (ANOVA)
            summary_list = []
            for col in hc_features:
                group_means = df_with_clusters.groupby('Cluster_Group')[col].mean()
                # Вызываем вашу математическую функцию из 3-й вкладки
                _, p_val = smart_compare_groups(df_with_clusters, 'Cluster_Group', col)
                
                row = {'Показатель': get_name(col), 'p-value (значимость)': p_val}
                for g_id, m_val in group_means.items():
                    row[f'Группа {g_id}'] = round(m_val, 2)
                summary_list.append(row)
                
            df_summary = pd.DataFrame(summary_list)
            
            # 4. Кнопка скачивания Excel
            buffer_cl = io.BytesIO()
            df_summary.to_excel(buffer_cl, index=False)
            st.download_button(
                "📥 Скачать профили групп и p-value (Excel)", 
                buffer_cl.getvalue(), 
                "cluster_profiles.xlsx", 
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # 5. ИНТЕРАКТИВНЫЕ СПИСКИ С ПОДСВЕТКОЙ (HTML/JS)
            import streamlit.components.v1 as components

            st.info("💡 Ниже представлены профили кластеров. **Наведите курсор на любую шкалу**, чтобы увидеть её позицию в других группах!")

            # Отрисовка шапок (стандартный Streamlit)
            cols_groups = st.columns(n_hc_clusters)
            for i in range(1, n_hc_clusters + 1):
                with cols_groups[i - 1]:
                    st.success(f"**Группа {i}**")
                    group_members = df_hc_res[df_hc_res['Cluster_ID'] == i]['Респондент'].tolist()
                    with st.expander(f"👥 Состав ({len(group_members)} чел.)"):
                        st.caption(", ".join(group_members))

            # --- ГЕНЕРАЦИЯ ПОЛНОЦЕННОГО HTML-ОТЧЕТА ---
            # Стили (те же, что и раньше + обертка для файла)
            css_style = """
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #fafafa; }
                h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .comparator { display: flex; gap: 15px; flex-wrap: wrap; }
                .col { flex: 1; min-width: 250px; background: #fff; border-radius: 8px; padding: 15px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
                .col-title { font-weight: bold; margin-bottom: 15px; color: #2980b9; font-size: 16px; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 8px;}
                .item { padding: 10px; margin-bottom: 6px; border-radius: 6px; border: 1px solid transparent; cursor: pointer; font-size: 13px; display: flex; justify-content: space-between; transition: all 0.2s ease; background: #fdfdfd; border: 1px solid #f0f0f0; }
                .item:hover { background-color: #fff9db; border-color: #ffe066; }
                .item.highlight { background-color: #fff3cd !important; border-color: #ffe69c !important; transform: scale(1.02); box-shadow: 0 4px 8px rgba(0,0,0,0.1); z-index: 10; position: relative;}
                .scale-name { color: #34495e; text-align: right; line-height: 1.2; font-size: 12px; margin-left: 10px;}
                .scale-val { font-weight: bold; color: #2980b9; font-size: 14px; min-width: 45px;}
                .legend { margin-top: 20px; font-size: 12px; color: #7f8c8d; font-style: italic; }
            </style>
            """

            # Сама структура списков
            html_body = f'<div class="comparator">'
            for i in range(1, n_hc_clusters + 1):
                group_col_name = f'Группа {i}'
                sorted_group = df_summary[['Показатель', group_col_name]].sort_values(by=group_col_name, ascending=False)
                
                html_body += f'<div class="col"><div class="col-title">Группа {i}</div>'
                for _, row in sorted_group.iterrows():
                    scale_name = row['Показатель']
                    val = row[group_col_name]
                    safe_id = "".join(c if c.isalnum() else "_" for c in scale_name)
                    html_body += f'''
                        <div class="item" data-scale="{safe_id}">
                            <span class="scale-val">{val:.2f}</span>
                            <span class="scale-name">{scale_name}</span>
                        </div>
                    '''
                html_body += '</div>'
            html_body += '</div>'

            # JavaScript
            js_script = """
            <script>
                const items = document.querySelectorAll('.item');
                items.forEach(item => {
                    item.addEventListener('mouseenter', () => {
                        const target = item.getAttribute('data-scale');
                        document.querySelectorAll(`.item[data-scale="${target}"]`).forEach(el => el.classList.add('highlight'));
                    });
                    item.addEventListener('mouseleave', () => {
                        const target = item.getAttribute('data-scale');
                        document.querySelectorAll(`.item[data-scale="${target}"]`).forEach(el => el.classList.remove('highlight'));
                    });
                });
            </script>
            """

            # Собираем всё в одну строку для отображения в приложении
            full_html_snippet = css_style + html_body + js_script
            
            # Собираем полноценный файл для скачивания (с тегами html/body)
            standalone_html = f"""
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <title>Интерактивный профиль кластеров</title>
                {css_style}
            </head>
            <body>
                <h2>📊 Психологические профили выделенных групп</h2>
                <p class="legend">Подсказка: наведите курсор на любую шкалу, чтобы подсветить её во всех группах.</p>
                {html_body}
                {js_script}
            </body>
            </html>
            """

            # Выводим в интерфейс
            components.html(full_html_snippet, height=max(500, len(hc_features) * 42), scrolling=True)

            # Кнопка скачивания HTML
            st.download_button(
                label="🌐 Скачать этот интерактивный отчет (HTML)",
                data=standalone_html,
                file_name="interactive_clusters.html",
                mime="text/html",
                help="Вы скачаете файл, который сохранит всю магию подсветки. Его можно открыть в любом браузере."
            )

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