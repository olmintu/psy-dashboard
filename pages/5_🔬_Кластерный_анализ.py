import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
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


def silhouette_comment(sil_value):
    """Возвращает (статус, текст) интерпретации значения силуэта."""
    if sil_value is None:
        return "⚪", "Силуэт не рассчитан."
    if sil_value >= 0.50:
        return "🟢", f"Силуэт = {sil_value:.3f}. Сильная кластерная структура — группы чётко разделены."
    if sil_value >= 0.25:
        return "🟡", f"Силуэт = {sil_value:.3f}. Умеренная структура — разбиение приемлемо, но границы между группами размыты."
    if sil_value >= 0.15:
        return "🟠", f"Силуэт = {sil_value:.3f}. Слабая структура — кластеры условны, интерпретировать с осторожностью."
    if sil_value >= 0:
        return "🔴", f"Силуэт = {sil_value:.3f}. Естественной кластерной структуры в данных практически нет. Разбиение носит описательный характер."
    return "🔴", f"Силуэт = {sil_value:.3f}. Разбиение хуже случайного — данные лучше не группировать."


def cluster_profile_table(df_with_clusters, features, cluster_col='Cluster'):
    """Строит таблицу профилей кластеров с p-value (Краскел-Уоллис/ANOVA через smart_compare_groups)."""
    rows = []
    for col in features:
        group_means = df_with_clusters.groupby(cluster_col)[col].mean()
        _, p_val = smart_compare_groups(df_with_clusters, cluster_col, col)
        try:
            p_val_num = float(p_val)
        except (TypeError, ValueError):
            p_val_num = None
        row = {'Показатель': get_name(col)}
        for g_id, m_val in group_means.items():
            row[f'Группа {g_id}'] = round(m_val, 2)
        if p_val_num is not None:
            stars = "***" if p_val_num < 0.001 else ("**" if p_val_num < 0.01 else ("*" if p_val_num < 0.05 else ""))
            row['p-value'] = f"{p_val_num:.4f} {stars}".strip()
        else:
            row['p-value'] = str(p_val)
        rows.append(row)
    return pd.DataFrame(rows)


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
        hc_plot_type = st.radio("Выберите тип графика:", [
            "Кластеризация переменных (Шкал)",
            "Кластеризация наблюдений (Респондентов)",
            "Тепловая карта + Деревья (Clustergram)"
        ], horizontal=True)

        # ========= ТИП 1: ПЕРЕМЕННЫЕ =========
        if hc_plot_type == "Кластеризация переменных (Шкал)":
            st.markdown("**Дендрограмма переменных:** показывает, какие психологические шкалы ведут себя похоже.")
            Z_vars = linkage(X_scaled.T, method='ward')
            fig_v, ax_v = plt.subplots(figsize=(10, 6))
            dendrogram(Z_vars, labels=var_names, leaf_rotation=45, leaf_font_size=10, ax=ax_v)
            ax_v.set_xticklabels(ax_v.get_xticklabels(), rotation=90, ha='right', fontsize=8)
            plt.title("Дендрограмма (Шкалы)")
            plt.tight_layout()
            st.pyplot(fig_v)

        # ========= ТИП 2: НАБЛЮДЕНИЯ =========
        elif hc_plot_type == "Кластеризация наблюдений (Респондентов)":
            st.markdown("**Дендрограмма наблюдений:** показывает, как респонденты объединяются в группы.")
            Z_obs = linkage(X_scaled, method='ward')

            show_full_tree = False
            if len(obs_names) > 50:
                show_full_tree = st.checkbox("🌳 Показать полное дерево (все респонденты без сжатия)", value=False)

            fig_o, ax_o = plt.subplots(figsize=(12, 7))
            if len(obs_names) > 50 and not show_full_tree:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o, truncate_mode='lastp', p=30, show_contracted=True)
                plt.title("Дендрограмма (Респонденты) - Показаны 30 верхних узловых групп")
            else:
                dendrogram(Z_obs, labels=obs_names, leaf_rotation=90, leaf_font_size=8, ax=ax_o)
                plt.title("Дендрограмма (Респонденты)")
            plt.tight_layout()
            st.pyplot(fig_o)

            # --- Продвинутая аналитика кластеров ---
            st.markdown("---")
            st.subheader("📊 Характеристики выделенных групп")

            n_hc_clusters = st.slider("Количество групп (веток) для анализа:", min_value=2, max_value=15, value=3, step=1)
            cluster_labels = fcluster(Z_obs, t=n_hc_clusters, criterion='maxclust')

            # --- ОЦЕНКА КАЧЕСТВА РАЗБИЕНИЯ ---
            if len(set(cluster_labels)) > 1:
                try:
                    sil_hc = silhouette_score(X_scaled, cluster_labels)
                except Exception:
                    sil_hc = None
            else:
                sil_hc = None

            status, comment = silhouette_comment(sil_hc)
            col_q1, col_q2 = st.columns([1, 3])
            with col_q1:
                st.metric("Коэффициент силуэта", f"{sil_hc:.3f}" if sil_hc is not None else "—")
            with col_q2:
                if sil_hc is None or sil_hc < 0.15:
                    st.warning(f"{status} {comment}")
                elif sil_hc < 0.25:
                    st.info(f"{status} {comment}")
                else:
                    st.success(f"{status} {comment}")

            # Подготовка данных
            df_hc_res = pd.DataFrame({'Респондент': obs_names, 'Cluster_ID': cluster_labels})
            df_with_clusters = df_hc.copy()
            df_with_clusters['Cluster_Group'] = cluster_labels.astype(str)

            # Таблица профилей (с p-value через smart_compare_groups)
            df_summary = cluster_profile_table(df_with_clusters, hc_features, cluster_col='Cluster_Group')

            # --- СКАЧИВАНИЕ: ДВА ФАЙЛА ---
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                buffer_profiles = io.BytesIO()
                df_summary.to_excel(buffer_profiles, index=False)
                st.download_button(
                    "📥 Скачать профили групп + p-value (Excel)",
                    buffer_profiles.getvalue(),
                    "hc_cluster_profiles.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col_d2:
                buffer_members = io.BytesIO()
                df_hc_res.to_excel(buffer_members, index=False)
                st.download_button(
                    "👥 Скачать состав групп (Excel)",
                    buffer_members.getvalue(),
                    "hc_cluster_members.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # --- БОКСПЛОТЫ ПО КЛЮЧЕВЫМ ПОКАЗАТЕЛЯМ ---
            st.markdown("#### 📦 Распределение ключевых показателей по группам")
            key_indicators = [c for c in ['IPL_Total', 'IPL_G', 'IPL_A', 'IPL_P'] if c in df_with_clusters.columns]
            if key_indicators:
                indicator = st.selectbox("Показатель для боксплота:", key_indicators, format_func=get_name, key="hc_boxplot")
                fig_box = px.box(df_with_clusters, x='Cluster_Group', y=indicator, color='Cluster_Group',
                                 points='all', title=f"{get_name(indicator)} по группам")
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

            # --- ИНТЕРАКТИВНЫЕ СПИСКИ С ПОДСВЕТКОЙ ---
            import streamlit.components.v1 as components

            st.info("💡 Ниже представлены профили кластеров. **Наведите курсор на любую шкалу**, чтобы увидеть её позицию в других группах!")

            cols_groups = st.columns(n_hc_clusters)
            for i in range(1, n_hc_clusters + 1):
                with cols_groups[i - 1]:
                    st.success(f"**Группа {i}**")
                    group_members = df_hc_res[df_hc_res['Cluster_ID'] == i]['Респондент'].tolist()
                    with st.expander(f"👥 Состав ({len(group_members)} чел.)"):
                        st.caption(", ".join(group_members))

            # --- ГЕНЕРАЦИЯ HTML-ОТЧЕТА ---
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

            html_body = '<div class="comparator">'
            for i in range(1, n_hc_clusters + 1):
                group_col_name = f'Группа {i}'
                if group_col_name not in df_summary.columns:
                    continue
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

            full_html_snippet = css_style + html_body + js_script
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

            components.html(full_html_snippet, height=max(500, len(hc_features) * 42), scrolling=True)

            st.download_button(
                label="🌐 Скачать этот интерактивный отчет (HTML)",
                data=standalone_html,
                file_name="interactive_clusters.html",
                mime="text/html",
                help="Вы скачаете файл, который сохранит всю магию подсветки. Его можно открыть в любом браузере."
            )

        # ========= ТИП 3: CLUSTERGRAM (тепловая карта + дендрограммы) =========
        elif hc_plot_type == "Тепловая карта + Деревья (Clustergram)":
            st.markdown("""
            **Clustergram:** тепловая карта, в которой и строки (респонденты), и столбцы (шкалы) упорядочены
            по результатам иерархической кластеризации. Сверху и слева — деревья (дендрограммы),
            показывающие структуру группировки. Это самый информативный вид: сразу видно, какие группы
            респондентов имеют похожий профиль и по каким именно шкалам они различаются.
            """)

            n_obs = X_scaled.shape[0]

            # Адаптивный размер и настройки для больших выборок
            show_row_labels = st.checkbox(
                "Подписывать строки (респондентов)",
                value=(n_obs <= 60),
                help="Для выборок больше 60 человек подписи лучше скрыть — они сольются."
            )

            # Цветовая схема
            cmap_choice = st.selectbox(
                "Цветовая шкала:",
                ["vlag (красный-белый-синий)", "coolwarm", "RdBu_r", "viridis"],
                index=0
            )
            cmap = cmap_choice.split(' ')[0]

            with st.spinner("Строим clustergram (для больших выборок может занять до 10 секунд)..."):
                # Данные для отображения: стандартизированные значения
                heatmap_df = pd.DataFrame(X_scaled, columns=var_names, index=obs_names)

                # Адаптивная высота
                fig_height = min(max(6, n_obs * 0.12), 22)
                fig_width = min(max(8, len(var_names) * 0.6), 18)

                try:
                    g = sns.clustermap(
                        heatmap_df,
                        method='ward',
                        metric='euclidean',
                        cmap=cmap,
                        figsize=(fig_width, fig_height),
                        yticklabels=show_row_labels,
                        xticklabels=True,
                        cbar_kws={'label': 'Z-оценка'},
                        dendrogram_ratio=(0.12, 0.18),
                        center=0
                    )
                    # Поворот подписей по X для читаемости
                    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                    if show_row_labels:
                        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)

                    g.fig.suptitle(
                        "Clustergram: респонденты × шкалы (Ward, евклидова дистанция)",
                        fontsize=12, y=1.02
                    )

                    st.pyplot(g.fig)
                    plt.close(g.fig)

                    # Скачивание картинки
                    img_buffer = io.BytesIO()
                    g.fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    st.download_button(
                        "🖼️ Скачать clustergram (PNG)",
                        img_buffer.getvalue(),
                        "clustergram.png",
                        "image/png"
                    )
                except Exception as e:
                    st.error(f"Ошибка при построении clustergram: {e}")
                    st.info("Частая причина — в данных остались константные столбцы (нулевая дисперсия). "
                            "Попробуйте исключить такие шкалы из выбора.")

            st.caption("""
            **Как читать:** красные ячейки — значения выше среднего по шкале, синие — ниже.
            Горизонтальные блоки одного цвета = группа респондентов с похожим профилем.
            Ветви верхней дендрограммы показывают, какие шкалы коррелируют (группируются).
            """)

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
        best_sil = max(sil_scores)

        c_info1, c_info2 = st.columns([1, 2])
        with c_info1:
            st.success(f"**Оптимальное число кластеров:** {best_k}")
            st.caption(f"Макс. силуэт = {best_sil:.3f}")
            n_clusters = st.slider("Согласиться или выбрать вручную:", 2, max_k, int(best_k))
            run_btn = st.button("🚀 Запустить K-Means")

        with c_info2:
            fig_sil = go.Figure(data=go.Scatter(x=list(K_range), y=sil_scores, mode='lines+markers'))
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color="green", annotation_text="Оптимум")
            fig_sil.update_layout(title="Метрика Силуэта (выше = лучше)", height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig_sil, use_container_width=True)

        if run_btn:
            # Новая сигнатура: функция возвращает кортеж (df, silhouette)
            res_clustered, sil_final = run_clustering_analysis(df_km, km_features, n_clusters)
            if res_clustered is not None:
                st.markdown("---")

                # --- ОЦЕНКА КАЧЕСТВА РАЗБИЕНИЯ ---
                status, comment = silhouette_comment(sil_final)
                col_q1, col_q2 = st.columns([1, 3])
                with col_q1:
                    st.metric("Коэффициент силуэта", f"{sil_final:.3f}" if sil_final is not None else "—")
                with col_q2:
                    if sil_final is None or sil_final < 0.15:
                        st.warning(f"{status} {comment}")
                    elif sil_final < 0.25:
                        st.info(f"{status} {comment}")
                    else:
                        st.success(f"{status} {comment}")

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

                # --- ПРОФИЛИ С P-VALUE ---
                st.markdown("#### 📊 Средние значения по группам и значимость различий")
                df_summary_km = cluster_profile_table(res_clustered, km_features, cluster_col='Cluster')
                st.dataframe(df_summary_km, use_container_width=True)

                # --- СКАЧИВАНИЕ ---
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    buffer_prof = io.BytesIO()
                    df_summary_km.to_excel(buffer_prof, index=False)
                    st.download_button(
                        "📥 Скачать профили групп + p-value (Excel)",
                        buffer_prof.getvalue(),
                        "kmeans_cluster_profiles.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col_d2:
                    # Состав: индекс + кластер + контекстные колонки
                    members_cols = ['Cluster'] + [c for c in ['FIO', 'Gender', 'Age', 'Course', 'Edu_Status']
                                                  if c in res_clustered.columns]
                    members = res_clustered[members_cols].reset_index().rename(columns={'index': 'ID'})
                    buffer_mem = io.BytesIO()
                    members.to_excel(buffer_mem, index=False)
                    st.download_button(
                        "👥 Скачать состав групп (Excel)",
                        buffer_mem.getvalue(),
                        "kmeans_cluster_members.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # --- БОКСПЛОТЫ ---
                st.markdown("#### 📦 Распределение ключевых показателей по кластерам")
                key_indicators = [c for c in ['IPL_Total', 'IPL_G', 'IPL_A', 'IPL_P'] if c in res_clustered.columns]
                if key_indicators:
                    indicator_km = st.selectbox("Показатель для боксплота:", key_indicators, format_func=get_name, key="km_boxplot")
                    fig_box_km = px.box(res_clustered, x='Cluster', y=indicator_km, color='Cluster',
                                        points='all', title=f"{get_name(indicator_km)} по кластерам")
                    fig_box_km.update_layout(showlegend=False)
                    st.plotly_chart(fig_box_km, use_container_width=True)

st.session_state.safe_hc_sel = st.session_state.hc_sel
st.session_state.safe_km_sel = st.session_state.km_sel