import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pingouin as pg
import io
from utils import render_sidebar, get_name, calc_correlation_matrices

st.set_page_config(page_title="Корреляции", layout="wide", page_icon="🔗")

df = render_sidebar()
if df is None: st.stop()

st.header("🔗 Корреляционный анализ (в стиле SPSS)")
num_cols = df.select_dtypes(include=np.number).columns.tolist()

if 'safe_corr_sel' not in st.session_state: st.session_state.safe_corr_sel = []
if 'corr_sel' not in st.session_state: st.session_state.corr_sel = st.session_state.safe_corr_sel

def add_prefix(prefix):
    current = st.session_state.corr_sel
    new_items = [c for c in num_cols if c.startswith(prefix) and c not in current]
    st.session_state.corr_sel = current + new_items

def clear_all(): st.session_state.corr_sel = []

st.write("**Быстрое добавление шкал:**")
btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
btn_c1.button("➕ Добавить Братуся", on_click=add_prefix, args=('B_',))
btn_c2.button("➕ Добавить Мильмана", on_click=add_prefix, args=('M_',))
btn_c3.button("➕ Добавить ИПЛ", on_click=add_prefix, args=('IPL_',))
btn_c4.button("❌ Очистить всё", on_click=clear_all)

corr_cols = st.multiselect("Выберите показатели:", num_cols, key="corr_sel", format_func=get_name)

col1, col2 = st.columns(2)
with col1: method = st.radio("Метод корреляции:", ["spearman", "pearson"], horizontal=True, help="Spearman лучше подходит для психологических опросников. Pearson — для строго нормально распределенных метрических данных.")
with col2: show_stars = st.checkbox("Показывать значимость (звездочки)", value=True)

if len(corr_cols) > 1:
    if st.button("🚀 Рассчитать матрицу корреляций", type="primary", use_container_width=True):
        with st.spinner("Считаем корреляции..."):
            df_corr = df[corr_cols].dropna()
            r_matrix, p_matrix = calc_correlation_matrices(df_corr, method)
            st.session_state['corr_results'] = {'r_matrix': r_matrix, 'p_matrix': p_matrix, 'cols': corr_cols, 'method': method}

    if 'corr_results' in st.session_state:
        res = st.session_state['corr_results']
        r_matrix, p_matrix, saved_cols = res['r_matrix'], res['p_matrix'], res['cols']
        
        if saved_cols != corr_cols:
            st.warning("⚠️ Вы изменили состав шкал. Нажмите кнопку 'Рассчитать матрицу', чтобы обновить данные.")
        
        annot_matrix, hover_matrix = np.empty_like(r_matrix, dtype=object), np.empty_like(r_matrix, dtype=object)
        
        for i in range(len(saved_cols)):
            for j in range(len(saved_cols)):
                r_val, p_val = r_matrix.iloc[i, j], p_matrix.iloc[i, j]
                stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "") if i != j else ""
                annot_matrix[i, j] = f"{r_val:.2f}{stars}" if show_stars else f"{r_val:.2f}"
                hover_matrix[i, j] = f"<b>X:</b> {get_name(saved_cols[j])}<br><b>Y:</b> {get_name(saved_cols[i])}<br><b>r =</b> {r_val:.3f} {stars}<br><b>p-value =</b> {p_val:.4f}"
        
        translated_cols = [get_name(c) for c in saved_cols]
        fig_corr = go.Figure(data=go.Heatmap(z=r_matrix.values, x=translated_cols, y=translated_cols, text=annot_matrix, texttemplate="%{text}", customdata=hover_matrix, hovertemplate="%{customdata}<extra></extra>", colorscale='RdBu_r', zmin=-1, zmax=1))
        fig_corr.update_layout(title=f"Матрица корреляций ({res['method'].capitalize()})", height=max(600, len(saved_cols) * 35), xaxis_tickangle=-45)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("**Как читать:** Красный = прямая связь, Синий = обратная.")
        
        st.markdown("---")
        st.subheader("🔍 Автоматический поиск связей")
        
        filter_c1, filter_c2, filter_c3 = st.columns(3)
        with filter_c1:
            strength_preset = st.selectbox("Сила связи (|r|):", ["Все значимые", "Слабая (0.1-0.3)", "Умеренная (0.3-0.5)", "Сильная (0.5-0.7)", "Очень сильная (>0.7)", "Своё значение..."])
            if strength_preset == "Своё значение...": min_r, max_r = st.number_input("Минимальный модуль |r|:", 0.0, 1.0, 0.4, 0.05), 1.0
            elif strength_preset == "Слабая (0.1-0.3)": min_r, max_r = 0.1, 0.3
            elif strength_preset == "Умеренная (0.3-0.5)": min_r, max_r = 0.3, 0.5
            elif strength_preset == "Сильная (0.5-0.7)": min_r, max_r = 0.5, 0.7
            elif strength_preset == "Очень сильная (>0.7)": min_r, max_r = 0.7, 1.0
            else: min_r, max_r = 0.0, 1.0

        with filter_c2:
            sig_level = st.selectbox("Уровень значимости (p):", ["p < 0.05 (*)", "p < 0.01 (**)", "p < 0.001 (***)", "Показывать незначимые"])
            if sig_level == "p < 0.05 (*)": p_thresh = 0.05
            elif sig_level == "p < 0.01 (**)": p_thresh = 0.01
            elif sig_level == "p < 0.001 (***)": p_thresh = 0.001
            else: p_thresh = 1.0 
            
        with filter_c3: link_type = st.radio("Направление связи:", ["Все", "Прямая (r > 0)", "Обратная (r < 0)"])

        links = []
        for i in range(len(saved_cols)):
            for j in range(i + 1, len(saved_cols)):
                c1, c2 = saved_cols[i], saved_cols[j]
                r, p = r_matrix.loc[c1, c2], p_matrix.loc[c1, c2]
                
                if min_r <= abs(r) <= max_r and (sig_level == "Показывать незначимые" or p < p_thresh):
                    if link_type == "Все" or (link_type == "Прямая (r > 0)" and r > 0) or (link_type == "Обратная (r < 0)" and r < 0): 
                        s = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                        links.append({'Показатель 1': get_name(c1), 'Показатель 2': get_name(c2), 'r': r, 'p-value': p, 'Значимость': s})
                            
        if not links: st.info("По заданным фильтрам связей не найдено.")
        else:
            links_df = pd.DataFrame(links).sort_values(by='r', key=abs, ascending=False).reset_index(drop=True)
            col_l1, col_l2 = st.columns([1, 1])
            with col_l1: st.write(f"**Найдено связей: {len(links_df)}**")
            with col_l2:
                buffer_links = io.BytesIO()
                links_df.to_excel(buffer_links, index=False, engine='openpyxl')
                st.download_button("📥 Скачать таблицу (Excel)", buffer_links.getvalue(), 'links.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            st.markdown("---")
            for _, row in links_df.iterrows():
                color_dot = "🔴" if row['r'] > 0 else "🔵"
                st.markdown(f"{color_dot} **{row['Показатель 1']}** ↔ **{row['Показатель 2']}** | `r = {row['r']:.2f}` {row['Значимость']} *(p={row['p-value']:.3f})*")

st.session_state.safe_corr_sel = st.session_state.corr_sel