import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import render_sidebar, get_name

st.set_page_config(page_title="Анализ методик", layout="wide", page_icon="🧩")

df = render_sidebar()
if df is None: st.stop()

st.header("Анализ методик")
    
# --- ВЫБОР РЕЖИМА И РЕСПОНДЕНТА ---
analysis_mode = st.radio("Режим анализа:", ["Сводный (Средние по отфильтрованной группе)", "Индивидуальный (Конкретный респондент)"], horizontal=True)

if analysis_mode == "Индивидуальный (Конкретный респондент)":
    respondent_list = df.apply(lambda x: f"[ID: {x.name}] {x.get('FIO', f'Строка {x.name}')} | Пол: {x.get('Gender', '?')} | Возраст: {x.get('Age', '?')}", axis=1)
    selected_id = st.selectbox("Выберите респондента:", respondent_list.index, format_func=lambda x: respondent_list[x])
    target_data = df.loc[[selected_id]] 
else:
    target_data = df 
    
subtab_b, subtab_m, subtab_i = st.tabs(["Братусь (Смыслы)", "Мильман (Мотивация)", "ИПЛ (Инновации)"])

with subtab_b:
    st.subheader("Жизненные смыслы")
    b_cols = [c for c in target_data.columns if c.startswith('B_')]
    if b_cols:
        means = target_data[b_cols].mean().sort_values(ascending=True) 
        labels = [get_name(c).replace('Братусь: ', '') for c in means.index]
        visual_weight = 25 - means.values
        
        fig_b = go.Figure(go.Bar(
            x=visual_weight, y=labels, orientation='h',
            marker=dict(color=means.values, colorscale=[[0, '#27ae60'], [0.5, '#f1c40f'], [1, '#e74c3c']], showscale=True, colorbar=dict(title="Балл", tickvals=[5, 15, 25])),
            text=np.round(means.values, 1), textposition='outside', textfont=dict(size=14),
            hovertemplate="Шкала: %{y}<br>Балл (сумма рангов): %{text}<extra></extra>"
        ))
        
        fig_b.update_layout(title="Жизненные смыслы (чем длиннее полоса, тем важнее смысл)", xaxis=dict(showticklabels=False, range=[0, 26]), yaxis=dict(autorange="reversed"), height=650, margin=dict(r=100), font=dict(size=14))
        st.plotly_chart(fig_b, use_container_width=True)
        st.markdown("<div style='text-align:center; font-size: 16px;'><span style='color:#27ae60'>■</span> ведущие (≤9) &nbsp;&nbsp; <span style='color:#f1c40f'>■</span> нейтральные (10-17) &nbsp;&nbsp; <span style='color:#e74c3c'>■</span> игнорируемые (≥18)</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 📋 Рейтинг жизненных смыслов")
        list_c1, list_c2 = st.columns(2)
        items = list(means.items())
        half = len(items) // 2 + (len(items) % 2) 
        
        for i, (col, val) in enumerate(items):
            scale_name = get_name(col).replace('Братусь: ', '')
            if val <= 9: color, status = "#27ae60", "Доминируют"
            elif val <= 17: color, status = "#f1c40f", "Представлены достаточно"
            else: color, status = "#e74c3c", "Представлены слабо"
            
            item_html = f"<div style='font-size: 15px; margin-bottom: 5px;'><b>{i+1}.</b> {scale_name} — <span style='color:{color}; font-weight:bold;'>{val:.1f} ({status})</span></div>"
            if i < half: list_c1.markdown(item_html, unsafe_allow_html=True)
            else: list_c2.markdown(item_html, unsafe_allow_html=True)
    else:
        st.info("Колонки B_ не найдены")

with subtab_m:
    st.subheader("Мотивационная структура (Мильман)")
    st.caption("Сравнение Идеального (чего хочу) и Реального (что имею) состояний.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1: show_group_m = st.checkbox("📊 Сравнить со средним по группе (серый фон)", key="m_gr") if analysis_mode == "Индивидуальный (Конкретный респондент)" else False
    with col_t2: show_frust = st.checkbox("🔥 Показать зоны фрустрации и текстовый анализ", key="m_fr")
    
    scale_keys = ['P', 'K', 'S', 'O', 'D', 'DR', 'OD']
    scale_names = ['Жизнеобеспечение', 'Комфорт', 'Статус', 'Общение', 'Дело', 'Творчество', 'Общ. польза']
    
    if f"M_{scale_keys[0]}_Zh-id" not in target_data.columns:
        st.warning("⚠️ Колонки для методики Мильмана не найдены.")
    else:
        col_m1, col_m2 = st.columns(2)
        fill_arg = 'tonexty' if show_frust else 'none'
        fill_color = 'rgba(231, 76, 60, 0.2)' if show_frust else None
        
        def get_mot_profile_type(ideal_vals):
            sum_develop = ideal_vals[4] + ideal_vals[5] + ideal_vals[6]
            sum_maintain = ideal_vals[0] + ideal_vals[1] + ideal_vals[2]
            diff = sum_develop - sum_maintain
            if diff >= 5: return "Прогрессивный"
            elif diff <= -5: return "Регрессивный"
            else:
                peaks = 0
                for i in range(7):
                    v = ideal_vals[i]
                    if i == 0:
                        if v >= ideal_vals[1] + 4: peaks += 1
                    elif i == 6:
                        if v >= ideal_vals[5] + 4: peaks += 1
                    else:
                        if v >= ideal_vals[i-1] + 2 and v >= ideal_vals[i+1] + 2: peaks += 1
                if peaks >= 3: return "Импульсивный"
                elif peaks == 2: return "Экспрессивный"
                else: return "Уплощенный"

        means_zh_id, means_zh_re = [target_data[f"M_{s}_Zh-id"].mean() for s in scale_keys], [target_data[f"M_{s}_Zh-re"].mean() for s in scale_keys]
        mot_type_zh = get_mot_profile_type(means_zh_id)
        means_rb_id, means_rb_re = [target_data[f"M_{s}_Rb-id"].mean() for s in scale_keys], [target_data[f"M_{s}_Rb-re"].mean() for s in scale_keys]
        mot_type_rb = get_mot_profile_type(means_rb_id)
        
        with col_m1:
            st.markdown("#### 🏠 Общежитейская сфера")
            st.info(f"🧠 **Тип профиля:** {mot_type_zh}") 
            fig_zh = go.Figure()
            if show_group_m: fig_zh.add_trace(go.Scatter(x=scale_names, y=[df[f"M_{s}_Zh-re"].mean() for s in scale_keys], mode='lines', name='Группа (Реал)', line=dict(color='rgba(180,180,180,0.6)', width=5)))
            fig_zh.add_trace(go.Scatter(x=scale_names, y=means_zh_id, mode='lines+markers', name='Желаемое', line=dict(color='blue', dash='dash', width=3), marker=dict(size=8)))
            fig_zh.add_trace(go.Scatter(x=scale_names, y=means_zh_re, mode='lines+markers', name='Реальное', fill=fill_arg, fillcolor=fill_color, line=dict(color='red', width=3), marker=dict(size=8)))
            fig_zh.update_layout(yaxis=dict(range=[0, max(max(means_zh_id), max(means_zh_re)) + 2]), height=400, margin=dict(l=20, r=20, t=10, b=20), font=dict(size=14))
            st.plotly_chart(fig_zh, use_container_width=True)
            
        with col_m2:
            st.markdown("#### 💼 Учебная/Рабочая сфера")
            st.info(f"🧠 **Тип профиля:** {mot_type_rb}") 
            fig_rb = go.Figure()
            if show_group_m: fig_rb.add_trace(go.Scatter(x=scale_names, y=[df[f"M_{s}_Rb-re"].mean() for s in scale_keys], mode='lines', name='Группа (Реал)', line=dict(color='rgba(180,180,180,0.6)', width=5)))
            fig_rb.add_trace(go.Scatter(x=scale_names, y=means_rb_id, mode='lines+markers', name='Желаемое', line=dict(color='green', dash='dash', width=3), marker=dict(size=8)))
            fig_rb.add_trace(go.Scatter(x=scale_names, y=means_rb_re, mode='lines+markers', name='Реальное', fill=fill_arg, fillcolor=fill_color, line=dict(color='orange', width=3), marker=dict(size=8)))
            fig_rb.update_layout(yaxis=dict(range=[0, max(max(means_rb_id), max(means_rb_re)) + 2]), height=400, margin=dict(l=20, r=20, t=10, b=20), font=dict(size=14))
            st.plotly_chart(fig_rb, use_container_width=True)
        
        if show_frust:
            st.markdown("---")
            st.markdown("#### 📝 Анализ фрустрации (неудовлетворенности)")
            def print_frustration_analysis(id_vals, re_vals, sphere_name):
                deltas = [i - r for i, r in zip(id_vals, re_vals)]
                max_delta_idx, max_delta_val = np.argmax(deltas), np.max(deltas)
                frust_items = [f"*{scale_names[idx]}* (дельта: **{d:.1f}**)" for idx, d in enumerate(deltas) if d > 0]
                if max_delta_val > 0:
                    st.markdown(f"**{sphere_name}:**")
                    st.error(f"⚠️ Наибольшая фрустрация выявлена в шкале: **«{scale_names[max_delta_idx]}»** (неудовлетворенность: **{max_delta_val:.1f}** балла).")
                    if len(frust_items) > 1: st.markdown(f"Также зафиксирована разница в шкалах: {', '.join(frust_items)}.")
                else:
                    st.success(f"**{sphere_name}:** Значимых зон фрустрации не выявлено.")

            col_f1, col_f2 = st.columns(2)
            with col_f1: print_frustration_analysis(means_zh_id, means_zh_re, "Общежитейская сфера")
            with col_f2: print_frustration_analysis(means_rb_id, means_rb_re, "Учебная/Рабочая сфера")
        
        emo_cols = ['M_Est', 'M_East', 'M_Fst', 'M_Fast']
        if all(col in target_data.columns for col in emo_cols):
            st.markdown("---")
            st.markdown("<h4 style='text-align: center;'>🎭 Эмоциональный профиль</h4>", unsafe_allow_html=True)
            emo_means = target_data[emo_cols].mean()
            e_st, e_ast, f_st, f_ast = emo_means['M_Est'], emo_means['M_East'], emo_means['M_Fst'], emo_means['M_Fast']
            
            if e_st > e_ast and f_st > f_ast: emo_type = "Стенический"
            elif e_ast > e_st and f_ast > f_st: emo_type = "Астенический"
            elif e_ast > e_st and f_st > f_ast: emo_type = "Смешанный стенический"
            elif e_st > e_ast and f_ast > f_st: emo_type = "Смешанный астенический"
            else: emo_type = "Не определён (баланс показателей)"
            
            st.markdown(f"<div style='text-align:center; margin-bottom: 10px; font-size: 16px; color: #34495e;'><b>Тип эмоционального профиля:</b> {emo_type}</div>", unsafe_allow_html=True)
            
            col_e1, col_e2, col_e3 = st.columns([1, 1.2, 1])
            with col_e2:
                fig_emo = go.Figure()
                if show_group_m:
                    grp_emo = df[emo_cols].mean()
                    fig_emo.add_trace(go.Scatter(x=["Обычное состояние", "Стресс"], y=[grp_emo['M_Est'], grp_emo['M_Fst']], mode='lines', name='Группа (Стен.)', line=dict(color='rgba(180,180,180,0.6)', width=4)))
                    fig_emo.add_trace(go.Scatter(x=["Обычное состояние", "Стресс"], y=[grp_emo['M_East'], grp_emo['M_Fast']], mode='lines', name='Группа (Астен.)', line=dict(color='rgba(180,180,180,0.6)', width=4, dash='dot')))
                
                fig_emo.add_trace(go.Scatter(x=["Обычное состояние", "Стресс"], y=[e_st, f_st], mode='lines+markers', name='Стенические', line=dict(color='#27ae60', width=5), marker=dict(size=12)))
                fig_emo.add_trace(go.Scatter(x=["Обычное состояние", "Стресс"], y=[e_ast, f_ast], mode='lines+markers', name='Астенические', line=dict(color='#e74c3c', width=4, dash='dot'), marker=dict(size=10, symbol='diamond')))
                fig_emo.update_layout(yaxis=dict(range=[0, emo_means.max() + 4]), height=480, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), font=dict(size=14))
                st.plotly_chart(fig_emo, use_container_width=True)

with subtab_i:
    st.subheader("Инновационный потенциал личности (ИПЛ)")
    show_group_i = st.checkbox("📊 Сравнить со средними показателями группы", key="i_gr") if analysis_mode == "Индивидуальный (Конкретный респондент)" else False
        
    if 'IPL_Total' in target_data.columns:
        total_mean = target_data['IPL_Total'].mean()
        if total_mean >= 118: ipl_level, ipl_color = "Высокий", "linear-gradient(90deg, #2ecc71, #27ae60)"
        elif total_mean >= 95: ipl_level, ipl_color = "Средний", "linear-gradient(90deg, #3498db, #2980b9)"
        else: ipl_level, ipl_color = "Низкий", "linear-gradient(90deg, #e74c3c, #c0392b)"
            
        ipl_percent = min(100, max(0, (total_mean / 180) * 100))
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:5px;">
                <div style="font-size:2.5em; font-weight:bold; color:#2c3e50;">{total_mean:.1f}</div>
                <div style="font-size:1.3em; font-weight:bold; color:#555;">Уровень: {ipl_level}</div>
            </div>
            <div style="background:#eee; border-radius:20px; height:25px; overflow:hidden; margin-bottom:5px; width: 100%;">
                <div style="height:100%; width:{ipl_percent}%; background:{ipl_color}; border-radius: 20px;"></div>
            </div>
            <div style="font-size:1em; color: gray; margin-bottom:10px;">Максимум 180 баллов. Нормы: < 95 (Низкий) | 95-117 (Средний) | ≥ 118 (Высокий)</div>
        """, unsafe_allow_html=True)
        
        if show_group_i: st.markdown(f"<div style='color:gray; font-weight:bold; margin-bottom:20px;'>Справочно: Средний ИПЛ по группе = {df['IPL_Total'].mean():.1f}</div>", unsafe_allow_html=True)
        else: st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            comp_cols = ['IPL_G', 'IPL_A', 'IPL_P']
            available_comp = [c for c in comp_cols if c in target_data.columns]
            if available_comp:
                comp_means = target_data[available_comp].mean()
                fig_comp = go.Figure(go.Bar(name='Респондент', x=comp_means.index, y=comp_means.values, text=[f"{v:.1f}" for v in comp_means.values], textposition='outside', marker_color='#3498db'))
                if show_group_i: fig_comp.add_trace(go.Bar(name='Группа', x=comp_means.index, y=df[available_comp].mean().values, text=[f"{v:.1f}" for v in df[available_comp].mean().values], textposition='outside', marker_color='lightgrey'))
                fig_comp.update_layout(barmode='group', title="Структура ИПЛ (Г-А-П)", showlegend=show_group_i, yaxis=dict(range=[0, comp_means.max() * 1.3]), font=dict(size=14))
                st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            level_cols = ['IPL_Level_Nature', 'IPL_Level_Social', 'IPL_Level_Culture', 'IPL_Level_Life']
            available_levels = [c for c in level_cols if c in target_data.columns]
            if available_levels:
                l_means = target_data[available_levels].mean()
                clean_index = [x.replace('IPL_Level_', '') for x in l_means.index]
                fig_lvl = go.Figure(go.Bar(name='Респондент', y=clean_index, x=l_means.values, orientation='h', text=[f"{v:.1f}" for v in l_means.values], textposition='outside', marker_color='#2ecc71'))
                if show_group_i: fig_lvl.add_trace(go.Bar(name='Группа', y=clean_index, x=df[available_levels].mean().values, orientation='h', text=[f"{v:.1f}" for v in df[available_levels].mean().values], textposition='outside', marker_color='lightgrey'))
                fig_lvl.update_layout(barmode='group', title="Средние баллы по уровням", showlegend=show_group_i, xaxis=dict(range=[0, l_means.max() * 1.3]), font=dict(size=14))
                st.plotly_chart(fig_lvl, use_container_width=True)

        st.markdown("---")
        st.subheader("Типы реализации возможностей")
        type_pairs = [('IPL_Type_OI', 'IPL_Type_FN', 'Поиск информации'), ('IPL_Type_PD', 'IPL_Type_NG', 'Оценка нового'), ('IPL_Type_IP', 'IPL_Type_VP', 'Действие')]
        cols_types = st.columns(3)
        for i, (k1, k2, title) in enumerate(type_pairs):
            if k1 in target_data.columns and k2 in target_data.columns:
                with cols_types[i]:
                    m1, m2 = target_data[k1].mean(), target_data[k2].mean()
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Bar(name=k1.replace('IPL_Type_', ''), x=[k1.replace('IPL_Type_', '')], y=[m1], text=[f"{m1:.1f}"], textposition='auto', marker_color='#9b59b6'))
                    fig_t.add_trace(go.Bar(name=k2.replace('IPL_Type_', ''), x=[k2.replace('IPL_Type_', '')], y=[m2], text=[f"{m2:.1f}"], textposition='auto', marker_color='#34495e'))
                    if show_group_i:
                        fig_t.add_trace(go.Bar(name='Группа', x=[k1.replace('IPL_Type_', '')], y=[df[k1].mean()], text=[f"{df[k1].mean():.1f}"], textposition='auto', marker_color='lightgrey', showlegend=False))
                        fig_t.add_trace(go.Bar(name='Группа', x=[k2.replace('IPL_Type_', '')], y=[df[k2].mean()], text=[f"{df[k2].mean():.1f}"], textposition='auto', marker_color='lightgrey', showlegend=False))
                    fig_t.update_layout(barmode='group', title=title, showlegend=show_group_i, height=300, margin=dict(l=10, r=10, t=40, b=10), font=dict(size=14), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
                    st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.warning("Колонки ИПЛ не найдены.")