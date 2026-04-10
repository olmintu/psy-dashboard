import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pingouin as pg
from utils import render_sidebar, get_name, smart_compare_groups

st.set_page_config(page_title="Сравнение групп", layout="wide", page_icon="🆚")

df = render_sidebar()
if df is None: st.stop()

st.header("🆚 Сравнение групп и проверка гипотез")
    
subtab_single, subtab_mass, subtab_auto = st.tabs([
    "🎯 Детальный анализ одной шкалы", 
    "📊 Сравнение профилей методик", 
    "💡 Авто-поиск всех различий"
])

cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 10]
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# ---------------------------------------------------------
# 1. ОДИНОЧНЫЙ ТЕСТ (Базовый)
# ---------------------------------------------------------
with subtab_single:
    st.subheader("Проверка гипотез (T-test / ANOVA) для конкретного показателя")
    
    col_set1, col_set2 = st.columns(2)
    with col_set1: group_var_single = st.selectbox("1. Разделить группы по:", cat_cols, key="grp_single")
    with col_set2: target_var_single = st.selectbox("2. Сравнить показатель:", num_cols, key="tgt_single", format_func=get_name)

    if group_var_single and target_var_single:
        fig_box = px.box(df, x=group_var_single, y=target_var_single, color=group_var_single, points="all", title=f"{get_name(target_var_single)} по группам {group_var_single}")
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### Результаты статистического теста")
        res_df, p_val_res = smart_compare_groups(df, group_var_single, target_var_single)
        
        if res_df is not None:
            st.table(res_df)
            if p_val_res < 0.05: st.success("✅ Обнаружены **статистически значимые различия** (p < 0.05).")
            else: st.warning("❌ Значимых различий не обнаружено (p > 0.05).")
        else: st.error("Не удалось провести тест (недостаточно данных).")

# ---------------------------------------------------------
# 2. МАССОВОЕ СРАВНЕНИЕ ПРОФИЛЕЙ
# ---------------------------------------------------------
with subtab_mass:
    st.subheader("Визуальное сравнение групп по всем шкалам")
    group_var_mass = st.selectbox("Выберите признак для группировки профилей:", cat_cols, key="grp_mass")
    
    if group_var_mass and len(df[group_var_mass].dropna().unique()) >= 2:
        
        st.markdown("### 1. Жизненные смыслы (Братусь)")
        b_cols = [c for c in df.columns if c.startswith('B_')]
        if b_cols:
            b_res = df.groupby(group_var_mass)[b_cols].mean().reset_index().melt(id_vars=group_var_mass)
            b_res['variable'] = b_res['variable'].apply(lambda x: get_name(x).replace('Братусь: ', ''))
            fig_b = px.bar(b_res, x='variable', y='value', color=group_var_mass, barmode='group', title="Средние ранги (Меньше = Важнее)")
            fig_b.update_layout(xaxis_title="", yaxis_title="Ранг")
            st.plotly_chart(fig_b, use_container_width=True)
        
        st.markdown("### 2. Мотивационная структура (Мильман)")
        m_scales = ['P', 'K', 'S', 'O', 'D', 'DR', 'OD']
        tab_m_life, tab_m_work, tab_m_emo = st.tabs(["🏠 Жизнь", "💼 Работа", "🎭 Эмоции"])
        
        with tab_m_life:
            col_ml1, col_ml2 = st.columns(2)
            with col_ml1:
                m_re_zh = [f"M_{s}_Zh-re" for s in m_scales]
                if all(c in df.columns for c in m_re_zh):
                    res_zh_re = df.groupby(group_var_mass)[m_re_zh].mean().reset_index().melt(id_vars=group_var_mass)
                    res_zh_re['variable'] = res_zh_re['variable'].apply(lambda x: x.split('_')[1].split('-')[0])
                    fig_m_z_re = px.line(res_zh_re, x='variable', y='value', color=group_var_mass, markers=True, title="Реальное состояние")
                    st.plotly_chart(fig_m_z_re, use_container_width=True)
            with col_ml2:
                m_id_zh = [f"M_{s}_Zh-id" for s in m_scales]
                if all(c in df.columns for c in m_id_zh):
                    res_zh_id = df.groupby(group_var_mass)[m_id_zh].mean().reset_index().melt(id_vars=group_var_mass)
                    res_zh_id['variable'] = res_zh_id['variable'].apply(lambda x: x.split('_')[1].split('-')[0])
                    fig_m_z_id = px.line(res_zh_id, x='variable', y='value', color=group_var_mass, markers=True, line_dash=group_var_mass, title="Идеальное состояние")
                    st.plotly_chart(fig_m_z_id, use_container_width=True)

        with tab_m_work:
            col_mw1, col_mw2 = st.columns(2)
            with col_mw1:
                m_re_rb = [f"M_{s}_Rb-re" for s in m_scales]
                if all(c in df.columns for c in m_re_rb):
                    res_rb_re = df.groupby(group_var_mass)[m_re_rb].mean().reset_index().melt(id_vars=group_var_mass)
                    res_rb_re['variable'] = res_rb_re['variable'].apply(lambda x: x.split('_')[1].split('-')[0])
                    fig_m_r_re = px.line(res_rb_re, x='variable', y='value', color=group_var_mass, markers=True, title="Реальное состояние")
                    st.plotly_chart(fig_m_r_re, use_container_width=True)
            with col_mw2:
                m_id_rb = [f"M_{s}_Rb-id" for s in m_scales]
                if all(c in df.columns for c in m_id_rb):
                    res_rb_id = df.groupby(group_var_mass)[m_id_rb].mean().reset_index().melt(id_vars=group_var_mass)
                    res_rb_id['variable'] = res_rb_id['variable'].apply(lambda x: x.split('_')[1].split('-')[0])
                    fig_m_r_id = px.line(res_rb_id, x='variable', y='value', color=group_var_mass, markers=True, line_dash=group_var_mass, title="Идеальное состояние")
                    st.plotly_chart(fig_m_r_id, use_container_width=True)
                    
        with tab_m_emo:
            emo_cols = ['M_Est', 'M_East', 'M_Fst', 'M_Fast']
            if all(c in df.columns for c in emo_cols):
                emo_res = df.groupby(group_var_mass)[emo_cols].mean().reset_index().melt(id_vars=group_var_mass)
                emo_res['variable'] = emo_res['variable'].apply(lambda x: get_name(x).replace('Мильман: ', ''))
                fig_emo = px.bar(emo_res, x='variable', y='value', color=group_var_mass, barmode='group', title="Эмоциональный профиль")
                fig_emo.update_layout(xaxis_title="")
                st.plotly_chart(fig_emo, use_container_width=True)

        st.markdown("### 3. Инновационный потенциал (ИПЛ)")
        c_ipl1, c_ipl2 = st.columns(2)
        with c_ipl1:
            i_cols = ['IPL_G', 'IPL_A', 'IPL_P']
            if all(c in df.columns for c in i_cols):
                ipl_means = df.groupby(group_var_mass)[i_cols].mean().reset_index()
                fig_i_radar = go.Figure()
                for i, row in ipl_means.iterrows():
                    fig_i_radar.add_trace(go.Scatterpolar(r=row[i_cols].values, theta=['Гносеологический', 'Аксиологический', 'Праксеологический'], fill='toself', name=str(row[group_var_mass])))
                fig_i_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Аспекты Г-А-П", margin=dict(t=40, b=20))
                st.plotly_chart(fig_i_radar, use_container_width=True)
        
        with c_ipl2:
            if 'IPL_Total' in df.columns:
                fig_i_tot = px.box(df, x=group_var_mass, y='IPL_Total', color=group_var_mass, title="Общий уровень ИПЛ")
                st.plotly_chart(fig_i_tot, use_container_width=True)

        c_ipl3, c_ipl4 = st.columns(2)
        with c_ipl3:
            types_cols = ['IPL_Type_OI', 'IPL_Type_FN', 'IPL_Type_PD', 'IPL_Type_NG', 'IPL_Type_IP', 'IPL_Type_VP']
            avail_types = [c for c in types_cols if c in df.columns]
            if avail_types:
                res_types = df.groupby(group_var_mass)[avail_types].mean().reset_index().melt(id_vars=group_var_mass)
                res_types['variable'] = res_types['variable'].apply(lambda x: x.split('_')[-1]) 
                fig_types = px.bar(res_types, x='variable', y='value', color=group_var_mass, barmode='group', title="Типы реализации")
                st.plotly_chart(fig_types, use_container_width=True)
                
        with c_ipl4:
            lvl_cols = ['IPL_Level_Nature', 'IPL_Level_Social', 'IPL_Level_Culture', 'IPL_Level_Life']
            avail_lvls = [c for c in lvl_cols if c in df.columns]
            if avail_lvls:
                res_lvl = df.groupby(group_var_mass)[avail_lvls].mean().reset_index().melt(id_vars=group_var_mass)
                res_lvl['variable'] = res_lvl['variable'].apply(lambda x: get_name(x).replace('ИПЛ: ', ''))
                fig_lvl = px.bar(res_lvl, x='value', y='variable', color=group_var_mass, orientation='h', barmode='group', title="Уровни взаимодействия")
                st.plotly_chart(fig_lvl, use_container_width=True)

# ---------------------------------------------------------
# 3. АВТО-ПОИСК РАЗЛИЧИЙ (Супер-функция)
# ---------------------------------------------------------
with subtab_auto:
    st.subheader("Автоматический сканер значимых различий")
    st.markdown("Выберите группирующую переменную, и алгоритм проверит все шкалы на наличие статистически значимых различий.")
    
    group_var_auto = st.selectbox("Признак для анализа:", cat_cols, key="grp_auto")
    
    if st.button("🚀 Начать сканирование"):
        with st.spinner("Считаем статистику..."):
            auto_results = []
            
            for col in num_cols:
                clean_df = df[[group_var_auto, col]].dropna()
                groups = clean_df[group_var_auto].unique()
                
                if len(groups) < 2: continue
                
                try:
                    is_normal = True
                    for g in groups:
                        g_data = clean_df[clean_df[group_var_auto] == g][col]
                        if len(g_data) >= 3 and pg.normality(g_data)['pval'].values[0] < 0.05:
                            is_normal = False
                            break

                    if len(groups) == 2:
                        g1, g2 = clean_df[clean_df[group_var_auto] == groups[0]][col], clean_df[clean_df[group_var_auto] == groups[1]][col]
                        if is_normal:
                            res = pg.ttest(g1, g2, correction=True)
                            p_val, eff, eff_name, test_used = res['p-val'].values[0], res['cohen-d'].values[0], "Cohen's d", "T-test"
                        else:
                            res = pg.mwu(g1, g2)
                            p_val, eff, eff_name, test_used = res['p-val'].values[0], res['RBC'].values[0], "Rank-Biserial", "Mann-Whitney"
                    else:
                        if is_normal:
                            res = pg.anova(data=clean_df, dv=col, between=group_var_auto)
                            p_val, eff, eff_name, test_used = res['p-unc'].values[0], res['np2'].values[0], "Eta-sq", "ANOVA"
                        else:
                            res = pg.kruskal(data=clean_df, dv=col, between=group_var_auto)
                            p_val, eff, eff_name, test_used = res['p-unc'].values[0], res['H'].values[0], "H-stat", "Kruskal-Wallis"
                        
                    if p_val < 0.05:
                        means = clean_df.groupby(group_var_auto)[col].mean()
                        max_group = means.idxmax()
                        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                        
                        auto_results.append({
                            "Показатель": get_name(col), "Тест": test_used, "p-value": f"{p_val:.4f} {stars}",
                            "Метрика эффекта": eff_name, "Размер эффекта": round(eff, 3), "Выше у группы": max_group
                        })
                except:
                    pass
            
            if not auto_results: st.info(f"Значимых различий (p < 0.05) между группами '{group_var_auto}' не найдено.")
            else:
                res_df = pd.DataFrame(auto_results).sort_values(by="Размер эффекта", ascending=False).reset_index(drop=True)
                st.success(f"**Найдено значимых различий: {len(res_df)}** (Отсортировано по силе эффекта)")
                st.dataframe(res_df.style.background_gradient(cmap='Blues', subset=['Размер эффекта']), use_container_width=True)
                st.caption("ℹ️ **Cohen's d**: 0.2 - слабый, 0.5 - средний, >0.8 - сильный. **Eta-sq**: 0.01 - слабый, 0.06 - средний, >0.14 - сильный.")