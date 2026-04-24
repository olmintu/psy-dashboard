import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pingouin as pg
from utils import render_sidebar, get_name, smart_compare_groups, run_auto_scan, DERIVED_CATEGORICAL_COLS

st.set_page_config(page_title="Сравнение групп", layout="wide", page_icon="🆚")

df = render_sidebar()
if df is None: st.stop()

st.header("🆚 Сравнение групп и проверка гипотез")

subtab_single, subtab_mass, subtab_auto, subtab_cross, subtab_cross_auto = st.tabs([
    "🎯 Детальный анализ одной шкалы",
    "📊 Сравнение профилей методик",
    "💡 Авто-поиск различий (число × группа)",
    "🔀 Пересечение типов (категория × категория)",
    "🔎 Авто-поиск пересечений типов"
])

# Категориальные колонки: текстовые (object/string) + числовые с малым числом уникальных + все наши производные типы
cat_cols = [c for c in df.columns if
            df[c].dtype == 'object'
            or str(df[c].dtype) == 'string'
            or df[c].nunique() < 10
            or c in DERIVED_CATEGORICAL_COLS]
# Числовые колонки БЕЗ производных категориальных (защита от случая, когда производные загружены как число)
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in DERIVED_CATEGORICAL_COLS]

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
        # Только числовые шкалы Братуся (без производных _Level колонок)
        b_cols = [c for c in df.columns if c.startswith('B_') and not c.endswith('_Level')]
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
# 3. АВТО-ПОИСК РАЗЛИЧИЙ (через run_auto_scan из utils)
# ---------------------------------------------------------
with subtab_auto:
    st.subheader("Автоматический сканер значимых различий: числовая шкала × группа")
    st.markdown("Выберите группирующую переменную, и алгоритм проверит все **числовые** шкалы на наличие "
                "статистически значимых различий. Для каждой шкалы автоматически выбирается подходящий тест "
                "(t-test / U-test / ANOVA / Kruskal-Wallis).")

    col_auto1, col_auto2 = st.columns(2)
    with col_auto1:
        group_var_auto = st.selectbox("Признак для анализа:", cat_cols, key="grp_auto")
    with col_auto2:
        alpha_level = st.select_slider(
            "Уровень значимости (α):",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            help="По умолчанию 0.05 — стандарт в психологических исследованиях."
        )

    if st.button("🚀 Начать сканирование"):
        with st.spinner("Считаем статистику по всем шкалам..."):
            # Единый вызов вместо 40+ строк дублированной логики
            scan_df = run_auto_scan(df, group_var_auto, num_cols, alpha=alpha_level)

            if scan_df.empty:
                st.info(f"Не удалось рассчитать тесты для выбранной группирующей переменной '{group_var_auto}'.")
            else:
                # Отбираем только значимые
                sig_df = scan_df[scan_df['_p_raw'] < alpha_level].copy()

                # Для каждой значимой шкалы определяем, в какой группе среднее выше
                if not sig_df.empty:
                    max_groups = []
                    for col in sig_df['Колонка']:
                        means = df[[group_var_auto, col]].dropna().groupby(group_var_auto)[col].mean()
                        max_groups.append(str(means.idxmax()) if not means.empty else "—")
                    sig_df['Выше у группы'] = max_groups

                    # Готовим DataFrame для отображения (без служебных колонок)
                    display_cols = [c for c in sig_df.columns if not c.startswith('_') and c != 'Колонка']
                    display_df = sig_df[display_cols].copy()
                    # Пересортировать по модулю размера эффекта (чтобы самые сильные эффекты были наверху)
                    display_df['_abs_effect'] = sig_df['_effect_raw'].abs()
                    display_df = display_df.sort_values('_abs_effect', ascending=False).drop(columns=['_abs_effect']).reset_index(drop=True)

                    st.success(f"**Найдено значимых различий: {len(display_df)}** (при α = {alpha_level}, отсортировано по силе эффекта)")

                    # Определяем колонку с размером эффекта для подсветки
                    effect_col = next((c for c in display_df.columns if 'Размер эффекта' in c), None)
                    if effect_col:
                        st.dataframe(display_df.style.background_gradient(cmap='Blues', subset=[effect_col]), use_container_width=True)
                    else:
                        st.dataframe(display_df, use_container_width=True)

                    # Скачивание полного скана (включая незначимые)
                    import io
                    buffer_scan = io.BytesIO()
                    scan_df.drop(columns=['_p_raw', '_effect_raw']).to_excel(buffer_scan, index=False)
                    st.download_button(
                        "📥 Скачать полный отчёт сканирования (Excel)",
                        buffer_scan.getvalue(),
                        f"auto_scan_{group_var_auto}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Включает все шкалы — и значимые, и незначимые."
                    )
                else:
                    st.info(f"Значимых различий (p < {alpha_level}) между группами '{group_var_auto}' не найдено. "
                            f"Всего протестировано шкал: {len(scan_df)}.")

                st.caption("ℹ️ **Cohen's d**: 0.2 — слабый, 0.5 — средний, >0.8 — сильный. "
                           "**Eta-sq**: 0.01 — слабый, 0.06 — средний, >0.14 — сильный.")


# ---------------------------------------------------------
# Вспомогательная функция для χ² + Cramér's V + residuals
# ---------------------------------------------------------
def chi2_analysis(df_src, col_a, col_b, min_cell=1):
    """
    Рассчитывает полный набор статистик для пересечения двух категориальных колонок.

    Returns dict с полями:
        contingency : pd.DataFrame  -- таблица сопряжённости (наблюдаемые частоты)
        expected    : pd.DataFrame  -- ожидаемые частоты
        residuals   : pd.DataFrame  -- стандартизованные остатки (|r|>2 = значимое отклонение)
        chi2, p, dof, n
        cramers_v   : float         -- сила связи (0..1), аналог корреляции для категорий
        min_expected : float        -- минимальная ожидаемая частота (для проверки условий χ²)
        warning     : str | None    -- предупреждение, если условия нарушены
    """
    from scipy import stats
    clean = df_src[[col_a, col_b]].dropna()
    if len(clean) == 0:
        return None
    ct = pd.crosstab(clean[col_a], clean[col_b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
    except Exception:
        return None
    n = ct.values.sum()
    # Cramér's V
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
    # Стандартизованные остатки
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals = (ct.values - expected) / np.sqrt(expected)
    residuals_df = pd.DataFrame(residuals, index=ct.index, columns=ct.columns)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
    min_exp = float(expected.min())
    warn = None
    # Классическая проверка Cochran: >20% ячеек с ожидаемой частотой <5 или любая <1 → χ² ненадёжен
    n_cells = ct.size
    low_cells = int((expected < 5).sum())
    if min_exp < 1:
        warn = (f"⚠️ Минимальная ожидаемая частота = {min_exp:.2f} (<1). "
                f"Результаты χ² ненадёжны, рекомендуется точный тест Фишера или объединение категорий.")
    elif low_cells / n_cells > 0.2:
        warn = (f"⚠️ В {low_cells} из {n_cells} ячеек ожидаемая частота <5 ({low_cells/n_cells*100:.0f}%). "
                f"Результаты χ² могут быть неточными.")
    return {
        'contingency': ct, 'expected': expected_df, 'residuals': residuals_df,
        'chi2': float(chi2), 'p': float(p), 'dof': int(dof), 'n': int(n),
        'cramers_v': cramers_v, 'min_expected': min_exp, 'warning': warn
    }


def cramers_v_label(v):
    """Интерпретация силы связи Cramér's V."""
    if v < 0.10: return "пренебрежимо слабая"
    if v < 0.20: return "слабая"
    if v < 0.30: return "умеренная"
    if v < 0.50: return "сильная"
    return "очень сильная"


# ---------------------------------------------------------
# 4. ПЕРЕСЕЧЕНИЕ ДВУХ ТИПОВ (ручной выбор)
# ---------------------------------------------------------
with subtab_cross:
    st.subheader("Пересечение двух категориальных переменных")
    st.markdown("Выберите любые две категориальные переменные (типы, профили, демографические признаки). "
                "Алгоритм построит таблицу сопряжённости, проверит связь **критерием χ²** и оценит силу "
                "связи через **Cramér's V**.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        var_a = st.selectbox("Переменная A (строки):", cat_cols, key="cross_a")
    with col_c2:
        # Исключаем уже выбранную из кандидатов
        other_cats = [c for c in cat_cols if c != var_a]
        default_b_idx = 0
        # Если можно, предлагаем сразу что-то из производных типов чтобы был сразу интересный пример
        for candidate in ['M_Emo_Profile', 'IPL_Style', 'IPL_IP_VP']:
            if candidate in other_cats and candidate != var_a:
                default_b_idx = other_cats.index(candidate)
                break
        var_b = st.selectbox("Переменная B (столбцы):", other_cats, index=default_b_idx, key="cross_b")

    # Настройки отображения
    show_opts = st.columns(4)
    with show_opts[0]:
        show_percent = st.checkbox("Показать проценты (по строкам)", value=True)
    with show_opts[1]:
        show_expected = st.checkbox("Показать ожидаемые частоты", value=False)
    with show_opts[2]:
        show_residuals = st.checkbox("Показать остатки (|r|>2 — значимое отклонение)", value=False)
    with show_opts[3]:
        show_chart = st.checkbox("Стековый barchart", value=True)

    if var_a and var_b and var_a != var_b:
        result = chi2_analysis(df, var_a, var_b)
        if result is None:
            st.error("Недостаточно данных или в одной из переменных меньше 2 уникальных значений.")
        else:
            # ===== РЕЗУЛЬТАТЫ ТЕСТА =====
            stat_cols = st.columns(4)
            with stat_cols[0]:
                p_val = result['p']
                stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "н.з."))
                if p_val < 0.05:
                    st.success(f"**p-value: {p_val:.4f}** {stars}")
                else:
                    st.info(f"**p-value: {p_val:.4f}** {stars}")
            with stat_cols[1]:
                st.metric("χ²", f"{result['chi2']:.2f}", f"df = {result['dof']}")
            with stat_cols[2]:
                v = result['cramers_v']
                st.metric("Cramér's V", f"{v:.3f}", cramers_v_label(v))
            with stat_cols[3]:
                st.metric("Наблюдений", result['n'])

            if result['warning']:
                st.warning(result['warning'])

            # Интерпретация
            if p_val < 0.05:
                st.markdown(f"💡 **Вывод:** Между **{get_name(var_a)}** и **{get_name(var_b)}** есть "
                            f"статистически значимая связь (сила связи {cramers_v_label(v).lower()}).")
            else:
                st.markdown(f"💡 **Вывод:** Статистически значимой связи между **{get_name(var_a)}** "
                            f"и **{get_name(var_b)}** не обнаружено.")

            # ===== СТЕКОВЫЙ BARCHART =====
            if show_chart:
                st.markdown("#### 📊 Распределение B внутри каждой группы A (%)")
                ct = result['contingency']
                ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                plot_data = ct_pct.reset_index().melt(id_vars=var_a, var_name=var_b, value_name='Процент')
                # Добавим реальное количество для подсказки
                ct_counts = ct.reset_index().melt(id_vars=var_a, var_name=var_b, value_name='N')
                plot_data = plot_data.merge(ct_counts, on=[var_a, var_b])
                fig_stack = px.bar(
                    plot_data, x=var_a, y='Процент', color=var_b,
                    text=plot_data.apply(lambda r: f"{r['Процент']:.0f}%" if r['Процент'] >= 5 else "", axis=1),
                    custom_data=['N']
                )
                fig_stack.update_traces(hovertemplate=f"<b>{var_a}</b>: %{{x}}<br><b>{var_b}</b>: %{{fullData.name}}"
                                                      "<br>Доля: %{y:.1f}%<br>Количество: %{customdata[0]}<extra></extra>",
                                        textposition='inside')
                fig_stack.update_layout(yaxis=dict(range=[0, 100], title="% респондентов"),
                                        xaxis_title=get_name(var_a), barmode='stack', height=450,
                                        legend_title=get_name(var_b))
                st.plotly_chart(fig_stack, use_container_width=True)

            # ===== ТАБЛИЦЫ =====
            tab_obs, tab_pct, tab_exp, tab_res = st.tabs([
                "Наблюдаемые частоты",
                "В процентах по строкам",
                "Ожидаемые частоты" if show_expected else "_",
                "Остатки" if show_residuals else "_"
            ])
            with tab_obs:
                # Добавляем итоги по строкам/столбцам
                ct_with_totals = result['contingency'].copy()
                ct_with_totals['Всего'] = ct_with_totals.sum(axis=1)
                ct_with_totals.loc['Всего'] = ct_with_totals.sum(axis=0)
                st.dataframe(ct_with_totals, use_container_width=True)
            with tab_pct:
                if show_percent:
                    ct_pct_disp = result['contingency'].div(result['contingency'].sum(axis=1), axis=0) * 100
                    st.dataframe(ct_pct_disp.round(1), use_container_width=True)
                    st.caption("Значения в строке суммируются до 100%. Показывает, как распределены категории B внутри каждой группы A.")
                else:
                    st.info("Включите галочку «Показать проценты» чтобы увидеть таблицу.")
            with tab_exp:
                if show_expected:
                    st.dataframe(result['expected'].round(1), use_container_width=True)
                    st.caption("Ожидаемые частоты — то, сколько было бы в каждой ячейке при отсутствии связи между переменными.")
                else:
                    st.info("Включите галочку «Показать ожидаемые частоты».")
            with tab_res:
                if show_residuals:
                    # Подсветка |r|>2
                    res_df = result['residuals'].round(2)
                    def highlight(v):
                        if abs(v) >= 2.5: return 'background-color: #d32f2f; color: white; font-weight: bold;'
                        if abs(v) >= 2.0: return 'background-color: #fbc02d; font-weight: bold;'
                        return ''
                    st.dataframe(res_df.style.map(highlight), use_container_width=True)
                    st.caption("**Стандартизованные остатки:** показывают, какие ячейки вносят вклад в значимость. "
                               "**|r| ≥ 2** — ячейка значимо отличается от ожидаемого (выделено жёлтым). "
                               "**|r| ≥ 2.5** — очень сильное отклонение (красное). "
                               "Положительное r = в этой комбинации людей больше, чем ожидалось. Отрицательное — меньше.")
                else:
                    st.info("Включите галочку «Показать остатки».")

            # ===== СКАЧИВАНИЕ =====
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                result['contingency'].to_excel(writer, sheet_name='Наблюдаемые')
                result['expected'].round(2).to_excel(writer, sheet_name='Ожидаемые')
                result['residuals'].round(3).to_excel(writer, sheet_name='Остатки')
                # Итоговая сводка
                summary = pd.DataFrame({
                    'Показатель': ['χ²', 'df', 'p-value', "Cramér's V", 'N наблюдений',
                                   'Минимальная ожидаемая частота'],
                    'Значение':   [f"{result['chi2']:.3f}", result['dof'], f"{result['p']:.4f}",
                                   f"{result['cramers_v']:.3f}", result['n'],
                                   f"{result['min_expected']:.2f}"]
                })
                summary.to_excel(writer, sheet_name='Статистика', index=False)
            st.download_button(
                "📥 Скачать анализ (Excel с 4 листами)",
                buffer.getvalue(),
                f"cross_{var_a}_{var_b}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ---------------------------------------------------------
# 5. АВТОПОИСК ПЕРЕСЕЧЕНИЙ (массовый χ²)
# ---------------------------------------------------------
with subtab_cross_auto:
    st.subheader("Автоматический поиск связей между всеми парами категориальных переменных")
    st.markdown("Алгоритм попарно проверяет все выбранные категориальные переменные на наличие связи "
                "(критерий χ²). Из-за большого числа тестов применяется **поправка Бонферрони** "
                "против ложноположительных результатов.")

    # Отбираем кандидатов: только переменные с 2-15 уникальными значениями
    candidate_cats = [c for c in cat_cols
                      if 2 <= df[c].dropna().nunique() <= 15
                      and df[c].dropna().shape[0] >= 20]

    col_sc1, col_sc2 = st.columns([2, 1])
    with col_sc1:
        # По умолчанию предлагаем только наши производные типы
        default_sel = [c for c in candidate_cats if c in DERIVED_CATEGORICAL_COLS]
        selected_cats = st.multiselect(
            "Переменные для попарного анализа:",
            candidate_cats,
            default=default_sel,
            help="По умолчанию выбраны все производные типы. Можно добавить демографические (Gender, Edu_Status и т.п.)."
        )
    with col_sc2:
        alpha_cross = st.select_slider("Уровень значимости α:", options=[0.01, 0.05, 0.10], value=0.05)

    if st.button("🚀 Запустить автопоиск пересечений", key="btn_cross_auto"):
        if len(selected_cats) < 2:
            st.warning("Выберите как минимум 2 переменные.")
        else:
            with st.spinner("Считаем χ² для всех пар..."):
                from itertools import combinations
                rows = []
                pairs = list(combinations(selected_cats, 2))
                for a, b in pairs:
                    r = chi2_analysis(df, a, b)
                    if r is None:
                        continue
                    rows.append({
                        'Переменная A': get_name(a),
                        'Переменная B': get_name(b),
                        'col_a': a, 'col_b': b,
                        'χ²': round(r['chi2'], 2),
                        'df': r['dof'],
                        'p-value': r['p'],
                        "Cramér's V": round(r['cramers_v'], 3),
                        'Сила связи': cramers_v_label(r['cramers_v']),
                        'n': r['n'],
                        'min.ожид': round(r['min_expected'], 2),
                        '_надёжность': '⚠️' if r['warning'] else '✓',
                    })

                if not rows:
                    st.info("Не удалось рассчитать тесты.")
                else:
                    scan = pd.DataFrame(rows)
                    n_tests = len(scan)
                    # Поправка Бонферрони
                    bonf_alpha = alpha_cross / n_tests
                    scan['p_bonf'] = (scan['p-value'] * n_tests).clip(upper=1.0)
                    scan['Значимо (α)'] = scan['p-value'].apply(
                        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < alpha_cross else "н.з.")))
                    scan['Значимо (Бонферрони)'] = scan['p-value'].apply(lambda p: "✓" if p < bonf_alpha else "—")
                    scan = scan.sort_values('p-value').reset_index(drop=True)

                    # Сводка
                    n_raw = int((scan['p-value'] < alpha_cross).sum())
                    n_bonf = int((scan['p-value'] < bonf_alpha).sum())
                    st.success(f"Всего пар протестировано: **{n_tests}**. "
                               f"Значимых при α={alpha_cross}: **{n_raw}**. "
                               f"Выдерживают поправку Бонферрони (α={bonf_alpha:.5f}): **{n_bonf}**.")

                    # Отображение
                    display_cols = ['Переменная A', 'Переменная B', 'χ²', 'df', 'p-value',
                                    "Cramér's V", 'Сила связи', 'Значимо (α)',
                                    'Значимо (Бонферрони)', 'n', 'min.ожид', '_надёжность']
                    display_df = scan[display_cols].copy()
                    display_df['p-value'] = display_df['p-value'].apply(lambda p: f"{p:.4f}")
                    display_df = display_df.rename(columns={'_надёжность': 'Надёжн.'})
                    st.dataframe(
                        display_df.style.background_gradient(cmap='Blues', subset=["Cramér's V"]),
                        use_container_width=True
                    )
                    st.caption("**Cramér's V**: 0.10 слабая | 0.20 умеренная | 0.30 сильная | 0.50 очень сильная. "
                               "**Надёжн. ⚠️** — условия применимости χ² нарушены (мало наблюдений в некоторых "
                               "ячейках), результаты такой пары интерпретируйте с осторожностью.")

                    # Скачивание
                    import io
                    buf = io.BytesIO()
                    scan.drop(columns=['col_a', 'col_b']).to_excel(buf, index=False)
                    st.download_button(
                        "📥 Скачать все результаты (Excel)",
                        buf.getvalue(),
                        "cross_autoscan.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )