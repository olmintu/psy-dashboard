import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pingouin as pg
from sklearn.preprocessing import StandardScaler
import io
from sklearn.decomposition import PCA
from utils import render_sidebar, get_name
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

st.set_page_config(page_title="Факторный анализ", layout="wide", page_icon="📐")

df = render_sidebar()
if df is None: st.stop()

st.header("📐 Факторный анализ и надёжность")
st.caption("Поиск латентной структуры данных (FA/PCA) + проверка внутренней согласованности шкал (Альфа Кронбаха).")

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


def evaluate_sample_adequacy(n_obs, n_vars):
    """
    Оценка соотношения наблюдений к переменным для факторного анализа.
    Стандарты в психометрике:
      - < 3:1   — критически мало, результаты ненадёжны
      - 3-5:1   — минимально допустимо, интерпретировать с осторожностью
      - 5-10:1  — приемлемо
      - > 10:1  — достаточно
    """
    if n_vars == 0:
        return None
    ratio = n_obs / n_vars
    if ratio < 3:
        return ("🔴", "Критически мало", f"Соотношение n/переменные = {ratio:.1f}:1 — это меньше минимального порога 3:1. "
                                          f"При {n_vars} переменных нужно минимум {n_vars * 3} наблюдений, а у вас {n_obs}. "
                                          f"Результаты факторного анализа будут ненадёжными.")
    if ratio < 5:
        return ("🟠", "Минимально допустимо", f"Соотношение n/переменные = {ratio:.1f}:1 — в нижней границе допустимого. "
                                               f"Для более надёжных результатов рекомендуется минимум {n_vars * 5} наблюдений. "
                                               f"Интерпретируйте с осторожностью.")
    if ratio < 10:
        return ("🟡", "Приемлемо", f"Соотношение n/переменные = {ratio:.1f}:1 — приемлемо для исследовательских целей.")
    return ("🟢", "Достаточно", f"Соотношение n/переменные = {ratio:.1f}:1 — хорошая статистическая мощность.")
@st.cache_data(show_spinner=False)
def parallel_analysis(n_obs, n_vars, n_iter=100, percentile=95, seed=42):
    """
    Параллельный анализ Хорна (Horn, 1965).
    Генерирует n_iter случайных датасетов размерности (n_obs, n_vars),
    считает собственные значения и возвращает массив пороговых eigenvalues
    на заданном перцентиле. Шкалы реальных данных, чьи eigenvalues выше
    порогов — это «настоящие» факторы, остальное — шум.
    """
    rng = np.random.default_rng(seed)
    random_evs = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        random_data = rng.normal(size=(n_obs, n_vars))
        pca_random = PCA()
        pca_random.fit(random_data)
        random_evs[i] = pca_random.explained_variance_
    return np.percentile(random_evs, percentile, axis=0)
def build_factor_interpretation(loadings, var_labels_raw, factor_names, threshold=0.3, df_fa=None):
    """
    Классифицирует шкалы по факторам на основе матрицы нагрузок.

    Возвращает:
        dict_by_factor: {factor_idx: [(col_raw, loading), ...]} — отсортировано по |loading|
        df_grouped: DataFrame для листа Excel "Группировка по факторам"
        df_cross:   DataFrame для листа Excel "Cross-loadings"
        cross_set:  set колонок с cross-loading (для подсветки в UI)
        unloaded_set: set колонок без сильных нагрузок (для сводки в UI)
    """
    n_vars, n_factors = loadings.shape

    cross_set, unloaded_set = set(), set()
    for i, col_raw in enumerate(var_labels_raw):
        sig_count = sum(1 for j in range(n_factors) if abs(loadings[i, j]) >= threshold)
        if sig_count == 0:
            unloaded_set.add(col_raw)
        elif sig_count >= 2:
            cross_set.add(col_raw)

    # Группировка нагрузок по факторам
    dict_by_factor = {j: [] for j in range(n_factors)}
    for i, col_raw in enumerate(var_labels_raw):
        for j in range(n_factors):
            if abs(loadings[i, j]) >= threshold:
                dict_by_factor[j].append((col_raw, float(loadings[i, j])))
    for j in range(n_factors):
        dict_by_factor[j].sort(key=lambda x: abs(x[1]), reverse=True)
    # Альфа Кронбаха для каждого фактора (с учётом знака нагрузок)
    alpha_by_factor = {}
    if df_fa is not None:
        for j in range(n_factors):
            items = dict_by_factor[j]
            if len(items) < 2:
                alpha_by_factor[j] = None
                continue
            # Инвертируем шкалы с отрицательной нагрузкой,
            # чтобы все шли в одну сторону (требование α Кронбаха)
            series_list = []
            for col_raw, ld in items:
                s = df_fa[col_raw]
                if ld < 0:
                    s = -s
                series_list.append(s.rename(col_raw))
            df_factor = pd.concat(series_list, axis=1).dropna()
            if len(df_factor) < 3:
                alpha_by_factor[j] = None
                continue
            try:
                a, _ = pg.cronbach_alpha(data=df_factor)
                alpha_by_factor[j] = float(a)
            except Exception:
                alpha_by_factor[j] = None

    # Длинная таблица для Excel
    rows = []
    for j in range(n_factors):
        for col_raw, ld in dict_by_factor[j]:
            rows.append({
                'Фактор': factor_names[j],
                'Шкала': get_name(col_raw),
                'Нагрузка': round(ld, 3),
                '|Нагрузка|': round(abs(ld), 3),
                'Знак': '+' if ld > 0 else '−',
                'Тип': 'Cross-loading' if col_raw in cross_set else 'Чистая'
            })
    df_grouped = pd.DataFrame(rows)

    # Таблица cross-loadings: каждая шкала со ВСЕМИ её значимыми нагрузками
    cross_rows = []
    for col_raw in cross_set:
        i = var_labels_raw.index(col_raw)
        for j in range(n_factors):
            if abs(loadings[i, j]) >= threshold:
                cross_rows.append({
                    'Шкала': get_name(col_raw),
                    'Фактор': factor_names[j],
                    'Нагрузка': round(loadings[i, j], 3),
                    '|Нагрузка|': round(abs(loadings[i, j]), 3),
                    'Знак': '+' if loadings[i, j] > 0 else '−'
                })
    df_cross = (pd.DataFrame(cross_rows)
                .sort_values(['Шкала', '|Нагрузка|'], ascending=[True, False])
                .reset_index(drop=True)) if cross_rows else pd.DataFrame()

    return dict_by_factor, df_grouped, df_cross, cross_set, unloaded_set, alpha_by_factor


def render_factor_interpretation(loadings, var_labels_raw, factor_names, threshold, model_label, df_fa=None):
    """Рисует карточки факторов + сводку + кнопку выгрузки в Excel."""
    dict_by_factor, df_grouped, df_cross, cross_set, unloaded_set, alpha_by_factor = build_factor_interpretation(
        loadings, var_labels_raw, factor_names, threshold, df_fa=df_fa
    )
    n_factors = len(factor_names)

    # Карточки по факторам — по 2 в ряд
    cols_per_row = 2
    for row_start in range(0, n_factors, cols_per_row):
        cols = st.columns(cols_per_row)
        for offset in range(cols_per_row):
            idx = row_start + offset
            if idx >= n_factors:
                break
            with cols[offset]:
                with st.container(border=True):
                    st.markdown(f"#### {factor_names[idx]}")

                    # Альфа Кронбаха в подзаголовке карточки
                    a = alpha_by_factor.get(idx)
                    if a is None:
                        st.caption("α Кронбаха: — (нужно ≥ 2 шкал)")
                    else:
                        if a >= 0.8:
                            a_emoji, a_label = "🟢", "высокая согласованность"
                        elif a >= 0.7:
                            a_emoji, a_label = "🟢", "приемлемая"
                        elif a >= 0.6:
                            a_emoji, a_label = "🟡", "сомнительная"
                        else:
                            a_emoji, a_label = "🔴", "низкая"
                        st.caption(f"{a_emoji} α Кронбаха = **{a:.2f}** ({a_label})")

                    items = dict_by_factor[idx]
                    if not items:
                        st.caption(f"Нет шкал с |нагрузкой| ≥ {threshold:.2f}")
                    else:
                        for col_raw, ld in items:
                            sign = "🔴 +" if ld > 0 else "🔵 −"
                            cross_marker = " 🟡" if col_raw in cross_set else ""
                            st.markdown(f"{sign} **{get_name(col_raw)}** — `{ld:+.2f}`{cross_marker}")
    # Сводка
    st.markdown("##### Сводка по структуре")
    s1, s2 = st.columns(2)
    with s1:
        if cross_set:
            st.warning(f"🟡 **Cross-loading ({len(cross_set)}):** "
                       + ", ".join(sorted(get_name(c) for c in cross_set)))
            st.caption("Шкалы с сильной нагрузкой на 2+ фактора. Их сложно однозначно интерпретировать.")
        else:
            st.success("✅ Cross-loading не обнаружено.")
    with s2:
        if unloaded_set:
            st.info(f"⚪ **Не вошли в структуру ({len(unloaded_set)}):** "
                    + ", ".join(sorted(get_name(c) for c in unloaded_set)))
            st.caption(f"Шкалы без нагрузок ≥ {threshold:.2f} ни на один фактор.")
        else:
            st.success("✅ Все шкалы вошли в структуру.")

    # Excel: 2 листа
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        if not df_grouped.empty:
            df_grouped.to_excel(writer, index=False, sheet_name='Группировка по факторам')
        if not df_cross.empty:
            df_cross.to_excel(writer, index=False, sheet_name='Cross-loadings')

    st.download_button(
        f"📥 Скачать интерпретацию ({model_label}) — Excel",
        data=buffer.getvalue(),
        file_name=f'factor_interpretation_{model_label.lower()}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key=f"dl_factor_{model_label.lower()}"
    )
# ФАКТОРНЫЙ АНАЛИЗ — первая вкладка; Кронбах — вторая
subtab_fa, subtab_alpha = st.tabs([
    "🧬 Факторная структура (FA / PCA)",
    "🔗 Внутренняя согласованность (Альфа Кронбаха)",
])

# ---------------------------------------------------------
# 1. ФАКТОРНЫЙ АНАЛИЗ
# ---------------------------------------------------------
with subtab_fa:
    st.subheader("Извлечение латентных факторов")
    st.markdown("Показывает, как исходные шкалы группируются в укрупнённые, скрытые (латентные) факторы.")

    st.write("**Быстрое добавление шкал:**")
    f_b1, f_b2, f_b3, f_b4 = st.columns(4)
    f_b1.button("➕ Братусь", on_click=add_to_state, args=('fa_sel', 'B_'), key="btn_fa_b")
    f_b2.button("➕ Мильман", on_click=add_to_state, args=('fa_sel', 'M_'), key="btn_fa_m")
    f_b3.button("➕ ИПЛ", on_click=add_to_state, args=('fa_sel', 'IPL_'), key="btn_fa_i")
    f_b4.button("❌ Очистить", on_click=clear_state, args=('fa_sel',), key="btn_fa_clear")

    fa_cols = st.multiselect("Выберите шкалы для факторного анализа:", num_cols, key="fa_sel", format_func=get_name)

    if len(fa_cols) >= 3:
        df_fa = df[fa_cols].dropna()
        n_obs = len(df_fa)
        n_vars = len(fa_cols)

        # --- ПРОВЕРКА СООТНОШЕНИЯ N / ПЕРЕМЕННЫЕ ---
        st.markdown("##### 📏 Размер выборки vs количество переменных")
        adequacy = evaluate_sample_adequacy(n_obs, n_vars)
        if adequacy is not None:
            status, label, message = adequacy
            col_n1, col_n2 = st.columns([1, 3])
            with col_n1:
                st.metric(f"{status} {label}", f"{n_obs} / {n_vars}")
                st.caption("наблюдений / переменных")
            with col_n2:
                if status == "🔴":
                    st.error(message)
                elif status == "🟠":
                    st.warning(message)
                elif status == "🟡":
                    st.info(message)
                else:
                    st.success(message)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_fa)
        translated_fa_cols = [get_name(c) for c in fa_cols]

        # --- АКАДЕМИЧЕСКАЯ ПРОВЕРКА ДАННЫХ (SPSS-style) ---
        st.markdown("##### 🔬 Диагностика применимости данных")
        try:
            # Тест Бартлетта
            chi_square_value, p_value = calculate_bartlett_sphericity(df_fa)
            # KMO
            kmo_all, kmo_model = calculate_kmo(df_fa)

            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                if kmo_model >= 0.8: kmo_status = "🟢 Отлично"
                elif kmo_model >= 0.6: kmo_status = "🟡 Приемлемо"
                else: kmo_status = "🔴 Неадекватно"
                st.metric("Мера адекватности KMO", f"{kmo_model:.3f}", kmo_status)
                st.caption("Показывает долю дисперсии, которая может быть вызвана скрытыми факторами (норма > 0.6).")

            with col_diag2:
                bartlett_status = "🟢 Значимо" if p_value < 0.05 else "🔴 Незначимо"
                st.metric("Критерий Бартлетта (p-value)", f"{p_value:.4f}", bartlett_status)
                st.caption("Доказывает, что шкалы коррелируют между собой и анализ имеет смысл (норма < 0.05).")
        except Exception as e:
            st.warning(f"⚠️ Невозможно рассчитать KMO/Бартлетта. Возможно, данных слишком мало или шкалы дублируют друг друга. Ошибка: {e}")

        st.caption("ℹ️ **Как связаны три проверки выше:** соотношение n/переменные говорит, хватит ли вам данных *в принципе*; "
                   "KMO отвечает на вопрос *достаточно ли общей дисперсии* у ваших шкал; "
                   "Бартлетт проверяет, *коррелируют ли шкалы* между собой. Все три должны пройти, чтобы факторный анализ имел смысл.")

        st.divider()
# --- ПАРАЛЛЕЛЬНЫЙ АНАЛИЗ ХОРНА (общие настройки для PCA и EFA) ---
        with st.expander("⚙️ Параметры параллельного анализа Хорна", expanded=False):
            col_pa1, col_pa2 = st.columns(2)
            with col_pa1:
                pa_iter = st.selectbox(
                    "Количество итераций:",
                    [50, 100, 500],
                    index=1,
                    help="Сколько случайных датасетов сгенерировать для оценки шума. "
                         "100 — стандарт. 50 — быстрее, но менее точно. 500 — для строгости."
                )
            with col_pa2:
                pa_percentile = st.slider(
                    "Перцентиль для отсечения шума:",
                    min_value=50, max_value=99, value=95, step=1,
                    help=(
                        "Фактор считается реальным, если его собственное значение выше, "
                        "чем у указанной доли случайных датасетов той же размерности. "
                        "**95-й** — стандарт (Glorfeld, 1995). "
                        "**50-й (медиана)** — оригинальный подход Хорна (1965), менее строгий. "
                        "**99-й** — очень строгий. "
                        "Обычно меняют только при методологическом обосновании."
                    )
                )

        random_evs = parallel_analysis(n_obs, n_vars, n_iter=pa_iter, percentile=pa_percentile)
        # Внутренние вкладки PCA / EFA
        fa_tab_pca, fa_tab_efa = st.tabs(["PCA (Главные компоненты)", "EFA (Факторный анализ)"])

        # --- БЛОК 1: PCA ---
        with fa_tab_pca:
            pca_full = PCA()
            pca_full.fit(data_scaled)
            eigenvalues = pca_full.explained_variance_

            kaiser_factors = int(sum(eigenvalues > 1.0))
            horn_factors = int(sum(eigenvalues > random_evs))
            recommended_pca = max(1, horn_factors)

            col_fa1, col_fa2 = st.columns([1, 2])
            with col_fa1:
                st.success(f"🟢 **По Хорну (parallel analysis):** {horn_factors}")
                st.caption(f"По Кайзеру (eigenvalue > 1): {kaiser_factors}")
                n_factors = st.number_input(
                    "Сколько компонент извлечь?",
                    min_value=1, max_value=len(fa_cols),
                    value=recommended_pca,
                    key="pca_n"
                )

            with col_fa2:
                x_axis = list(range(1, len(fa_cols) + 1))
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Scatter(
                    x=x_axis, y=eigenvalues,
                    mode='lines+markers', name='Реальные данные',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_scree.add_trace(go.Scatter(
                    x=x_axis, y=random_evs,
                    mode='lines+markers', name=f'Случайные (Хорн, p{pa_percentile})',
                    line=dict(color='#ff7f0e', width=2, dash='dot')
                ))
                fig_scree.add_hline(y=1.0, line_dash="dash", line_color="red",
                                    annotation_text="Порог Кайзера (1.0)")
                fig_scree.update_layout(
                    title="График 'Каменистой осыпи' (PCA)",
                    xaxis_title="Номер компоненты",
                    yaxis_title="Собственное значение",
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_scree, use_container_width=True)

            pca_final = PCA(n_components=n_factors)
            pca_final.fit(data_scaled)
            loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)

            factor_names = [f"Компонента {i+1} ({pca_final.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_factors)]

            fig_loadings = go.Figure(data=go.Heatmap(z=loadings, x=factor_names, y=translated_fa_cols, colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(loadings, 2), texttemplate="%{text}", hovertemplate="Шкала: %{y}<br>Компонента: %{x}<br>Нагрузка: %{z:.3f}<extra></extra>"))
            fig_loadings.update_layout(title="Матрица нагрузок PCA", height=max(400, len(fa_cols) * 35))
            st.plotly_chart(fig_loadings, use_container_width=True)
            # --- АВТОМАТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ КОМПОНЕНТ ---
            st.markdown("---")
            st.markdown("##### 🧭 Автоматическая интерпретация компонент")
            threshold_pca = st.slider(
                "Порог значимости нагрузки |loading|:",
                min_value=0.20, max_value=0.70, value=0.30, step=0.05,
                key="pca_threshold",
                help="Шкалы с |нагрузкой| ≥ порога относятся к компоненте. 0.30 — мягкий, 0.40 — средний, 0.50 — строгий."
            )
            render_factor_interpretation(loadings, fa_cols, factor_names, threshold_pca, "PCA", df_fa=df_fa)

        # --- БЛОК 2: EFA ---
        with fa_tab_efa:
            # Инициализируем EFA без вращения для получения собственных значений
            efa_full = FactorAnalyzer(n_factors=len(fa_cols), rotation=None)
            efa_full.fit(data_scaled)
            ev, v = efa_full.get_eigenvalues()

            kaiser_factors_efa = int(sum(ev > 1.0))
            horn_factors_efa = int(sum(ev > random_evs))
            recommended_efa = max(1, horn_factors_efa)

            col_efa1, col_efa2 = st.columns([1, 2])
            with col_efa1:
                st.success(f"🟢 **По Хорну (parallel analysis):** {horn_factors_efa}")
                st.caption(f"По Кайзеру (eigenvalue > 1): {kaiser_factors_efa}")
                n_factors_efa = st.number_input(
                    "Сколько факторов извлечь?",
                    min_value=1, max_value=len(fa_cols),
                    value=recommended_efa,
                    key="efa_n"
                )

            with col_efa2:
                x_axis = list(range(1, len(fa_cols) + 1))
                fig_scree_efa = go.Figure()
                fig_scree_efa.add_trace(go.Scatter(
                    x=x_axis, y=ev,
                    mode='lines+markers', name='Реальные данные',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_scree_efa.add_trace(go.Scatter(
                    x=x_axis, y=random_evs,
                    mode='lines+markers', name=f'Случайные (Хорн, p{pa_percentile})',
                    line=dict(color='#ff7f0e', width=2, dash='dot')
                ))
                fig_scree_efa.add_hline(y=1.0, line_dash="dash", line_color="red",
                                       annotation_text="Порог Кайзера (1.0)")
                fig_scree_efa.update_layout(
                    title="График 'Каменистой осыпи' (EFA)",
                    xaxis_title="Номер фактора",
                    yaxis_title="Собственное значение",
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_scree_efa, use_container_width=True)

            # --- РАСШИРЕННЫЕ НАСТРОЙКИ (SPSS-style) ---
            with st.expander("⚙️ Расширенные настройки EFA (SPSS-style)"):
                st.markdown("Используйте эти настройки для тонкой калибровки, если этого требует методология исследования.")
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    efa_method = st.selectbox(
                        "Метод извлечения факторов:",
                        options=["minres", "ml", "principal"],
                        format_func=lambda x: {
                            "minres": "Minres (Минимум остатков - Рекомендуется)",
                            "ml": "Maximum Likelihood (Макс. правдоподобие)",
                            "principal": "Principal Axis (Главные оси)"
                        }[x],
                        help="Minres — современный стандарт EFA. ML хорош для нормально распределённых данных. Principal Axis — классика из старых версий SPSS."
                    )
                with col_opt2:
                    efa_rotation = st.selectbox(
                        "Метод вращения:",
                        options=["varimax", "promax", "oblimin", None],
                        format_func=lambda x: {
                            "varimax": "Varimax (Ортогональное - факторы независимы)",
                            "promax": "Promax (Косоугольное - факторы связаны)",
                            "oblimin": "Oblimin (Косоугольное)",
                            None: "Без вращения"
                        }[x],
                        help="Varimax делает структуру максимально чёткой. Promax и Oblimin разрешают факторам коррелировать (что часто бывает в психологии)."
                    )

            # Словарь с описаниями типов вращения
            rotation_info = {
                "varimax": {
                    "name": "Varimax",
                    "desc": "🔍 **Вращение Varimax:** максимизирует дисперсию нагрузок. Это делает факторы 'чище', заставляя каждую шкалу сильно коррелировать только с одним фактором, что упрощает психологическую интерпретацию."
                },
                "promax": {
                    "name": "Promax",
                    "desc": "🔍 **Вращение Promax:** косоугольное вращение. Позволяет факторам коррелировать между собой. Это часто лучше отражает психологическую реальность, где черты личности редко бывают полностью независимыми."
                },
                "oblimin": {
                    "name": "Oblimin",
                    "desc": "🔍 **Вращение Oblimin:** косоугольное вращение. Гибкий метод для выявления сложной структуры взаимосвязей между факторами."
                },
                None: {
                    "name": "Без вращения",
                    "desc": "⚠️ **Без вращения:** отображается исходная структура. Факторы обычно сложнее интерпретировать, так как шкалы могут иметь высокие нагрузки сразу на несколько факторов."
                }
            }

            # Финальная модель EFA с выбранными настройками
            efa_final = FactorAnalyzer(n_factors=n_factors_efa, rotation=efa_rotation, method=efa_method)
            efa_final.fit(data_scaled)
            loadings_efa = efa_final.loadings_

            factor_names_efa = [f"Фактор {i+1}" for i in range(n_factors_efa)]

            current_rotation_name = rotation_info[efa_rotation]["name"]

            fig_loadings_efa = go.Figure(data=go.Heatmap(
                z=loadings_efa,
                x=factor_names_efa,
                y=translated_fa_cols,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=np.round(loadings_efa, 2),
                texttemplate="%{text}",
                hovertemplate="Шкала: %{y}<br>Фактор: %{x}<br>Нагрузка: %{z:.3f}<extra></extra>"
            ))

            fig_loadings_efa.update_layout(
                title=f"Матрица факторных нагрузок EFA (Вращение: {current_rotation_name})",
                height=max(400, len(fa_cols) * 35)
            )

            st.plotly_chart(fig_loadings_efa, use_container_width=True)

            st.caption(rotation_info[efa_rotation]["desc"])
            # --- АВТОМАТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ ФАКТОРОВ ---
            st.markdown("---")
            st.markdown("##### 🧭 Автоматическая интерпретация факторов")
            threshold_efa = st.slider(
                "Порог значимости нагрузки |loading|:",
                min_value=0.20, max_value=0.70, value=0.30, step=0.05,
                key="efa_threshold",
                help="Шкалы с |нагрузкой| ≥ порога относятся к фактору. 0.30 — мягкий, 0.40 — средний, 0.50 — строгий."
            )
            # Имена факторов с долей дисперсии (если доступно)
            try:
                ev_proportions = efa_final.get_factor_variance()[1]
                factor_names_efa_rich = [f"Фактор {i+1} ({ev_proportions[i]*100:.1f}%)"
                                         for i in range(n_factors_efa)]
            except Exception:
                factor_names_efa_rich = factor_names_efa
            render_factor_interpretation(loadings_efa, fa_cols, factor_names_efa_rich, threshold_efa, "EFA", df_fa=df_fa)
            # --- ЭКСПОРТ ФАКТОРНЫХ ОЦЕНОК В ДАТАСЕТ ---
            st.markdown("---")
            st.markdown("##### 💾 Сохранить факторы как переменные")
            st.caption(
                "Рассчитывает факторные оценки (взвешенные суммы исходных шкал по матрице нагрузок) "
                "для каждого респондента и добавляет их в датасет как новые колонки. "
                "Скачайте файл и загрузите его на главной странице — новые «супер-шкалы» появятся "
                "во всех вкладках (корреляции, сравнение групп, кластеризация и т.д.)."
            )

            with st.form("factor_export_form"):
                st.markdown("**Названия факторов** (используются как названия колонок):")
                factor_custom_names = []
                cols_per_row = 2
                for row_start in range(0, n_factors_efa, cols_per_row):
                    name_cols = st.columns(cols_per_row)
                    for offset in range(cols_per_row):
                        idx = row_start + offset
                        if idx >= n_factors_efa:
                            break
                        with name_cols[offset]:
                            default_name = f"Фактор_{idx + 1}"
                            custom = st.text_input(
                                f"Фактор {idx + 1}",
                                value=default_name,
                                key=f"factor_name_{idx}"
                            )
                            factor_custom_names.append(custom)

                generate_export = st.form_submit_button(
                    "🔧 Рассчитать факторные оценки",
                    type="primary",
                    use_container_width=True
                )

            if generate_export:
                # Проверка на дубликаты названий
                clean_names = [n.strip() if n.strip() else f"Фактор_{i+1}"
                               for i, n in enumerate(factor_custom_names)]
                if len(set(clean_names)) != len(clean_names):
                    st.error("❌ Названия факторов должны быть уникальными. Исправьте дубликаты.")
                else:
                    # Рассчитываем факторные оценки методом регрессии
                    factor_scores = efa_final.transform(data_scaled)
                    score_col_names = [f"FA_{name.replace(' ', '_')}" for name in clean_names]

                    # Берём весь df (а не только df_fa) и вписываем оценки только для тех строк,
                    # которые попали в анализ (без пропусков по выбранным шкалам)
                    df_export = df.copy()
                    for col in score_col_names:
                        df_export[col] = np.nan
                    df_export.loc[df_fa.index, score_col_names] = factor_scores

                    # Сохраняем результат в session_state, чтобы кнопка скачивания
                    # не сбрасывалась при перерисовке страницы
                    st.session_state['fa_export_df'] = df_export
                    st.session_state['fa_export_meta'] = {
                        'n_factors': n_factors_efa,
                        'n_respondents': len(df_fa),
                        'n_total': len(df_export),
                        'method': efa_method,
                        'rotation': str(efa_rotation),
                        'col_names': score_col_names,
                        'human_names': clean_names,
                    }

            # Если оценки уже посчитаны — показываем их и кнопку скачивания
            if 'fa_export_df' in st.session_state:
                meta = st.session_state['fa_export_meta']
                st.success(
                    f"✅ Рассчитано {meta['n_factors']} факторов "
                    f"для {meta['n_respondents']} из {meta['n_total']} респондентов "
                    f"(остальные исключены из-за пропусков по выбранным шкалам)."
                )

                with st.expander("👀 Превью факторных оценок (первые 10 строк)", expanded=False):
                    preview_df = st.session_state['fa_export_df'][meta['col_names']].head(10)
                    st.dataframe(preview_df, use_container_width=True)

                with st.expander("📊 Описательная статистика факторных оценок", expanded=False):
                    desc = st.session_state['fa_export_df'][meta['col_names']].describe().round(3)
                    st.dataframe(desc, use_container_width=True)
                    st.caption("Факторные оценки стандартизированы (M ≈ 0, SD ≈ 1).")

                # Excel с двумя листами: данные + метаданные о расчёте
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state['fa_export_df'].to_excel(writer, index=False, sheet_name='Данные')

                    meta_rows = [
                        ['Дата расчёта', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
                        ['Метод извлечения', meta['method']],
                        ['Метод вращения', meta['rotation']],
                        ['Количество факторов', meta['n_factors']],
                        ['Респондентов с факторными оценками', meta['n_respondents']],
                        ['Всего респондентов в выборке', meta['n_total']],
                        ['', ''],
                        ['Исходные шкалы для FA:', ''],
                    ]
                    for c in fa_cols:
                        meta_rows.append(['', get_name(c)])
                    meta_rows.append(['', ''])
                    meta_rows.append(['Названия факторов:', ''])
                    for human, code in zip(meta['human_names'], meta['col_names']):
                        meta_rows.append([code, human])

                    pd.DataFrame(meta_rows, columns=['Параметр', 'Значение']).to_excel(
                        writer, index=False, sheet_name='Метаданные'
                    )

                st.download_button(
                    "📥 Скачать датасет с факторами (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"data_with_factors_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )

                if st.button("🗑️ Очистить расчёт", key="clear_fa_export"):
                    del st.session_state['fa_export_df']
                    del st.session_state['fa_export_meta']
                    st.rerun()

    else:
        st.info("Для факторного анализа требуется минимум 3 шкалы.")

# ---------------------------------------------------------
# 2. АЛЬФА КРОНБАХА
# ---------------------------------------------------------
with subtab_alpha:
    st.subheader("Макро-согласованность шкал (Альфа Кронбаха)")
    st.markdown("Позволяет проверить, образуют ли выбранные шкалы единый теоретический конструкт.")

    st.warning("⚠️ **Аналитический контекст:** в систему загружены финальные баллы по шкалам (не ответы на отдельные пункты). "
               "Поэтому альфа здесь измеряет не классическую надёжность отдельных вопросов, а **макро-согласованность** — "
               "степень того, насколько выбранные шкалы ведут себя как части одного общего конструкта.")

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
            st.error("Недостаточно данных для расчёта.")
    else:
        st.info("Выберите минимум 2 шкалы.")

st.session_state.safe_alpha_sel = st.session_state.alpha_sel
st.session_state.safe_fa_sel = st.session_state.fa_sel