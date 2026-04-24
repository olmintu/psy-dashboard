import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from scipy import stats
import pingouin as pg
from utils import render_sidebar, get_name, DERIVED_CATEGORICAL_COLS

st.set_page_config(page_title="Регрессионный анализ", layout="wide", page_icon="📈")

df = render_sidebar()
if df is None:
    st.stop()

st.header("📈 Регрессионный анализ")
st.caption("Построение моделей предсказания целевой переменной с множественными подходами: "
           "стандартная регрессия, иерархическая (по блокам), частные корреляции, "
           "регуляризованные модели (Ridge/Lasso).")

# Числовые колонки без производных категориальных
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in DERIVED_CATEGORICAL_COLS]

with st.expander("ℹ️ Как пользоваться: когда какой подход выбирать"):
    st.markdown("""
**🎯 Множественная регрессия (стандартная)** — базовый подход. Выбираешь целевую переменную (Y) и набор предикторов (X). Получаешь R², β-коэффициенты и их значимость. Используй, если у тебя есть список потенциальных предикторов и нужно понять, насколько хорошо они вместе объясняют Y и кто из них значим.

**🧱 Иерархическая регрессия** — продвинутый подход, стандарт в психологических исследованиях. Разбиваешь предикторы на **блоки** (например: блок 1 — демография, блок 2 — смыслы, блок 3 — мотивация) и добавляешь их **по очереди**. Видишь, сколько дополнительной дисперсии (прирост R²) объясняет каждый блок. Отвечает на вопрос: «Объясняют ли мотивационные переменные ИПЛ сверх того, что уже объясняется смыслами и демографией?»

**🕸 Частные корреляции** — ответ на вопрос «какая остаётся связь между X и Y, если контролировать другие переменные?». Например, связь Братуся с ИПЛ может оказаться только видимостью, вызванной общей связью с мотивацией. Частная корреляция удаляет это влияние.

**🛡 Регуляризованная регрессия (Ridge/Lasso)** — когда предикторов много и они сами между собой коррелируют (мультиколлинеарность). Ridge уменьшает коэффициенты пропорционально, Lasso обнуляет самые слабые — делает автоматический отбор переменных.
""")

tab_simple, tab_hier, tab_partial, tab_reg = st.tabs([
    "🎯 Множественная регрессия",
    "🧱 Иерархическая регрессия",
    "🕸 Частные корреляции",
    "🛡 Регуляризованная регрессия (Ridge/Lasso)"
])


# =============================================================================
# Вспомогательные функции
# =============================================================================

def fit_ols_with_stats(X, y, var_names):
    """
    Линейная регрессия с полным набором статистик для каждого предиктора.
    Использует формулы OLS для расчёта стандартных ошибок и t-тестов.

    Returns dict с: coefs, std_errors, t_stats, p_values, ci_lower, ci_upper,
                    r2, r2_adj, f_stat, f_pval, n, k, residuals, fitted
    """
    n, k = X.shape
    X_design = np.column_stack([np.ones(n), X])  # добавляем константу
    # OLS решение
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    fitted = X_design @ beta
    residuals = y - fitted
    rss = np.sum(residuals ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else 0.0
    dof = n - k - 1
    r2_adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else 0.0

    # Стандартные ошибки коэффициентов
    sigma2 = rss / dof if dof > 0 else 0.0
    cov = sigma2 * np.linalg.pinv(X_design.T @ X_design)
    std_err = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stats = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err > 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof)) if dof > 0 else np.ones_like(beta)
    ci_half = stats.t.ppf(0.975, dof) * std_err if dof > 0 else np.zeros_like(beta)

    # F-статистика модели (только для предикторов, не для константы)
    if k > 0 and dof > 0:
        ms_model = (tss - rss) / k
        ms_error = rss / dof
        f_stat = ms_model / ms_error if ms_error > 0 else 0.0
        f_pval = 1 - stats.f.cdf(f_stat, k, dof)
    else:
        f_stat, f_pval = 0.0, 1.0

    return {
        'intercept': float(beta[0]),
        'coefs': beta[1:],
        'std_errors': std_err[1:],
        't_stats': t_stats[1:],
        'p_values': p_values[1:],
        'ci_lower': (beta - ci_half)[1:],
        'ci_upper': (beta + ci_half)[1:],
        'r2': float(r2),
        'r2_adj': float(r2_adj),
        'f_stat': float(f_stat),
        'f_pval': float(f_pval),
        'n': int(n),
        'k': int(k),
        'dof': int(dof),
        'residuals': residuals,
        'fitted': fitted,
        'var_names': var_names,
    }


def format_p(p):
    if p < 0.001: return "< 0.001 ***"
    if p < 0.01: return f"{p:.4f} **"
    if p < 0.05: return f"{p:.4f} *"
    return f"{p:.4f}"


def vif_scores(X):
    """Коэффициент инфляции дисперсии для диагностики мультиколлинеарности."""
    n_feat = X.shape[1]
    vif = np.zeros(n_feat)
    for i in range(n_feat):
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)
        if X_other.shape[1] == 0:
            vif[i] = 1.0
            continue
        X_design = np.column_stack([np.ones(len(y_i)), X_other])
        try:
            beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_i
            fitted = X_design @ beta
            rss = np.sum((y_i - fitted) ** 2)
            tss = np.sum((y_i - y_i.mean()) ** 2)
            r2 = 1 - rss / tss if tss > 0 else 0.0
            vif[i] = 1 / (1 - r2) if r2 < 1 else np.inf
        except Exception:
            vif[i] = np.nan
    return vif


def build_selector(label, key_state, default_target='IPL_Total'):
    """Селектор целевой переменной с удобной подстановкой стандартных ИПЛ-шкал по умолчанию."""
    target_options = [c for c in num_cols]
    default_idx = target_options.index(default_target) if default_target in target_options else 0
    return st.selectbox(label, target_options, index=default_idx, key=key_state, format_func=get_name)


def multiselect_predictors(label, exclude=None, key_state=None):
    """Мультиселект предикторов с кнопками быстрого добавления методик."""
    exclude = set(exclude or [])
    available = [c for c in num_cols if c not in exclude]

    safe_key = f"safe_{key_state}"
    if safe_key not in st.session_state:
        st.session_state[safe_key] = []
    if key_state not in st.session_state:
        st.session_state[key_state] = st.session_state[safe_key]

    def add_prefix(prefix):
        cur = st.session_state[key_state]
        new = [c for c in available if c.startswith(prefix) and c not in cur]
        st.session_state[key_state] = cur + new

    def clear_all():
        st.session_state[key_state] = []

    cols = st.columns(5)
    cols[0].button("➕ Братусь", on_click=add_prefix, args=('B_',), key=f"{key_state}_b")
    cols[1].button("➕ Мильман", on_click=add_prefix, args=('M_',), key=f"{key_state}_m")
    cols[2].button("➕ ИПЛ", on_click=add_prefix, args=('IPL_',), key=f"{key_state}_i")
    cols[3].button("➕ Демо", on_click=lambda: st.session_state.update(
        {key_state: list(set(st.session_state[key_state] + [c for c in ['Age', 'Course'] if c in available]))}
    ), key=f"{key_state}_d")
    cols[4].button("❌ Очистить", on_click=clear_all, key=f"{key_state}_clear")

    return st.multiselect(label, available, key=key_state, format_func=get_name)


# =============================================================================
# 1. СТАНДАРТНАЯ МНОЖЕСТВЕННАЯ РЕГРЕССИЯ
# =============================================================================
with tab_simple:
    st.subheader("Множественная линейная регрессия")
    st.markdown("Классическая модель: `Y = β₀ + β₁·X₁ + β₂·X₂ + ... + ε`. "
                "Показывает вклад каждого предиктора в объяснение целевой переменной.")

    target = build_selector("🎯 Целевая переменная (Y):", "simple_target")
    predictors = multiselect_predictors("📈 Предикторы (X):", exclude=[target], key_state="simple_preds")

    standardize = st.checkbox(
        "Стандартизовать предикторы (получить β, а не b)", value=True,
        help="При стандартизации коэффициенты сравнимы между собой — β=0.3 всегда больше влияет, чем β=0.1. "
             "Без стандартизации — сырые коэффициенты в единицах измерения шкал."
    )

    if len(predictors) >= 2 and st.button("🚀 Построить модель", key="simple_run"):
        data_cols = [target] + predictors
        data = df[data_cols].dropna()

        if len(data) < len(predictors) + 10:
            st.error(f"Недостаточно данных: после удаления пропусков осталось {len(data)} "
                     f"наблюдений для {len(predictors)} предикторов. Нужно хотя бы {len(predictors) + 10}.")
        else:
            X_raw = data[predictors].values.astype(float)
            y = data[target].values.astype(float)

            scaler = StandardScaler()
            if standardize:
                X = scaler.fit_transform(X_raw)
                y_std = (y - y.mean()) / y.std()
                result = fit_ols_with_stats(X, y_std, predictors)
                coef_label = "β (станд.)"
            else:
                X = X_raw
                result = fit_ols_with_stats(X, y, predictors)
                coef_label = "b (сыр.)"

            # ОСНОВНЫЕ МЕТРИКИ
            st.markdown("### 📊 Качество модели")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                color = "normal" if result['r2'] > 0.2 else "inverse"
                st.metric("R² (объяснённая дисперсия)", f"{result['r2']:.3f}",
                          f"{result['r2'] * 100:.1f}%", delta_color=color)
            with c2:
                st.metric("R² adjusted", f"{result['r2_adj']:.3f}",
                          help="R² с поправкой на число предикторов (честнее при больших моделях)")
            with c3:
                st.metric("F-статистика", f"{result['f_stat']:.2f}",
                          f"p = {format_p(result['f_pval'])}",
                          help="Значимость модели в целом (F-тест)")
            with c4:
                st.metric("N наблюдений", result['n'], f"df = {result['dof']}")

            if result['r2'] < 0.1:
                st.warning("⚠️ R² очень низкий — модель объясняет менее 10% дисперсии. "
                           "Возможно, выбранные предикторы действительно слабо связаны с Y.")
            elif result['r2'] > 0.8:
                st.warning("⚠️ R² подозрительно высокий. Проверьте, нет ли среди предикторов "
                           "переменных, которые математически входят в целевую (например, компоненты общего балла).")

            # ТАБЛИЦА КОЭФФИЦИЕНТОВ
            st.markdown("### 📋 Коэффициенты предикторов")
            coef_df = pd.DataFrame({
                'Предиктор': [get_name(p) for p in predictors],
                coef_label: np.round(result['coefs'], 3),
                'Станд. ошибка': np.round(result['std_errors'], 3),
                't': np.round(result['t_stats'], 2),
                'p-value': [format_p(p) for p in result['p_values']],
                '95% CI нижняя': np.round(result['ci_lower'], 3),
                '95% CI верхняя': np.round(result['ci_upper'], 3),
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # ВИЗУАЛИЗАЦИЯ КОЭФФИЦИЕНТОВ
            st.markdown("### 🎯 Визуализация вклада предикторов")
            viz_df = pd.DataFrame({
                'Предиктор': [get_name(p) for p in predictors],
                coef_label: result['coefs'],
                'CI_low': result['ci_lower'],
                'CI_high': result['ci_upper'],
                'Значимость': ['Значим (p<0.05)' if p < 0.05 else 'Не значим'
                                for p in result['p_values']]
            }).sort_values(coef_label, key=abs, ascending=True)

            fig = go.Figure()
            # Доверительные интервалы
            for i, row in viz_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['CI_low'], row['CI_high']],
                    y=[row['Предиктор'], row['Предиктор']],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False, hoverinfo='skip'
                ))
            # Точечные оценки
            colors_sig = ['#27ae60' if s == 'Значим (p<0.05)' else '#95a5a6'
                          for s in viz_df['Значимость']]
            fig.add_trace(go.Scatter(
                x=viz_df[coef_label],
                y=viz_df['Предиктор'],
                mode='markers',
                marker=dict(size=14, color=colors_sig,
                            line=dict(color='black', width=1)),
                showlegend=False,
                text=[f"β={v:.3f}" for v in viz_df[coef_label]],
                hovertemplate='%{y}<br>Коэффициент: %{x:.3f}<extra></extra>'
            ))
            fig.add_vline(x=0, line_dash='dash', line_color='red')
            fig.update_layout(
                xaxis_title=f"{coef_label} (с 95% CI)",
                height=max(350, 40 * len(predictors)),
                margin=dict(l=10, r=10, t=10, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Зелёный = значимый предиктор (p < 0.05), 🔘 Серый = не значим. "
                       "Горизонтальная линия — 95% доверительный интервал.")

            # ДИАГНОСТИКА: МУЛЬТИКОЛЛИНЕАРНОСТЬ
            with st.expander("🔬 Диагностика: мультиколлинеарность (VIF)"):
                st.markdown("**VIF** (Variance Inflation Factor) показывает, насколько каждый предиктор "
                            "коррелирует с остальными. Высокий VIF делает коэффициенты нестабильными.")
                vifs = vif_scores(X)
                vif_df = pd.DataFrame({
                    'Предиктор': [get_name(p) for p in predictors],
                    'VIF': np.round(vifs, 2),
                    'Статус': ['🟢 ОК' if v < 5 else ('🟡 Умеренно' if v < 10 else '🔴 Проблема')
                                for v in vifs]
                })
                st.dataframe(vif_df, use_container_width=True, hide_index=True)
                st.caption("VIF < 5 — нормально. 5–10 — умеренная мультиколлинеарность. "
                           ">10 — предикторы дублируют друг друга, коэффициенты ненадёжны.")

            # ДИАГНОСТИКА: ОСТАТКИ
            with st.expander("🔬 Диагностика: остатки модели"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    fig_fit = px.scatter(x=result['fitted'], y=result['residuals'],
                                          labels={'x': 'Предсказанные значения', 'y': 'Остатки'},
                                          title='Остатки vs предсказания')
                    fig_fit.add_hline(y=0, line_dash='dash', line_color='red')
                    st.plotly_chart(fig_fit, use_container_width=True)
                    st.caption("Идеально: облако точек без паттерна вокруг красной линии.")
                with col_d2:
                    # Q-Q plot
                    std_resid = result['residuals'] / result['residuals'].std()
                    theoretical = stats.norm.ppf((np.arange(1, len(std_resid) + 1) - 0.5) / len(std_resid))
                    observed = np.sort(std_resid)
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(x=theoretical, y=observed, mode='markers', name='Остатки'))
                    fig_qq.add_trace(go.Scatter(x=theoretical, y=theoretical, mode='lines',
                                                 name='Норма', line=dict(dash='dash', color='red')))
                    fig_qq.update_layout(title='Q-Q plot остатков (проверка нормальности)',
                                          xaxis_title='Теоретические квантили',
                                          yaxis_title='Наблюдаемые квантили')
                    st.plotly_chart(fig_qq, use_container_width=True)
                    # Тест Шапиро-Уилка на остатки
                    if 3 < len(result['residuals']) < 5000:
                        sw_stat, sw_p = stats.shapiro(result['residuals'])
                        if sw_p < 0.05:
                            st.warning(f"Шапиро-Уилк: W={sw_stat:.3f}, p={sw_p:.4f} — остатки "
                                       f"отличаются от нормального распределения. "
                                       f"Стандартные ошибки могут быть неточными.")
                        else:
                            st.success(f"Шапиро-Уилк: W={sw_stat:.3f}, p={sw_p:.4f} — остатки близки к норме. ✓")

            # СКАЧИВАНИЕ
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                summary = pd.DataFrame({
                    'Метрика': ['R²', 'R² adjusted', 'F', 'F p-value', 'N', 'df'],
                    'Значение': [result['r2'], result['r2_adj'], result['f_stat'],
                                 result['f_pval'], result['n'], result['dof']]
                })
                summary.to_excel(writer, sheet_name='Модель', index=False)
                coef_df.to_excel(writer, sheet_name='Коэффициенты', index=False)
            st.download_button("📥 Скачать результаты (Excel)", buf.getvalue(),
                                f"regression_{target}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =============================================================================
# 2. ИЕРАРХИЧЕСКАЯ РЕГРЕССИЯ
# =============================================================================
with tab_hier:
    st.subheader("Иерархическая регрессия: блок за блоком")
    st.markdown("Стандарт психологических исследований. Добавляешь предикторы по блокам "
                "(например: Блок 1 = демография, Блок 2 = смыслы, Блок 3 = мотивация). "
                "Смотришь, сколько **дополнительной** дисперсии объясняет каждый блок.")

    target_h = build_selector("🎯 Целевая переменная (Y):", "hier_target")

    n_blocks = st.slider("Количество блоков:", 2, 5, 3, key="hier_nblocks")

    # Подготовка session_state для каждого блока
    for i in range(1, 6):
        key = f"hier_block_{i}"
        safe_key = f"safe_{key}"
        if safe_key not in st.session_state: st.session_state[safe_key] = []
        if key not in st.session_state: st.session_state[key] = st.session_state[safe_key]

    blocks = []
    block_names = []
    for i in range(1, n_blocks + 1):
        with st.expander(f"📦 Блок {i}", expanded=(i == 1)):
            name = st.text_input(f"Название блока {i}:", value=f"Блок {i}", key=f"hier_name_{i}")
            block_names.append(name)

            key = f"hier_block_{i}"
            # ВАЖНО: предлагаем ВСЕ возможные переменные (без исключения по другим блокам).
            # Проверка пересечений делается при запуске — так Streamlit не сбрасывает выбор.
            available = [c for c in num_cols if c != target_h]

            def make_add(prefix, block_idx=i):
                def _add():
                    key_name = f"hier_block_{block_idx}"
                    cur = list(st.session_state.get(key_name, []))
                    # Собираем кандидатов: подходят по префиксу и не в этом блоке
                    candidates = [c for c in num_cols
                                  if c != target_h
                                  and c.startswith(prefix)
                                  and c not in cur]
                    # Исключаем те, что уже в ДРУГИХ блоках
                    for j in range(1, n_blocks + 1):
                        if j != block_idx:
                            candidates = [c for c in candidates
                                          if c not in st.session_state.get(f"hier_block_{j}", [])]
                    st.session_state[key_name] = cur + candidates
                return _add

            def make_clear(block_idx=i):
                def _clear():
                    st.session_state[f"hier_block_{block_idx}"] = []
                return _clear

            btns = st.columns(4)
            btns[0].button("➕ Братусь", on_click=make_add('B_'), key=f"hb_{i}_b")
            btns[1].button("➕ Мильман", on_click=make_add('M_'), key=f"hb_{i}_m")
            btns[2].button("➕ ИПЛ", on_click=make_add('IPL_'), key=f"hb_{i}_i")
            btns[3].button("❌ Очистить", on_click=make_clear(), key=f"hb_{i}_c")

            selected = st.multiselect(f"Переменные блока {i}:", available,
                                       key=key, format_func=get_name)
            blocks.append(selected)

    # Проверка на дубликаты между блоками
    all_selected = [v for b in blocks for v in b]
    duplicates = set([v for v in all_selected if all_selected.count(v) > 1])
    if duplicates:
        st.error(f"⚠️ Одна и та же переменная выбрана в нескольких блоках: "
                 f"{', '.join(get_name(d) for d in duplicates)}. "
                 f"Уберите дубликаты — переменная должна быть только в одном блоке.")

    can_run = all(len(b) > 0 for b in blocks) and not duplicates
    if can_run and st.button("🚀 Построить иерархическую модель", key="hier_run"):
        all_vars = [target_h] + [v for b in blocks for v in b]
        data = df[all_vars].dropna()

        if len(data) < sum(len(b) for b in blocks) + 10:
            st.error(f"Недостаточно данных: {len(data)} наблюдений.")
        else:
            y = data[target_h].values.astype(float)
            # Строим модели шаг за шагом
            step_results = []
            cumulative_vars = []
            prev_r2 = 0.0
            prev_k = 0

            for step, block_vars in enumerate(blocks, 1):
                cumulative_vars.extend(block_vars)
                X_cum = data[cumulative_vars].values.astype(float)
                X_std = StandardScaler().fit_transform(X_cum)
                y_std = (y - y.mean()) / y.std()
                res = fit_ols_with_stats(X_std, y_std, cumulative_vars)

                delta_r2 = res['r2'] - prev_r2
                # F-change test (значимость прироста R²)
                added_k = len(block_vars)
                new_k = res['k']
                n_obs = res['n']
                dof_denom = n_obs - new_k - 1
                if delta_r2 > 0 and dof_denom > 0 and (1 - res['r2']) > 0:
                    f_change = (delta_r2 / added_k) / ((1 - res['r2']) / dof_denom)
                    p_change = 1 - stats.f.cdf(f_change, added_k, dof_denom)
                else:
                    f_change, p_change = 0.0, 1.0

                step_results.append({
                    'Шаг': step,
                    'Блок': block_names[step - 1],
                    'Добавлено переменных': added_k,
                    'Всего в модели': new_k,
                    'R²': round(res['r2'], 3),
                    'R² adjusted': round(res['r2_adj'], 3),
                    'ΔR²': round(delta_r2, 3),
                    'F-change': round(f_change, 2),
                    'p (F-change)': format_p(p_change),
                    '_last_result': res,
                })
                prev_r2 = res['r2']
                prev_k = new_k

            st.markdown("### 📊 Пошаговые результаты")
            display_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                                         for r in step_results])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Визуализация прироста R²
            st.markdown("### 📈 Прирост объяснённой дисперсии")
            fig_hier = go.Figure()
            cumul = 0
            for r in step_results:
                fig_hier.add_trace(go.Bar(
                    name=f"{r['Блок']} (+{r['ΔR²'] * 100:.1f}%)",
                    x=[r['Блок']],
                    y=[r['ΔR²'] * 100],
                    text=f"ΔR² = {r['ΔR²']:.3f}<br>{r['p (F-change)']}",
                    textposition='auto',
                    hovertemplate=f"<b>{r['Блок']}</b><br>ΔR² = {r['ΔR²']:.3f}<br>"
                                   f"p(F-change) = {r['p (F-change)']}<extra></extra>"
                ))
            fig_hier.update_layout(
                barmode='stack',
                xaxis_title="Блок предикторов",
                yaxis_title="Прирост R² (%)",
                title="Сколько дисперсии добавляет каждый блок",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_hier, use_container_width=True)

            # Вывод-интерпретация
            final = step_results[-1]
            st.markdown(f"### 💡 Интерпретация")
            st.markdown(f"Итоговая модель (все {len(blocks)} блоков) объясняет "
                        f"**{final['R²'] * 100:.1f}%** дисперсии {get_name(target_h)}.")
            for r in step_results:
                signif = "✓ значимо" if "***" in r['p (F-change)'] or "**" in r['p (F-change)'] \
                         or "*" in r['p (F-change)'] else "✗ не значимо"
                st.markdown(f"- **{r['Блок']}** добавляет {r['ΔR²'] * 100:.1f}% дисперсии "
                            f"сверх предыдущих блоков ({signif}).")

            # Коэффициенты финальной модели
            st.markdown("### 📋 Коэффициенты финальной модели")
            final_res = step_results[-1]['_last_result']
            final_coefs = pd.DataFrame({
                'Предиктор': [get_name(v) for v in final_res['var_names']],
                'β': np.round(final_res['coefs'], 3),
                't': np.round(final_res['t_stats'], 2),
                'p-value': [format_p(p) for p in final_res['p_values']],
            })
            st.dataframe(final_coefs, use_container_width=True, hide_index=True)


# =============================================================================
# 3. ЧАСТНЫЕ КОРРЕЛЯЦИИ
# =============================================================================
with tab_partial:
    st.subheader("Частные корреляции: связь X↔Y при контроле других переменных")
    st.markdown("Отвечает на вопрос: «Какая остаётся связь между X и Y, если учесть влияние Z?». "
                "Помогает увидеть уникальный вклад шкалы, не объяснимый через другие.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        x_var = st.selectbox("Переменная X:", num_cols, key="partial_x",
                              format_func=get_name,
                              index=num_cols.index('B_Self-realization') if 'B_Self-realization' in num_cols else 0)
    with col_p2:
        y_var = st.selectbox("Переменная Y:", [c for c in num_cols if c != x_var],
                              key="partial_y", format_func=get_name,
                              index=0 if 'IPL_Total' not in num_cols else
                              [c for c in num_cols if c != x_var].index('IPL_Total') if 'IPL_Total' in num_cols else 0)

    control_vars = multiselect_predictors(
        "🎛 Контрольные переменные Z (исключаем их влияние):",
        exclude=[x_var, y_var], key_state="partial_controls"
    )

    method = st.radio("Метод корреляции:", ["pearson", "spearman"], horizontal=True)

    if x_var and y_var and st.button("🔍 Рассчитать", key="partial_run"):
        data_cols = [x_var, y_var] + control_vars
        data = df[data_cols].dropna()

        if len(data) < len(control_vars) + 10:
            st.error("Недостаточно данных после удаления пропусков.")
        else:
            # Простая корреляция (без контроля)
            if method == "pearson":
                r_simple, p_simple = stats.pearsonr(data[x_var], data[y_var])
            else:
                r_simple, p_simple = stats.spearmanr(data[x_var], data[y_var])

            # Частная корреляция (при наличии контрольных)
            st.markdown("### 📊 Сравнение: простая vs частная корреляция")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(f"r простая ({method})", f"{r_simple:.3f}",
                          f"p = {format_p(p_simple)}",
                          help="Обычная корреляция без учёта контрольных переменных")

            if len(control_vars) > 0:
                try:
                    pc = pg.partial_corr(data=data, x=x_var, y=y_var,
                                          covar=control_vars, method=method)
                    r_part = float(pc['r'].values[0])
                    p_part = float(pc['p-val'].values[0])
                    ci = pc['CI95%'].values[0]
                except Exception as e:
                    st.error(f"Ошибка расчёта: {e}")
                    r_part, p_part = None, None

                if r_part is not None:
                    with c2:
                        delta = r_part - r_simple
                        st.metric(f"r частная ({method})", f"{r_part:.3f}",
                                  f"p = {format_p(p_part)}",
                                  help=f"Связь X↔Y при контроле: {', '.join(get_name(c) for c in control_vars)}")
                    with c3:
                        if abs(r_simple) > 0.01:
                            reduction = (abs(r_simple) - abs(r_part)) / abs(r_simple) * 100
                            st.metric("Падение силы связи", f"{reduction:+.1f}%",
                                      help="Сколько связи было обусловлено контрольными переменными")

                    st.markdown("### 💡 Интерпретация")
                    if abs(r_part) < abs(r_simple) - 0.05:
                        st.info(f"**Падение связи при контроле.** Изначальная связь {get_name(x_var)} × {get_name(y_var)} "
                                f"частично объясняется контрольными переменными: "
                                f"r снизилось с {r_simple:.3f} до {r_part:.3f}. "
                                f"Это значит, что {get_name(x_var)} **не является независимым предиктором** "
                                f"{get_name(y_var)} — часть её 'эффекта' на самом деле идёт через контролируемые переменные.")
                    elif abs(r_part) > abs(r_simple) + 0.05:
                        st.success(f"**Усиление связи (супрессия).** После контроля связь стала сильнее: "
                                   f"с {r_simple:.3f} до {r_part:.3f}. Это редкий, но важный паттерн — "
                                   f"контрольные переменные маскировали истинную связь.")
                    else:
                        st.success(f"**Связь устойчива.** Сила связи почти не изменилась: "
                                   f"r = {r_simple:.3f} → {r_part:.3f}. "
                                   f"Это **уникальный вклад** {get_name(x_var)} в объяснение {get_name(y_var)}, "
                                   f"не сводимый к другим переменным.")
            else:
                with c2:
                    st.info("Добавьте контрольные переменные, чтобы увидеть частную корреляцию.")


# =============================================================================
# 4. РЕГУЛЯРИЗОВАННАЯ РЕГРЕССИЯ (Ridge/Lasso)
# =============================================================================
with tab_reg:
    st.subheader("Ridge и Lasso регрессия")
    st.markdown("Подходит, когда **предикторов много и они коррелируют между собой**. "
                "Обычная регрессия в этих условиях даёт нестабильные коэффициенты. "
                "Ridge уменьшает их пропорционально, Lasso **обнуляет** самые слабые — делает автоматический отбор.")

    target_r = build_selector("🎯 Целевая переменная (Y):", "reg_target")
    predictors_r = multiselect_predictors("📈 Предикторы (X):", exclude=[target_r], key_state="reg_preds")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        method_r = st.radio("Метод регуляризации:", ["Ridge", "Lasso"], horizontal=True)
    with col_r2:
        alpha_r = st.select_slider(
            "Сила регуляризации (α):",
            options=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], value=1.0,
            help="0 = обычная регрессия. Чем выше α, тем сильнее штраф за большие коэффициенты."
        )

    if len(predictors_r) >= 2 and st.button("🚀 Построить регуляризованную модель", key="reg_run"):
        data_cols = [target_r] + predictors_r
        data = df[data_cols].dropna()

        if len(data) < len(predictors_r) + 10:
            st.error("Недостаточно данных.")
        else:
            X_raw = data[predictors_r].values.astype(float)
            y = data[target_r].values.astype(float)
            X = StandardScaler().fit_transform(X_raw)

            # Сравним OLS, Ridge и Lasso при одном и том же α
            ols = LinearRegression().fit(X, y)
            ridge = Ridge(alpha=alpha_r).fit(X, y)
            lasso = Lasso(alpha=alpha_r, max_iter=5000).fit(X, y)

            active_model = ridge if method_r == "Ridge" else lasso
            r2_ols = r2_score(y, ols.predict(X))
            r2_active = r2_score(y, active_model.predict(X))

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("R² (OLS без регул.)", f"{r2_ols:.3f}")
            with c2:
                st.metric(f"R² ({method_r}, α={alpha_r})", f"{r2_active:.3f}",
                          f"{(r2_active - r2_ols) * 100:+.1f}%")
            with c3:
                if method_r == "Lasso":
                    n_zeroed = int(np.sum(np.abs(lasso.coef_) < 1e-6))
                    st.metric("Обнулено предикторов", f"{n_zeroed} из {len(predictors_r)}",
                              help="Lasso автоматически обнуляет малозначимые переменные")
                else:
                    max_shrink = np.max(np.abs(ols.coef_ - ridge.coef_))
                    st.metric("Макс. усадка коэфф.", f"{max_shrink:.3f}",
                              help="Насколько Ridge уменьшил самый большой коэффициент")

            # Сравнительный график коэффициентов
            coef_compare = pd.DataFrame({
                'Предиктор': [get_name(p) for p in predictors_r],
                'OLS': ols.coef_,
                method_r: active_model.coef_,
            })
            coef_compare_melt = coef_compare.melt(id_vars='Предиктор', var_name='Метод', value_name='β')

            fig_compare = px.bar(coef_compare_melt, x='β', y='Предиктор', color='Метод',
                                  orientation='h', barmode='group',
                                  title=f"Сравнение коэффициентов: OLS vs {method_r} (α={alpha_r})")
            fig_compare.add_vline(x=0, line_dash='dash', line_color='gray')
            fig_compare.update_layout(height=max(400, 35 * len(predictors_r)))
            st.plotly_chart(fig_compare, use_container_width=True)

            # Таблица коэффициентов
            st.markdown("### 📋 Коэффициенты")
            coef_table = coef_compare.copy()
            coef_table['Разница'] = (coef_table[method_r] - coef_table['OLS']).round(3)
            if method_r == "Lasso":
                coef_table['Статус'] = coef_table[method_r].apply(
                    lambda v: '🔴 Обнулён' if abs(v) < 1e-6 else '🟢 Оставлен'
                )
            coef_table = coef_table.sort_values(method_r, key=abs, ascending=False)
            coef_table[['OLS', method_r]] = coef_table[['OLS', method_r]].round(3)
            st.dataframe(coef_table, use_container_width=True, hide_index=True)

            # Путь регуляризации
            with st.expander("🔬 Путь регуляризации: как коэффициенты меняются при разных α"):
                alphas = np.logspace(-3, 2, 50)
                coefs_path = []
                for a in alphas:
                    m = Ridge(alpha=a) if method_r == "Ridge" else Lasso(alpha=a, max_iter=5000)
                    m.fit(X, y)
                    coefs_path.append(m.coef_)
                coefs_path = np.array(coefs_path)

                fig_path = go.Figure()
                for i, name in enumerate(predictors_r):
                    fig_path.add_trace(go.Scatter(
                        x=alphas, y=coefs_path[:, i],
                        mode='lines', name=get_name(name)
                    ))
                fig_path.add_vline(x=alpha_r, line_dash='dash', line_color='red',
                                    annotation_text=f"Текущее α = {alpha_r}")
                fig_path.update_layout(
                    xaxis_type='log',
                    xaxis_title='α (сила регуляризации, лог-шкала)',
                    yaxis_title='Коэффициент β',
                    title=f'Как коэффициенты {method_r} меняются с ростом α',
                    height=500
                )
                st.plotly_chart(fig_path, use_container_width=True)
                st.caption(f"С ростом α коэффициенты приближаются к 0. "
                           f"{'Lasso обнуляет слабые предикторы первыми — это отбор переменных.' if method_r == 'Lasso' else 'Ridge уменьшает пропорционально, никогда не обнуляет полностью.'}")
