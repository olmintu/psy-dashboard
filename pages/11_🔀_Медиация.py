import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
import io
from utils import render_sidebar, get_name, DERIVED_CATEGORICAL_COLS

st.set_page_config(page_title="Медиационный анализ", layout="wide", page_icon="🔀")

df = render_sidebar()
if df is None:
    st.stop()

st.header("🔀 Медиационный анализ")
st.caption("Проверка гипотезы 'X влияет на Y через M' — один из самых сильных инструментов "
           "причинно-следственного анализа в психологии.")

# Числовые колонки
num_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c not in DERIVED_CATEGORICAL_COLS]


with st.expander("ℹ️ Что такое медиация и когда её применять", expanded=False):
    st.markdown("""
### Модель медиации

В простой модели `X → Y` вы проверяете прямую связь. Но часто за прямой связью скрывается промежуточное звено — **медиатор M**. Модель становится:

```
X ─[a]─→ M ─[b]─→ Y
 └─────[c']──────→ Y
```

- **c** — общий эффект (X → Y, без учёта M)
- **a** — эффект X на M
- **b** — эффект M на Y (при контроле X)
- **c'** — прямой эффект X на Y (при контроле M)
- **a·b** — косвенный эффект (через медиатор)
- **c = c' + a·b** — общий эффект раскладывается на два компонента

### Пять возможных исходов

| Паттерн | a·b | c' | Интерпретация |
|---|---|---|---|
| **Полная медиация** | значим | ≈ 0 | X влияет на Y ТОЛЬКО через M |
| **Частичная медиация** | значим | значим | X влияет и напрямую, и через M |
| **Прямой эффект без медиации** | ≈ 0 | значим | Медиатор не нужен |
| **Нет эффекта** | ≈ 0 | ≈ 0 | Связи не существует |
| **Супрессия** (редкая) | значим | значим, противоположный знак | Медиатор маскировал связь |

### Когда применять

- Есть теоретическая гипотеза, почему X должен влиять на Y именно через M
- Корреляционный/регрессионный анализ показал связи X↔M и M↔Y
- Хотите понять **механизм**, а не только наличие связи
""")


# =============================================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# =============================================================================

def ols_simple(X, y):
    """Обычная OLS, возвращает beta, se, t, p, r2."""
    n = len(y)
    k = X.shape[1] if X.ndim > 1 else 1
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_design = np.column_stack([np.ones(n), X])
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    fitted = X_design @ beta
    resid = y - fitted
    rss = np.sum(resid ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else 0.0
    dof = n - k - 1
    if dof <= 0:
        return {'beta': beta, 'se': np.zeros_like(beta), 't': np.zeros_like(beta),
                'p': np.ones_like(beta), 'r2': r2}
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X_design.T @ X_design)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return {'beta': beta, 'se': se, 't': t, 'p': p, 'r2': r2, 'resid': resid, 'fitted': fitted}


def mediation_single(x, m, y, n_bootstrap=5000, seed=42, standardize=True):
    """
    Классический медиационный анализ (Baron & Kenny + бутстрап Хайеса).

    Входы — массивы numpy. Возвращает dict с путями, CI и интерпретацией.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)

    if standardize:
        x = (x - x.mean()) / x.std()
        m = (m - m.mean()) / m.std()
        y = (y - y.mean()) / y.std()

    n = len(x)

    # Путь c (общий эффект): Y = c·X
    res_c = ols_simple(x, y)
    c = res_c['beta'][1]; c_se = res_c['se'][1]; c_p = res_c['p'][1]

    # Путь a: M = a·X
    res_a = ols_simple(x, m)
    a = res_a['beta'][1]; a_se = res_a['se'][1]; a_p = res_a['p'][1]

    # Пути b и c': Y = c'·X + b·M
    X_both = np.column_stack([x, m])
    res_cp = ols_simple(X_both, y)
    cp = res_cp['beta'][1]; cp_se = res_cp['se'][1]; cp_p = res_cp['p'][1]
    b  = res_cp['beta'][2]; b_se  = res_cp['se'][2]; b_p  = res_cp['p'][2]

    indirect = a * b

    # Тест Собеля (классический, приблизительный)
    sobel_se = np.sqrt((a**2) * (b_se**2) + (b**2) * (a_se**2))
    sobel_z  = indirect / sobel_se if sobel_se > 0 else 0.0
    sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # Бутстрап-CI для косвенного эффекта a·b
    rng = np.random.default_rng(seed)
    boot_ab = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        xb, mb, yb = x[idx], m[idx], y[idx]
        try:
            a_b = ols_simple(xb, mb)['beta'][1]
            b_b = ols_simple(np.column_stack([xb, mb]), yb)['beta'][2]
            boot_ab[i] = a_b * b_b
        except Exception:
            boot_ab[i] = np.nan
    boot_ab = boot_ab[~np.isnan(boot_ab)]
    ci_lower = np.percentile(boot_ab, 2.5)
    ci_upper = np.percentile(boot_ab, 97.5)
    boot_significant = (ci_lower > 0) or (ci_upper < 0)  # 95% CI не включает 0

    # Доля медиации
    prop_mediated = indirect / c if abs(c) > 1e-9 else np.nan

    # Интерпретация
    c_sig  = c_p  < 0.05
    a_sig  = a_p  < 0.05
    b_sig  = b_p  < 0.05
    cp_sig = cp_p < 0.05
    ab_sig = boot_significant

    if ab_sig and not cp_sig:
        interp_type = "Полная медиация"
        interp_text = (f"Эффект X на Y **полностью** опосредован медиатором. "
                       f"Прямой путь c' = {cp:.3f} не значим (p = {cp_p:.3f}), "
                       f"а косвенный a·b = {indirect:.3f} значим (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]).")
    elif ab_sig and cp_sig:
        same_sign = np.sign(indirect) == np.sign(cp)
        if same_sign:
            interp_type = "Частичная медиация"
            interp_text = (f"Эффект X на Y идёт **и напрямую, и через медиатор**. "
                           f"Прямой эффект c' = {cp:.3f} значим, косвенный a·b = {indirect:.3f} также значим. "
                           f"Через медиатор проходит {abs(prop_mediated)*100:.1f}% общего эффекта.")
        else:
            interp_type = "Супрессия (противоположные знаки)"
            interp_text = (f"**Редкий, но важный паттерн.** Прямой и косвенный эффекты имеют "
                           f"противоположные знаки: c' = {cp:.3f}, a·b = {indirect:.3f}. "
                           f"Медиатор частично **маскирует** истинную связь. Общий эффект c = {c:.3f} "
                           f"на самом деле слабее, чем прямой.")
    elif not ab_sig and cp_sig:
        interp_type = "Нет медиации (только прямой эффект)"
        interp_text = (f"Прямой эффект c' = {cp:.3f} значим, но косвенный a·b = {indirect:.3f} "
                       f"не достиг значимости (CI включает 0). Медиатор не объясняет связь X → Y.")
    elif not ab_sig and not cp_sig and not c_sig:
        interp_type = "Нет связи"
        interp_text = (f"Ни общий эффект c = {c:.3f}, ни косвенный a·b = {indirect:.3f} не значимы. "
                       f"Выбранная тройка переменных не демонстрирует связи.")
    else:
        interp_type = "Неоднозначный паттерн"
        interp_text = "Картина не укладывается в классические сценарии. Проверьте данные и теоретическое обоснование."

    return {
        'n': n,
        'c': c, 'c_se': c_se, 'c_p': c_p,
        'a': a, 'a_se': a_se, 'a_p': a_p,
        'b': b, 'b_se': b_se, 'b_p': b_p,
        'cp': cp, 'cp_se': cp_se, 'cp_p': cp_p,
        'indirect': indirect,
        'sobel_z': sobel_z, 'sobel_p': sobel_p,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'boot_significant': bool(boot_significant),
        'prop_mediated': prop_mediated,
        'boot_distribution': boot_ab,
        'interp_type': interp_type,
        'interp_text': interp_text,
        'a_sig': a_sig, 'b_sig': b_sig, 'c_sig': c_sig,
        'cp_sig': cp_sig, 'ab_sig': ab_sig,
        'r2_y_x': res_c['r2'],
        'r2_m_x': res_a['r2'],
        'r2_y_xm': res_cp['r2'],
    }


def mediation_multiple(x, M_arr, y, mediator_names, n_bootstrap=5000, seed=42, standardize=True):
    """
    Множественная медиация: один X, несколько медиаторов, один Y.
    Каждый медиатор получает свой a·b с отдельным CI.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    M_arr = np.asarray(M_arr, dtype=float)

    if standardize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        M_arr = (M_arr - M_arr.mean(axis=0)) / M_arr.std(axis=0)

    n = len(x)
    k = M_arr.shape[1]

    # Путь c (общий)
    res_c = ols_simple(x, y)
    c = res_c['beta'][1]; c_p = res_c['p'][1]

    # Пути a: отдельные регрессии M_i ~ X
    a_coefs = np.zeros(k)
    a_ses   = np.zeros(k)
    a_ps    = np.zeros(k)
    for i in range(k):
        r = ols_simple(x, M_arr[:, i])
        a_coefs[i] = r['beta'][1]; a_ses[i] = r['se'][1]; a_ps[i] = r['p'][1]

    # Y ~ X + M1 + M2 + ... : даёт b_i и c'
    X_full = np.column_stack([x, M_arr])
    res_full = ols_simple(X_full, y)
    cp = res_full['beta'][1]; cp_p = res_full['p'][1]
    b_coefs = res_full['beta'][2:2+k]
    b_ses   = res_full['se'][2:2+k]
    b_ps    = res_full['p'][2:2+k]

    indirects = a_coefs * b_coefs
    total_indirect = indirects.sum()

    # Бутстрап для каждого a_i*b_i
    rng = np.random.default_rng(seed)
    boot_indirects = np.zeros((n_bootstrap, k))
    boot_total     = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        xb, Mb, yb = x[idx], M_arr[idx], y[idx]
        try:
            a_b = np.array([ols_simple(xb, Mb[:, j])['beta'][1] for j in range(k)])
            r_full_b = ols_simple(np.column_stack([xb, Mb]), yb)
            b_b = r_full_b['beta'][2:2+k]
            boot_indirects[i] = a_b * b_b
            boot_total[i] = (a_b * b_b).sum()
        except Exception:
            boot_indirects[i] = np.nan
            boot_total[i] = np.nan

    boot_indirects = boot_indirects[~np.isnan(boot_indirects).any(axis=1)]
    boot_total     = boot_total[~np.isnan(boot_total)]

    rows = []
    for i, name in enumerate(mediator_names):
        ci_lo = np.percentile(boot_indirects[:, i], 2.5)
        ci_hi = np.percentile(boot_indirects[:, i], 97.5)
        sig = (ci_lo > 0) or (ci_hi < 0)
        rows.append({
            'Медиатор': get_name(name),
            'Колонка': name,
            'a': a_coefs[i], 'a_p': a_ps[i],
            'b': b_coefs[i], 'b_p': b_ps[i],
            'a·b': indirects[i],
            'CI_low': ci_lo, 'CI_high': ci_hi,
            'Значим (CI)': '✓' if sig else '—',
        })

    ci_tot_lo = np.percentile(boot_total, 2.5)
    ci_tot_hi = np.percentile(boot_total, 97.5)

    return {
        'n': n,
        'c': c, 'c_p': c_p,
        'cp': cp, 'cp_p': cp_p,
        'total_indirect': total_indirect,
        'total_ci': (ci_tot_lo, ci_tot_hi),
        'mediators': pd.DataFrame(rows),
        'r2_y_x':   res_c['r2'],
        'r2_y_xm':  res_full['r2'],
    }


def check_assumptions(x, m, y, labels=('X', 'M', 'Y')):
    """
    Проверки предпосылок, нужные для выбора между бутстрапом и тестом Собеля
    и для понимания робустности модели.
    """
    diag = {}
    # Нормальность каждой переменной (Шапиро-Уилк)
    for data, name in zip([x, m, y], labels):
        if 3 < len(data) < 5000:
            w, p = stats.shapiro(data)
            diag[f'shapiro_{name}'] = (float(w), float(p))
        else:
            diag[f'shapiro_{name}'] = (None, None)

    # Линейность (корреляция vs ранговая)
    pearson_xm = stats.pearsonr(x, m)
    spearman_xm = stats.spearmanr(x, m)
    pearson_my = stats.pearsonr(m, y)
    spearman_my = stats.spearmanr(m, y)
    pearson_xy = stats.pearsonr(x, y)
    spearman_xy = stats.spearmanr(x, y)
    diag['pearson_xm'] = (float(pearson_xm[0]), float(pearson_xm[1]))
    diag['spearman_xm'] = (float(spearman_xm[0]), float(spearman_xm[1]))
    diag['pearson_my'] = (float(pearson_my[0]), float(pearson_my[1]))
    diag['spearman_my'] = (float(spearman_my[0]), float(spearman_my[1]))
    diag['pearson_xy'] = (float(pearson_xy[0]), float(pearson_xy[1]))
    diag['spearman_xy'] = (float(spearman_xy[0]), float(spearman_xy[1]))

    # Гомоскедастичность (тест Бреуша-Пагана — упрощённый): корреляция |resid| с fitted
    # Для Y ~ X + M
    X_full = np.column_stack([x, m])
    res = ols_simple(X_full, y)
    abs_resid = np.abs(res['resid'])
    bp_r, bp_p = stats.spearmanr(abs_resid, res['fitted'])
    diag['heteroscedasticity'] = (float(bp_r), float(bp_p))

    return diag


def format_p(p):
    if p < 0.001: return "< 0.001 ***"
    if p < 0.01:  return f"{p:.4f} **"
    if p < 0.05:  return f"{p:.4f} *"
    return f"{p:.4f}"


def format_sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "н.з."


def draw_mediation_diagram(result, x_name, m_name, y_name):
    """Классическая диаграмма путей X → M → Y с подписями коэффициентов."""
    a_sig = result['a_sig']; b_sig = result['b_sig']
    cp_sig = result['cp_sig']; ab_sig = result['ab_sig']

    def color(sig): return '#27ae60' if sig else '#bdc3c7'
    def width(sig): return 4 if sig else 2

    fig = go.Figure()
    # Узлы
    boxes = {
        'X': (0.05, 0.5, x_name, '#3498db'),
        'M': (0.5, 0.9, m_name, '#9b59b6'),
        'Y': (0.95, 0.5, y_name, '#e67e22'),
    }
    for key, (xp, yp, name, col) in boxes.items():
        fig.add_shape(type='rect',
                      x0=xp-0.09, x1=xp+0.09, y0=yp-0.08, y1=yp+0.08,
                      fillcolor=col, opacity=0.2, line=dict(color=col, width=2))
        fig.add_annotation(x=xp, y=yp, text=f"<b>{key}</b><br>{name}",
                           showarrow=False, font=dict(size=12, color=col), align='center')

    # Стрелки с коэффициентами
    def arrow(x0, y0, x1, y1, label, sig, curvature=0, shift_y=0):
        # Саму стрелку делаем через annotation
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                           xref='x', yref='y', axref='x', ayref='y',
                           showarrow=True, arrowhead=3, arrowsize=1.5,
                           arrowwidth=width(sig), arrowcolor=color(sig))
        # Подпись посередине
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + shift_y
        fig.add_annotation(x=mx, y=my, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(size=12, color=color(sig)),
                           bgcolor='white', bordercolor=color(sig), borderwidth=1, borderpad=4)

    # a: X → M
    arrow(0.14, 0.5 + 0.05, 0.41, 0.9 - 0.05,
          f"a = {result['a']:.3f} {format_sig(result['a_p'])}", a_sig, shift_y=0.04)
    # b: M → Y
    arrow(0.59, 0.9 - 0.05, 0.86, 0.5 + 0.05,
          f"b = {result['b']:.3f} {format_sig(result['b_p'])}", b_sig, shift_y=0.04)
    # c': X → Y
    arrow(0.14, 0.5, 0.86, 0.5,
          f"c' = {result['cp']:.3f} {format_sig(result['cp_p'])}", cp_sig, shift_y=-0.1)

    # Подпись косвенного эффекта
    ab_color = '#27ae60' if ab_sig else '#bdc3c7'
    fig.add_annotation(x=0.5, y=0.15,
                       text=f"<b>Косвенный эффект a·b = {result['indirect']:.3f}</b><br>"
                            f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]<br>"
                            f"<b>{'✓ Значим' if ab_sig else '— Не значим'}</b>",
                       showarrow=False,
                       font=dict(size=12, color=ab_color),
                       bgcolor='white', bordercolor=ab_color, borderwidth=2, borderpad=6)

    fig.update_xaxes(visible=False, range=[-0.02, 1.02])
    fig.update_yaxes(visible=False, range=[0, 1.05])
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor='white', showlegend=False)
    return fig


# =============================================================================
# ВКЛАДКИ
# =============================================================================
tab_single, tab_multi, tab_auto, tab_help = st.tabs([
    "🎯 Простая медиация (X → M → Y)",
    "🧩 Множественная медиация (несколько M)",
    "🔎 Автопоиск медиаций (разведочный)",
    "📖 Шпаргалка по интерпретации",
])

# -----------------------------------------------------------------------------
# ВКЛАДКА 1: ПРОСТАЯ МЕДИАЦИЯ
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("Классическая модель X → M → Y с бутстрап-CI")

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        default_x = 'B_Self-realization' if 'B_Self-realization' in num_cols else num_cols[0]
        x_var = st.selectbox("Независимая X:", num_cols,
                              index=num_cols.index(default_x),
                              key='med_x', format_func=get_name)
    with col_sel2:
        m_options = [c for c in num_cols if c != x_var]
        default_m = 'M_DR_Zh-id' if 'M_DR_Zh-id' in m_options else m_options[0]
        m_var = st.selectbox("Медиатор M:", m_options,
                              index=m_options.index(default_m) if default_m in m_options else 0,
                              key='med_m', format_func=get_name)
    with col_sel3:
        y_options = [c for c in num_cols if c != x_var and c != m_var]
        default_y = 'IPL_Total' if 'IPL_Total' in y_options else y_options[0]
        y_var = st.selectbox("Зависимая Y:", y_options,
                              index=y_options.index(default_y) if default_y in y_options else 0,
                              key='med_y', format_func=get_name)

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        n_boot = st.select_slider("Количество бутстрап-итераций:",
                                   options=[1000, 2000, 5000, 10000], value=5000,
                                   help="Больше итераций = точнее CI, но дольше расчёт. 5000 — стандарт.")
    with col_opt2:
        standardize = st.checkbox("Стандартизовать переменные", value=True,
                                   help="Коэффициенты станут стандартизованными β — сопоставимыми между разными шкалами.")
    with col_opt3:
        seed = st.number_input("Seed (для воспроизводимости):", value=42, step=1)

    if st.button("🚀 Запустить анализ", key="run_single_med"):
        data = df[[x_var, m_var, y_var]].dropna()
        if len(data) < 30:
            st.error(f"Недостаточно данных для надёжной медиации: {len(data)} наблюдений (нужно ≥30).")
        else:
            with st.spinner(f"Бутстрап {n_boot} итераций..."):
                x = data[x_var].values.astype(float)
                m = data[m_var].values.astype(float)
                y = data[y_var].values.astype(float)
                result = mediation_single(x, m, y, n_bootstrap=n_boot,
                                           seed=int(seed), standardize=standardize)
                diag = check_assumptions(x, m, y)
                st.session_state['last_mediation'] = {
                    'result': result, 'diag': diag,
                    'x_var': x_var, 'm_var': m_var, 'y_var': y_var,
                    'data': data,
                }

    if 'last_mediation' in st.session_state:
        state = st.session_state['last_mediation']
        result = state['result']
        diag = state['diag']
        x_var = state['x_var']; m_var = state['m_var']; y_var = state['y_var']

        # ДИАГРАММА
        st.markdown("### 📐 Диаграмма путей")
        st.plotly_chart(draw_mediation_diagram(result, get_name(x_var),
                                                 get_name(m_var), get_name(y_var)),
                         use_container_width=True)

        # ИНТЕРПРЕТАЦИЯ
        if "Полная" in result['interp_type']:
            st.success(f"### 🎯 {result['interp_type']}\n\n{result['interp_text']}")
        elif "Частичная" in result['interp_type']:
            st.info(f"### 🔀 {result['interp_type']}\n\n{result['interp_text']}")
        elif "Супрессия" in result['interp_type']:
            st.warning(f"### ⚠️ {result['interp_type']}\n\n{result['interp_text']}")
        else:
            st.info(f"### ℹ️ {result['interp_type']}\n\n{result['interp_text']}")

        # СВОДНЫЕ МЕТРИКИ
        st.markdown("### 📊 Сводные показатели")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Общий эффект c", f"{result['c']:.3f}",
                      format_sig(result['c_p']))
        with mc2:
            st.metric("Косвенный a·b", f"{result['indirect']:.3f}",
                      f"CI: [{result['ci_lower']:.3f}; {result['ci_upper']:.3f}]")
        with mc3:
            st.metric("Прямой c'", f"{result['cp']:.3f}",
                      format_sig(result['cp_p']))
        with mc4:
            if not np.isnan(result['prop_mediated']):
                st.metric("% через медиатор", f"{result['prop_mediated']*100:+.1f}%",
                          help="Доля общего эффекта, проходящая через медиатор")

        # ПУТИ В ТАБЛИЦЕ
        with st.expander("🔍 Детали всех путей (a, b, c, c')", expanded=True):
            paths_df = pd.DataFrame({
                'Путь': ['a (X → M)', 'b (M → Y | X)', 'c (X → Y)', "c' (X → Y | M)",
                         'a·b (косвенный)'],
                'Коэффициент': [result['a'], result['b'], result['c'],
                                result['cp'], result['indirect']],
                'SE': [result['a_se'], result['b_se'], result['c_se'],
                       result['cp_se'], '—'],
                'p-value': [format_p(result['a_p']), format_p(result['b_p']),
                            format_p(result['c_p']), format_p(result['cp_p']),
                            f"95% CI: [{result['ci_lower']:.3f}; {result['ci_upper']:.3f}]"],
                'Значим': [format_sig(result['a_p']), format_sig(result['b_p']),
                           format_sig(result['c_p']), format_sig(result['cp_p']),
                           '✓' if result['ab_sig'] else '—'],
            })
            paths_df['Коэффициент'] = paths_df['Коэффициент'].apply(lambda v: f"{v:.4f}")
            paths_df['SE'] = paths_df['SE'].apply(lambda v: f"{v:.4f}" if v != '—' else '—')
            st.dataframe(paths_df, use_container_width=True, hide_index=True)

        # БУТСТРАП-РАСПРЕДЕЛЕНИЕ
        with st.expander("📈 Бутстрап-распределение косвенного эффекта", expanded=False):
            fig_boot = go.Figure()
            fig_boot.add_trace(go.Histogram(x=result['boot_distribution'],
                                             nbinsx=50, marker_color='#3498db',
                                             opacity=0.75, name='Бутстрап'))
            fig_boot.add_vline(x=result['indirect'], line_dash='solid',
                                line_color='red', line_width=2,
                                annotation_text=f"a·b = {result['indirect']:.3f}")
            fig_boot.add_vline(x=result['ci_lower'], line_dash='dash', line_color='green',
                                annotation_text=f"2.5%: {result['ci_lower']:.3f}")
            fig_boot.add_vline(x=result['ci_upper'], line_dash='dash', line_color='green',
                                annotation_text=f"97.5%: {result['ci_upper']:.3f}")
            fig_boot.add_vline(x=0, line_dash='dot', line_color='black')
            fig_boot.update_layout(
                title=f"Распределение a·b по {len(result['boot_distribution'])} ресэмплам",
                xaxis_title="a·b", yaxis_title="Частота", height=400)
            st.plotly_chart(fig_boot, use_container_width=True)
            st.caption("Если 95% CI (зелёные линии) не пересекает 0 (чёрная пунктирная) — косвенный эффект значим.")

        # ТЕСТ СОБЕЛЯ (КЛАССИКА)
        with st.expander("🧮 Классический тест Собеля (для сравнения)", expanded=False):
            st.markdown(f"""
**Z-статистика Собеля:** {result['sobel_z']:.3f}
**p-value:** {format_p(result['sobel_p'])}

Собель — старая классика, предполагает нормальность распределения *a·b*, что часто нарушается.
**Бутстрап надёжнее** и является современным стандартом (Хайес, 2018).

- Собель говорит: значимо ли *a·b* в предположении нормальности?
- Бутстрап CI: не включает ли 0 эмпирическое 95% CI?

В большинстве случаев оба дадут одинаковый вывод, но при отклонениях от нормальности
бутстрап может показывать значимость там, где Собель — нет (и наоборот).
""")

        # ПРОВЕРКИ ПРЕДПОСЫЛОК
        with st.expander("🔬 Проверки предпосылок (нормальность, линейность, гомоскедастичность)",
                          expanded=False):
            st.markdown("#### Нормальность переменных (Шапиро-Уилк)")
            norm_rows = []
            for key, label in [('shapiro_X', get_name(x_var)), ('shapiro_M', get_name(m_var)),
                                ('shapiro_Y', get_name(y_var))]:
                w, p = diag[key]
                if w is None:
                    norm_rows.append({'Переменная': label, 'W': '—', 'p': '—', 'Нормально?': '—'})
                else:
                    norm_rows.append({'Переменная': label, 'W': f"{w:.3f}",
                                      'p': f"{p:.4f}",
                                      'Нормально?': '✓' if p > 0.05 else '✗'})
            st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)
            st.caption("Если одна или несколько переменных не нормальны — **предпочитайте бутстрап**, "
                       "а не тест Собеля. Бутстрап не зависит от предположения о нормальности.")

            st.markdown("#### Линейность: Пирсон vs Спирмен")
            lin_rows = []
            for a, b, label in [('xm', 'xm', 'X ↔ M'), ('my', 'my', 'M ↔ Y'),
                                 ('xy', 'xy', 'X ↔ Y')]:
                pr, pp = diag[f'pearson_{a}']
                sr, sp = diag[f'spearman_{b}']
                diff = abs(pr) - abs(sr)
                mark = '✓ линейно' if abs(diff) < 0.10 else '⚠️ возможная нелинейность'
                lin_rows.append({
                    'Пара': label,
                    'r (Пирсон)': f"{pr:.3f}",
                    'rho (Спирмен)': f"{sr:.3f}",
                    '|r| - |rho|': f"{diff:+.3f}",
                    'Статус': mark,
                })
            st.dataframe(pd.DataFrame(lin_rows), use_container_width=True, hide_index=True)
            st.caption("Сильное расхождение Пирсона и Спирмена (разница >0.10) намекает на нелинейность. "
                       "Линейная медиация тогда упускает часть связи — стоит визуально проверить scatter.")

            st.markdown("#### Гомоскедастичность остатков")
            bp_r, bp_p = diag['heteroscedasticity']
            if bp_p < 0.05:
                st.warning(f"⚠️ Корреляция |остатков| с предсказанием значима (rho = {bp_r:.3f}, p = {bp_p:.4f}). "
                           f"Дисперсия остатков меняется с уровнем Y — это гетероскедастичность. "
                           f"**Бутстрап робустен к ней**, OLS-SE — нет.")
            else:
                st.success(f"✓ Дисперсия остатков стабильна (rho = {bp_r:.3f}, p = {bp_p:.4f}).")

        # СКАЧИВАНИЕ
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            summary = pd.DataFrame({
                'Показатель': ['X', 'M', 'Y', 'N',
                                'c (общий)', 'c p', 'a (X→M)', 'a p',
                                'b (M→Y|X)', 'b p', "c' (X→Y|M)", "c' p",
                                'a·b (косвенный)', 'CI нижняя', 'CI верхняя',
                                'Бутстрап значим?', 'Собель z', 'Собель p',
                                '% медиации', 'Тип'],
                'Значение': [get_name(x_var), get_name(m_var), get_name(y_var), result['n'],
                              result['c'], result['c_p'], result['a'], result['a_p'],
                              result['b'], result['b_p'], result['cp'], result['cp_p'],
                              result['indirect'], result['ci_lower'], result['ci_upper'],
                              'Да' if result['ab_sig'] else 'Нет',
                              result['sobel_z'], result['sobel_p'],
                              f"{result['prop_mediated']*100:.1f}%" if not np.isnan(result['prop_mediated']) else '—',
                              result['interp_type']]
            })
            summary.to_excel(writer, sheet_name='Медиация', index=False)
        st.download_button("📥 Скачать отчёт (Excel)", buf.getvalue(),
                            f"mediation_{x_var}_{m_var}_{y_var}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# -----------------------------------------------------------------------------
# ВКЛАДКА 2: МНОЖЕСТВЕННАЯ МЕДИАЦИЯ
# -----------------------------------------------------------------------------
with tab_multi:
    st.subheader("Несколько медиаторов одновременно (parallel mediation)")
    st.markdown("Если есть **несколько кандидатов в медиаторы**, можно проверить их вместе. "
                "Для каждого получите отдельный косвенный эффект a·b и 95% CI. "
                "Это сильнее чем запускать одиночные медиации подряд: контролируется взаимное влияние медиаторов.")

    col_mm1, col_mm2 = st.columns(2)
    with col_mm1:
        default_x = 'B_Self-realization' if 'B_Self-realization' in num_cols else num_cols[0]
        x_mm = st.selectbox("Независимая X:", num_cols,
                             index=num_cols.index(default_x),
                             key='mm_x', format_func=get_name)
    with col_mm2:
        y_options_mm = [c for c in num_cols if c != x_mm]
        default_y = 'IPL_Total' if 'IPL_Total' in y_options_mm else y_options_mm[0]
        y_mm = st.selectbox("Зависимая Y:", y_options_mm,
                             index=y_options_mm.index(default_y) if default_y in y_options_mm else 0,
                             key='mm_y', format_func=get_name)

    # Медиаторы с кнопками быстрого выбора
    if 'mm_meds_safe' not in st.session_state: st.session_state['mm_meds_safe'] = []
    if 'mm_meds' not in st.session_state: st.session_state['mm_meds'] = st.session_state['mm_meds_safe']

    def add_prefix_mm(prefix):
        cur = st.session_state['mm_meds']
        new = [c for c in num_cols if c not in (x_mm, y_mm) and c.startswith(prefix) and c not in cur]
        st.session_state['mm_meds'] = cur + new

    def clear_mm():
        st.session_state['mm_meds'] = []

    bc = st.columns(4)
    bc[0].button("➕ Все Мильман", on_click=add_prefix_mm, args=('M_',), key='mm_btn_m')
    bc[1].button("➕ Мильман развивающие (Д, ДР, ОД)", on_click=lambda: st.session_state.update(
        {'mm_meds': list(set(st.session_state['mm_meds'] +
                             [c for c in num_cols if c.startswith('M_D_') or c.startswith('M_DR_')
                              or c.startswith('M_OD_')]))}), key='mm_btn_dev')
    bc[2].button("➕ Все ИПЛ", on_click=add_prefix_mm, args=('IPL_',), key='mm_btn_ipl')
    bc[3].button("❌ Очистить", on_click=clear_mm, key='mm_btn_clear')

    mediators_mm = st.multiselect("Медиаторы (от 2 и больше):",
                                    [c for c in num_cols if c != x_mm and c != y_mm],
                                    key='mm_meds', format_func=get_name)

    col_mm_opt1, col_mm_opt2, col_mm_opt3 = st.columns(3)
    with col_mm_opt1:
        n_boot_mm = st.select_slider("Бутстрап итераций:",
                                       options=[1000, 2000, 5000, 10000], value=5000,
                                       key='mm_nboot')
    with col_mm_opt2:
        std_mm = st.checkbox("Стандартизовать", value=True, key='mm_std')
    with col_mm_opt3:
        seed_mm = st.number_input("Seed:", value=42, step=1, key='mm_seed')

    if st.button("🚀 Запустить множественную медиацию", key="run_mm"):
        if len(mediators_mm) < 2:
            st.warning("Выберите минимум 2 медиатора.")
        else:
            all_cols = [x_mm, y_mm] + mediators_mm
            data = df[all_cols].dropna()
            if len(data) < len(mediators_mm) + 20:
                st.error(f"Недостаточно данных: {len(data)} наблюдений.")
            else:
                with st.spinner(f"Бутстрап {n_boot_mm} итераций для {len(mediators_mm)} медиаторов..."):
                    x_arr = data[x_mm].values.astype(float)
                    y_arr = data[y_mm].values.astype(float)
                    M_arr = data[mediators_mm].values.astype(float)
                    mm_result = mediation_multiple(x_arr, M_arr, y_arr, mediators_mm,
                                                    n_bootstrap=n_boot_mm, seed=int(seed_mm),
                                                    standardize=std_mm)
                    st.session_state['last_mm'] = {
                        'result': mm_result, 'x': x_mm, 'y': y_mm, 'meds': mediators_mm,
                    }

    if 'last_mm' in st.session_state:
        res_mm = st.session_state['last_mm']['result']
        x_mm_s = st.session_state['last_mm']['x']
        y_mm_s = st.session_state['last_mm']['y']

        st.markdown("### 📊 Общая картина")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Общий эффект c", f"{res_mm['c']:.3f}", format_sig(res_mm['c_p']))
        with m_c2:
            ci_lo, ci_hi = res_mm['total_ci']
            ts = "✓" if (ci_lo > 0 or ci_hi < 0) else "—"
            st.metric("Суммарный косвенный Σa·b", f"{res_mm['total_indirect']:.3f}",
                      f"CI: [{ci_lo:.3f}; {ci_hi:.3f}] {ts}")
        with m_c3:
            st.metric("Прямой c'", f"{res_mm['cp']:.3f}", format_sig(res_mm['cp_p']))
        with m_c4:
            st.metric("N", res_mm['n'])

        # Таблица медиаторов
        st.markdown("### 🧩 Отдельные медиаторы")
        tbl = res_mm['mediators'].copy()
        tbl_display = tbl[['Медиатор', 'a', 'a_p', 'b', 'b_p', 'a·b', 'CI_low', 'CI_high', 'Значим (CI)']].copy()
        for col in ['a', 'b', 'a·b', 'CI_low', 'CI_high']:
            tbl_display[col] = tbl_display[col].apply(lambda v: f"{v:.3f}")
        tbl_display['a_p'] = tbl_display['a_p'].apply(format_p)
        tbl_display['b_p'] = tbl_display['b_p'].apply(format_p)
        st.dataframe(tbl_display, use_container_width=True, hide_index=True)

        # Визуализация всех a·b с CI (forest plot)
        st.markdown("### 🌲 Forest plot: косвенные эффекты с 95% CI")
        fp = tbl.sort_values('a·b').copy()
        fig_fp = go.Figure()
        for i, row in fp.iterrows():
            sig_color = '#27ae60' if (row['CI_low'] > 0 or row['CI_high'] < 0) else '#95a5a6'
            fig_fp.add_trace(go.Scatter(x=[row['CI_low'], row['CI_high']],
                                          y=[row['Медиатор'], row['Медиатор']],
                                          mode='lines',
                                          line=dict(color=sig_color, width=3),
                                          showlegend=False, hoverinfo='skip'))
        fig_fp.add_trace(go.Scatter(x=fp['a·b'], y=fp['Медиатор'], mode='markers',
                                     marker=dict(size=12,
                                                 color=['#27ae60' if (r['CI_low'] > 0 or r['CI_high'] < 0)
                                                        else '#95a5a6' for _, r in fp.iterrows()],
                                                 line=dict(color='black', width=1)),
                                     showlegend=False,
                                     text=[f"a·b = {v:.3f}" for v in fp['a·b']],
                                     hovertemplate='%{y}<br>%{text}<extra></extra>'))
        fig_fp.add_vline(x=0, line_dash='dash', line_color='red')
        fig_fp.update_layout(xaxis_title="Косвенный эффект a·b (с 95% CI)",
                              height=max(350, 45 * len(fp)),
                              margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_fp, use_container_width=True)
        st.caption("🟢 Зелёные = значимые медиаторы (CI не пересекает 0). 🔘 Серые = не значимы.")

        # СКАЧИВАНИЕ
        buf_mm = io.BytesIO()
        with pd.ExcelWriter(buf_mm, engine='openpyxl') as writer:
            tbl.to_excel(writer, sheet_name='Медиаторы', index=False)
            pd.DataFrame({
                'Показатель': ['X', 'Y', 'N', 'c', 'c p', "c'", "c' p",
                                'Σa·b', 'CI низ', 'CI верх'],
                'Значение': [get_name(x_mm_s), get_name(y_mm_s), res_mm['n'],
                              res_mm['c'], res_mm['c_p'], res_mm['cp'], res_mm['cp_p'],
                              res_mm['total_indirect'], res_mm['total_ci'][0], res_mm['total_ci'][1]],
            }).to_excel(writer, sheet_name='Сводка', index=False)
        st.download_button("📥 Скачать (Excel)", buf_mm.getvalue(),
                            f"multi_mediation_{x_mm_s}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# -----------------------------------------------------------------------------
# ВКЛАДКА 3: АВТОПОИСК МЕДИАЦИЙ
# -----------------------------------------------------------------------------
with tab_auto:
    st.subheader("Эксплораторный автопоиск медиационных моделей")

    st.warning("""
⚠️ **Это разведочный инструмент, а не основной метод исследования.**

Автопоиск перебирает множество комбинаций X→M→Y и находит статистически значимые.
Это даёт **гипотезы для проверки**, но НЕ готовые результаты для работы:

- **Множественные сравнения** раздувают ложноположительные. Из 100 случайных тестов
  ~5 окажутся «значимыми» просто из-за шума.
- **Нет теории** — значимая модель, найденная перебором, может быть статистическим артефактом.
- **Направление X→M→Y** остаётся на твоей ответственности. Статистика не определяет причинность.

**Как использовать правильно:** запустить автопоиск → посмотреть, какие паттерны
эмпирически самые сильные → выбрать из них 2-3 **теоретически обоснованные** → проверить
их в отдельной вкладке «Простая медиация» и описать в работе.
""")

    st.markdown("### 🎛 Настройка поиска")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Кандидаты в X (независимые):**")
        # Кнопки быстрого выбора
        if 'auto_x_list' not in st.session_state: st.session_state['auto_x_list'] = []
        ax_c1, ax_c2, ax_c3 = st.columns(3)

        def ax_add(prefix):
            cur = st.session_state['auto_x_list']
            new = [c for c in num_cols if c.startswith(prefix) and c not in cur]
            st.session_state['auto_x_list'] = cur + new

        ax_c1.button("➕ Братусь", on_click=ax_add, args=('B_',), key='ax_b')
        ax_c2.button("➕ Мильман", on_click=ax_add, args=('M_',), key='ax_m')
        ax_c3.button("❌", on_click=lambda: st.session_state.update({'auto_x_list': []}), key='ax_clr')

        x_candidates = st.multiselect("X-кандидаты:", num_cols, key='auto_x_list',
                                        format_func=get_name,
                                        label_visibility='collapsed')

    with col_a2:
        st.markdown("**Кандидаты в M (медиаторы):**")
        if 'auto_m_list' not in st.session_state: st.session_state['auto_m_list'] = []
        am_c1, am_c2, am_c3 = st.columns(3)

        def am_add(prefix):
            cur = st.session_state['auto_m_list']
            new = [c for c in num_cols if c.startswith(prefix) and c not in cur]
            st.session_state['auto_m_list'] = cur + new

        am_c1.button("➕ Мильман", on_click=am_add, args=('M_',), key='am_m')
        am_c2.button("➕ ИПЛ", on_click=am_add, args=('IPL_',), key='am_i')
        am_c3.button("❌", on_click=lambda: st.session_state.update({'auto_m_list': []}), key='am_clr')

        m_candidates = st.multiselect("M-кандидаты:", num_cols, key='auto_m_list',
                                        format_func=get_name,
                                        label_visibility='collapsed')

    y_candidates = st.multiselect("Кандидаты в Y (зависимые):",
                                    num_cols,
                                    default=['IPL_Total'] if 'IPL_Total' in num_cols else [],
                                    key='auto_y_list', format_func=get_name)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        auto_nboot = st.select_slider("Бутстрап итераций:",
                                        options=[500, 1000, 2000, 5000], value=2000,
                                        key='auto_nboot',
                                        help="Меньше итераций = быстрее перебор, но точность CI ниже. "
                                             "Для разведки хватит 2000.")
    with col_b2:
        fdr_method = st.selectbox("Поправка на множественные сравнения:",
                                    ['BH (FDR)', 'Бонферрони', 'Без поправки'],
                                    help="BH (Бенджамини-Хохберг) контролирует долю ложных открытий — "
                                         "более мягкая чем Бонферрони, лучше для разведки. "
                                         "Бонферрони очень строгий.")
    with col_b3:
        min_effect = st.number_input("Мин. |a·b| для показа:",
                                       value=0.05, step=0.01,
                                       help="Фильтр по величине эффекта. 0.05 = только содержательно заметные.")
    with col_b4:
        top_n = st.number_input("Показать топ-N:", value=30, step=5, min_value=5,
                                  help="Сколько лучших моделей выводить.")

    # Оценка числа моделей
    n_models_est = len(x_candidates) * max(0, len(m_candidates)) * len(y_candidates)
    # Вычитаем пересечения (где X=M или M=Y или X=Y — невалидные)
    overlap = len(set(x_candidates) & set(m_candidates)) * len(y_candidates) \
              + len(set(m_candidates) & set(y_candidates)) * len(x_candidates)
    n_valid_est = max(0, n_models_est - overlap)

    if n_valid_est > 0:
        est_time = n_valid_est * auto_nboot / 5000 * 0.5  # грубая оценка секунд
        info_msg = f"К проверке: **~{n_valid_est}** моделей. Примерно **{est_time:.0f} сек**."
        if n_valid_est > 500:
            st.warning(info_msg + " Очень много моделей — уменьши список кандидатов.")
        else:
            st.info(info_msg)

    if st.button("🚀 Запустить автопоиск", key='run_auto', disabled=(n_valid_est == 0)):
        if len(x_candidates) == 0 or len(m_candidates) == 0 or len(y_candidates) == 0:
            st.error("Добавь хотя бы по одной переменной в каждую категорию.")
        else:
            progress = st.progress(0.0)
            status = st.empty()
            rows = []

            # Перебираем тройки
            total = 0
            pairs = []
            for xv in x_candidates:
                for mv in m_candidates:
                    if mv == xv: continue
                    for yv in y_candidates:
                        if yv == xv or yv == mv: continue
                        pairs.append((xv, mv, yv))
                        total += 1

            for idx, (xv, mv, yv) in enumerate(pairs):
                status.caption(f"Проверяется {idx+1}/{total}: {get_name(xv)[:25]} → {get_name(mv)[:25]} → {get_name(yv)[:25]}")
                progress.progress((idx + 1) / total)

                try:
                    data = df[[xv, mv, yv]].dropna()
                    if len(data) < 30:
                        continue
                    x = data[xv].values.astype(float)
                    m = data[mv].values.astype(float)
                    y = data[yv].values.astype(float)
                    # Стандартизуем
                    x = (x - x.mean()) / x.std()
                    m = (m - m.mean()) / m.std()
                    y = (y - y.mean()) / y.std()

                    # Пути
                    beta_c  = ols_simple(x, y)
                    beta_a  = ols_simple(x, m)
                    beta_cm = ols_simple(np.column_stack([x, m]), y)
                    c  = beta_c['beta'][1];  c_p  = beta_c['p'][1]
                    a  = beta_a['beta'][1];  a_p  = beta_a['p'][1]
                    cp = beta_cm['beta'][1]; cp_p = beta_cm['p'][1]
                    b  = beta_cm['beta'][2]; b_p  = beta_cm['p'][2]
                    indirect = a * b

                    # Быстрый бутстрап
                    rng = np.random.default_rng(42)
                    n = len(x)
                    boot = np.zeros(auto_nboot)
                    for i in range(auto_nboot):
                        ix = rng.integers(0, n, n)
                        xb = x[ix]; mb = m[ix]; yb = y[ix]
                        ab = ols_simple(xb, mb)['beta'][1]
                        bb = ols_simple(np.column_stack([xb, mb]), yb)['beta'][2]
                        boot[i] = ab * bb

                    ci_lo = float(np.percentile(boot, 2.5))
                    ci_hi = float(np.percentile(boot, 97.5))
                    # Эмпирическое p-value для a·b (двусторонний тест через бутстрап)
                    # количество бутстрап-значений, попадающих по "противоположную сторону от 0"
                    if indirect > 0:
                        p_boot = 2 * np.mean(boot <= 0)
                    else:
                        p_boot = 2 * np.mean(boot >= 0)
                    p_boot = min(p_boot, 1.0)

                    ab_sig_ci = (ci_lo > 0) or (ci_hi < 0)

                    # Классификация типа
                    if ab_sig_ci and cp_p >= 0.05:
                        mtype = "Полная"
                    elif ab_sig_ci and cp_p < 0.05:
                        if np.sign(indirect) == np.sign(cp):
                            mtype = "Частичная"
                        else:
                            mtype = "Супрессия"
                    elif not ab_sig_ci and cp_p < 0.05:
                        mtype = "Только прямой"
                    else:
                        mtype = "Нет связи"

                    rows.append({
                        'X': get_name(xv), 'M': get_name(mv), 'Y': get_name(yv),
                        'x_col': xv, 'm_col': mv, 'y_col': yv,
                        'n': len(data),
                        'c': c, 'c_p': c_p,
                        'a': a, 'a_p': a_p,
                        'b': b, 'b_p': b_p,
                        "c'": cp, "c'_p": cp_p,
                        'a·b': indirect,
                        'CI_low': ci_lo, 'CI_high': ci_hi,
                        'p_boot': p_boot,
                        'ab_sig_ci': ab_sig_ci,
                        'type': mtype,
                    })
                except Exception:
                    continue

            progress.empty(); status.empty()

            if not rows:
                st.error("Ни одной валидной модели не получилось рассчитать.")
            else:
                result_df = pd.DataFrame(rows)
                # Поправка на множественные сравнения
                m_tests = len(result_df)
                if fdr_method == 'Бонферрони':
                    result_df['p_adj'] = (result_df['p_boot'] * m_tests).clip(upper=1.0)
                    crit = 0.05
                    threshold_text = f"Бонферрони-порог: p_adj < 0.05"
                elif fdr_method == 'BH (FDR)':
                    # Benjamini-Hochberg
                    sorted_ = result_df.sort_values('p_boot').reset_index(drop=True).copy()
                    sorted_['rank'] = np.arange(1, m_tests + 1)
                    sorted_['p_adj'] = sorted_['p_boot'] * m_tests / sorted_['rank']
                    # Монотонность: p_adj должно неубывать
                    sorted_['p_adj'] = sorted_['p_adj'][::-1].cummin()[::-1]
                    sorted_['p_adj'] = sorted_['p_adj'].clip(upper=1.0)
                    # Возвращаем в исходный порядок по ключам
                    result_df = result_df.merge(
                        sorted_[['x_col', 'm_col', 'y_col', 'p_adj']],
                        on=['x_col', 'm_col', 'y_col'], how='left'
                    )
                    crit = 0.05
                    threshold_text = f"BH (FDR)-порог: p_adj < 0.05"
                else:
                    result_df['p_adj'] = result_df['p_boot']
                    crit = 0.05
                    threshold_text = "Без поправки"

                result_df['Значим (скорр.)'] = result_df['p_adj'].apply(
                    lambda p: '✓' if p < crit else '—')
                result_df['|a·b|'] = result_df['a·b'].abs()

                # Сводка
                n_sig_raw = int((result_df['p_boot'] < 0.05).sum())
                n_sig_adj = int((result_df['p_adj'] < crit).sum())
                n_big = int((result_df['|a·b|'] >= min_effect).sum())

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: st.metric("Проверено моделей", m_tests)
                with sc2: st.metric("Значимы (без попр.)", n_sig_raw)
                with sc3: st.metric("Значимы (скорр.)", n_sig_adj)
                with sc4: st.metric(f"Крупные (|a·b|≥{min_effect})", n_big)

                st.caption(threshold_text)

                # Фильтр по величине эффекта и сортировка
                filtered = result_df[result_df['|a·b|'] >= min_effect].copy()
                filtered = filtered.sort_values(['ab_sig_ci', '|a·b|'],
                                                  ascending=[False, False]).head(int(top_n))

                if len(filtered) == 0:
                    st.info(f"Нет моделей с |a·b| ≥ {min_effect}. Попробуй снизить порог.")
                else:
                    st.markdown(f"### 🏆 Топ-{len(filtered)} моделей (отсортировано по |a·b|)")

                    display = filtered[['X', 'M', 'Y', 'a', 'b', "c'", 'a·b',
                                          'CI_low', 'CI_high', 'type',
                                          'p_adj', 'Значим (скорр.)', 'n']].copy()
                    display['a'] = display['a'].apply(lambda v: f"{v:+.3f}")
                    display['b'] = display['b'].apply(lambda v: f"{v:+.3f}")
                    display["c'"] = display["c'"].apply(lambda v: f"{v:+.3f}")
                    display['a·b'] = display['a·b'].apply(lambda v: f"{v:+.3f}")
                    display['CI'] = display.apply(
                        lambda r: f"[{r['CI_low']:+.3f}; {r['CI_high']:+.3f}]", axis=1)
                    display['p_adj'] = display['p_adj'].apply(
                        lambda p: '<0.001' if p < 0.001 else f'{p:.3f}')
                    display = display.drop(columns=['CI_low', 'CI_high'])
                    display = display[['X', 'M', 'Y', 'type', 'a', 'b', "c'", 'a·b',
                                        'CI', 'p_adj', 'Значим (скорр.)', 'n']]

                    st.dataframe(display, use_container_width=True, hide_index=True)

                    # Forest plot для топа
                    st.markdown("### 🌲 Forest plot: косвенные эффекты лучших моделей")
                    plot_df = filtered.head(20).copy()
                    plot_df['label'] = plot_df.apply(
                        lambda r: f"{r['X'][:15]} → {r['M'][:15]} → {r['Y'][:12]}", axis=1)
                    plot_df = plot_df.sort_values('a·b')

                    fig_auto = go.Figure()
                    for _, row in plot_df.iterrows():
                        col = '#27ae60' if row['ab_sig_ci'] else '#95a5a6'
                        fig_auto.add_trace(go.Scatter(
                            x=[row['CI_low'], row['CI_high']],
                            y=[row['label'], row['label']],
                            mode='lines',
                            line=dict(color=col, width=3),
                            showlegend=False, hoverinfo='skip'))
                    fig_auto.add_trace(go.Scatter(
                        x=plot_df['a·b'], y=plot_df['label'], mode='markers',
                        marker=dict(size=12,
                                      color=['#27ae60' if s else '#95a5a6'
                                             for s in plot_df['ab_sig_ci']],
                                      line=dict(color='black', width=1)),
                        showlegend=False,
                        text=[f"a·b = {v:.3f}" for v in plot_df['a·b']],
                        hovertemplate='%{y}<br>%{text}<extra></extra>'))
                    fig_auto.add_vline(x=0, line_dash='dash', line_color='red')
                    fig_auto.update_layout(
                        xaxis_title="Косвенный эффект a·b (с 95% CI)",
                        height=max(400, 30 * len(plot_df)),
                        margin=dict(l=10, r=10, t=10, b=30))
                    st.plotly_chart(fig_auto, use_container_width=True)

                    st.info("💡 **Что делать с результатом:** выбери 2-3 наиболее интересных модели, "
                            "которые имеют теоретическое обоснование, и перейди во вкладку "
                            "«🎯 Простая медиация» для углублённого анализа — там будет диаграмма путей, "
                            "проверки предпосылок и детальная интерпретация.")

                    # Скачивание полного отчёта
                    buf_auto = io.BytesIO()
                    with pd.ExcelWriter(buf_auto, engine='openpyxl') as writer:
                        result_df.drop(columns=['x_col', 'm_col', 'y_col']).to_excel(
                            writer, sheet_name='Все модели', index=False)
                        filtered.drop(columns=['x_col', 'm_col', 'y_col']).to_excel(
                            writer, sheet_name='Топ моделей', index=False)
                    st.download_button(
                        "📥 Скачать полный отчёт (Excel)",
                        buf_auto.getvalue(),
                        "autoscan_mediation.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


# -----------------------------------------------------------------------------
# ВКЛАДКА 4: ШПАРГАЛКА
# -----------------------------------------------------------------------------
with tab_help:
    st.subheader("Шпаргалка по медиационному анализу")

    st.markdown("""
### 1. Когда имеет смысл проверять медиацию

- Есть **теоретическая причина** полагать, что X действует на Y через M
- Корреляционный анализ показал значимые связи X↔M и M↔Y
- Хотите перейти от «Y связан с X» к **«вот как именно X влияет на Y»**

### 2. Как выбирать между бутстрапом и Собелем

| Вопрос | Ответ |
|---|---|
| Переменные нормально распределены? | Если да, оба метода дают близкие результаты |
| Переменные ненормальны? | **Только бутстрап** — Собель требует нормальности произведения a·b |
| N < 100? | Бутстрап надёжнее (Собель консервативен на малых выборках) |
| В руководителе диплома очень классический? | Показать оба, сослаться на бутстрап (Hayes, 2018) |

**Современный стандарт — бутстрап**, Собель оставляется для сравнения.

### 3. Как читать типы исхода

- **Полная медиация (c' ≈ 0, a·b значим):** сильнейший результат, медиатор полностью объясняет связь
- **Частичная медиация (c' значим, a·b значим):** тоже хорошо — медиатор часть связи объясняет
- **Только прямой (c' значим, a·b незначим):** M не медиатор в этой связи
- **Ничего (всё незначимо):** нет данных о связи
- **Супрессия (c' и a·b противоположных знаков):** важная находка — медиатор маскировал истинную связь

### 4. Частые ошибки

- **Направление: X → M → Y требует теории.** Статистика не скажет что причина, а что следствие — она работает с любым направлением
- **Кросс-секционные данные (как у тебя) не дают причинности.** Можно говорить только о совместимости данных с моделью
- **`c` не обязан быть значим.** Есть сильная традиция (Baron & Kenny) требовать значимого c, но современные стандарты (Hayes) показали — бутстрап работает и без этого
- **`N` должно быть достаточным.** При N<50 бутстрап может быть нестабилен даже на 5000 итераций

### 5. Что писать в раздел результатов

> Для проверки гипотезы о медиирующей роли M в связи X → Y был проведён медиационный анализ
> с бутстрап-оценкой доверительных интервалов (5000 ресэмплов). Общий эффект X на Y составил
> c = ... (p = ...). При контроле медиатора прямой эффект c' = ... (p = ...), тогда как косвенный
> эффект a·b = ..., 95% CI [..., ...]. Поскольку CI не включает ноль, косвенный эффект статистически
> значим. Анализ свидетельствует о [полной/частичной/отсутствии] медиации: влияние X на Y
> [полностью/частично/не] опосредовано M.

### 6. Ключевые ссылки

- Hayes, A. F. (2018). *Introduction to mediation, moderation, and conditional process analysis* (2nd ed.). Guilford Press.
- Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects. *Behavior Research Methods*.
- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research. *Journal of Personality and Social Psychology*.
""")