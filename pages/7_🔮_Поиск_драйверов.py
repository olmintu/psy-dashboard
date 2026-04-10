import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from utils import render_sidebar, get_name

st.set_page_config(page_title="Поиск драйверов", layout="wide", page_icon="🔮")

df = render_sidebar()
if df is None: st.stop()

st.header("🔮 Поиск скрытых драйверов (Random Forest)")

with st.expander("ℹ️ Как пользоваться этой вкладкой (Инструкция)"):
    st.markdown("""
    **Цель этой вкладки** — найти скрытые закономерности и понять, какие именно психологические факторы сильнее всего "драйвят" или тормозят конкретный показатель.

    **Шаг 1:** Выберите **Целевой показатель (Y)**. Это то, что вы хотите изучить.
    **Шаг 2:** Выберите **Факторы влияния (X)**. Это шкалы, среди которых алгоритм будет искать причины.
    
    ⚠️ **Правило чистоты данных:** Не пытайтесь предсказать *Общий балл* методики, используя в качестве факторов её же *Подшкалы*. 
    """)

num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Инициализация сейфа для вкладки 7
if 'safe_cb_b_7' not in st.session_state: st.session_state.safe_cb_b_7 = False
if 'safe_cb_m_7' not in st.session_state: st.session_state.safe_cb_m_7 = False
if 'safe_cb_i_7' not in st.session_state: st.session_state.safe_cb_i_7 = False
if 'safe_pred_vars' not in st.session_state: st.session_state.safe_pred_vars = []

# Восстановление из сейфа при загрузке страницы
if 'cb_b_7' not in st.session_state: st.session_state.cb_b_7 = st.session_state.safe_cb_b_7
if 'cb_m_7' not in st.session_state: st.session_state.cb_m_7 = st.session_state.safe_cb_m_7
if 'cb_i_7' not in st.session_state: st.session_state.cb_i_7 = st.session_state.safe_cb_i_7
if 'pred_vars' not in st.session_state: st.session_state.pred_vars = st.session_state.safe_pred_vars

c_pred1, c_pred2 = st.columns([1, 1.5])
with c_pred1:
    target_var = st.selectbox(
        "🎯 Целевой показатель:", num_cols, 
        index=num_cols.index('IPL_Total') if 'IPL_Total' in num_cols else 0, 
        key="target_var_7", format_func=get_name
    )

# Умные функции-колбэки для галочек
def toggle_b_7():
    cols = [c for c in num_cols if c.startswith('B_') and c != st.session_state.target_var_7]
    if st.session_state.cb_b_7: st.session_state.pred_vars += [c for c in cols if c not in st.session_state.pred_vars]
    else: st.session_state.pred_vars = [c for c in st.session_state.pred_vars if c not in cols]

def toggle_m_7():
    cols = [c for c in num_cols if c.startswith('M_') and c != st.session_state.target_var_7]
    if st.session_state.cb_m_7: st.session_state.pred_vars += [c for c in cols if c not in st.session_state.pred_vars]
    else: st.session_state.pred_vars = [c for c in st.session_state.pred_vars if c not in cols]

def toggle_i_7():
    cols = [c for c in num_cols if c.startswith('IPL_') and c != st.session_state.target_var_7]
    if st.session_state.cb_i_7: st.session_state.pred_vars += [c for c in cols if c not in st.session_state.pred_vars]
    else: st.session_state.pred_vars = [c for c in st.session_state.pred_vars if c not in cols]

with c_pred2:
    st.markdown("**⚡ Быстрый выбор методик-предикторов:**")
    cb1, cb2, cb3 = st.columns(3)
    with cb1: st.checkbox("Шкалы Братуся", key='cb_b_7', on_change=toggle_b_7)
    with cb2: st.checkbox("Шкалы Мильмана", key='cb_m_7', on_change=toggle_m_7)
    with cb3: st.checkbox("Шкалы ИПЛ", key='cb_i_7', on_change=toggle_i_7)

    predictor_vars = st.multiselect("📈 Факторы влияния:", [c for c in num_cols if c != target_var], key="pred_vars", format_func=get_name)

if st.button("🚀 Найти ключевые драйверы"):
    if len(predictor_vars) < 2:
        st.warning("⚠️ Для обучения модели выберите хотя бы 2 фактора влияния.")
    else:
        with st.spinner("Обучаем модель Random Forest и вычисляем направления..."):
            df_pred = df[[target_var] + predictor_vars].dropna()
            
            if len(df_pred) < 15:
                st.error("❌ Недостаточно данных для обучения.")
            else:
                X = df_pred[predictor_vars]
                y = df_pred[target_var]
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                rf.fit(X, y)
                r2 = r2_score(y, rf.predict(X))

                dirs = []
                for c in predictor_vars:
                    corr = df_pred[c].corr(df_pred[target_var])
                    if corr > 0: dirs.append("📈") 
                    else: dirs.append("📉") 

                importance = pd.DataFrame({
                    'Фактор': [get_name(c) for c in predictor_vars],
                    'Колонка': predictor_vars, 
                    'Важность': rf.feature_importances_ * 100,
                    'Знак': dirs
                })
                
                importance['Фактор_с_иконкой'] = importance['Знак'] + " " + importance['Фактор']
                importance = importance.sort_values(by='Важность', ascending=True)

                st.session_state['rf_results'] = {
                    'target_var': target_var, 'df_pred': df_pred,
                    'importance': importance, 'r2': r2, 'model': rf, 'features': predictor_vars
                }

if 'rf_results' in st.session_state:
    res = st.session_state['rf_results']
    t_var, df_p, imp, r_sq = res['target_var'], res['df_pred'], res['importance'], res['r2']

    st.markdown("---")
    res_c1, res_c2 = st.columns([2, 1])

    with res_c1:
        st.markdown(f"#### 📊 Рейтинг влияния на «{get_name(t_var)}»")
        fig_rf = px.bar(imp, x='Важность', y='Фактор_с_иконкой', orientation='h', color='Важность', color_continuous_scale='Viridis')
        fig_rf.update_layout(xaxis_title="Сила влияния (в %)", yaxis_title="", coloraxis_showscale=False, height=max(400, len(imp) * 35))
        st.plotly_chart(fig_rf, use_container_width=True, config={'displayModeBar': False})

    with res_c2:
        st.markdown("#### ⚙️ Качество модели")
        if r_sq > 0.5: st.success(f"**Точность объяснения (R²):** {r_sq*100:.1f}%")
        elif r_sq > 0.2: st.warning(f"**Точность (R²):** {r_sq*100:.1f}%")
        else: st.error(f"**Точность (R²):** {r_sq*100:.1f}%")

        st.markdown("---")
        st.markdown("🏆 **Топ-3 драйвера:**")
        top3 = imp.sort_values(by='Важность', ascending=False).head(3)
        for i, row in top3.iterrows():
            st.markdown(f"{row['Знак']} **{row['Фактор']}** ({row['Важность']:.1f}%)")

    st.markdown("---")
    st.markdown("### 🔍 Как именно работают Топ-3 драйвера?")
    st.markdown("`📈 Катализатор` (тянет вверх) | `📉 Блокатор` (тянет вниз)")
    
    graph_cols = st.columns(3)
    for idx, (index, row) in enumerate(top3.iterrows()):
        with graph_cols[idx]:
            fig_scatter = px.scatter(df_p, x=row['Колонка'], y=t_var, trendline="ols", trendline_color_override="red", opacity=0.7)
            fig_scatter.update_layout(title=dict(text=f"№{idx+1}: {row['Фактор_с_иконкой']}", font=dict(size=12)), xaxis_title=row['Фактор'], yaxis_title=get_name(t_var) if idx == 0 else "", margin=dict(l=10, r=10, t=40, b=10), height=300)
            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

    st.markdown("### 🎛️ Проверить другие факторы")
    ordered_features = imp.sort_values(by='Важность', ascending=False)['Фактор'].tolist()
    selected_feature_ru = st.selectbox("Выберите любую шкалу из рейтинга для детального анализа:", ordered_features)
    
    if selected_feature_ru:
        selected_row = imp[imp['Фактор'] == selected_feature_ru].iloc[0]
        fig_detail = px.scatter(df_p, x=selected_row['Колонка'], y=t_var, trendline="ols", trendline_color_override="red", opacity=0.7, hover_data=df_p.columns)
        fig_detail.update_layout(title=dict(text=f"Взаимосвязь {selected_row['Знак']}: {selected_feature_ru} ➔ {get_name(t_var)}", font=dict(size=16)), xaxis_title=selected_feature_ru, yaxis_title=get_name(t_var), margin=dict(l=20, r=20, t=50, b=20), height=450)
        st.plotly_chart(fig_detail, use_container_width=True)

    # ==========================================
    # СИМУЛЯТОР "ЧТО-ЕСЛИ"
    # ==========================================
    st.markdown("---")
    st.markdown("### 🎛️ Симулятор «Что-если» (Интерактивный прогноз)")
    st.markdown("Смоделируйте идеального респондента! Изменяйте значения ползунками, а затем нажмите кнопку пересчета, чтобы увидеть новый прогноз.")

    if 'model' in res:
        rf_model, all_features = res['model'], res['features']
        top5_cols = imp.sort_values(by='Важность', ascending=False).head(5)['Колонка'].tolist()
        
        sim_col1, sim_col2 = st.columns([1.5, 1])
        
        with sim_col1:
            st.markdown("#### Управление факторами (Топ-5)")
            with st.form("simulator_form"):
                user_inputs = {}
                for col in top5_cols:
                    user_inputs[col] = st.slider(get_name(col), min_value=float(df_p[col].min()), max_value=float(df_p[col].max()), value=float(df_p[col].mean()), step=0.5)
                submit_sim = st.form_submit_button("🔄 Пересчитать прогноз")
        
        with sim_col2:
            sim_data = {col: user_inputs.get(col, df_p[col].mean()) for col in all_features}
            predicted_target = rf_model.predict(pd.DataFrame([sim_data]))[0]
            
            t_min, t_max, t_mean = df_p[t_var].min(), df_p[t_var].max(), df_p[t_var].mean()
            
            st.markdown(f"#### Прогноз: {get_name(t_var)}")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=predicted_target, delta={'reference': t_mean, 'position': "top"},
                domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Ожидаемый балл", 'font': {'size': 18}},
                gauge={'axis': {'range': [t_min, t_max], 'tickwidth': 1}, 'bar': {'color': "darkblue"}, 'steps': [{'range': [t_min, t_mean], 'color': "lightgray"}, {'range': [t_mean, t_max], 'color': "lightgreen"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': predicted_target}}
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            
            if submit_sim: st.success("Прогноз успешно обновлен!")
            else: st.caption("Сдвиньте ползунки слева и нажмите «Пересчитать прогноз», чтобы обновить спидометр.")

# Сохранение бэкапа 
st.session_state.safe_cb_b_7 = st.session_state.cb_b_7
st.session_state.safe_cb_m_7 = st.session_state.cb_m_7
st.session_state.safe_cb_i_7 = st.session_state.cb_i_7
st.session_state.safe_pred_vars = st.session_state.pred_vars