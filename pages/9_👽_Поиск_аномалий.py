import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import render_sidebar, get_name, calculate_mahalanobis_distances

st.set_page_config(page_title="Детектор аномалий", layout="wide", page_icon="👽")

df = render_sidebar()
if df is None: st.stop()

st.header("👽 Детектор аномалий")
st.markdown("Поиск нетипичных респондентов с парадоксальными сочетаниями мотивов и смыслов.")

method_outlier = st.radio(
    "Метод поиска многомерных выбросов:",
    ["Isolation Forest (ML)", "Расстояние Махаланобиса (классический)"],
    horizontal=True,
    help="Isolation Forest — алгоритм машинного обучения, ищет нетипичные паттерны. "
         "Махаланобис — классический статистический метод, основанный на χ²-распределении. "
         "Для академической работы рекомендуется Махаланобис (его понимают рецензенты-психологи)."
)

with st.expander("ℹ️ Как это работает?"):
    if method_outlier == "Isolation Forest (ML)":
        st.markdown("""
        Алгоритм **Isolation Forest** ищет *многомерные* выбросы через построение случайных деревьев. 
        Например, высокий мотив 'Комфорт' — это нормально. Высокая 'Творческая активность' — тоже нормально. Но если у человека **оба** эти показателя зашкаливают (что психологически парадоксально), алгоритм пометит его как аномалию.
        
        Это **разведочный** инструмент — хорош для поиска интересных кейсов, но не имеет строгого статистического основания.
        """)
    else:
        st.markdown("""
        **Расстояние Махаланобиса** — классический статистический метод обнаружения многомерных выбросов.
        Для каждого респондента считается, насколько он удалён от центра «облака» данных, с учётом ковариаций между переменными.

        - Распределение D² ≈ χ² с df = k (число переменных)
        - **Строгий критерий:** p < 0.001 (рекомендуется для академической работы)
        - **Мягкий критерий:** p < 0.01 (для предварительного просмотра)

        Это **стандартный** метод для подготовки данных к регрессионному и факторному анализу 
        (Tabachnick & Fidell, 2019). Используется в психометрии и SEM.
        """)

num_cols = df.select_dtypes(include=np.number).columns.tolist()

# Инициализация сейфа для вкладки 8
if 'safe_cb_b_8' not in st.session_state: st.session_state.safe_cb_b_8 = False
if 'safe_cb_m_8' not in st.session_state: st.session_state.safe_cb_m_8 = False
if 'safe_cb_i_8' not in st.session_state: st.session_state.safe_cb_i_8 = False
if 'safe_anom_vars' not in st.session_state: st.session_state.safe_anom_vars = []

if 'cb_b_8' not in st.session_state: st.session_state.cb_b_8 = st.session_state.safe_cb_b_8
if 'cb_m_8' not in st.session_state: st.session_state.cb_m_8 = st.session_state.safe_cb_m_8
if 'cb_i_8' not in st.session_state: st.session_state.cb_i_8 = st.session_state.safe_cb_i_8
if 'anom_vars' not in st.session_state: st.session_state.anom_vars = st.session_state.safe_anom_vars

# Колбэки
def toggle_b_8():
    cols = [c for c in num_cols if c.startswith('B_')]
    if st.session_state.cb_b_8: st.session_state.anom_vars += [c for c in cols if c not in st.session_state.anom_vars]
    else: st.session_state.anom_vars = [c for c in st.session_state.anom_vars if c not in cols]

def toggle_m_8():
    cols = [c for c in num_cols if c.startswith('M_')]
    if st.session_state.cb_m_8: st.session_state.anom_vars += [c for c in cols if c not in st.session_state.anom_vars]
    else: st.session_state.anom_vars = [c for c in st.session_state.anom_vars if c not in cols]

def toggle_i_8():
    cols = [c for c in num_cols if c.startswith('IPL_')]
    if st.session_state.cb_i_8: st.session_state.anom_vars += [c for c in cols if c not in st.session_state.anom_vars]
    else: st.session_state.anom_vars = [c for c in st.session_state.anom_vars if c not in cols]

st.markdown("**1. Выберите шкалы для анализа на аномалии:**")
cb_a1, cb_a2, cb_a3 = st.columns(3)
with cb_a1: st.checkbox("Шкалы Братуся (Аномалии)", key='cb_b_8', on_change=toggle_b_8)
with cb_a2: st.checkbox("Шкалы Мильмана (Аномалии)", key='cb_m_8', on_change=toggle_m_8)
with cb_a3: st.checkbox("Шкалы ИПЛ (Аномалии)", key='cb_i_8', on_change=toggle_i_8)

anom_vars = st.multiselect("Анализируемые метрики:", num_cols, key="anom_vars", format_func=get_name)

# Параметры — отличаются для разных методов
if method_outlier == "Isolation Forest (ML)":
    contamination_label = "Какой процент выборки считать 'странным' (чувствительность)?"
    contamination = st.slider(contamination_label, min_value=1, max_value=15, value=5, step=1)
    mahal_threshold_label = None
else:
    mahal_threshold = st.radio(
        "Порог значимости для D²:",
        ["Строгий (p < 0.001)", "Умеренный (p < 0.01)"],
        index=0,
        horizontal=True,
        help="Строгий — стандарт для академических работ. Умеренный — для предварительного просмотра."
    )

# КНОПКА ПОИСКА (Сохраняем результаты в память сессии)
if st.button("🔍 Найти аномалии", type="primary"):
    if len(anom_vars) < 2:
        st.warning("⚠️ Для поиска многомерных аномалий нужно выбрать хотя бы 2 шкалы.")
    else:
        with st.spinner("Прочесываем данные в поисках аномалий..."):
            df_anom = df.dropna(subset=anom_vars).copy()

            if len(df_anom) < max(20, len(anom_vars) + 5):
                st.error(f"❌ Слишком мало данных. Нужно минимум {max(20, len(anom_vars) + 5)} наблюдений для {len(anom_vars)} переменных.")
            else:
                X_anom = df_anom[anom_vars]

                if method_outlier == "Isolation Forest (ML)":
                    # === ВЕТКА 1: Isolation Forest ===
                    iso = IsolationForest(contamination=contamination/100.0, random_state=42)
                    df_anom['Anomaly'] = iso.fit_predict(X_anom)
                    df_anom['Anomaly_Label'] = df_anom['Anomaly'].map({1: 'Норма', -1: 'Аномалия'})
                    df_anom['Anomaly_Score'] = iso.decision_function(X_anom)
                    df_anom['mahal_d2'] = np.nan
                    df_anom['mahal_p'] = np.nan
                else:
                    # === ВЕТКА 2: Махаланобис ===
                    mahal_result = calculate_mahalanobis_distances(X_anom)
                    df_anom = df_anom.join(mahal_result, how='left')

                    if mahal_threshold == "Строгий (p < 0.001)":
                        df_anom['Anomaly'] = df_anom['is_outlier_strict'].map({True: -1, False: 1})
                    else:
                        df_anom['Anomaly'] = df_anom['is_outlier_moderate'].map({True: -1, False: 1})

                    df_anom['Anomaly_Label'] = df_anom['Anomaly'].map({1: 'Норма', -1: 'Аномалия'})
                    # Anomaly_Score: чем больше D², тем более выбросовый (для сортировки)
                    # Инвертируем знак чтобы отрицательные = выбросы (как в Isolation Forest)
                    df_anom['Anomaly_Score'] = -df_anom['mahal_d2']

                if 'FIO' in df_anom.columns:
                    df_anom['Display_Name'] = df_anom['FIO'].fillna("Аноним") + " (ID: " + df_anom.index.astype(str) + ")"
                else:
                    df_anom['Display_Name'] = "Респондент ID: " + df_anom.index.astype(str)

                # PCA для визуализации (2D) — общая для обоих методов
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_anom)
                pca_anom = PCA(n_components=2)
                components = pca_anom.fit_transform(X_scaled)

                df_anom['PCA1'] = components[:, 0]
                df_anom['PCA2'] = components[:, 1]

                st.session_state['anom_results'] = {
                    'df_anom': df_anom,
                    'anom_vars': anom_vars,
                    'method_used': method_outlier,
                }

# ОТРИСОВКА ИНТЕРФЕЙСА (Берем данные из памяти)
if 'anom_results' in st.session_state:
    res_a = st.session_state['anom_results']
    df_a = res_a['df_anom']
    a_vars = res_a['anom_vars']
    method_used = res_a.get('method_used', 'Isolation Forest (ML)')

    st.markdown("---")
    st.caption(f"Использованный метод: **{method_used}**")
    c_res1, c_res2 = st.columns([2, 1])

    with c_res1:
        st.markdown("#### 🌌 Карта респондентов (Проекция)")
        
        fig_anom = px.scatter(
            df_a, x='PCA1', y='PCA2', color='Anomaly_Label',
            color_discrete_map={'Норма': '#3498db', 'Аномалия': '#e74c3c'},
            hover_name='Display_Name',
            hover_data={'PCA1': False, 'PCA2': False, 'Anomaly_Label': False},
            opacity=0.8
        )
        fig_anom.update_layout(
            title=dict(text="Красные точки — многомерные выбросы", font=dict(size=14)),
            xaxis_title="Скрытая компонента 1", yaxis_title="Скрытая компонента 2",
            legend_title="", margin=dict(l=10, r=10, t=40, b=10), height=400
        )
        fig_anom.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_anom, use_container_width=True, config={'displayModeBar': False})
        
    with c_res2:
        num_anom = len(df_a[df_a['Anomaly'] == -1])
        st.metric("Найдено аномальных профилей", f"{num_anom} чел.")
        st.markdown("На графике слева ИИ сжал все выбранные шкалы в 2D-пространство. **Красные точки**, оторванные от синего облака — это респонденты с нетипичным мышлением.")
        st.caption("Наведите мышку на точку, чтобы увидеть, кто это.")
        
    st.markdown("#### 🕵️‍♂️ Досье на нетипичных респондентов")
    outliers = df_a[df_a['Anomaly'] == -1].sort_values('Anomaly_Score')

    # Если использовался Махаланобис — показываем таблицу с D² и p-values
    if method_used == "Расстояние Махаланобиса (классический)" and not outliers.empty:
        st.markdown("##### 📋 Сводка по выбросам (D² и p-value)")
        mahal_summary = outliers[['Display_Name', 'mahal_d2', 'mahal_p']].copy()
        mahal_summary.columns = ['Респондент', 'D² (Махаланобис)', 'p-value (χ²)']
        mahal_summary['D² (Махаланобис)'] = mahal_summary['D² (Махаланобис)'].round(2)
        mahal_summary['p-value (χ²)'] = mahal_summary['p-value (χ²)'].apply(
            lambda p: f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
        )
        st.dataframe(mahal_summary, use_container_width=True, hide_index=True)
        st.caption(
            f"💡 Расстояния D² подчиняются распределению χ² с df={len(a_vars)}. "
            f"Чем больше D², тем сильнее респондент отклонён от центра выборки."
        )
        st.markdown("---")
    
    if outliers.empty:
        st.success("При заданных настройках ярких аномалий не найдено. Выборка очень однородна.")
    else:
        st.markdown("---")
        st.markdown("### 🔬 Рентген аномалии (В чем их странность?)")
        st.markdown("Выберите респондента из списка аномалий. Алгоритм сравнит его с «нормальной» частью выборки и покажет, какие шкалы у него зашкаливают.")
        
        selected_anom = st.selectbox("Выберите респондента для анализа:", outliers['Display_Name'].tolist())
        
        if selected_anom:
            person = outliers[outliers['Display_Name'] == selected_anom].iloc[0]
            normal_df = df_a[df_a['Anomaly'] == 1]
            
            deviations = []
            for col in a_vars:
                val = person[col]
                mean_norm = normal_df[col].mean()
                std_norm = normal_df[col].std()
                
                z_score = (val - mean_norm) / std_norm if std_norm > 0 else 0
                    
                deviations.append({
                    'Шкала': get_name(col),
                    'Балл респондента': val,
                    'Среднее по норме': round(mean_norm, 1),
                    'Отклонение (Z)': z_score,
                    'Абс_Отклонение': abs(z_score) 
                })
            
            dev_df = pd.DataFrame(deviations).sort_values(by='Абс_Отклонение', ascending=False).head(5)
            dev_df['Направление'] = dev_df['Отклонение (Z)'].apply(lambda x: 'Выше нормы' if x > 0 else 'Ниже нормы')
            
            fig_dev = px.bar(
                dev_df, x='Отклонение (Z)', y='Шкала', orientation='h',
                color='Направление',
                color_discrete_map={'Выше нормы': '#27ae60', 'Ниже нормы': '#e74c3c'},
                text='Балл респондента'
            )
            
            fig_dev.update_layout(
                title=dict(text=f"Топ-5 экстремальных отклонений: {selected_anom}", font=dict(size=16)),
                xaxis_title="Сила отклонения (в стандартных отклонениях Z)",
                yaxis_title="", height=350, margin=dict(l=10, r=10, t=40, b=10)
            )
            fig_dev.add_vline(x=0, line_width=2, line_color="black", opacity=0.5)
            
            st.plotly_chart(fig_dev, use_container_width=True, config={'displayModeBar': False})
            st.info("💡 **Как читать этот график:** Черная вертикальная линия (0) — это средний балл большинства людей. Зеленые столбцы показывают, что показатель человека аномально **завышен**. Красные — аномально **занижен**. Число на столбце — это сырой балл респондента по методике.")
        
        with st.expander("Посмотреть сырые данные всех аномальных респондентов (Таблица)"):
            cols_to_show = []
            if 'FIO' in df_a.columns: cols_to_show.append('FIO')
            if 'Gender' in df_a.columns: cols_to_show.append('Gender')
            cols_to_show.extend(a_vars)
            
            renamed_outliers = outliers[cols_to_show].rename(columns={c: get_name(c) for c in a_vars})
            st.dataframe(renamed_outliers, use_container_width=True)

# Сохранение бэкапа 
st.session_state.safe_cb_b_8 = st.session_state.cb_b_8
st.session_state.safe_cb_m_8 = st.session_state.cb_m_8
st.session_state.safe_cb_i_8 = st.session_state.cb_i_8
st.session_state.safe_anom_vars = st.session_state.anom_vars