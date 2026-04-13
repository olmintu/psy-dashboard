import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import render_sidebar, get_name, get_descriptive_stats 

st.set_page_config(page_title="Обзор выборки", layout="wide", page_icon="📊")

# 1. ДОСТАЕМ ОТФИЛЬТРОВАННЫЕ ДАННЫЕ ИЗ ПАМЯТИ
df = render_sidebar()

# Если файла нет, страница просто остановится и попросит его загрузить
if df is None:
    st.stop()

# 2. ЧИСТЫЙ КОД ВКЛАДКИ
st.header("Обзор выборки")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_n = len(df)
mean_age = df['Age'].mean() if 'Age' in df.columns else 0

if 'Gender' in df.columns and not df['Gender'].empty:
    top_gender = df['Gender'].mode()[0]
    top_gender_pct = (df['Gender'] == top_gender).sum() / total_n * 100
    gender_text = f"{top_gender} ({top_gender_pct:.0f}%)"
else:
    gender_text = "Н/Д"
    
if 'Work' in df.columns and not df['Work'].empty:
    working_pct = (df['Work'].astype(str).str.contains('Да', case=False, na=False)).sum() / total_n * 100
    work_text = f"{working_pct:.0f}%"
else:
    work_text = "Н/Д"

kpi1.metric("👥 Всего респондентов", f"{total_n} чел.")
kpi2.metric("🎂 Средний возраст", f"{mean_age:.1f} лет")
kpi3.metric("⚧ Преобладающий пол", gender_text)
kpi4.metric("💼 Работающих", work_text)

st.markdown("---")

tab1_demo, tab1_stats, tab1_cross = st.tabs([
    "👥 Социально-демографический портрет", 
    "📈 Статистика и Распределения", 
    "🔀 Кросс-табуляция (Срезы)"
])

with tab1_demo:
    st.subheader("Социально-демографическая структура")
    c1, c2 = st.columns([1, 1.2]) 
    
    with c1:
        st.markdown("#### Возрастной состав")
        if 'Age' in df.columns:
            fig_age = px.histogram(df, x='Age', nbins=15, color_discrete_sequence=['#3498db'])
            fig_age.update_layout(yaxis_title="Количество чел.", xaxis_title="Возраст", margin=dict(t=20, b=10))
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("Данные о возрасте отсутствуют.")
            
    with c2:
        st.markdown("#### Конструктор демографии")
        demo_dict = {
            'Gender': 'Пол', 'Family': 'Семейное положение', 'Children': 'Наличие детей',
            'Work': 'Трудоустройство', 'Edu_Status': 'Статус обучения', 'Edu_Level': 'Уровень образования', 'Is_KMNS': 'Принадлежность к КМНС'
        }
        available_demo = {k: v for k, v in demo_dict.items() if k in df.columns}
        
        if available_demo:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1: sel_var = st.selectbox("1. Показатель:", list(available_demo.keys()), format_func=lambda x: available_demo[x])
            with col_sel2: sel_viz = st.selectbox("2. Вид графика:", ["Круговая диаграмма (Pie)", "Вертикальные столбцы (Bar)", "Горизонтальные столбцы (Bar)", "Древовидная карта (Treemap)"])
            
            var_counts = df[sel_var].value_counts().reset_index()
            var_counts.columns = ['Категория', 'Количество']
            
            if sel_viz == "Круговая диаграмма (Pie)":
                fig_dyn = px.pie(var_counts, names='Категория', values='Количество', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_dyn.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig_dyn.update_layout(showlegend=False, margin=dict(t=20, b=10))
            elif sel_viz == "Вертикальные столбцы (Bar)":
                fig_dyn = px.bar(var_counts, x='Категория', y='Количество', text='Количество', color='Категория', color_discrete_sequence=px.colors.qualitative.Set2)
                fig_dyn.update_traces(textposition='outside', textfont_size=14)
                fig_dyn.update_layout(showlegend=False, xaxis_title="", yaxis_title="Количество чел.", margin=dict(t=20, b=10))
            elif sel_viz == "Горизонтальные столбцы (Bar)":
                fig_dyn = px.bar(var_counts, y='Категория', x='Количество', orientation='h', text='Количество', color='Категория', color_discrete_sequence=px.colors.qualitative.Set2)
                fig_dyn.update_traces(textposition='outside', textfont_size=14)
                fig_dyn.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'}, yaxis_title="", xaxis_title="Количество чел.", margin=dict(t=20, b=10))
            elif sel_viz == "Древовидная карта (Treemap)":
                fig_dyn = px.treemap(var_counts, path=['Категория'], values='Количество', color='Количество', color_continuous_scale='Blues')
                fig_dyn.update_traces(textinfo="label+value+percent entry", textfont_size=14)
                fig_dyn.update_layout(margin=dict(t=20, b=10, l=10, r=10))
            
            st.plotly_chart(fig_dyn, use_container_width=True)
        else:
            st.info("Нет доступных категориальных данных для построения.")

with tab1_stats:
    st.subheader("Описательная статистика и форма распределения")
    c1_s, c2_s = st.columns([1, 1.2]) 
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    default_cols = [c for c in ['Age', 'IPL_Total', 'M_Sum_Zh', 'B_Altruistic'] if c in df.columns]
    
    with c1_s:
        st.markdown("#### 🎻 Визуализация распределения")
        var_to_plot = st.selectbox("Выберите показатель:", numeric_cols, format_func=get_name)
        plot_type = st.radio("Тип графика:", ["Violin (Скрипичный)", "Boxplot (Ящик с усами)", "Гистограмма"], horizontal=True)
        
        if var_to_plot:
            if plot_type == "Violin (Скрипичный)": fig_dist = px.violin(df, y=var_to_plot, box=True, points="all", title=get_name(var_to_plot), color_discrete_sequence=['#9b59b6'])
            elif plot_type == "Boxplot (Ящик с усами)": fig_dist = px.box(df, y=var_to_plot, points="all", title=get_name(var_to_plot), color_discrete_sequence=['#e67e22'])
            else: fig_dist = px.histogram(df, x=var_to_plot, marginal="box", title=get_name(var_to_plot), color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig_dist, use_container_width=True)

    with c2_s:
        st.markdown("#### 📋 Подробная статистика")
        sel_stat_cols = st.multiselect("Выберите показатели для таблицы:", numeric_cols, default=default_cols, format_func=get_name)
        if sel_stat_cols:
            stats_df = get_descriptive_stats(df, sel_stat_cols)
            formatted_df = stats_df.copy()
            for c in ['Среднее', 'SD', 'Медиана', 'Skew', 'Kurtosis']:
                if c in formatted_df.columns: formatted_df[c] = formatted_df[c].round(2)
            st.dataframe(formatted_df.style.background_gradient(cmap='Blues', subset=['Среднее', 'SD']), use_container_width=True, hide_index=True)
            st.caption("ℹ️ **Skew**: 0 = симметрично. **Kurtosis**: >0 = острый пик. **p > 0.05** = нормальное распределение.")

with tab1_cross:
    st.subheader("Кросс-табуляция (пересечение признаков)")
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 10]
    cross_c1, cross_c2 = st.columns(2)
    with cross_c1: cross_x = st.selectbox("1. Разбить столбцы по (Ось X):", cat_cols, index=0)
    with cross_c2: cross_y = st.selectbox("2. Заливка цветом (Группы):", cat_cols, index=1 if len(cat_cols)>1 else 0)
        
    if cross_x and cross_y:
        st.markdown("---")
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.markdown("#### 📊 Структура в процентах")
            ct_pct = pd.crosstab(df[cross_x], df[cross_y], normalize='index') * 100
            fig_cross = px.bar(ct_pct, barmode="stack", title=f"Из чего состоит '{cross_x}' (%)", labels={'value': '%', cross_x: cross_x}, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_cross, use_container_width=True)
        with col_x2:
            st.markdown("#### 🔢 Таблица абсолютных значений")
            ct = pd.crosstab(df[cross_y], df[cross_x], margins=True, margins_name="Всего")
            st.dataframe(ct.style.background_gradient(cmap='YlGnBu', axis=None), use_container_width=True)