import streamlit as st
from utils import load_data, render_sidebar

st.set_page_config(page_title="Анализ Психодиагностики Pro", layout="wide", page_icon="🧠")

# 1. Вызываем сайдбар
df = render_sidebar()

st.title("🧠 Аналитическая система: Психодиагностика v3.0")

st.markdown("### 📥 Шаг 1. Загрузка данных")

# 2. Проверяем, есть ли УЖЕ данные в глобальной памяти
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    
    # --- ЛОГИКА ДЕМО-РЕЖИМА ---
    if st.session_state.get('is_demo', False):
        st.warning(f"⚠️ **ВНИМАНИЕ: Активирован ДЕМО-РЕЖИМ.** Загружены тестовые данные (TEST_RESULTS.xlsx) для демонстрации работы алгоритмов. Все совпадения случайны. Размер выборки: {len(st.session_state['df_raw'])} чел.")
    else:
        st.success(f"✅ Реальные данные успешно загружены и находятся в памяти (Размер выборки: {len(st.session_state['df_raw'])} чел.)")
    
    # Кнопка сброса
    if st.button("🗑️ Очистить память и загрузить свой файл"):
        st.session_state['df_raw'] = None
        st.session_state['is_demo'] = False # Сбрасываем флаг
        st.rerun() 
        
else:
    # 3. Если данных нет, показываем загрузчик
    uploaded_file = st.file_uploader("Загрузить свой файл с ответами (XLSX)", type=['xlsx'])

    if uploaded_file:
        # Сохраняем РЕАЛЬНЫЙ файл в память
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.session_state['is_demo'] = False # Это реальные данные
        st.rerun() 
    else:
        # Попытка загрузить тестовый файл по умолчанию (ДЕМО-РЕЖИМ)
        try:
            local_df = load_data('TEST_RESULTS.xlsx')
            if local_df is not None:
                st.session_state['df_raw'] = local_df
                st.session_state['is_demo'] = True # Включаем флаг демо-режима!
                st.rerun()
        except:
            pass

# 4. Приветствие
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    st.markdown("---")
    st.markdown("### 🚀 Данные готовы к работе!")
    st.info("👈 Используйте **навигационное меню в левой панели** (выше фильтров), чтобы переключаться между модулями аналитики.")
else:
    st.warning("👈 Пожалуйста, загрузите Excel-файл для начала работы.")