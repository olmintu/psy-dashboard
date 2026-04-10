import streamlit as st
from utils import load_data, render_sidebar

st.set_page_config(page_title="Анализ Психодиагностики Pro", layout="wide", page_icon="🧠")

# 1. ВЫЗЫВАЕМ САЙДБАР СРАЗУ (Чтобы кнопка руководства появилась мгновенно)
df = render_sidebar()

st.title("🧠 Аналитическая система: Психодиагностика v3.0")

st.markdown("### 📥 Шаг 1. Загрузка данных")

# 2. Проверяем, есть ли УЖЕ данные в глобальной памяти
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    st.success(f"✅ Данные успешно загружены и находятся в памяти (Размер выборки: {len(st.session_state['df_raw'])} чел.)")
    
    # Даем кнопку, чтобы можно было удалить файл из памяти и загрузить новый
    if st.button("🗑️ Очистить память и загрузить другой файл"):
        st.session_state['df_raw'] = None
        st.rerun() # Мгновенная перезагрузка страницы
        
else:
    # 3. Если данных в памяти нет, показываем загрузчик
    uploaded_file = st.file_uploader("Загрузить файл с ответами (XLSX)", type=['xlsx'])

    if uploaded_file:
        # Сохраняем в память
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.rerun() # Перезагружаем, чтобы применить данные
    else:
        # Попытка загрузить локальный файл по умолчанию
        try:
            local_df = load_data('FINAL_RESULTS.xlsx')
            if local_df is not None:
                st.session_state['df_raw'] = local_df
                st.rerun()
        except:
            pass

# 4. Если данные загружены, выводим приветствие
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    st.markdown("---")
    st.markdown("### 🚀 Данные готовы к работе!")
    st.info("👈 Используйте **навигационное меню в левой панели** (выше фильтров), чтобы переключаться между модулями аналитики.")
else:
    st.warning("👈 Пожалуйста, загрузите Excel-файл для начала работы.")