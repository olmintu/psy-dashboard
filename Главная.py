import streamlit as st
from utils import load_data, render_sidebar, init_session_state

st.set_page_config(page_title="Анализ Психодиагностики Pro", layout="wide", page_icon="🧠")

# 1. ИНИЦИАЛИЗАЦИЯ (сначала готовим данные в памяти)
init_session_state()

# 2. ОТРИСОВКА САЙДБАРА (теперь он сразу увидит данные из init_session_state)
df = render_sidebar()

st.title("🧠 Аналитическая система: Психодиагностика v3.0")

st.markdown("### 📥 Шаг 1. Загрузка данных")

# Если данные уже в памяти (неважно, демо или свои)
if st.session_state['df_raw'] is not None:
    if st.session_state.get('is_demo', False):
        st.warning(f"⚠️ **ДЕМО-РЕЖИМ.** Загружены тестовые данные (TEST_RESULTS.xlsx).")
    else:
        st.success(f"✅ Данные успешно загружены! (Выборка: {len(st.session_state['df_raw'])} чел.)")
    
    if st.button("🗑️ Очистить память и загрузить свой файл"):
        st.session_state['df_raw'] = None
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = True 
        st.rerun() 
else:
    # Если памяти пусто (после очистки)
    uploaded_file = st.file_uploader("Загрузить свой файл с ответами (XLSX)", type=['xlsx'])
    if uploaded_file:
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = False 
        st.rerun()

    if st.button("🔄 Вернуть демо-данные"):
        st.session_state['disable_auto_demo'] = False
        st.rerun()

# Приветствие
if st.session_state['df_raw'] is not None:
    st.markdown("---")
    st.markdown("### 🚀 Данные готовы к работе!")
    st.info("👈 Используйте навигационное меню в левой панели.")