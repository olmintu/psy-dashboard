import streamlit as st
from utils import load_data, render_sidebar, init_session_state

st.set_page_config(page_title="Анализ Психодиагностики Pro", layout="wide", page_icon="🧠")

# Функция для сброса старых фильтров при загрузке нового файла
def reset_filters():
    keys_to_clear = ['f_gender', 'f_age', 'f_work', 'f_edu', 'f_kmns','f_fast', 'f_extra']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# 1. ИНИЦИАЛИЗАЦИЯ
init_session_state()

# 2. ОТРИСОВКА САЙДБАРА
df = render_sidebar()

st.title("🧠 Аналитическая система: Психодиагностика v3.0")
st.markdown("### 📥 Шаг 1. Загрузка данных")

# Если данные уже в памяти
if st.session_state['df_raw'] is not None:
    if st.session_state.get('is_demo', False):
        st.warning("⚠️ **ДЕМО-РЕЖИМ.** Загружены тестовые данные (TEST_RESULTS.xlsx).")
    else:
        st.success(f"✅ Данные успешно загружены! (Выборка: {len(st.session_state['df_raw'])} чел.)")
    
    if st.button("🗑️ Очистить память и загрузить свой файл"):
        st.session_state['df_raw'] = None
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = True 
        reset_filters() # СБРАСЫВАЕМ ФИЛЬТРЫ!
        st.rerun() 
else:
    # Если памяти пусто
    uploaded_file = st.file_uploader("Загрузить свой файл с ответами (XLSX)", type=['xlsx'])
    if uploaded_file:
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = False 
        reset_filters() # СБРАСЫВАЕМ ФИЛЬТРЫ!
        st.rerun()

    if st.button("🔄 Вернуть демо-данные"):
        st.session_state['disable_auto_demo'] = False
        reset_filters() # СБРАСЫВАЕМ ФИЛЬТРЫ!
        st.rerun()

# Приветствие
if st.session_state['df_raw'] is not None:
    st.markdown("---")
    st.markdown("### 🚀 Данные готовы к работе!")
    st.info("👈 Используйте навигационное меню в левой панели.")