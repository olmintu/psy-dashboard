import streamlit as st
from utils import load_data, render_sidebar

st.set_page_config(page_title="Анализ Психодиагностики Pro", layout="wide", page_icon="🧠")

# Инициализируем флаг запрета автозагрузки демо-данных, если его еще нет
if 'disable_auto_demo' not in st.session_state:
    st.session_state['disable_auto_demo'] = False

# 1. Вызываем сайдбар
df = render_sidebar()

st.title("🧠 Аналитическая система: Психодиагностика v3.0")

st.markdown("### 📥 Шаг 1. Загрузка данных")

# 2. Проверяем, есть ли УЖЕ данные в глобальной памяти
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    
    # Логика отображения статуса
    if st.session_state.get('is_demo', False):
        st.warning(f"⚠️ **ДЕМО-РЕЖИМ.** Загружены тестовые данные (TEST_RESULTS.xlsx). Размер выборки: {len(st.session_state['df_raw'])} чел.")
    else:
        st.success(f"✅ Данные успешно загружены! (Размер выборки: {len(st.session_state['df_raw'])} чел.)")
    
    # Кнопка сброса
    if st.button("🗑️ Очистить память и загрузить свой файл"):
        st.session_state['df_raw'] = None
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = True # ЗАПРЕЩАЕМ автоматический подхват демо-файла
        st.rerun() 
        
else:
    # 3. Если данных нет, показываем загрузчик
    uploaded_file = st.file_uploader("Загрузить свой файл с ответами (XLSX)", type=['xlsx'])

    if uploaded_file:
        st.session_state['df_raw'] = load_data(uploaded_file)
        st.session_state['is_demo'] = False
        st.session_state['disable_auto_demo'] = False # Сбрасываем запрет, так как файл загружен вручную
        st.rerun() 
        
    # 4. Пытаемся загрузить демо-файл только если это не было запрещено кнопкой "Очистить"
    elif not st.session_state['disable_auto_demo']:
        try:
            import os
            if os.path.exists('TEST_RESULTS.xlsx'):
                local_df = load_data('TEST_RESULTS.xlsx')
                if local_df is not None:
                    st.session_state['df_raw'] = local_df
                    st.session_state['is_demo'] = True
                    st.rerun()
        except:
            pass

# 5. Приветствие и инструкции
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    st.markdown("---")
    st.markdown("### 🚀 Данные готовы к работе!")
    st.info("👈 Используйте **навигационное меню в левой панели**, чтобы переключаться между модулями аналитики.")
else:
    st.info("💡 Вы очистили память. Теперь вы можете перетащить свой собственный файл в поле выше.")
    # Добавим кнопку ручного возврата демо-режима на случай, если пользователь передумает
    if st.button("🔄 Вернуть демо-данные"):
        st.session_state['disable_auto_demo'] = False
        st.rerun()
