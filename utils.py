import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score
import networkx as nx
import io

# ==========================================
# МОДУЛЬ СПРАВКИ (Интерактивный и подробный)
# ==========================================
@st.dialog("📖 Подробное руководство пользователя", width="large")
def show_help_dialog():
    menu_col, content_col = st.columns([1, 2.8])
    
    # Железобетонный список разделов
    topics = [
        "📥 1. Загрузка и Фильтры", "📊 2. Обзор выборки", "🧩 3. Анализ методик",
        "🆚 4. Сравнение групп", "🔗 5. Корреляции", "🔬 6. Кластерный анализ",
        "📐 7. Факторный анализ", "🔮 8. Поиск драйверов", "👽 9. Детектор аномалий", "🕸️ 10. Сетевой анализ"
    ]
    
    with menu_col:
        st.markdown("### 📚 Разделы:")
        selected_topic = st.radio("Навигация по справке", topics, label_visibility="collapsed")
        
    with content_col:
        if selected_topic == topics[0]:
            st.markdown("### 📥 Загрузка данных и Фильтрация")
            st.info("💡 **Основной принцип:** Любые изменения в левой боковой панели автоматически пересчитывают данные на всех вкладках дашборда.")
            
            st.markdown("#### Пошаговая инструкция:")
            st.markdown("""
            1. **Загрузка файла:** Перейдите на страницу «Главная». Перетащите ваш файл Excel (`.xlsx`) в поле загрузки.
            2. **Настройка фильтров:** Выберите нужные параметры в левой панели (например, Пол: *Мужской*, Возраст: *18-25*). Можно оставить поля пустыми, чтобы анализировать всех.
            """)
            
            st.markdown("3. **Применение:** Обязательно нажмите синюю кнопку ниже. Без этого данные не обновятся!")
            st.button("Применить фильтры (Пример кнопки)", type="primary", disabled=True)
            
            st.markdown("""
            4. **Выгрузка:** Чтобы скачать отфильтрованную таблицу для работы в других программах, нажмите кнопку скачивания в самом низу панели.
            """)
            st.download_button("📥 Скачать отфильтрованные данные (Пример)", data=b"example", disabled=True)

        elif selected_topic == topics[1]:
            st.markdown("### 📊 Вкладка 1: Обзор выборки")
            st.markdown("Раздел для анализа социально-демографического состава и распределения баллов.")
            
            st.markdown("#### 📌 Пример панели показателей (KPI):")
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("👥 Всего респондентов", "100 чел.")
            kpi2.metric("🎂 Средний возраст", "22.4 года")
            kpi3.metric("⚧ Преобладающий пол", "Женщины (65%)")
            
            st.markdown("#### 🛠️ Доступные инструменты:")
            st.markdown("""
            * **Конструктор демографии:** Выберите признак из выпадающего списка, чтобы построить круговую (Pie) или столбчатую (Bar) диаграмму, а также древовидную карту (Treemap).
            * **Анализ распределений:**
              * *Скрипичный график (Violin):* Показывает плотность ответов. Широкая часть — там, где ответов больше всего.
              * *Ящик с усами (Boxplot):* Точка в центре — медиана. Границы ящика — основная масса людей (50%). Точки за усами — выбросы.
            * **Кросс-табуляция:** Выберите два признака (Строки и Колонки), чтобы получить сводную таблицу (например, распределение семейного положения по полу).
            """)

        elif selected_topic == topics[2]:
            st.markdown("### 🧩 Вкладка 2: Анализ методик")
            st.markdown("Детальный разбор психологических шкал (Братусь, Мильман, ИПЛ).")
            
            st.warning("⚡ **Режимы работы:** Вверху страницы выберите, чьи данные вы хотите видеть: всей группы (средние значения) или конкретного человека (выбор по ФИО/ID).")
            
            st.markdown("#### 🔍 Как читать графики (на примере Мильмана):")
            st.markdown("""
            * 🟢 **Зеленая пунктирная линия:** Желаемое (Идеал).
            * 🔴 **Красная сплошная линия:** Действительное (Реальность).
            """)
            
            st.markdown("""
            <div style="padding:15px; border-left: 5px solid #e74c3c; background-color: #fadbd8; border-radius: 5px; margin-bottom: 15px;">
                <b>🔥 Зоны фрустрации (Красная заливка):</b><br>
                Разрыв: Чем дальше красная точка от зеленой на одной шкале, тем выше уровень неудовлетворенности (фрустрации) в этой сфере. Система автоматически рассчитает шкалу с максимальной болью.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Братусь (Жизненные смыслы):** Чем короче полоса (меньше сумма рангов), тем важнее смысл для человека.\n\n**ИПЛ:** Анализ общего балла, структуры (Г-А-П) и типов реализации инновационного потенциала.")

        elif selected_topic == topics[3]:
            st.markdown("### 🆚 Вкладка 3: Сравнение групп")
            st.markdown("Автоматический поиск статистически значимых различий (p-value < 0.05).")
            
            st.info("⚙️ **Под капотом:** Система сама проверяет нормальность распределения и выбирает критерий (t-Стьюдента или U-Манна-Уитни для 2 групп; ANOVA или Kruskal-Wallis для 3+ групп). Вам не нужно знать математику.")
            
            st.markdown("#### 💡 Авто-поиск всех различий (Сканер)")
            st.markdown("""
            1. Выберите **Группирующий признак** (например, Пол).
            2. Настройте **уровень значимости** (α). По умолчанию 0.05 — стандарт в психологии.
            3. Нажмите **«Начать сканирование»**.
            4. **Результат:** таблица только тех шкал, по которым группы достоверно отличаются.
               Отсортирована по силе эффекта (Cohen's d или η²).
            5. **Скачайте полный отчёт** в Excel — в него попадают *все* шкалы, включая незначимые,
               что удобно для отчёта по исследованию.
            """)
            
            st.markdown("#### Как читать результат:")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h4 style='text-align: center; color: #27ae60;'>p < α</h4>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'><b>Различия есть.</b> Это не случайность.</p>", unsafe_allow_html=True)
            with c2:
                st.markdown("<h4 style='text-align: center; color: #e74c3c;'>p > α</h4>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'><b>Различий нет.</b> Группы статистически одинаковы.</p>", unsafe_allow_html=True)

        elif selected_topic == topics[4]:
            st.markdown("### 🔗 Вкладка 4: Корреляции")
            st.markdown("Поиск взаимосвязей между шкалами (от -1.0 до 1.0).")
            
            st.markdown("""
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div style="flex: 1; padding: 15px; background: #ebf5fb; border-radius: 10px;">
                    <h4 style="color: #2980b9; margin-top: 0;">🔵 Синий цвет (r < 0)</h4>
                    <b>Обратная связь.</b> Одно растет — второе падает. <i>(Например: Стресс и Комфорт)</i>
                </div>
                <div style="flex: 1; padding: 15px; background: #fdedec; border-radius: 10px;">
                    <h4 style="color: #c0392b; margin-top: 0;">🔴 Красный цвет (r > 0)</h4>
                    <b>Прямая связь.</b> Растут и падают вместе. <i>(Например: Активность и Творчество)</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🔍 Инструмент: Фильтр связей (Внизу страницы)")
            st.markdown("""
            Не пытайтесь искать цифры глазами на огромной карте!
            1. Выберите силу связи (например, `Сильная 0.5-0.7`).
            2. Задайте уровень значимости (обычно `p < 0.05`).
            3. Получите чистую таблицу только с самыми сильными и надежными корреляциями, готовую для скачивания в Excel.
            * **🔀 Межметодические связи:** Используйте эту галочку, чтобы мгновенно скрыть корреляции внутри одного теста. Это позволит вам увидеть «чистую» психологическую картину: как смыслы из одной методики влияют на результаты другой.
            """)

        elif selected_topic == topics[5]:
            st.markdown("### 🔬 Вкладка 5: Кластерный анализ")
            st.markdown("Разделение людей на схожие типажи с помощью машинного обучения. "
                        "Оба метода (иерархический и K-Means) теперь показывают **коэффициент силуэта** — "
                        "индикатор качества разбиения от −1 до 1 (чем выше, тем чётче разделены группы).")

            st.info("""
            **📏 Как читать силуэт:**
            🟢 **≥ 0.50** — сильная кластерная структура
            🟡 **0.25–0.50** — умеренная структура
            🟠 **0.15–0.25** — слабая (интерпретировать осторожно)
            🔴 **< 0.15** — естественной структуры нет, разбиение только описательное
            """)

            st.markdown("#### 🌳 Иерархическая кластеризация (Дендрограммы)")
            st.markdown("""Доступны три режима:

* **Кластеризация шкал** — показывает, какие психологические шкалы ведут себя похоже.
* **Кластеризация респондентов** — типажи людей; можно задать число веток ползунком и получить:
  - Коэффициент силуэта для выбранного разбиения
  - Профили групп с автоматическим расчётом p-value (тест выбирается автоматически)
  - Боксплоты ключевых показателей ИПЛ по группам
  - Интерактивный HTML-отчёт с подсветкой при наведении
  - Скачивание Excel: отдельно профили+p-value, отдельно состав групп
* **Clustergram** — тепловая карта «респонденты × шкалы» с дендрограммами по обеим осям.
  Самый информативный формат: сразу видно, какие группы людей имеют какой профиль и по каким шкалам они различаются.""")

            st.markdown("#### 🎯 K-Means (Авто-выбор)")
            st.markdown("""
            * Выберите шкалы — алгоритм рассчитает силуэт для k от 2 до 10 и предложит **оптимальное количество кластеров**.
            * **Числовой силуэт** для выбранного k с цветовой интерпретацией качества.
            * Карта кластеров (PCA) и радар профилей.
            * **Таблица средних значений с p-value** — показывает, по каким именно шкалам кластеры значимо различаются.
            * **Боксплоты ИПЛ** с разбивкой по кластерам.
            * Скачивание Excel — профили и состав групп (с ID, ФИО, Полом и другими контекстными колонками).
            """)

        elif selected_topic == topics[6]:
            st.markdown("### 📐 Вкладка 6: Факторный анализ и надёжность")
            st.markdown("Поиск латентной структуры данных + проверка согласованности шкал.")

            st.markdown("#### 🧬 1. Факторная структура (главное)")
            st.markdown("""
            **Три слоя проверки** перед запуском FA:

            1. **📏 Размер выборки vs переменные** — соотношение n/vars (правило не менее 5:1 для надёжных результатов).
            2. **🔬 KMO** — мера адекватности данных (норма > 0.6). Низкое KMO = у шкал нет общей дисперсии для факторного анализа.
            3. **🔬 Критерий Бартлетта** — p-value должен быть < 0.05, иначе шкалы не коррелируют и факторов искать бессмысленно.

            Все три проверки должны пройти, чтобы факторный анализ имел смысл.

            **Внутренние вкладки:**
            * **PCA** — быстрая первичная компоновка; scree plot + критерий Кайзера (λ > 1) подскажут число факторов.
            * **EFA** — полноценный факторный анализ с выбором метода извлечения (Minres/ML/Principal Axis) и вращения (Varimax/Promax/Oblimin).

            Ищите нагрузки **|λ| ≥ 0.4** — они показывают, какие шкалы вошли в каждый латентный фактор.
            """)

            st.markdown("#### 🔗 2. Альфа Кронбаха")
            st.markdown("Проверка внутренней согласованности набора шкал.")
            st.progress(75)
            st.caption("Норма: α > 0.7. Если показатель высокий, вы имеете право объединить выбранные шкалы в один суммарный индекс.")
            st.warning("⚠️ В дашборде загружены финальные баллы по шкалам, а не ответы на отдельные вопросы. "
                       "Поэтому α здесь измеряет **макро-согласованность** (насколько шкалы ведут себя как части одного конструкта), "
                       "а не классическую надёжность отдельных пунктов.")

        elif selected_topic == topics[7]:
            st.markdown("### 🔮 Вкладка 7: Поиск драйверов (ИИ)")
            st.markdown("Определяет, какие факторы сильнее всего влияют на выбранный целевой показатель с помощью алгоритма Random Forest.")

            st.markdown("#### 🛠️ Инструкция:")
            st.markdown("""
            1. Выберите **Целевой показатель** (например, Общий балл ИПЛ).
            2. Выберите **метод корреляции** для определения направления влияния:
               * **Spearman** — ранговая корреляция (по умолчанию). Работает при любых распределениях и устойчива к выбросам. Рекомендуется для психологических данных.
               * **Pearson** — линейная корреляция. Предполагает нормальное распределение и чувствительна к выбросам.
            3. Выберите **Факторы влияния** (например, все шкалы Мильмана).
            4. Нажмите **«Найти ключевые драйверы»**.
            """)

            st.info("📈 **Катализатор:** Фактор повышает целевой показатель.\n📉 **Блокатор:** Фактор снижает целевой показатель.")

            st.markdown("#### 🔬 Подробная таблица: сравнение Spearman и Pearson")
            st.markdown("""
            В expander «Подробная таблица» показаны оба коэффициента рядом. Если у какой-то шкалы
            Spearman и Pearson дают **противоположные знаки** — строка подсвечивается жёлтым.
            Это сигнал о **нелинейной связи** или **сильном влиянии выбросов** — значит, с такой
            шкалой нужно работать отдельно, смотреть её скаттер и думать, что именно происходит.
            """)

            st.markdown("#### 🎛️ Симулятор «Что-если» (Внизу страницы)")
            st.markdown("Двигайте ползунки найденных факторов (например, искусственно поднимите 'Комфорт') и нажимайте «Пересчитать», чтобы увидеть на спидометре прогноз изменения целевого показателя.")

        elif selected_topic == topics[8]:
            st.markdown("### 👽 Вкладка 8: Детектор аномалий")
            st.markdown("Поиск респондентов с нетипичными профилями (выбросов).")
            
            st.markdown("""
            <div style="padding:10px; border-left: 5px solid #8e44ad; background: #f4ecf7; margin-bottom: 15px;">
                <b>Как работает Isolation Forest:</b> Высокий мотив 'Комфорт' — норма. Высокое 'Творчество' — норма. Но если у человека зашкаливают <b>оба</b> взаимоисключающих мотива, ИИ пометит его красной точкой как аномалию.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🔬 Рентген аномалии")
            st.markdown("""
            1. Задайте чувствительность ползунком (рекомендуется 5%) и запустите поиск.
            2. В блоке **«Рентген аномалии»** выберите найденного человека из списка.
            3. **Результат:** График Z-оценки покажет, какие именно баллы у этого человека аномально завышены (зеленый столбец) или занижены (красный столбец) по сравнению с нормальной группой.
            """)

        elif selected_topic == topics[9]:
            st.markdown("### 🕸️ Вкладка 9: Сетевой анализ")
            st.markdown("Визуализация структуры мотивов в виде графа.")
            
            st.markdown("#### 🔍 Как читать граф:")
            st.markdown("""
            * **Узлы (Кружки):** Шкалы теста. Чем больше кружок, тем больше у него связей (Центральный мотив).
            * **Линии:** Корреляции между шкалами. 
              * 🟢 **Зеленая:** Прямая связь (синхронность).
              * 🔴 **Красная:** Обратная связь (конфликт).
            * **Ползунок порога:** Если линий слишком много (каша), увеличьте порог отсечения (например, до 0.4), чтобы оставить только самые сильные связи.
            * **🔀 Только межметодические связи:** Если граф превратился в слишком плотный «клубок», включите этот фильтр. Он уберет ребра между шкалами одного теста, оставив только мосты между разными методиками. Это самый наглядный способ увидеть структуру личности.
            """)
            
            st.button("💾 Скачать этот граф как интерактивный файл (HTML) - Пример", disabled=True)
            st.caption("Скачанный файл можно открыть в любом браузере, чтобы приближать узлы и смотреть связи без Python.")

# ==========================================
# ГЛОБАЛЬНЫЙ СЛОВАРЬ РАСШИФРОВКИ ШКАЛ
# ==========================================
SCALE_NAMES_RU = {
    'Age': 'Возраст', 'Course': 'Курс',
    'B_Altruistic': 'Братусь: Альтруистические', 'B_Existential': 'Братусь: Экзистенциальные',
    'B_Hedonistic': 'Братусь: Гедонистические', 'B_Self-realization': 'Братусь: Самореализации',
    'B_Status': 'Братусь: Статусные', 'B_Communicative': 'Братусь: Коммуникативные',
    'B_Family': 'Братусь: Семейные', 'B_Cognitive': 'Братусь: Когнитивные',
    'M_P_Zh-id': 'Мильман: Поддержание (Жизнь, Идеал)', 'M_P_Zh-re': 'Мильман: Поддержание (Жизнь, Реал)',
    'M_K_Zh-id': 'Мильман: Комфорт (Жизнь, Идеал)', 'M_K_Zh-re': 'Мильман: Комфорт (Жизнь, Реал)',
    'M_S_Zh-id': 'Мильман: Статус (Жизнь, Идеал)', 'M_S_Zh-re': 'Мильман: Статус (Жизнь, Реал)',
    'M_O_Zh-id': 'Мильман: Общение (Жизнь, Идеал)', 'M_O_Zh-re': 'Мильман: Общение (Жизнь, Реал)',
    'M_D_Zh-id': 'Мильман: Активность (Жизнь, Идеал)', 'M_D_Zh-re': 'Мильман: Активность (Жизнь, Реал)',
    'M_DR_Zh-id': 'Мильман: Творчество (Жизнь, Идеал)', 'M_DR_Zh-re': 'Мильман: Творчество (Жизнь, Реал)',
    'M_OD_Zh-id': 'Мильман: Общ. польза (Жизнь, Идеал)', 'M_OD_Zh-re': 'Мильман: Общ. польза (Жизнь, Реал)',
    'M_P_Rb-id': 'Мильман: Поддержание (Работа, Идеал)', 'M_P_Rb-re': 'Мильман: Поддержание (Работа, Реал)',
    'M_K_Rb-id': 'Мильман: Комфорт (Работа, Идеал)', 'M_K_Rb-re': 'Мильман: Комфорт (Работа, Реал)',
    'M_S_Rb-id': 'Мильман: Статус (Работа, Идеал)', 'M_S_Rb-re': 'Мильман: Статус (Работа, Реал)',
    'M_O_Rb-id': 'Мильман: Общение (Работа, Идеал)', 'M_O_Rb-re': 'Мильман: Общение (Работа, Реал)',
    'M_D_Rb-id': 'Мильман: Активность (Работа, Идеал)', 'M_D_Rb-re': 'Мильман: Активность (Работа, Реал)',
    'M_DR_Rb-id': 'Мильман: Творчество (Работа, Идеал)', 'M_DR_Rb-re': 'Мильман: Творчество (Работа, Реал)',
    'M_OD_Rb-id': 'Мильман: Общ. польза (Работа, Идеал)', 'M_OD_Rb-re': 'Мильман: Общ. польза (Работа, Реал)',
    'M_Est': 'Мильман: Стенические', 'M_East': 'Мильман: Астенические',
    'M_Fst': 'Мильман: Стенические (Фрустрация)', 'M_Fast': 'Мильман: Астенические (Фрустрация)',
    'IPL_Total': 'ИПЛ: Общий балл', 'IPL_G': 'ИПЛ: Гносеологический (Г)',
    'IPL_A': 'ИПЛ: Аксиологический (А)', 'IPL_P': 'ИПЛ: Праксеологический (П)',
    'IPL_Type_OI': 'ИПЛ: Осмысленно-интенсивный (ОИ)', 'IPL_Type_FN': 'ИПЛ: Формально-накопительский (ФН)',
    'IPL_Type_PD': 'ИПЛ: Позитивно-дифференцированный (ПД)', 'IPL_Type_NG': 'ИПЛ: Негативно-генерализованный (НГ)',
    'IPL_Type_IP': 'ИПЛ: Инициативно-преобразовательный (ИП)', 'IPL_Type_VP': 'ИПЛ: Вынужденно-приспособительный (ВП)',
    'IPL_Level_Nature': 'ИПЛ: Природный уровень', 'IPL_Level_Social': 'ИПЛ: Социальный уровень',
    'IPL_Level_Culture': 'ИПЛ: Культурный уровень', 'IPL_Level_Life': 'ИПЛ: Уровень жизни',
    # --- Производные типы и профили ---
    'B_Altruistic_Level': 'Братусь: Уровень альтруист.', 'B_Existential_Level': 'Братусь: Уровень экзистенц.',
    'B_Hedonistic_Level': 'Братусь: Уровень гедонист.', 'B_Self-realization_Level': 'Братусь: Уровень самореализ.',
    'B_Status_Level': 'Братусь: Уровень статусных', 'B_Communicative_Level': 'Братусь: Уровень коммуник.',
    'B_Family_Level': 'Братусь: Уровень семейных', 'B_Cognitive_Level': 'Братусь: Уровень когнитивных',
    'M_Profile_Zh': 'Мильман: Мотив. профиль (жизнь)',
    'M_Profile_Rb': 'Мильман: Мотив. профиль (работа)',
    'M_Emo_Profile': 'Мильман: Эмоциональный профиль',
    'IPL_OI_FN': 'ИПЛ: Стиль поиска (ОИ/ФН)',
    'IPL_PD_NG': 'ИПЛ: Стиль оценки (ПД/НГ)',
    'IPL_IP_VP': 'ИПЛ: Стиль действия (ИП/ВП)',
    'IPL_Style': 'ИПЛ: Полный стиль',
    'IPL_Structure': 'ИПЛ: Структура Г/А/П',
}

def get_name(col):
    return SCALE_NAMES_RU.get(col, col)


# ==========================================
# КАТЕГОРИАЛЬНЫЕ КОЛОНКИ С ТИПАМИ И ПРОФИЛЯМИ
# ==========================================
# Эти колонки — текстовые результаты производных вычислений (типы, профили).
# Их НЕ нужно включать в числовые расчёты (.mean(), корреляции и т.д.).
# Используется для исключения при фильтрации по префиксам 'B_', 'M_', 'IPL_'.

DERIVED_CATEGORICAL_COLS = {
    # Уровни Братуся (8)
    'B_Altruistic_Level', 'B_Existential_Level', 'B_Hedonistic_Level',
    'B_Self-realization_Level', 'B_Status_Level', 'B_Communicative_Level',
    'B_Family_Level', 'B_Cognitive_Level',
    # Мильман профили (3)
    'M_Profile_Zh', 'M_Profile_Rb', 'M_Emo_Profile',
    # ИПЛ стили и структура (5)
    'IPL_OI_FN', 'IPL_PD_NG', 'IPL_IP_VP',
    'IPL_Style', 'IPL_Structure',
}


def get_numeric_scales(columns, prefix=None):
    """
    Возвращает список числовых шкал, отфильтрованных по префиксу,
    с исключением производных категориальных колонок.

    Пример:
        num_bratus = get_numeric_scales(df.columns, 'B_')
        # Вернёт только 8 баллов Братуся, без _Level

    Параметры:
        columns: iterable с именами колонок (df.columns)
        prefix: опциональный префикс ('B_', 'M_', 'IPL_'); None = без фильтра
    """
    result = [c for c in columns if c not in DERIVED_CATEGORICAL_COLS]
    if prefix is not None:
        result = [c for c in result if c.startswith(prefix)]
    return result


# ==========================================
# ЯДРО АНАЛИТИКИ (ФУНКЦИИ РАСЧЕТА)
# ==========================================

# --- Функции расчёта производных типов и профилей ---
# (дублируют логику process_survey.py для обратной совместимости со старыми файлами)

def _bratus_level(score):
    """Уровень категории смыслов: 3-9 Доминирует, 10-17 Нейтральный, 18-24 Игнорируется."""
    if score is None or pd.isna(score):
        return None
    if 3 <= score <= 9:
        return 'Доминирует'
    if 10 <= score <= 17:
        return 'Нейтральный'
    if 18 <= score <= 24:
        return 'Игнорируется'
    return None


def _milman_motivational_profile(p, k, s, o, d, dr, od):
    """Мотивационный профиль по Мильману (Прогрессивный/Регрессивный/Импульсивный/Экспрессивный/Уплощённый/Неопределённый)."""
    vals = [p, k, s, o, d, dr, od]
    diff = (d + dr + od) - (p + k + s)
    if diff >= 5:
        return 'Прогрессивный'
    if diff <= -5:
        return 'Регрессивный'

    peaks = 0
    for i in range(len(vals)):
        left = vals[i - 1] if i > 0 else None
        right = vals[i + 1] if i < len(vals) - 1 else None
        if i == 0:
            if right is not None and vals[i] - right >= 4:
                peaks += 1
        elif i == len(vals) - 1:
            if left is not None and vals[i] - left >= 4:
                peaks += 1
        else:
            if vals[i] - left >= 2 and vals[i] - right >= 2:
                peaks += 1
    if peaks >= 3:
        return 'Импульсивный'
    if peaks == 2:
        return 'Экспрессивный'
    if max(vals) - min(vals) <= 3:
        return 'Уплощённый'
    return 'Неопределённый'


def _milman_emotional_profile(e_st, e_ast, f_st, f_ast):
    """Эмоциональный профиль: симметричная обработка равенств."""
    emo_eq = (e_st == e_ast)
    fru_eq = (f_st == f_ast)
    if emo_eq and fru_eq:
        return 'Не определён'
    emo_sten = e_st > e_ast
    fru_sten = f_st > f_ast
    if emo_eq:
        return 'Смешанный стенический' if fru_sten else 'Смешанный астенический'
    if fru_eq:
        return 'Смешанный астенический' if emo_sten else 'Смешанный стенический'
    if emo_sten and fru_sten:
        return 'Стенический'
    if not emo_sten and not fru_sten:
        return 'Астенический'
    if not emo_sten and fru_sten:
        return 'Смешанный стенический'
    return 'Смешанный астенический'


def _ipl_dimension(a, b, label_a, label_b):
    """Одно измерение ИПЛ. При равенстве возвращает 'Неопределённый'."""
    if a > b:
        return label_a
    if b > a:
        return label_b
    return 'Неопределённый'


def _ipl_full_style(oi_fn, pd_ng, ip_vp):
    """Полный стиль ИПЛ — склейка трёх измерений. При наличии неопределённого измерения — 'Неопределённый'."""
    if 'Неопределённый' in (oi_fn, pd_ng, ip_vp):
        return 'Неопределённый'
    return f'{oi_fn}+{pd_ng}+{ip_vp}'


def _ipl_structure(g, a, p):
    """Структура Г/А/П в порядке возрастания. Равенства обозначаются '='."""
    components = [('Г', g), ('А', a), ('П', p)]
    components.sort(key=lambda x: x[1])
    parts = [components[0][0]]
    for i in range(1, 3):
        sep = '=' if components[i][1] == components[i - 1][1] else '<'
        parts.append(sep)
        parts.append(components[i][0])
    return ''.join(parts)


def compute_derived_types(df):
    """
    Добавляет в DataFrame производные категориальные колонки с типами и профилями.
    Вызывается из load_data автоматически, если эти колонки отсутствуют
    (для обратной совместимости со старыми файлами FINAL_RESULTS).

    Новые файлы, сформированные обновлённым process_survey.py, уже содержат
    все эти колонки — в этом случае функция ничего не пересчитывает.
    """
    # --- Братусь: уровни по 8 категориям ---
    bratus_cats = ['Altruistic', 'Existential', 'Hedonistic', 'Self-realization',
                   'Status', 'Communicative', 'Family', 'Cognitive']
    for cat in bratus_cats:
        level_col = f'B_{cat}_Level'
        score_col = f'B_{cat}'
        if level_col not in df.columns and score_col in df.columns:
            df[level_col] = df[score_col].apply(_bratus_level)

    # --- Мильман: мотивационные профили (Ж и Рб) ---
    m_scales = ['P', 'K', 'S', 'O', 'D', 'DR', 'OD']

    def _has_milman_cols(sphere_suffix):
        return all(f'M_{s}_{sphere_suffix}-id' in df.columns and f'M_{s}_{sphere_suffix}-re' in df.columns
                   for s in m_scales)

    if 'M_Profile_Zh' not in df.columns and _has_milman_cols('Zh'):
        def _compute_zh(row):
            vals = {s: row[f'M_{s}_Zh-id'] + row[f'M_{s}_Zh-re'] for s in m_scales}
            return _milman_motivational_profile(
                vals['P'], vals['K'], vals['S'], vals['O'],
                vals['D'], vals['DR'], vals['OD'])
        df['M_Profile_Zh'] = df.apply(_compute_zh, axis=1)

    if 'M_Profile_Rb' not in df.columns and _has_milman_cols('Rb'):
        def _compute_rb(row):
            vals = {s: row[f'M_{s}_Rb-id'] + row[f'M_{s}_Rb-re'] for s in m_scales}
            return _milman_motivational_profile(
                vals['P'], vals['K'], vals['S'], vals['O'],
                vals['D'], vals['DR'], vals['OD'])
        df['M_Profile_Rb'] = df.apply(_compute_rb, axis=1)

    # --- Мильман: эмоциональный профиль ---
    emo_cols = ['M_Est', 'M_East', 'M_Fst', 'M_Fast']
    if 'M_Emo_Profile' not in df.columns and all(c in df.columns for c in emo_cols):
        df['M_Emo_Profile'] = df.apply(
            lambda r: _milman_emotional_profile(r['M_Est'], r['M_East'], r['M_Fst'], r['M_Fast']),
            axis=1)

    # --- ИПЛ: три измерения ---
    ipl_type_pairs = [
        ('IPL_OI_FN', 'IPL_Type_OI', 'IPL_Type_FN', 'ОИ', 'ФН'),
        ('IPL_PD_NG', 'IPL_Type_PD', 'IPL_Type_NG', 'ПД', 'НГ'),
        ('IPL_IP_VP', 'IPL_Type_IP', 'IPL_Type_VP', 'ИП', 'ВП'),
    ]
    for target_col, col_a, col_b, label_a, label_b in ipl_type_pairs:
        if target_col not in df.columns and col_a in df.columns and col_b in df.columns:
            df[target_col] = df.apply(
                lambda r: _ipl_dimension(r[col_a], r[col_b], label_a, label_b), axis=1)

    # --- ИПЛ: полный стиль ---
    if 'IPL_Style' not in df.columns and all(c in df.columns for c in ['IPL_OI_FN', 'IPL_PD_NG', 'IPL_IP_VP']):
        df['IPL_Style'] = df.apply(
            lambda r: _ipl_full_style(r['IPL_OI_FN'], r['IPL_PD_NG'], r['IPL_IP_VP']), axis=1)

    # --- ИПЛ: структура Г/А/П ---
    if 'IPL_Structure' not in df.columns and all(c in df.columns for c in ['IPL_G', 'IPL_A', 'IPL_P']):
        df['IPL_Structure'] = df.apply(
            lambda r: _ipl_structure(r['IPL_G'], r['IPL_A'], r['IPL_P']), axis=1)

    return df


@st.cache_data(ttl=3600)
def load_data(file):
    try:
        df = pd.read_excel(file)

        # Колонки с производными типами/профилями НЕ трогаем .capitalize() —
        # иначе значения вроде 'ОИ+ПД+ИП' превратятся в 'Ои+пд+ип',
        # а 'Г<А<П' в 'Г<а<п'. Используем глобальную константу.

        # В pandas 2.x текстовые колонки могут иметь dtype 'string' (StringDtype),
        # а не 'object'. Включаем оба для совместимости.
        try:
            text_cols = df.select_dtypes(include=['object', 'string']).columns
        except TypeError:
            # На всякий случай fallback для старых версий pandas
            text_cols = df.select_dtypes(include=['object']).columns

        for col in text_cols:
            if col in DERIVED_CATEGORICAL_COLS:
                continue  # не меняем регистр у категориальных типов
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            df[col] = df[col].replace({'Nan': np.nan, 'None': np.nan})

        # Рассчитываем производные типы/профили, если их нет в файле
        # (актуально для старых файлов FINAL_RESULTS, обработанных до обновления process_survey)
        df = compute_derived_types(df)

        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

def get_descriptive_stats(df, columns):
    stats_list = []
    for col in columns:
        if col not in df.columns: continue
        series = df[col].dropna()
        if len(series) < 3: continue
        desc = series.describe()
        try:
            normality = pg.normality(series)
            p_val = normality['pval'].values[0]
            is_normal = p_val > 0.05
        except:
            p_val = 0
            is_normal = False
        stats_list.append({
            "Показатель": col, "N": int(desc['count']), "Среднее": desc['mean'],
            "SD": desc['std'], "Медиана": desc['50%'], "Min": desc['min'], "Max": desc['max'],
            "Skew": series.skew(), "Kurtosis": series.kurtosis(),
            "Нормальность (p)": f"{p_val:.3f} ({'Да' if is_normal else 'Нет'})"
        })
    return pd.DataFrame(stats_list)

def smart_compare_groups(df, group_col, metric_col):
    """
    Выбирает подходящий тест сравнения групп на основе нормальности.

    Returns
    -------
    result_df : pd.DataFrame | None
        Таблица с полями Метрика/Тест/p-value/Эффект (для отображения).
    p_val : float | str
        Числовое p-value, либо строка с описанием ошибки.
    """
    clean_df = df[[group_col, metric_col]].dropna()
    groups = clean_df[group_col].unique()
    if len(groups) < 2:
        return None, "Меньше 2 групп"
    try:
        is_normal = True
        for g in groups:
            g_data = clean_df[clean_df[group_col] == g][metric_col]
            if len(g_data) >= 3:
                if pg.normality(g_data)['pval'].values[0] < 0.05:
                    is_normal = False
                    break
        if len(groups) == 2:
            g1 = clean_df[clean_df[group_col] == groups[0]][metric_col]
            g2 = clean_df[clean_df[group_col] == groups[1]][metric_col]
            if is_normal:
                res = pg.ttest(g1, g2, correction=True)
                test_name, eff_label = "T-test (Welch)", "Cohen's d"
                p_val, eff_size = res['p-val'].values[0], res['cohen-d'].values[0]
            else:
                res = pg.mwu(g1, g2)
                test_name, eff_label = "Mann-Whitney U (Непараметрический)", "Rank-Biserial"
                p_val, eff_size = res['p-val'].values[0], res['RBC'].values[0]
        else:
            if is_normal:
                res = pg.anova(data=clean_df, dv=metric_col, between=group_col)
                test_name, eff_label = "One-way ANOVA", "Eta-sq (η²)"
                p_val, eff_size = res['p-unc'].values[0], res['np2'].values[0]
            else:
                res = pg.kruskal(data=clean_df, dv=metric_col, between=group_col)
                test_name, eff_label = "Kruskal-Wallis (Непараметрический)", "H-stat"
                p_val, eff_size = res['p-unc'].values[0], res['H'].values[0]

        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        result_df = pd.DataFrame({
            "Метрика": [metric_col],
            "Тест": [test_name],
            "p-value": [f"{p_val:.4f} {stars}"],
            f"Эффект ({eff_label})": [f"{eff_size:.3f}"]
        })
        return result_df, p_val
    except Exception:
        return None, None


def _compare_groups_raw(df, group_col, metric_col):
    """
    Внутренняя версия smart_compare_groups для массового сканирования.
    Возвращает словарь с числовыми значениями теста или None при ошибке.
    Используется run_auto_scan для агрегирования результатов.
    """
    clean_df = df[[group_col, metric_col]].dropna()
    groups = clean_df[group_col].unique()
    if len(groups) < 2:
        return None
    try:
        is_normal = True
        for g in groups:
            g_data = clean_df[clean_df[group_col] == g][metric_col]
            if len(g_data) >= 3:
                if pg.normality(g_data)['pval'].values[0] < 0.05:
                    is_normal = False
                    break
        if len(groups) == 2:
            g1 = clean_df[clean_df[group_col] == groups[0]][metric_col]
            g2 = clean_df[clean_df[group_col] == groups[1]][metric_col]
            if is_normal:
                res = pg.ttest(g1, g2, correction=True)
                return {'p_val': float(res['p-val'].values[0]),
                        'test': 'T-test (Welch)',
                        'effect_size': float(res['cohen-d'].values[0]),
                        'effect_label': "Cohen's d"}
            res = pg.mwu(g1, g2)
            return {'p_val': float(res['p-val'].values[0]),
                    'test': 'Mann-Whitney U',
                    'effect_size': float(res['RBC'].values[0]),
                    'effect_label': 'Rank-Biserial'}
        if is_normal:
            res = pg.anova(data=clean_df, dv=metric_col, between=group_col)
            return {'p_val': float(res['p-unc'].values[0]),
                    'test': 'One-way ANOVA',
                    'effect_size': float(res['np2'].values[0]),
                    'effect_label': 'Eta-sq (η²)'}
        res = pg.kruskal(data=clean_df, dv=metric_col, between=group_col)
        return {'p_val': float(res['p-unc'].values[0]),
                'test': 'Kruskal-Wallis',
                'effect_size': float(res['H'].values[0]),
                'effect_label': 'H-stat'}
    except Exception:
        return None


def run_auto_scan(df, group_col, metric_cols, alpha=0.05):
    """
    Массовое сравнение групп по всем metric_cols с автоматическим выбором теста.
    Устраняет дублирование логики тестирования: теперь страница 3 вызывает
    эту функцию, а не копирует выбор тестов заново.

    Returns
    -------
    pd.DataFrame
        Колонки: Показатель, Колонка, Тест, p-value, Размер эффекта (с меткой),
                 Значимость, _p_raw, _effect_raw.
        Отсортирован по p-value (возрастание), только удачные расчёты.
    """
    rows = []
    for col in metric_cols:
        details = _compare_groups_raw(df, group_col, col)
        if details is None:
            continue
        p_val = details['p_val']
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < alpha else ""))
        rows.append({
            'Показатель': get_name(col),
            'Колонка': col,
            'Тест': details['test'],
            'p-value': round(p_val, 4),
            f"Размер эффекта ({details['effect_label']})": round(details['effect_size'], 3),
            'Значимость': stars or "н.з.",
            '_p_raw': p_val,
            '_effect_raw': details['effect_size'],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('_p_raw').reset_index(drop=True)


def run_clustering_analysis(df, cols, n_clusters):
    """
    Запускает K-Means кластеризацию.
    Возвращает DataFrame с кластерами, PCA-координатами, силуэтом и оригинальными
    демографическими/текстовыми колонками (если они есть в df).
    """
    from sklearn.metrics import silhouette_score
    data = df[cols].dropna()
    if data.empty or len(data) <= n_clusters:
        return None, None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_scaled)

    # Силуэт для выбранного разбиения
    try:
        sil = silhouette_score(data_scaled, clusters)
    except Exception:
        sil = None

    res_df = data.copy()
    res_df['Cluster'] = clusters.astype(str)
    res_df['PC1'], res_df['PC2'] = components[:, 0], components[:, 1]

    # Подтягиваем все контекстные колонки, которые есть в df и не использовались для кластеризации
    context_cols = [c for c in ['FIO', 'Gender', 'Age', 'Course', 'Edu_Status',
                                'Edu_Level', 'University', 'Speciality']
                    if c in df.columns and c not in cols]
    if context_cols:
        res_df = res_df.join(df[context_cols], how='left')

    return res_df, sil

def calc_correlation_matrices(df_subset, method):
    cols = df_subset.columns
    r_matrix = df_subset.corr(method=method)
    p_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), columns=cols, index=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            res = pg.corr(df_subset[cols[i]], df_subset[cols[j]], method=method)
            p_val = res['p-val'].values[0]
            p_matrix.iloc[i, j] = p_val
            p_matrix.iloc[j, i] = p_val 
    return r_matrix, p_matrix

@st.cache_data
def apply_fdr_correction(p_matrix: pd.DataFrame, method: str = 'fdr_bh') -> pd.DataFrame:
    """
    Применяет поправку на множественные сравнения к матрице p-значений.

    Из квадратной матрицы p-values берёт только верхний треугольник
    (без диагонали), применяет к нему поправку, затем возвращает
    скорректированную симметричную матрицу.

    method:
        'fdr_bh'      — Benjamini-Hochberg (FDR), рекомендуется
        'bonferroni'  — более строгая поправка
    """
    from statsmodels.stats.multitest import multipletests

    cols = p_matrix.columns
    n = len(cols)

    # Извлекаем индексы верхнего треугольника (i < j)
    iu = np.triu_indices(n, k=1)
    p_values_flat = p_matrix.values[iu]

    # Если все NaN или массив пустой — возвращаем оригинал
    if len(p_values_flat) == 0 or np.all(np.isnan(p_values_flat)):
        return p_matrix.copy()

    # Применяем поправку
    valid_mask = ~np.isnan(p_values_flat)
    corrected_flat = np.full_like(p_values_flat, fill_value=np.nan, dtype=float)

    if valid_mask.sum() > 0:
        _, corrected_valid, _, _ = multipletests(
            p_values_flat[valid_mask],
            method=method,
            alpha=0.05
        )
        corrected_flat[valid_mask] = corrected_valid

    # Собираем обратно в матрицу
    p_corrected = p_matrix.copy()
    p_corrected.values[iu] = corrected_flat
    # Симметрия: нижний треугольник = верхний транспонированный
    il = np.tril_indices(n, k=-1)
    p_corrected.values[il] = p_corrected.T.values[il]
    # Диагональ — нули (корреляция переменной с самой собой)
    np.fill_diagonal(p_corrected.values, 0.0)

    return p_corrected
@st.cache_data
def calculate_mahalanobis_distances(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Расчёт расстояния Махаланобиса для каждого наблюдения.

    Махаланобис измеряет, насколько каждый респондент удалён от
    многомерного центра выборки с учётом ковариаций между переменными.
    Результат: значения D² распределены приблизительно как χ²(df=k),
    где k — число переменных.

    Возвращает DataFrame с колонками:
        - mahal_d2:   квадрат расстояния Махаланобиса (D²)
        - mahal_p:    p-value по χ² (с k степенями свободы)
        - is_outlier_strict:    p < 0.001 (рекомендация для psychology research)
        - is_outlier_moderate:  p < 0.01  (мягкий критерий)
    """
    from scipy.stats import chi2

    X = df_subset.dropna().values
    n, k = X.shape

    if n < k + 2:
        # Слишком мало наблюдений для надёжной оценки
        return pd.DataFrame(index=df_subset.dropna().index, data={
            'mahal_d2': np.nan, 'mahal_p': np.nan,
            'is_outlier_strict': False, 'is_outlier_moderate': False,
        })

    # Центр и обратная ковариационная матрица
    mean_vec = X.mean(axis=0)
    cov_matrix = np.cov(X, rowvar=False)

    # Псевдо-инверсия — устойчива к мультиколлинеарности
    try:
        inv_cov = np.linalg.pinv(cov_matrix)
    except np.linalg.LinAlgError:
        return pd.DataFrame(index=df_subset.dropna().index, data={
            'mahal_d2': np.nan, 'mahal_p': np.nan,
            'is_outlier_strict': False, 'is_outlier_moderate': False,
        })

    # D² для каждого наблюдения: (x-μ)ᵀ Σ⁻¹ (x-μ)
    diff = X - mean_vec
    d2_values = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)

    # p-value по распределению χ² с k степенями свободы
    # Большой D² → маленькое p → выброс
    p_values = 1 - chi2.cdf(d2_values, df=k)

    result = pd.DataFrame(
        index=df_subset.dropna().index,
        data={
            'mahal_d2': d2_values,
            'mahal_p': p_values,
            'is_outlier_strict': p_values < 0.001,
            'is_outlier_moderate': p_values < 0.01,
        }
    )
    return result

def init_session_state():
    """Инициализация базовых переменных и автозагрузка демо-данных"""
    if 'disable_auto_demo' not in st.session_state:
        st.session_state['disable_auto_demo'] = False
    if 'df_raw' not in st.session_state:
        st.session_state['df_raw'] = None
    if 'is_demo' not in st.session_state:
        st.session_state['is_demo'] = False

    # Автозагрузка демо-файла, если данных нет и это не запрещено
    if st.session_state['df_raw'] is None and not st.session_state['disable_auto_demo']:
        try:
            import os
            if os.path.exists('TEST_RESULTS.xlsx'):
                # Используем load_data для единообразной обработки текстовых полей
                # (без load_data демо-данные и пользовательские данные обрабатывались
                # по-разному, что могло приводить к расхождениям в фильтрах и тестах)
                with open('TEST_RESULTS.xlsx', 'rb') as f:
                    st.session_state['df_raw'] = load_data(f)
                st.session_state['is_demo'] = True
        except Exception:
            pass
def render_sidebar():
    """Функция отрисовки глобального бокового меню и фильтров с кэшированием между страницами"""
    import io
    # Сначала всегда проверяем состояние (на случай, если функция вызвана без предварительной инициализации)
    if 'df_raw' not in st.session_state:
        init_session_state()
    # 1. РУКОВОДСТВО И ДЕМО-ПЛАШКА ДОСТУПНЫ ВСЕГДА
    with st.sidebar:
        if st.button("📖 Открыть руководство", use_container_width=True, type="primary"):
            show_help_dialog()
            
        # --- ПЛАШКА ДЕМО-РЕЖИМА ---
        if st.session_state.get('is_demo', False):
            st.error("⚠️ **ДЕМО-РЕЖИМ** (Тестовые данные)")
            
    # 2. ПРОВЕРКА ДАННЫХ
    if st.session_state['df_raw'] is None:
        st.sidebar.warning("👈 Пожалуйста, загрузите данные на Главной странице.")
        return None
        
    df_raw = st.session_state['df_raw']
    
    # ==========================================
    # ИНИЦИАЛИЗАЦИЯ ПАМЯТИ ФИЛЬТРОВ
    # ==========================================
    all_genders = sorted(df_raw['Gender'].dropna().astype(str).unique().tolist()) if 'Gender' in df_raw.columns else []
    if 'f_gender' not in st.session_state: st.session_state['f_gender'] = all_genders
    
    if 'Age' in df_raw.columns and not df_raw['Age'].dropna().empty:
        min_a, max_a = int(df_raw['Age'].min()), int(df_raw['Age'].max())
    else:
        min_a, max_a = 0, 100
    if 'f_age' not in st.session_state: st.session_state['f_age'] = (min_a, max_a)
        
    all_work = sorted(df_raw['Work'].dropna().astype(str).unique().tolist()) if 'Work' in df_raw.columns else []
    if 'f_work' not in st.session_state: st.session_state['f_work'] = all_work
        
    all_edu = sorted(df_raw['Edu_Status'].dropna().astype(str).unique().tolist()) if 'Edu_Status' in df_raw.columns else []
    if 'f_edu' not in st.session_state: st.session_state['f_edu'] = all_edu

    all_kmns = sorted(df_raw['Is_KMNS'].dropna().astype(str).unique().tolist()) if 'Is_KMNS' in df_raw.columns else []
    if 'f_kmns' not in st.session_state: st.session_state['f_kmns'] = all_kmns

    all_fast = sorted(df_raw['Fast_Clicker'].dropna().astype(str).unique().tolist()) if 'Fast_Clicker' in df_raw.columns else []
    if 'f_fast' not in st.session_state: st.session_state['f_fast'] = []

    extra_cols = {
        'KMNS_Name': 'Конкретный народ (КМНС)', 'Family': 'Семейное положение',
        'Children': 'Наличие детей', 'University': 'Название ВУЗа',
        'Speciality': 'Специальность', 'Edu_Level': 'Уровень обучения', 'Edu_Basis': 'Основа обучения'
    }
    if 'f_extra' not in st.session_state: st.session_state['f_extra'] = {col: [] for col in extra_cols}

    # Фильтры по производным типам и профилям
    type_filter_cols = {
        'M_Emo_Profile': 'Эмоциональный профиль',
        'M_Profile_Zh': 'Мотив. профиль (жизнь)',
        'M_Profile_Rb': 'Мотив. профиль (работа)',
        'IPL_Style': 'Полный стиль ИПЛ',
        'IPL_OI_FN': 'Стиль поиска (ОИ/ФН)',
        'IPL_PD_NG': 'Стиль оценки (ПД/НГ)',
        'IPL_IP_VP': 'Стиль действия (ИП/ВП)',
        'IPL_Structure': 'Структура Г/А/П',
    }
    # Уровни Братуся — каждая категория отдельным фильтром
    bratus_level_cols = {
        'B_Altruistic_Level': 'Альтруистические смыслы',
        'B_Existential_Level': 'Экзистенциальные смыслы',
        'B_Hedonistic_Level': 'Гедонистические смыслы',
        'B_Self-realization_Level': 'Смыслы самореализации',
        'B_Status_Level': 'Статусные смыслы',
        'B_Communicative_Level': 'Коммуникативные смыслы',
        'B_Family_Level': 'Семейные смыслы',
        'B_Cognitive_Level': 'Когнитивные смыслы',
    }
    if 'f_types' not in st.session_state:
        st.session_state['f_types'] = {col: [] for col in {**type_filter_cols, **bratus_level_cols}}

    # ==========================================
    # ОТРИСОВКА ИНТЕРФЕЙСА
    # ==========================================
    with st.sidebar:
            
        st.header("Фильтры")
        
        with st.form("filters_form"):
            # Берем значения по умолчанию строго из st.session_state
            sel_gender = st.multiselect("Пол", all_genders, default=st.session_state['f_gender']) if all_genders else []
            sel_age = st.slider("Возраст", min_a, max_a, st.session_state['f_age']) if min_a < max_a else (min_a, max_a)
            sel_work = st.multiselect("Работа", all_work, default=st.session_state['f_work']) if all_work else []
            sel_edu = st.multiselect("Образование", all_edu, default=st.session_state['f_edu']) if all_edu else []
            sel_kmns = st.multiselect("Относится к КМНС?", all_kmns, default=st.session_state['f_kmns']) if all_kmns else []

            sel_fast = st.multiselect(
                "Анти-фрод (Fast Clicker)", 
                all_fast, 
                default=st.session_state['f_fast'], 
                help="Оставьте пустым, чтобы показать всех. Выберите 'No', чтобы оставить только надежные ответы."
            ) if all_fast else []

            with st.expander("Дополнительные фильтры", expanded=False):
                sel_extra = {}
                for col, label in extra_cols.items():
                    if col in df_raw.columns:
                        options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                        saved_vals = [v for v in st.session_state['f_extra'].get(col, []) if v in options]
                        sel_extra[col] = st.multiselect(label, options, default=saved_vals)
                    else:
                        sel_extra[col] = []

            with st.expander("🧬 Фильтры по типам профилей", expanded=False):
                st.caption("Оставьте пустым, чтобы не фильтровать. Выберите значения, чтобы оставить только людей с нужным типом.")
                sel_types = {}

                # Мильман и ИПЛ
                st.markdown("**Профили (Мильман, ИПЛ)**")
                for col, label in type_filter_cols.items():
                    if col in df_raw.columns:
                        options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                        saved = [v for v in st.session_state['f_types'].get(col, []) if v in options]
                        sel_types[col] = st.multiselect(label, options, default=saved, key=f"ft_{col}")
                    else:
                        sel_types[col] = []

                # Братусь — уровни
                st.markdown("**Уровни категорий смыслов (Братусь)**")
                for col, label in bratus_level_cols.items():
                    if col in df_raw.columns:
                        options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                        saved = [v for v in st.session_state['f_types'].get(col, []) if v in options]
                        sel_types[col] = st.multiselect(label, options, default=saved, key=f"ft_{col}")
                    else:
                        sel_types[col] = []

            submit = st.form_submit_button("Применить фильтры")

        # ==========================================
        # ОБНОВЛЕНИЕ ПАМЯТИ ПРИ НАЖАТИИ КНОПКИ
        # ==========================================
        if submit:
            st.session_state['f_gender'] = sel_gender
            st.session_state['f_age'] = sel_age
            st.session_state['f_work'] = sel_work
            st.session_state['f_edu'] = sel_edu
            st.session_state['f_kmns'] = sel_kmns
            st.session_state['f_fast'] = sel_fast
            st.session_state['f_extra'] = sel_extra
            st.session_state['f_types'] = sel_types

        # ==========================================
        # ПРИМЕНЕНИЕ МАСКИ НА ОСНОВЕ ПАМЯТИ СЕССИИ
        # ==========================================
        mask = pd.Series(True, index=df_raw.index)
        if 'Gender' in df_raw.columns and st.session_state['f_gender']: 
            mask &= df_raw['Gender'].astype(str).isin(st.session_state['f_gender'])
        if 'Age' in df_raw.columns: 
            mask &= df_raw['Age'].between(st.session_state['f_age'][0], st.session_state['f_age'][1])
        if 'Work' in df_raw.columns and st.session_state['f_work']: 
            mask &= df_raw['Work'].astype(str).isin(st.session_state['f_work'])
        if 'Edu_Status' in df_raw.columns and st.session_state['f_edu']: 
            mask &= df_raw['Edu_Status'].astype(str).isin(st.session_state['f_edu'])
        if 'Is_KMNS' in df_raw.columns and st.session_state['f_kmns']: 
            mask &= df_raw['Is_KMNS'].astype(str).isin(st.session_state['f_kmns'])
        if 'Fast_Clicker' in df_raw.columns and st.session_state['f_fast']: 
            mask &= df_raw['Fast_Clicker'].astype(str).isin(st.session_state['f_fast'])
        
        for col, selected in st.session_state['f_extra'].items():
            if col in df_raw.columns and selected:
                mask &= df_raw[col].astype(str).isin(selected)
        # Фильтры по типам профилей
        for col, selected in st.session_state.get('f_types', {}).items():
            if col in df_raw.columns and selected:
                mask &= df_raw[col].astype(str).isin(selected)
        df = df_raw[mask].copy()
        
        # ---  ФИЛЬТР ПО ID (ИНДЕКСАМ) ---
        st.sidebar.divider()
        id_filter_raw = st.sidebar.text_area(
            "🎯 Фильтр по ID респондентов", 
            placeholder="Например: 5, 12, 44",
            help="Введите ID через запятую или пробел. Позволяет оставить только конкретную группу людей."
        )

        if id_filter_raw.strip():
            try:
                # Парсим строку: заменяем запятые на пробелы и разбиваем
                target_ids = [int(x.strip()) for x in id_filter_raw.replace(',', ' ').split() if x.strip().isdigit()]
                if target_ids:
                    # Оставляем только те ID, которые реально есть в текущем (уже отфильтрованном) наборе
                    existing_ids = [i for i in target_ids if i in df.index]
                    if existing_ids:
                        df = df.loc[existing_ids]
                        # Сообщение об успехе выводим в сайдбаре
                        st.sidebar.success(f"Найдено ID: {len(df)}")
                    
                    # Если ввели ID, которых нет в базе, предупреждаем
                    if len(existing_ids) < len(target_ids):
                        missing = set(target_ids) - set(existing_ids)
                        st.sidebar.warning(f"ID не найдены: {list(missing)}")
            except Exception as e:
                st.sidebar.error(f"Ошибка в формате ID: {e}")

        st.markdown("---")
        st.metric("Выборка", f"{len(df)} чел.", delta=len(df)-len(df_raw))
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Скачать текущую выборку",
            data=buffer.getvalue(), file_name='filtered_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    return df