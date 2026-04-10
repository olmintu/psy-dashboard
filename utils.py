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
        "📐 7. Психометрика", "🔮 8. Поиск драйверов", "👽 9. Детектор аномалий", "🕸️ 10. Сетевой анализ"
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
            2. Нажмите **«Найти значимые различия»**.
            3. **Результат:** Таблица покажет *только те шкалы*, по которым группы достоверно отличаются.
            """)
            
            st.markdown("#### Как читать результат:")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h4 style='text-align: center; color: #27ae60;'>p < 0.05</h4>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'><b>Различия есть.</b> Это не случайность.</p>", unsafe_allow_html=True)
            with c2:
                st.markdown("<h4 style='text-align: center; color: #e74c3c;'>p > 0.05</h4>", unsafe_allow_html=True)
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
            """)

        elif selected_topic == topics[5]:
            st.markdown("### 🔬 Вкладка 5: Кластерный анализ")
            st.markdown("Разделение людей на схожие типажи с помощью машинного обучения.")
            
            st.markdown("#### 🌳 Иерархическая кластеризация (Дендрограммы)")
            st.markdown("Позволяет визуально оценить естественные группировки (деревья) в данных. Ствол дерева делится на крупные ветки — это и есть основные типажи в вашей выборке.")
            
            st.markdown("#### 🎯 K-Means (Авто-выбор)")
            st.markdown("""
            * Выберите шкалы. Алгоритм сам рассчитает метрику Силуэта и предложит **оптимальное количество кластеров**.
            * Радарные диаграммы (паутинки) покажут психологический профиль каждого кластера (чем отличается Группа 1 от Группы 2).
            """)

        elif selected_topic == topics[6]:
            st.markdown("### 📐 Вкладка 6: Психометрика")
            st.markdown("Проверка надежности и факторной структуры самого теста.")
            
            st.markdown("#### 1. Альфа Кронбаха (Внутренняя согласованность)")
            st.markdown("Метрика надежности. Показывает, измеряют ли выбранные шкалы единый конструкт.")
            st.progress(75)
            st.caption("Норма: α > 0.7. Если показатель высокий, вы имеете право объединить эти шкалы в один суммарный индекс.")
            
            st.markdown("#### 2. Метод главных компонент (PCA)")
            st.markdown("Сжимает все шкалы в 2-3 крупных латентных фактора и выводит таблицу «нагрузок». Смотрите на цифры **> 0.4** — они показывают, какие именно шкалы вошли в состав каждого укрупненного фактора.")

        elif selected_topic == topics[7]:
            st.markdown("### 🔮 Вкладка 7: Поиск драйверов (ИИ)")
            st.markdown("Определяет, какие факторы сильнее всего влияют на выбранный целевой показатель с помощью алгоритма Random Forest.")
            
            st.markdown("#### 🛠️ Инструкция:")
            st.markdown("""
            1. Выберите **Целевой показатель** (например, Общий балл ИПЛ).
            2. Выберите **Факторы влияния** (например, все шкалы Мильмана).
            3. Нажмите **«Найти ключевые драйверы»**.
            """)
            
            st.info("📈 **Катализатор:** Фактор повышает целевой показатель.\n📉 **Блокатор:** Фактор снижает целевой показатель.")
            
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
    'IPL_Level_Culture': 'ИПЛ: Культурный уровень', 'IPL_Level_Life': 'ИПЛ: Уровень жизни'
}

def get_name(col):
    return SCALE_NAMES_RU.get(col, col)

# ==========================================
# ЯДРО АНАЛИТИКИ (ФУНКЦИИ РАСЧЕТА)
# ==========================================
@st.cache_data(ttl=3600)
def load_data(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return None

@st.cache_data
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

@st.cache_data
def smart_compare_groups(df, group_col, metric_col):
    clean_df = df[[group_col, metric_col]].dropna()
    groups = clean_df[group_col].unique()
    if len(groups) < 2: return None, "Меньше 2 групп"
    try:
        is_normal = True
        for g in groups:
            g_data = clean_df[clean_df[group_col] == g][metric_col]
            if len(g_data) >= 3:
                if pg.normality(g_data)['pval'].values[0] < 0.05:
                    is_normal = False
                    break
        if len(groups) == 2:
            g1, g2 = clean_df[clean_df[group_col] == groups[0]][metric_col], clean_df[clean_df[group_col] == groups[1]][metric_col]
            if is_normal:
                res = pg.ttest(g1, g2, correction=True)
                test_name, p_val, eff_size, eff_label = "T-test (Welch)", res['p-val'].values[0], res['cohen-d'].values[0], "Cohen's d"
            else:
                res = pg.mwu(g1, g2)
                test_name, p_val, eff_size, eff_label = "Mann-Whitney U (Непараметрический)", res['p-val'].values[0], res['RBC'].values[0], "Rank-Biserial"
        else:
            if is_normal:
                res = pg.anova(data=clean_df, dv=metric_col, between=group_col)
                test_name, p_val, eff_size, eff_label = "One-way ANOVA", res['p-unc'].values[0], res['np2'].values[0], "Eta-sq (η²)"
            else:
                res = pg.kruskal(data=clean_df, dv=metric_col, between=group_col)
                test_name, p_val, eff_size, eff_label = "Kruskal-Wallis (Непараметрический)", res['p-unc'].values[0], res['H'].values[0], "H-stat"
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        result_df = pd.DataFrame({"Метрика": [metric_col], "Тест": [test_name], "p-value": [f"{p_val:.4f} {stars}"], f"Эффект ({eff_label})": [f"{eff_size:.3f}"]})
        return result_df, p_val
    except Exception as e:
        return None, str(e)

@st.cache_data
def run_clustering_analysis(df, cols, n_clusters):
    data = df[cols].dropna()
    if data.empty: return None
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    pca = PCA(n_components=2)
    components = pca.fit_transform(data_scaled)
    res_df = data.copy()
    res_df['Cluster'] = clusters.astype(str)
    res_df['PC1'], res_df['PC2'] = components[:, 0], components[:, 1]
    res_df = res_df.join(df[['Gender', 'Age']], how='left')
    return res_df

@st.cache_data
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

def render_sidebar():
    """Функция отрисовки глобального бокового меню и фильтров с кэшированием между страницами"""
    import io
    
    # 1. РУКОВОДСТВО ДОСТУПНО ВСЕГДА (Даже до загрузки файла)
    with st.sidebar:
        if st.button("📖 Открыть руководство", use_container_width=True, type="primary"):
            show_help_dialog()
            
    # 2. ПРОВЕРКА ДАННЫХ
    if 'df_raw' not in st.session_state or st.session_state['df_raw'] is None:
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

    extra_cols = {
        'KMNS_Name': 'Конкретный народ (КМНС)', 'Family': 'Семейное положение',
        'Children': 'Наличие детей', 'University': 'Название ВУЗа',
        'Speciality': 'Специальность', 'Edu_Level': 'Уровень обучения', 'Edu_Basis': 'Основа обучения'
    }
    if 'f_extra' not in st.session_state: st.session_state['f_extra'] = {col: [] for col in extra_cols}

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

            with st.expander("Дополнительные фильтры", expanded=False):
                sel_extra = {}
                for col, label in extra_cols.items():
                    if col in df_raw.columns:
                        options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                        # Защита: проверяем, что сохраненные фильтры всё ещё существуют в опциях
                        saved_vals = [v for v in st.session_state['f_extra'].get(col, []) if v in options]
                        sel_extra[col] = st.multiselect(label, options, default=saved_vals)
                    else:
                        sel_extra[col] = []
                        
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
            st.session_state['f_extra'] = sel_extra

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
        
        for col, selected in st.session_state['f_extra'].items():
            if col in df_raw.columns and selected:
                mask &= df_raw[col].astype(str).isin(selected)

        df = df_raw[mask].copy()

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