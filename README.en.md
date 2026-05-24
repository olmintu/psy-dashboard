# 🧠 Psy Dashboard

[Русский](README.md) | **English**

> ⚠️ **Note:** The dashboard interface is in Russian. This README is for international visitors who want to understand the project's scope and capabilities.

An interactive Streamlit dashboard for analyzing psychological survey data: descriptive statistics, group comparisons, correlations, cluster and factor analysis, regression, driver and anomaly detection, network and mediation analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://psy-res-dashboard.streamlit.app/)

## 🔗 Live Demo

👉 **[psy-res-dashboard.streamlit.app](https://psy-res-dashboard.streamlit.app/)**

Demo data (`TEST_RESULTS.xlsx`) is loaded automatically on first visit — you can explore all features immediately without uploading your own file.

> 💡 Inside the dashboard, the **"📖 Open Guide"** button in the sidebar opens a step-by-step manual covering all 11 modules with visual examples.

## 📖 About

The dashboard processes data collected by my [survey project](https://github.com/olmintu/psychology-research), which implements three psychological methodologies (Russian-language academic instruments):

- **Bratus Life Meanings Inventory** — 8 categories of life meanings (altruistic, existential, hedonistic, self-realization, status-related, communicative, family-related, cognitive)
- **Milman's Motivational Profile** — seven motivational dimensions (maintenance, comfort, status, communication, activity, creativity, social benefit) across two spheres (life / work) in two modes (ideal / actual), plus an emotional profile (sthenic / asthenic)
- **PIP (Personal Innovative Potential)** — composite score with components G/A/P (Gnoseological / Axiological / Praxeological), six realization types, and three style dimensions

The input is an Excel file with processed survey results; the dashboard handles everything else.

## 🚀 Features

The dashboard consists of 11 pages, each a separate analytical module. All work on top of a unified filtering system (gender, age, education, employment, indigenous status, anti-fraud flag, plus filters by derived types and profiles).

### 📊 1. Overview
- KPI metrics (total respondents, average age, dominant gender)
- Demographic chart builder: pie / bar / treemap
- Distribution plots: violin, boxplot
- Cross-tabulation for any two categorical features

### 🧩 2. Methodologies
- Modes "whole group" / "individual respondent"
- Milman profiles with frustration zones (gap between ideal and actual)
- Bratus life meanings ranking
- PIP structure and realization types

### 🆚 3. Group Comparison
- Automatic test selection: normality check → Welch's t-test / Mann-Whitney / ANOVA / Kruskal-Wallis
- Effect sizes (Cohen's d, η², rank-biserial, H-stat)
- Auto-scanner across all scales, sorted by p-value
- Excel export of the full report (including non-significant scales)

### 🔗 4. Correlations
- Spearman and Pearson with heatmaps
- FDR correction (Benjamini-Hochberg) and Bonferroni for multiple comparisons
- Filter connections by strength and significance
- "Cross-method only" mode to remove within-instrument noise

### 🔬 5. Cluster Analysis
- **Hierarchical clustering**: dendrograms of scales, respondents, and clustergram
- **K-Means**: automatic optimal *k* selection via silhouette score (2 to 10)
- Silhouette coefficient for quality assessment
- Cluster profiles with automatic p-value computation
- Cluster map in PCA coordinates
- PIP boxplots by groups
- Excel export of profiles and group composition

### 📐 6. Factor Analysis
- Three-level data adequacy check: n/vars ratio, KMO, Bartlett's test
- PCA with scree plot and Kaiser criterion
- EFA with extraction methods (Minres / ML / Principal Axis) and rotations (Varimax / Promax / Oblimin)
- Cronbach's alpha for internal consistency

### 🔮 7. Driver Discovery
- Random Forest for identifying the most influential factors
- Spearman vs. Pearson comparison (highlights non-linear relationships)
- Catalysts vs. blockers
- "What-if" simulator for predicting changes in the target variable

### 📈 8. Regression
Four approaches in a single tab:
- **Multiple regression (OLS)** — R², β-coefficients with p-values, confidence intervals, and model F-test
- **Hierarchical regression** — adding predictors in blocks with ΔR² (standard in psychological research)
- **Partial correlations** — relationship between X and Y while controlling for other variables
- **Regularized regression (Ridge / Lasso)** — for multicollinearity and automatic variable selection

### 👽 9. Anomaly Detection
- **Isolation Forest** for finding atypical profiles
- **Mahalanobis distance** (with χ² criterion at p < 0.001 / p < 0.01) — recommended for academic work
- "Anomaly X-ray": Z-scores across all scales for a specific respondent

### 🕸️ 10. Network Analysis
- Motive network graph: nodes = scales, edges = correlations
- Threshold slider for connection density
- Cross-method filter
- HTML export of interactive graph

### 🔀 11. Mediation
Tests the hypothesis "X affects Y through M" — five possible outcomes:
- Full mediation / Partial mediation / Direct effect without mediation / No effect / Suppression
- Computes direct (c'), indirect (a·b), and total (c) effects
- Significance test for the indirect effect

## 🛠️ Tech Stack

- **UI**: streamlit 1.52
- **Data**: pandas, numpy, openpyxl
- **Statistics**: pingouin, statsmodels, scipy
- **Factor analysis**: factor_analyzer
- **ML**: scikit-learn (PCA, KMeans, RandomForest, IsolationForest, Ridge, Lasso)
- **Graphs**: networkx
- **Visualization**: plotly

## 📦 Local Setup

```bash
# Clone the repository
git clone https://github.com/olmintu/psy-dashboard.git
cd psy-dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run Главная.py
```

The dashboard will open in your browser at `http://localhost:8501`.

> The entry-point file is named `Главная.py` (Russian for "Home") — Streamlit handles Unicode filenames without issues.

## 📁 Project Structure

```
psy-dashboard/
├── Главная.py                       # Entry point (file upload, filters)
├── pages/                           # 11 analytical modules
│   ├── 1_📊_Обзор.py                # Overview
│   ├── 2_🧩_Методики.py             # Methodologies
│   ├── 3_🆚_Сравнение.py            # Group comparison
│   ├── 4_🔗_Корреляции.py           # Correlations
│   ├── 5_🔬_Кластерный_анализ.py    # Cluster analysis
│   ├── 6_📐_Факторный_анализ.py     # Factor analysis
│   ├── 7_🔮_Поиск_драйверов.py      # Driver discovery
│   ├── 8_📈_Регрессия.py            # Regression
│   ├── 9_👽_Поиск_аномалий.py       # Anomaly detection
│   ├── 10_🕸️_Сетевой_анализ.py     # Network analysis
│   └── 11_🔀_Медиация.py            # Mediation
├── utils.py                         # Core: calculations, tests, caching, help
├── TEST_RESULTS.xlsx                # Demo data (loaded automatically)
├── requirements.txt                 # Dependencies
└── .devcontainer/                   # Codespaces configuration
```

## 📥 Input Data Format

The dashboard accepts an `.xlsx` file produced by the [survey project](https://github.com/olmintu/psychology-research). Expected column groups:

- **Demographics**: `FIO`, `Gender`, `Age`, `Course`, `Work`, `Edu_Status`, `Edu_Level`, `Edu_Basis`, `University`, `Speciality`, `Family`, `Children`, `Is_KMNS`, `KMNS_Name`, `Fast_Clicker`
- **Bratus**: `B_Altruistic`, `B_Existential`, `B_Hedonistic`, `B_Self-realization`, `B_Status`, `B_Communicative`, `B_Family`, `B_Cognitive`
- **Milman**: `M_{scale}_{sphere}-{mode}` (e.g., `M_K_Zh-id` — comfort, life, ideal) + `M_Est`, `M_East`, `M_Fst`, `M_Fast`
- **PIP**: `IPL_Total`, `IPL_G`, `IPL_A`, `IPL_P`, six types `IPL_Type_*`, four levels `IPL_Level_*`

Derived columns (Bratus levels, Milman motivational/emotional profiles, PIP styles, and structure) are computed automatically if missing from the file — for backward compatibility with older exports.

## 🔗 Related Projects

- 🧪 [psychology-research](https://github.com/olmintu/psychology-research) — the survey that collects data for this dashboard

## 📝 License

<!-- TODO: specify license (MIT / Apache-2.0 / unlicensed) -->

## 👤 Author

**olmintu** — [GitHub](https://github.com/olmintu)
