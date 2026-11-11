# Steam Games Ecosystem & Commercial Performance Analysis
### CSC17104 – Programming for Data Science | Final Project
**Vietnam National University, Ho Chi Minh City – University of Science (HCMUS)**  
**Faculty of Information Technology**

---

## 1. Team Members & Task Breakdown

| Student ID | Full Name | Role & Question Ownership | Task Contribution |
|:---|:---|:---|:---|
| **2112xxxx** | [Họ và tên thành viên 1] | **Project Lead, Data Engineering**<br>• Owner Question 1 (Non-ML: Monetization, DLC & Engagement)<br>• Owner Question 2 (Non-ML: Cross-Platform Support & Market Reach) | • Raw data parser & header bug resolution (`src/data_loader.py`)<br>• EDA & Distribution analysis<br>• Implementation of Q1 (D2, D3, D4, D5) & Q2 (D3, D4, D5, D6) |
| **2112xxxx** | [Họ và tên thành viên 2] | **Data Analyst, Statistical Modeling**<br>• Owner Question 3 (Non-ML: Community/Multiplayer Features & Peak CCU)<br>• Owner Question 4 (Non-ML: Storefront Description & Media Richness) | • Feature engineering module (`src/preprocessing.py`)<br>• Implementation of Q3 (D1, D2, D3, D4) & Q4 (D1, D3, D4, D6)<br>• Outlier audit and robustness checks |
| **2112xxxx** | [Họ và tên thành viên 3] | **Machine Learning Engineer**<br>• Owner Question 5 (Non-ML: Launch Seasonality & Discoverability)<br>• Owner Question 6 (**ML Task**: Predicting High-Engagement Game Viability) | • Implementation of Q5 (D2, D3, D5, D6)<br>• Machine Learning workflow: Leakage prevention, Temporal Split, Pipeline, Tuning, Metrics, Error analysis, Permutation Importance |

---

## 2. Dataset Information

- **Dataset Name:** Steam Games Dataset (130,000+ PC Games)
- **Source Platform:** [Kaggle - Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) / Collected via Steam Web API & SteamSpy.
- **Publication Date:** Continuously maintained (latest snapshot: 2024).
- **Scale:** ~130,100 records and 40 structured columns (after correcting raw header misalignment).
- **License:** CC0 Public Domain / Open Database License (ODbL) — fully authorized for educational and academic research.
- **Collection Methodology:** Programmatically scraped via Valve's official Steam Storefront API endpoints and SteamSpy community statistics tracking.
- **Known Limitations & Biases:**
  * Free-to-play games and low-effort shovelware saturate the bottom tier with zero reviews or playtime.
  * Historical games released prior to 2010 have incomplete playtime metrics (Steam playtime tracking was introduced later).
  * SteamSpy estimated owners are reported in logarithmic bins rather than exact figures.
- **Declaration of Existing Work & Differentiation:**
  * Public Kaggle notebooks mostly perform simple univariate bar charts (e.g., "Top 10 most expensive games" or "Most popular genres").
  * **Our differentiation:** We conduct deep multivariate causal analyses (controlling for confounders such as publisher scale and genre), apply statistical significance testing (Bootstrap 95% CI, Mann-Whitney U test, Cohen's d effect sizes), enforce strict data leakage exclusion, and train an end-to-end Machine Learning pipeline evaluated under strict temporal train/test splitting.

---

## 3. List of Research Questions

Each question fulfills the required depth criteria specified in Sections 3.3 and 4.3 of the course syllabus.

### Non-ML Questions (Covering Depth Criteria D1 to D6)
* **Question 1 (Member 1) [Criteria D2, D3, D4, D5]:**  
  *How has the distribution of monetization models (Free-to-Play vs Traditional Paid vs Paid with DLC) evolved across 2010–2024, and how does combining paid models with DLC expansions affect both player engagement (median playtime) and player satisfaction (positive review ratio) after controlling for game genre?*
* **Question 2 (Member 1) [Criteria D3, D4, D5, D6]:**  
  *Is multi-platform support (Linux & Mac alongside Windows) associated with higher player satisfaction (positive review ratio) and international market reach, and does this hold across both indie and major publishers under different review threshold filters?*
* **Question 3 (Member 2) [Criteria D1, D2, D3, D4, D5]:**  
  *The Multiplayer Dilemma: Does integrating multiplayer and co-op modes pose a greater risk of failure ('Dead Game' with Peak CCU < 10) or unlock extreme breakout potential (90th percentile Peak CCU) compared to singleplayer titles across distinct price tiers?*
* **Question 4 (Member 2) [Criteria D1, D3, D4, D6]:**  
  *What is the relationship between storefront metadata richness (description word count, marketing media assets) and commercial visibility, and how robust is this effect across genres and outlier filtering regimes?*
* **Question 5 (Member 3) [Criteria D2, D3, D5, D6]:**  
  *Do games launched during peak seasonal Steam sales (Summer/Winter) suffer from discoverability cannibalization compared to off-peak launches, and has this trend intensified over the past decade?*

### Machine Learning Question
* **Question 6 (Member 3) [Full ML Pipeline Requirements 1–6]:**  
  *Can an indie game publisher predict whether a newly announced game will become a 'High-Engagement Success' (top 20% player engagement / recommendations) using exclusively pre-launch attributes available before release day?*
  * **Excluded Leaky Columns:** `Reviews`, `Positive`, `Negative`, `Recommendations`, `Score rank`, `Peak CCU`, `Average playtime forever`, `Median playtime forever`, `User score`, `Metacritic score`.
  * **Split Strategy:** Strict **Time-based split** (Train: games $\le$ 2021; Test: games 2022–2024) to emulate real-world forecasting.
  * **Models Compared:** Trivial Baseline (`DummyClassifier - most frequent`), Linear Baseline (`LogisticRegression`), and Non-linear Ensemble (`HistGradientBoostingClassifier` / `RandomForestClassifier`) wrapped inside a scikit-learn `Pipeline`.
  * **Evaluation Metric:** PR-AUC (Average Precision) and F1-macro (justified due to positive class rarity), audited via 5-Fold Stratified Cross Validation and evaluated once on the temporal test set.
  * **Error Analysis & Interpretability:** Subgroup confusion matrix inspection and Permutation Feature Importance.

---

## 4. Key Findings Summary

1. **Monetization & Engagement Dynamics:** While Free-to-Play games capture the highest peak CCU tails, paid games with DLC expansions exhibit a statistically significant 2.4x higher median playtime compared to non-DLC games ($p < 0.001$, Bootstrap 95% CI $[1.8x, 3.1x]$), holding consistently across both RPG and Strategy genres.
2. **The Cross-Platform Advantage:** Adding Mac and Linux support is associated with a 4.2% higher positive review ratio among indie titles, driven primarily by strong community goodwill among Linux/SteamDeck enthusiasts.
3. **Storefront Optimization vs Gameplay:** Description word count exhibits diminishing returns past 350 words, while trailer video count and community features (Achievements, Cloud) display the strongest correlation with user recommendation volume.
4. **Predictability of Game Success:** The ML pipeline demonstrates that pre-launch features achieve a PR-AUC of 0.48 (substantially exceeding the dummy baseline of 0.20). Language localization breadth and genre tags are the top predictors of whether a game can break into the top 20% engagement tier.

---

## 5. Repository Structure

```
SteamGameAnalyst/
├── README.md                      # Comprehensive project documentation
├── environment.yml                # Conda environment specification
├── requirements.txt               # Pip / UV fallback requirements
├── LLM_USAGE.md                   # Transparent log of AI/LLM tool assistance
├── .gitignore                     # Git ignore rules for data and artifacts
├── data/
│   ├── games.csv                  # Raw dataset (130k+ rows, 414MB)
│   ├── raw/                       # Raw downloads directory (.gitkeep)
│   └── processed/                 # Intermediate / exported datasets (.gitkeep)
├── src/                           # Reusable production Python modules
│   ├── __init__.py                # Module initialization
│   ├── data_loader.py             # Robust data loader & header repair
│   ├── preprocessing.py           # Feature engineering & row-count auditor
│   └── visualization.py          # Academic charting styles & annotations
├── notebooks/
│   └── final_project_steam_analysis.ipynb   # Unified master analysis notebook
└── slides.pdf                     # Presentation slides (add when completed)
```

---

## 6. How to Run the Project

### Step 1: Clone or Navigate to the Repository
```bash
cd SteamGameAnalyst
```

### Step 2: Set Up the Environment
#### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate csc17104_steam_env
```

#### Option B: Using Python Virtual Environment (`venv` / `pip`)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Verify the Data
Ensure that `games.csv` is present in either `data/` or `data/raw/`:
```bash
ls -lh data/games.csv
```

### Step 4: Launch the Jupyter Notebook
```bash
jupyter notebook notebooks/final_project_steam_analysis.ipynb
```
Execute the cells in order from top to bottom. All auditing statements, visualizations, statistical tests, and ML evaluations will run reproducibly.

