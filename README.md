# Premier League Betting AI ⚽🔮

Welcome! This is a state-of-the-art sports betting AI engine built for the **Premier League (soccer)**. 

It uses a **Hybrid Machine Learning Architecture** combining:
1. **Expected Goals (xG) Regressors:** To predict goal counts, totals (Over/Under 2.5), and Both Teams to Score (BTTS).
2. **Direct 1X2 Outcome Classifier:** A highly tuned multi-class model that bypasses traditional Poisson independence limitations to directly predict the probability of a **Home Win**, **Draw**, or **Away Win**.
3. **Kelly Criterion Staking:** Suggests mathematical bet sizing based on your edge against the bookmakers' odds.

---

## 🛠️ Step 1: One-Time Setup (For Beginners)

Before running the AI, you need to prepare your computer. Follow these quick steps:

### 1. Create a Virtual Environment
A virtual environment keeps all the project's packages isolated. Open your terminal in this folder and run:
* **Windows:**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
* **Mac / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Install Packages
Install the required machine learning libraries (XGBoost, Scikit-learn, etc.):
```bash
pip install -r requirements.txt
```

### 3. Add your API Key
Create a file named `.env` in the root folder of this project (or edit the existing one) and add your [The Odds API](https://the-odds-api.com/) key:
```env
ODDS_API_KEY="your_api_key_here"
```

---

## 🚀 Step 2: The Daily Betting Routine (How to Run)

The project includes four simple scripts. Run them inside your active terminal (`venv` active) in this order:

### 📊 1. Run the Automated Live Odds Scanner (`main.py`)
This script downloads historical Premier League results, trains the AI, grabs live odds from bookmakers, and prints a grid showing exactly where mathematical edges exist.
```bash
python main.py
```
* **What it does:** Scans bookmakers to find cases where the AI's calculated probability is significantly higher than the bookmaker's implied probability (giving you a mathematical edge of +5% or more) and suggests how much money to bet.

### 🔮 2. Run the Manual Match Predictor (`predict_match.py`)
Want to predict a matchup manually (including games not currently in the live feeds)?
```bash
python predict_match.py
```
* **How to use it:** Enter the Home Team name, Away Team name, and the decimal odds offered by your bookmaker.
* **Important:** Make sure to use the **official team names** exactly as written (e.g. `Man City`, `Chelsea`, `Arsenal`, `Nott'm Forest`, `Man Utd`).

### 🔗 3. Calculate Multi-Leg Parlays (`parlay_calc.py`)
If you want to combine multiple predictions into a single parlay bet:
```bash
python parlay_calc.py
```
* **What it does:** Asks you how many legs are in the parlay, lets you input the AI probabilities and bookmaker decimal odds for each leg, and outputs the overall combined success probability and expected return.

### 🔄 4. Update and Sync Results (`update_results.py`)
Run this script the morning **after** matches are played to calculate your bankroll performance and P&L.
```bash
python update_results.py
```
* **What it does:** Automatically fetches actual match results, reconciles them against your logged predictions in `data/prediction_log.csv`, grades H2H, Totals, and BTTS bets (`WON` or `LOST`), and outputs a complete bankroll audit (Total Staked, Net P&L, and ROI %).

### 📈 5. Run the Multi-Season Walk-Forward Backtest (`walkforward_backtest.py`)
Want to verify the mathematical reliability of the model across past Premier League seasons?
```bash
python walkforward_backtest.py
```
* **What it does:** Performs an expanding-window chronological walk-forward test across 4 historical seasons. Computes out-of-sample Brier scores, compares against Bet365 closing lines, and simulates Tenth-Kelly bankroll trajectory.

---

## 🏟️ Official Team Names Reference
When using the manual predictor (`predict_match.py`), you must type these names exactly as written (case-sensitive):

| | | | |
| :--- | :--- | :--- | :--- |
| **Arsenal** | **Chelsea** | **Leeds** | **Nott'm Forest** |
| **Aston Villa** | **Crystal Palace** | **Liverpool** | **Sunderland** |
| **Bournemouth** | **Everton** | **Man City** | **Tottenham** |
| **Brentford** | **Fulham** | **Man Utd** | **West Ham** |
| **Brighton** | **Ipswich** | **Newcastle** | **Wolves** |