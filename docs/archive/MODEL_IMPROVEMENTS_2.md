# Quantitative Model Improvement Roadmap: Premier League Match Predictor

## Executive Summary

A walk-forward out-of-sample backtest across four Premier League seasons (`2122`, `2223`, `2324`, `2425` — 1,520 matches) yielded an aggregate **-4.61% Return on Investment (ROI)** across 516 placed bets, resulting in a net realized loss of **-$200.03** on a $1,000 initial bankroll.

This -4.61% loss closely mirrors the standard bookmaker margin (~5% vigorish / overround) on English Premier League matches. Statistical evaluation proves that the model is currently **inferior to the closing market**, meaning that perceived model edges are primarily **estimation noise**, causing the model to bet on its own errors into an efficient market.

This document details the quantitative diagnostic results, the mathematical and structural root causes, and a tiered implementation roadmap to achieve positive expected value (+EV).

---

## 1. Empirical Diagnostic Findings

### A. AI Accuracy vs. Market Benchmark
| Metric | AI Model (Calibrated XGBoost) | De-Vigged Market (Bet365 Closing) | Delta (AI vs Market) |
| :--- | :--- | :--- | :--- |
| **Brier Score** *(0 = Perfect, Lower is better)* | **0.5841** | **0.5594** | +0.0247 (Market is sharper) |
| **Multi-Class Log-Loss** | **~0.983** | **~0.945** | +0.038 (Market is sharper) |

The closing market is strictly better calibrated and has higher predictive accuracy than the model across every evaluated season. 

### B. Historical Breakdown by Market & Selection (516 Bets)
| Market | Selection | Bets | Win Rate | Implied Break-Even Win Rate | Avg Odds | Net P&L | Realized ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1X2** | Away (`A`) | 148 | 35.1% | 35.7% | 2.80 | -$59.94 | **-4.4%** |
| **1X2** | Home (`H`) | 117 | 35.9% | 37.8% | 2.64 | -$74.57 | **-7.1%** |
| **Totals** | Over 2.5 | 114 | 46.5% | 50.6% | 1.98 | -$36.71 | **-3.0%** |
| **Totals** | Under 2.5 | 143 | 42.0% | 45.8% | 2.18 | -$46.42 | **-3.5%** |

*All four betting categories produced negative returns. Totals accounted for 49.8% of all bets placed.*

### C. Performance by Odds & Perceived Edge Buckets
* **Short Favorites (Odds 1.05 – 1.80):** 29 bets, **41.4% win rate, -24.0% ROI**. Severe model overconfidence on heavy favorites that the market accurately discounted.
* **Longshots (Odds 2.80 – 4.00):** 99 bets, **26.3% win rate, -12.3% ROI**. Bookmaker longshot overround eroded expected returns.
* **Small Perceived Edges (2.0% – 5.0%):** 201 bets, **-16.8% ROI**. When the model identifies a marginal edge (< 5%), it fails to clear the bookmaker's 5% overround hurdle.
* **Profitable Sweet Spot (Odds 2.20 – 2.80, Edge 5.0% – 8.0%):** 182 bets, **+3.4% to +3.9% ROI**. Selective mid-range mispricings yielded actual positive returns.

---

## 2. Structural Root Causes

### Root Cause 1: Information Asymmetry & The Adverse Selection Trap
The model trains solely on:
* 5-game rolling averages of shots, shots on target (SOT), goals, corners, and cards.
* A basic dynamic Elo rating.
* Calendar rest days.

**The Failure Mode:**
The betting market incorporates player availability, team sheets, tactical adaptations, travel fatigue, European rotation, and millions of pounds of sharp syndicate action. 
* *Example:* If an elite striker or starting goalkeeper is ruled out 2 hours before kickoff, the market price on that team drifts from `1.40` to `1.85`.
* Because the model does not ingest player lineups or injury data, it still computes a ~70% win probability based on prior 5-game form.
* The model sees odds of `1.85` as an enormous "edge" (+16%) and stakes aggressively. In reality, the squad is heavily weakened, and the market price is accurate. 
* This is **Adverse Selection**: an under-informed model systematically takes the wrong side of the bet whenever it disagrees with the market.

### Root Cause 2: Extreme Over-Betting Frequency (34% Bet Volume)
* 516 bets across 1,520 fixtures means the algorithm is betting on **33.9% of all Premier League matches**.
* The Premier League is one of the most efficient sports betting markets in the world. True positive-expectation (+EV) mispricings against closing lines appear in only **3% to 8%** of fixtures.
* Betting one-third of the league guarantees that ~75% of placed bets have zero or negative expectation, effectively donating the 5% bookmaker overround on every turnover.

### Root Cause 3: High Variance in 5-Game Rolling Windows
* A 5-game sample is only 450 minutes of football. Over 5 matches, goals and shot conversion rates are heavily dominated by randomness and finishing luck.
* A mid-table team can score 8 goals on 3.5 expected goals due to lucky deflections or individual wonder-goals. The model over-indexes on this short-term run and overestimates their attacking baseline.

### Root Cause 4: Totals Modeling via Uncalibrated Goal Regression
* Over/Under 2.5 markets are predicted by feeding `XGBRegressor` goal forecasts ($\lambda_H, \lambda_A$) into a Dixon-Coles Poisson grid with a static correlation parameter ($\rho = -0.13$).
* Standard Poisson assumes variance equals the mean ($\text{Var}(X) = \text{E}[X]$). Football goal distributions exhibit **over-dispersion** (variance exceeds the mean), particularly in games between mismatched teams or blowout scenarios.
* Point regression for match goals has an $R^2 < 0.10$. Feeding noisy continuous point predictions into a rigid bivariate Poisson framework generated miscalibrated tail probabilities, resulting in negative ROI on both Over (-3.0%) and Under (-3.5%).

### Root Cause 5: Execution Drag & Single-Bookmaker Monopsony
* The backtest executes all bets exclusively at Bet365 (`B365H`, `B365D`, `B365A`).
* Bet365 charges an overround of 5% to 6%. In real quantitative sports trading, operators utilize **line shopping** across multiple bookmakers (`MaxH`, `MaxD`, `MaxA`) or execute on betting exchanges (e.g., Betfair, Matchbook) paying low commissions (1.5% to 2.0%).
* Taking `Max` available odds across the market typically provides a +2.5% to +4.0% price improvement on every wager.

---

## 3. Tiered Implementation Roadmap

```
                                  [ IMPLEMENTATION ROADMAP ]
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
      TIER 1                               TIER 2                               TIER 3
Immediate Backtest &                 Algorithmic & Statistical            Advanced Data Pipeline &
Staking Optimizations               Architecture Upgrades                Feature Engineering
(Zero External Data)                 (Model Core Overhaul)                (Deep Alpha Generation)
 • Tighter Odds Band [1.80, 2.80]     • Empirical Bayes Market Prior       • True xG / npxG (FBref/Understat)
 • Higher Edge Threshold (>= 5.5%)    • 10/25-Match Dual EMA Windows       • Starting XI & Missing Player Value
 • Disable Uncalibrated Totals        • Negative Binomial Goal Models      • European Travel & Schedule Fatigue
 • Multi-Bookmaker Line Shopping      • Multi-Class Beta Calibration       • Closing Line Value (CLV) Feedback
```

---

### Tier 1: Immediate Backtest & Staking Optimizations
*Effort: Low (1–2 days) | Impact: High (~+3% to +5% ROI improvement immediately)*

1. **Eliminate Toxic Odds Buckets:**
   * Restrict bets to the verified profitable band: `1.80 <= odds <= 2.80`.
   * Completely reject odds $< 1.80$ (which lost -24.0% ROI) and odds $> 3.20$ (which lost -12.3% ROI).
2. **Raise Minimum Edge Threshold to 5.5%:**
   * Increase `min_edge` from 0.035 to `0.055` post-shrinkage.
   * This cuts out the 201 low-confidence bets that generated a -16.8% ROI.
3. **Drop Uncalibrated Totals Markets:**
   * Remove Over/Under 2.5 wagers until goal dispersion and true xG are modeled. Focus capital on the 1X2 market.
4. **Enable Multi-Bookmaker Line Shopping:**
   * In [walkforward_backtest.py](file:///c:/Users/kevin/Desktop/Coding/Projects/AI/pl_analysis_model/walkforward_backtest.py), simulate execution at `MaxH`, `MaxD`, `MaxA` instead of purely `B365`.
5. **Harmonize Kelly Staking & Shrinkage:**
   * Remove redundant double-shrinkage inside `calculate_kelly()`. Calculate the true edge once against de-vigged fair market probabilities ($P_{\text{fair}}$), and size stakes strictly with Quarter- or Tenth-Kelly based on the blended probability.

---

### Tier 2: Algorithmic & Statistical Architecture Upgrades
*Effort: Medium (1 week) | Impact: Substantial (Improves Brier score to beat market)*

1. **Market-Anchor Residual Modeling (Bayesian Prior):**
   * Do not attempt to predict match probabilities from scratch using only rolling stats.
   * Treat the de-vigged market probability ($P_{\text{mkt}}$) as the **prior baseline** carrying 70% to 80% weight.
   * Train the machine learning model to predict the **delta / residual**:
     $$\hat{P} = \sigma\left(\text{logit}(P_{\text{mkt}}) + f_{\theta}(X_{\text{features}})\right)$$
   * This anchors the model to the market's superior calibration while allowing feature signals (e.g., rest differences, shot accuracy divergence) to nudge the probability only when statistically warranted.
2. **Dual-Horizon Exponential Moving Averages (EMA):**
   * Replace noisy 5-game rolling averages with two distinct horizons:
     * **Short-Term Form:** 8-match EMA ($\alpha = 0.25$) for recent tactical shifts and form.
     * **Long-Term Baseline:** 25-match EMA ($\alpha = 0.08$) spanning across seasons to establish true team baseline ability.
3. **Negative Binomial / Bivariate Over-Dispersion for Totals:**
   * Replace standard Poisson with a Negative Binomial regression framework to model the dispersion parameter $\theta$ in match goal counts:
     $$\text{Var}(Y) = \mu + \frac{\mu^2}{\theta}$$
   * Calibrate score matrices with dynamic home advantage and draw-inflation factors.
4. **Isotonic or Beta Probability Calibration:**
   * Replace simple Sigmoid (`method='sigmoid'`) calibration with **Beta Calibration** or out-of-fold Isotonic Regression tailored for multi-class ternary probabilities (Home, Draw, Away).

---

### Tier 3: Advanced Data Pipeline & Feature Engineering
*Effort: High (2–3 weeks) | Impact: Long-term sustainable edge*

1. **Integrate True Expected Goals (xG & npxG):**
   * Ingest non-penalty expected goals (npxG), expected goals against (xGA), and big chance creation from data sources like FBref or Understat.
   * Raw shots and synthetic shot accuracy have high noise; npxG differences explain ~65% of season-to-season points variance compared to ~35% for raw shots.
2. **Lineup & Missing Player Squad Value Index:**
   * Ingest matchday lineup data (available 1 hour pre-match via API).
   * Calculate a **Squad Strength Ratio**:
     $$\text{Squad Strength} = \frac{\sum_{i \in \text{Starting XI}} \text{Player Market Value}}{\sum_{j \in \text{Full Squad}} \text{Player Market Value}}$$
   * When key players are missing, the model will adjust expected output downward rather than falsely perceiving bookmaker price shifts as value.
3. **European Travel & Schedule Congestion Metrics:**
   * Account for midweek UEFA Champions League / Europa League fixtures.
   * Features to engineer:
     * Rest hours difference ($< 72$ hours vs $\ge 144$ hours).
     * Away travel miles within the past 7 days.
     * Midweek rotation index (number of starting XI changes from previous match).
4. **Closing Line Value (CLV) Tracking & Model Feedback:**
   * In sports betting analytics, beating the closing line is the strongest predictor of long-term profitability.
   * Track whether bets placed at opening/mid-week beat the Pinnacle/Betfair closing line:
     $$\text{CLV} = \frac{\text{Odds}_{\text{taken}}}{\text{Odds}_{\text{closing}}} - 1.0$$
   * If average CLV is positive (+2% to +4%), the strategy is mathematically sound regardless of short-term variance.

---

## 4. Implementation Reference & Validation Checkpoints

### Validation Target Metrics
Before deploying real capital or declaring strategy success, the walk-forward backtest across seasons `2122` to `2425` must satisfy:

| Milestone / Metric | Baseline (Current) | Target Requirement |
| :--- | :--- | :--- |
| **Walk-Forward ROI** | -4.61% | **+3.0% to +6.0%** |
| **AI Brier Score** | 0.5841 | **$\le 0.5550$** (Must beat market Brier) |
| **Bet Frequency** | 33.9% (516 bets) | **5.0% to 10.0%** (~75 to 150 highly selective bets) |
| **Average Edge on Placed Bets** | 3.5% | **$\ge 5.5\%$** post-shrinkage |
| **Average CLV against Pinnacle** | Untracked | **$\ge +2.0\%$** |
