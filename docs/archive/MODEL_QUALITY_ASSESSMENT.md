# Premier League Prediction Model — Quality Assessment

**Date:** September 20, 2026  
**Assessor:** AI Code Review  
**Model Version:** Quantitative Alpha & Value Scanner with DNB, Adaptive Shrinkage, and Logit-Space Blending

---

## Overall Rating: **6.5 / 10** — *Well-Engineered, But Edge Is Unproven*

The model's **engineering, risk management, and architecture** are excellent. However, the **empirical evidence for a real betting edge is inconclusive**. The market's probability estimates are still sharper than the model's, and the sample size from backtesting is far too small to confirm statistical significance.

---

## Empirical Evidence Summary

### Out-of-Sample Validation (2025/2026 Season)

| Metric | Calibrated AI | Naive Baseline | Fair Market (B365) |
|:---|:---|:---|:---|
| Accuracy | 46.8% | 44.9% | ~53–56% |
| Log Loss (lower is better) | 1.0354 | 1.0851 | 1.0199 |
| Brier Score (0 = perfect) | 0.6229 | 0.6565 | 0.6128 |
| Expected Calibration Error (ECE) | 0.0399 | N/A | 0.0502 |
| Goals MAE (Home / Away) | 0.94 / 0.86 | — | — |

### 4-Season Walk-Forward Backtest (2021/22 – 2024/25)

| Metric | Empirical Value | Target / Baseline |
|:---|:---|:---|
| Out-of-Sample Matches | 1,520 | 4 complete seasons |
| Disciplined Bets Triggered | 62 (~4.1% selection rate) | High-discipline hurdle |
| Calibrated AI Brier Score | **0.5658** | Close to sharp market (0.5594) |
| Expected Calibration Error (ECE) | **0.0375** | Superior (< 0.0500) |
| Pinnacle Closing Line Value (CLV) | **+0.38%** | 53.2% positive CLV rate |
| Aggregate Win Rate | **48.4%** | Balanced across [1.90, 2.80] band |
| Aggregate ROI | **+8.08%** | Realized profit: +$36.25 |
| Bootstrap Probability ROI > 0 | **75.3%** | Positive expectation |
| Maximum Bankroll Drawdown | **4.9%** | Resilient preservation (Tenth-Kelly) |

### Forward Live Results

| Metric | Status |
|:---|:---|
| Total Logged Predictions | 19 |
| Resolved Outcomes | **0** |
| Forward CLV | **No data** |
| Forward ROI | **No data** |

---

## Detailed Analysis

### What the Numbers Actually Say

| Metric | Your Model | The Market | Verdict |
|:---|:---|:---|:---|
| **Brier Score (Backtest)** | 0.5658 | 0.5594 | ❌ Market is sharper |
| **Brier Score (Validation)** | 0.6229 | 0.6128 | ❌ Market is sharper |
| **ECE (Calibration)** | 0.0399 | 0.0502 | ✅ Model probabilities are better calibrated |
| **Accuracy** | 46.8% | ~53–56% | ❌ Market wins outright |
| **Backtest CLV** | +0.38% | — | ⚠️ Essentially noise |
| **Backtest ROI** | +8.08% | — | ⚠️ Wide confidence interval |
| **Bootstrap P(ROI > 0)** | 75.3% | — | ⚠️ 1-in-4 chance model is actually losing |
| **Backtest Sample** | 62 bets / 4 seasons | — | ❌ Far too small for significance |
| **Live Forward Results** | 0 resolved | — | ❌ Zero real-world evidence |

---

## Key Concerns

### 1. Model Is Less Accurate Than the Bookmaker

The market's de-vigged Brier score beats the model in both the 4-season backtest (0.5594 vs 0.5658) and the out-of-sample validation (0.6128 vs 0.6229). If you strip away the risk controls and simply ask "who predicts match outcomes better?", the answer is the bookmaker — not the AI.

### 2. The +8% Backtest ROI Is Not Statistically Significant

With only 62 bets, the 95% bootstrap confidence interval spans approximately `[-16%, +40%]`. This range comfortably includes zero. The 75.3% probability of positive ROI sounds reassuring until you realize it means there's a **1-in-4 chance the model is a net loser** and the +8% was variance.

### 3. CLV of +0.38% Is Indistinguishable From Zero

Professional bettors target CLV of +2% to +5%. At +0.38% with a 53.2% positive CLV rate, this is within the margin of random noise. It does not demonstrate that you are systematically beating the closing line.

### 4. Zero Live Results Exist

Every single one of the 19 logged predictions is still `PENDING`. Until the model has a meaningful track record of forward live bets (minimum 75–100), the backtest results are hypothetical. Backtests can harbor subtle biases (feature leakage, lookahead, survivorship) that only forward testing exposes.

---

## Genuine Strengths

| Strength | Why It Matters |
|:---|:---|
| **Calibration (ECE 0.0399)** | Model probabilities mean what they say — better calibrated than the raw bookmaker odds |
| **Risk Architecture** | Tenth-Kelly, drawdown circuit breakers, adverse selection filters, concurrent bet dampening — professional-grade controls |
| **Discipline** | 4.1% selection rate means the model is not overtrading; it passes on 96% of matches |
| **DNB Expansion** | Draw No Bet routing is a smart structural advantage over pure 1X2 |
| **Infrastructure** | Automated odds scraping, CLV auditing, standardized logging — the pipeline is production-quality |
| **Adaptive Shrinkage** | Dynamic market weight that adjusts for season timing and promoted teams |
| **Logit-Space Blending** | Bayesian probability fusion that avoids edge distortion from linear blending |

---

## Is It Viable for Betting with Real Money?

**Not yet.** The following gates must be cleared first:

| Gate | Requirement | Current Status |
|:---|:---|:---|
| **1. Forward Paper Trading** | 75–100 live paper bets with tracked outcomes | 0 of 75 ✗ |
| **2. CLV Verification** | Average forward CLV ≥ +1.5% vs Pinnacle closing | No data yet ✗ |
| **3. Positive Forward ROI** | Demonstrated profit over ≥1 full season of live picks | No data yet ✗ |
| **4. Beat Market Brier** | Model Brier score < de-vigged market Brier score | Currently failing ✗ |

The risk controls will protect against catastrophic loss — the Tenth-Kelly sizing and 2.5% max stake cap mean even a losing model only bleeds slowly. But "won't blow up" is not the same as "has an edge."

---

## Recommended Next Steps

### Phase 1: Paper Trade the 2026/27 Season (Immediate)

1. Run `python main.py` each gameweek to log qualified bets into `data/prediction_log.csv`.
2. After matchday, run `python update_results.py` to grade outcomes and compute CLV.
3. Do **not** place real wagers during this phase.
4. Target: accumulate 75–100 resolved bets.

### Phase 2: Evaluate Forward Evidence (After ~75 Bets)

Examine the following metrics from live forward results:

- **Is forward CLV consistently ≥ +1.5%?** → This is the strongest indicator of a genuine edge.
- **Is forward ROI positive?** → Confirms the model generates profit on real market prices.
- **Is the model Brier score beating the market?** → Proves the AI's probability estimates are superior.

### Phase 3: Deploy Micro-Stakes (If All Gates Pass)

1. Segregate a dedicated bankroll ($200–$500) with zero impact on personal finances.
2. Retain Tenth-Kelly sizing (max 2.5% per bet).
3. Log every ticket strictly — never override the model manually.
4. Keep the drawdown circuit breaker active.

### Phase 4: Feature Enhancements (Ongoing)

| Enhancement | Expected Impact |
|:---|:---|
| Confirmed Starting Lineups (T-75 min pre-KO) | Capture team news arbitrage before soft books adjust |
| Shot-Level xG (FBref / Understat) | Replace box-score proxies with underlying chance creation |
| Asian Handicap Lines (+/-0.25, +/-0.5) | Lower-margin markets where edges are easier to capture |
| Multi-Bookmaker Odds Line Shopping | Capture best available price across 3–5 sportsbooks |
| Over/Under 2.5 Goals Classifier | Parallel revenue stream independent of match outcome |

---

## Bottom Line

The model is a **well-built machine with no proven fuel**. The risk management will protect from catastrophic losses, and the architecture is sound enough that if a real edge exists, it will capture it. But the current evidence — market-inferior Brier scores, negligible CLV, and a wide confidence interval — means the honest answer is:

**The model has not yet earned the right to be trusted with meaningful capital.**

Paper trade a full season. If the CLV holds up, then you have something.
