# Model Flaws & Recommended Solutions

This document details the critical flaws identified in the Premier League prediction model and provides actionable solutions for each, ordered by severity and priority.

---

## Flaw 1: Backtest ROI Is Not Statistically Significant

**Severity: 🔴 Critical**

### Problem

The walk-forward backtest reports **+4.58% ROI across only 96 bets**. With this sample size, the 95% confidence interval on true ROI spans roughly **-15% to +25%** (assuming standard deviation typical of 1X2 bets at ~2.30 average odds). This means the result is entirely compatible with **pure luck** — there is no statistical evidence of genuine edge.

Additionally, the model has **~8+ tunable hyperparameters** (odds band bounds, edge threshold, market prior weight, EMA alphas, Kelly fraction, XGBoost depth/learning rate) that were calibrated against these same 96 outcomes, creating a high risk of **curve-fitting** to historical noise.

### Recommended Solution

1. **Expand the backtest window** — include seasons prior to `2122` (e.g., `1819`, `1920`, `2021`) to increase the total out-of-sample bet count to **≥300–500 bets**. This narrows the confidence interval enough to distinguish a 4–5% edge from zero.

2. **Compute bootstrap confidence intervals** — resample the 96 bet outcomes with replacement (10,000 iterations) and compute the 2.5th and 97.5th percentile ROI. If the lower bound is negative, the result is not significant.

   ```python
   import numpy as np

   # bet_returns: array of per-bet returns (e.g., +1.30 for a win at 2.30 odds, -1.0 for a loss)
   boot_rois = []
   for _ in range(10_000):
       sample = np.random.choice(bet_returns, size=len(bet_returns), replace=True)
       boot_rois.append(sample.mean())
   ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])
   print(f"95% CI: [{ci_low:+.2%}, {ci_high:+.2%}]")
   ```

3. **Freeze hyperparameters** — once parameters are set, do not re-tune them on the same data. Any future adjustments should be validated on a completely held-out season not used during tuning.

---

## Flaw 2: Inconsistent Market-Anchor Blending Weights

**Severity: 🔴 Critical**

### Problem

The empirical Bayes market-anchor prior uses **different blending weights** across the three deployment scripts:

| Script | Market Weight | Model Weight | Code Location |
| :--- | :--- | :--- | :--- |
| `walkforward_backtest.py` | 50% | 50% | Line 110 |
| `predict_match.py` | 75% | 25% | Line 93 |
| `main.py` (live scanner) | 30% | 70% | Line 299 |

This means the backtest validates a 50/50 strategy, but live predictions execute a completely different 75/25 or 70/30 strategy. **The reported +4.58% ROI does not apply to the live system.**

### Recommended Solution

1. **Define a single `MARKET_PRIOR_WEIGHT` constant** in `utils.py` and import it everywhere:

   ```python
   # utils.py
   MARKET_PRIOR_WEIGHT = 0.60  # Cross-validated optimal weight
   ```

2. **Use it consistently** in all three scripts:

   ```python
   from utils import MARKET_PRIOR_WEIGHT

   blend = MARKET_PRIOR_WEIGHT * np.array(fair_mkt) + (1 - MARKET_PRIOR_WEIGHT) * np.array(model_probs)
   blend /= blend.sum()
   ```

3. **Cross-validate the weight** — run the walk-forward backtest with weights from `0.40` to `0.80` in steps of `0.05` and select the weight that maximizes ROI across all four test seasons. Then **freeze** it.

4. **Re-run the backtest** with the unified weight to get a single honest ROI number that matches the live system.

---

## Flaw 3: No Closing Line Value (CLV) Tracking

**Severity: 🔴 Critical**

### Problem

Closing Line Value — the percentage price improvement of the bet taken versus the closing market price — is the **single most reliable indicator of long-term profitability** in quantitative sports betting. A bettor who consistently beats the closing line by +2% to +3% will be profitable long-term regardless of short-term variance.

The prediction log schema in `utils.py` has `Closing_Odds` and `CLV` fields, but they are **never populated with real data**. Without CLV, there is no way to confirm the model has a genuine informational edge versus noise.

### Recommended Solution

1. **Record closing odds** — modify `update_results.py` to fetch Pinnacle or Betfair closing prices (available post-match from odds-archive sources like oddsportal.com or football-data.co.uk's closing columns `PSH`, `PSD`, `PSA`).

2. **Compute CLV on every settled bet**:

   ```python
   clv = (odds_taken / closing_odds) - 1.0
   ```

3. **Add a CLV summary to the backtest output**:

   ```
   Average CLV on Placed Bets:   +2.3%   (Target: >= +2.0%)
   ```

4. **Decision rule** — if average CLV across ≥100 bets is ≤ 0%, the strategy has no genuine edge and should not be deployed with real capital, regardless of P&L.

---

## Flaw 4: Model Calibration Still Loses to the Market

**Severity: 🟡 Medium-High**

### Problem

After all improvements, the model's aggregate Brier score is **0.5667** versus the market's **0.5594**. The bookmakers' probability estimates are still more accurate than the model's on average. The positive ROI comes entirely from **selective bet filtering** (narrow odds band + edge threshold) rather than from superior probability estimation.

This creates fragility — the model's profitability depends entirely on the specific filter parameters being correctly calibrated, not on a fundamental forecasting advantage.

### Recommended Solution

1. **Integrate xG data** — replace raw shot counts with non-penalty expected goals (npxG) from FBref or Understat. xG explains ~65% of season-to-season points variance versus ~35% for raw shots. This is the single highest-ROI feature improvement.

   ```python
   # Example: scrape FBref season xG summary
   # Features: npxG_for, npxG_against, xG_diff, big_chance_creation
   ```

2. **Add Isotonic or Beta calibration** — replace `method='sigmoid'` in `CalibratedClassifierCV` with `method='isotonic'` and evaluate whether Brier score improves. Sigmoid calibration can be poorly suited for multi-class 3-way probabilities.

3. **Hyperparameter tune XGBoost** — the current settings (`n_estimators=100`, `max_depth=3`, `learning_rate=0.03`) are hand-set. Use time-series cross-validation with a grid search:

   ```python
   param_grid = {
       'n_estimators': [80, 120, 160],
       'max_depth': [2, 3, 4],
       'learning_rate': [0.02, 0.05, 0.08],
       'reg_alpha': [0.5, 1.0, 2.0],
       'reg_lambda': [1.0, 2.0, 4.0]
   }
   ```

4. **Target**: Brier score ≤ 0.5550 (beating the market). Until this is achieved, the model should be treated as a filtering tool, not a forecasting engine.

---

## Flaw 5: No Lineup or Injury Data (Adverse Selection Vulnerability)

**Severity: 🟡 Medium-High**

### Problem

The model uses **zero information** about player availability, injuries, or squad rotation. The Premier League betting market incorporates lineup news within minutes of announcement (typically 1 hour pre-match). When a key player is ruled out:

1. The market adjusts odds (e.g., team price drifts from 1.40 to 1.85).
2. The model still sees historical form and computes a high win probability.
3. The model interprets the odds drift as "value" and bets aggressively.
4. In reality, the squad is weakened and the market is correct.

This is the **adverse selection trap** — the model systematically takes the wrong side whenever it disagrees with the market due to information it cannot see.

### Recommended Solution

1. **Integrate matchday lineup data** — available 1 hour pre-match via free APIs (e.g., API-Football, FotMob).

2. **Compute a Squad Strength Index** based on player market values or minutes-weighted contribution:

   ```python
   squad_strength = sum(starter_values) / sum(full_squad_values)
   # When key players missing, this ratio drops and the model adjusts
   ```

3. **Add a "Missing Player Impact" feature** — binary flag or continuous metric indicating whether ≥1 player in the typical starting XI is absent. Even a simple binary feature would reduce adverse selection.

4. **Short-term alternative** — until lineup integration is built, add a **market movement filter**: if the model's predicted probability diverges from the market by more than 15%, skip the bet (the market likely has information the model lacks).

   ```python
   if abs(model_prob - fair_market_prob) > 0.15:
       continue  # Market likely has private information
   ```

---

## Flaw 6: Season 2023/2024 Drawdown (-20.2% ROI)

**Severity: 🟡 Medium**

### Problem

Three of four backtest seasons produced double-digit positive returns, but 2023/2024 lost **-20.2% ROI on 31 bets** ($-91.25 net P&L). In real-capital terms, a 20% drawdown would:

- Erase approximately 2–3 seasons of accumulated profits.
- Create severe psychological pressure to abandon the strategy.
- Trigger bookmaker scrutiny if staking patterns change.

A robust strategy should have **maximum drawdown ≤ 10%** on any single season.

### Recommended Solution

1. **Analyze the 2023/2024 losing bets** — extract and examine each of the 31 bets to identify patterns (e.g., were losses concentrated on certain teams, odds ranges, or match weeks?).

2. **Add a drawdown circuit breaker** — if cumulative unrealized P&L within a rolling window of N bets drops below a threshold, reduce stake size or pause betting:

   ```python
   MAX_DRAWDOWN_PCT = 0.10  # 10% of bankroll
   LOOKBACK_BETS = 20

   recent_pnl = sum(last_20_bet_pnls)
   if recent_pnl / bankroll < -MAX_DRAWDOWN_PCT:
       stake *= 0.50  # Half-Kelly during drawdown
   ```

3. **Diversify across bet types** — re-enable Totals (Over/Under) once a Negative Binomial goal model with xG features is implemented (Tier 2/3 in roadmap). Concentration in 1X2-only creates correlated risk.

4. **Set a stop-loss rule** — define in advance: *"If season-to-date ROI drops below -15% at any point, pause all betting for 4 weeks and re-evaluate model parameters on fresh data."*

---

## Flaw 7: Zero Automated Tests

**Severity: 🟡 Medium**

### Problem

The codebase has **no unit tests, no integration tests, and no regression tests**. Critical numerical functions (`compute_ema`, `devig_odds_power`, `calculate_kelly`, feature generation) are untested. A single silent bug in any of these functions could cause the model to bet on incorrect probabilities without any warning.

### Recommended Solution

1. **Add unit tests** for core mathematical functions:

   ```python
   # test_utils.py
   def test_devig_odds_power_fair_market():
       """A fair market (no vig) should return the raw implied probabilities."""
       odds = [2.0, 3.5, 4.5]
       probs = devig_odds_power(odds)
       assert abs(sum(probs) - 1.0) < 1e-6
       assert all(p > 0 for p in probs)

   def test_kelly_rejects_odds_outside_band():
       """Odds outside [1.90, 2.80] should return zero stake."""
       assert calculate_kelly(prob=0.55, odds=1.50, bankroll=1000) == 0.0
       assert calculate_kelly(prob=0.55, odds=3.50, bankroll=1000) == 0.0

   def test_ema_single_value():
       """EMA of a single observation should return that observation."""
       assert compute_ema([3.0], alpha=0.25) == 3.0
   ```

2. **Add a regression test** — serialize the feature output for a known match and assert that future code changes reproduce the same features.

3. **Run tests in CI** — add a `pytest` step to any deployment or pre-commit workflow.

---

## Priority Execution Order

| Priority | Flaw | Effort | Impact |
| :--- | :--- | :--- | :--- |
| **P0** | #2 — Unify blending weights | 1 hour | Fixes strategy inconsistency |
| **P0** | #3 — Implement CLV tracking | 1 day | Validates whether edge is real |
| **P1** | #1 — Expand backtest + bootstrap CI | 1 day | Quantifies statistical confidence |
| **P1** | #7 — Add unit tests | 1 day | Prevents silent numerical bugs |
| **P2** | #4 — Integrate xG data | 1 week | Largest calibration improvement |
| **P2** | #5 — Add lineup/injury data | 1–2 weeks | Eliminates adverse selection |
| **P3** | #6 — Drawdown circuit breaker | 2 hours | Risk management safeguard |
