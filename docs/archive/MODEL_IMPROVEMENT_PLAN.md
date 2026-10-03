# Premier League Prediction Model — Strategic Improvement Plan

**Goal:** Transform the model from a well-engineered 6.5/10 system into a production-grade 9/10 system that consistently beats the sharp bookmaker closing lines (CLV > +2.0%) and generates statistically significant positive ROI.

This roadmap is divided into three tiers: **Data (The Fuel)**, **Modeling (The Engine)**, and **Market/Execution (The Edge)**.

---

## 1. Data Enhancements (High Impact)
*Your model is currently limited by its inputs. The market knows who is injured; your model only knows what the team did over the last 10 games.*

### A. Integrate Starting Lineups & Player Value
*   **The Flaw:** The model evaluates "Man City" as a static entity. If Haaland and De Bruyne are injured, the bookmaker adjusts immediately. Your model does not, leading to false "value" bets (adverse selection).
*   **The Fix:** Incorporate Transfermarkt squad values or FIFA/FM player ratings. Weight the team's Elo or strength based on the *actual starting XI* rather than the club badge.
*   **Implementation:** Fetch lineup data at T-60 mins (when announced) and apply a scalar multiplier to the team's baseline ratings.

### B. Upgrade from "Shots" to "Shot-Level Expected Goals (xG)"
*   **The Flaw:** The model uses `Home_V_Shots` and `Home_V_SOT`. A 35-yard prayer counts the same as a tap-in.
*   **The Fix:** Integrate FBref/Understat xG data.
*   **Implementation:** Replace aggregate shot counts with Non-Penalty Expected Goals (npxG) and Expected Threat (xT) to better measure true underlying performance.

### C. Use "Wisdom of the Crowds" as a Feature
*   **The Flaw:** The model builds its probabilities completely isolated from the market, then compares at the end.
*   **The Fix:** Use the bookmaker's opening line (or Pinnacle's early line) as an input feature in your XGBoost model.
*   **Implementation:** Let the AI learn *when* the bookmaker is typically wrong, rather than trying to out-predict the bookmaker from scratch.

---

## 2. Modeling Enhancements (Medium Impact)
*Your XGBoost setup and sigmoid calibration are solid, but football is a low-scoring game best modeled by goals, not categories.*

### A. Move to Bivariate Poisson / Dixon-Coles
*   **The Flaw:** `XGBClassifier` treats Home Win, Draw, and Away Win as discrete categories. It loses the structural relationship between goals scored.
*   **The Fix:** Model the *number of goals* each team scores (using Poisson distribution), then derive the 1X2 and Draw No Bet probabilities mathematically from the goal matrix.
*   **Implementation:** Introduce a Dixon-Coles time-decay model. This also naturally provides probabilities for Over/Under 2.5, Correct Score, and Asian Handicaps for free.

### B. Implement Time-Decayed Sample Weights
*   **The Flaw:** A match played in August 2021 might hold too much influence over a prediction for May 2026.
*   **The Fix:** Apply exponential time decay to `Sample_Weight` in the training phase.
*   **Implementation:** `weight = exp(-alpha * days_since_match)`. This forces the model to respect recent form more than historical data, while still maintaining a large dataset.

### C. Ensemble Architecture
*   **The Flaw:** Relying purely on gradient boosting (XGBoost) can overfit to specific feature interactions.
*   **The Fix:** Blend the XGBoost output with a purely statistical model (like Elo + Poisson).
*   **Implementation:** If XGBoost predicts 45% Home Win, and Dixon-Coles predicts 43%, take the average (44%). Ensembling reduces variance and improves the Brier score.

---

## 3. Market & Betting Strategy (High Impact)
*Even the best models lose money if they bet into the wrong markets.*

### A. Pivot to Asian Handicaps (AH)
*   **The Flaw:** You are betting into the 1X2 market, which has high margins (vig) and high variance (draws ruin bets). Draw No Bet is better, but still limited.
*   **The Fix:** Expand into Asian Handicaps (e.g., Home -0.5, Away +1.25). 
*   **Implementation:** Asian Handicap markets are the most liquid and mathematically pure markets in football. Edges here are smaller but much more consistent.

### B. Pinpoint Line Shopping
*   **The Flaw:** If you only take odds from one or two soft bookies, you might be taking 2.10 when 2.25 is available elsewhere.
*   **The Fix:** Use The Odds API to scan 10+ bookmakers simultaneously.
*   **Implementation:** Only execute the bet if the best available price exceeds your minimum edge threshold. A +0.15 difference in odds can turn a losing model into a winning one.

### C. Track Pinnacle Closing Line Value (CLV)
*   **The Flaw:** You don't know if your backtest ROI is luck or skill.
*   **The Fix:** Treat Pinnacle's closing odds (right before kickoff) as the absolute source of truth.
*   **Implementation:** If you bet Home at 2.20, and Pinnacle closes at 2.05, you have generated +7.3% CLV. If your model consistently generates CLV over 200+ bets, you have a winning model, regardless of short-term variance.

---

## The Immediate Roadmap (Next 30 Days)

1. **Week 1:** Replace basic shot stats with FBref xG data in the historical dataset.
2. **Week 2:** Implement an exponential time-decay function for training weights.
3. **Week 3:** Modify the Odds API scanner to track Asian Handicap markets alongside 1X2.
4. **Week 4:** Set up automated CLV logging against Pinnacle 5 minutes before kickoff for all paper bets.
