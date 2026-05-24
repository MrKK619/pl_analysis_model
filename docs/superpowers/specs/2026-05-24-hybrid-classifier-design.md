# Design Spec: Hybrid Sports Betting AI Model (Goal Regressors + Direct 1X2 Classifier)

**Date:** 2026-05-24  
**Status:** Approved  
**Topic:** Transitioning from a pure Poisson-derived probability model to a hybrid architecture featuring two goal regressors (for totals/BTTS) and a direct match outcome classifier (for H2H).

---

## 1. Executive Summary
The goal of this design is to improve prediction accuracy for match outcomes (Home Win, Draw, Away Win) in the Premier League Betting AI. The current Poisson-based probability model assumes goal independence between teams, which historically underrepresents draws and fails to capture the true joint probability distribution. 

By introducing a dedicated `XGBClassifier` trained directly on match outcomes, we can significantly reduce Log Loss on H2H predictions while retaining the existing `XGBRegressor` goal models to calculate Poisson grids for Totals (Over/Under 2.5) and Both Teams to Score (BTTS).

---

## 2. Model Architecture
The system will run three distinct models trained concurrently:

1. **Home Goal Regressor (`model_h` - XGBRegressor):** Predicts expected home goals (`FTHG`).
2. **Away Goal Regressor (`model_a` - XGBRegressor):** Predicts expected away goals (`FTAG`).
3. **Outcome Classifier (`model_clf` - XGBClassifier):** Predicts probabilities for Home Win, Draw, and Away Win directly.

```
                  [ Pre-Match Rolling Stats ]
                 /             |             \
                /              |              \
 [ Home Goals Regressor ]  [ Away Goals Regressor ]  [ Direct 1X2 Classifier ]
        |                          |                         |
  (Predicts xGH)             (Predicts xGA)           (Predicts H/D/A Probs)
         \                        /                          |
        [ Poisson Probabilities Grid ]                       |
         /                        \                          |
 (Over/Under 2.5)              (BTTS)                        |
        \                         /                          /
         \                       /                          /
        [ Multi-Market Kelly Staking & Betting Scanner ]
```

---

## 3. Feature Selection & Target Mapping

### A. Goal Regressors (Unchanged for compatibility)
* **Home Goals Features (`features_h`):**
  * `HomeID`, `AwayID`, `Home_Roll_Goals_Scored`, `Away_Roll_Goals_Conceded`, `Home_Roll_Shots`, `Away_Roll_Shots`, `Home_Roll_SOT`, `Away_Roll_SOT`, `Home_Roll_Corners`, `Away_Roll_Corners`
* **Away Goals Features (`features_a`):**
  * `HomeID`, `AwayID`, `Away_Roll_Goals_Scored`, `Home_Roll_Goals_Conceded`, `Home_Roll_Shots`, `Away_Roll_Shots`, `Home_Roll_SOT`, `Away_Roll_SOT`, `Home_Roll_Corners`, `Away_Roll_Corners`

### B. Outcome Classifier
* **Classifier Features (`features_clf`):**
  * `HomeID`, `AwayID`
  * `Home_Roll_Goals_Scored`, `Home_Roll_Goals_Conceded`, `Home_Roll_Shots`, `Home_Roll_SOT`, `Home_Roll_Corners`
  * `Away_Roll_Goals_Scored`, `Away_Roll_Goals_Conceded`, `Away_Roll_Shots`, `Away_Roll_SOT`, `Away_Roll_Corners`
* **Target Variable (`FTR` - Full-Time Result):**
  * Mapped as:
    * `H` (Home Win) $\rightarrow$ `0`
    * `D` (Draw) $\rightarrow$ `1`
    * `A` (Away Win) $\rightarrow$ `2`

---

## 4. Betting Strategy & Kelly Criterion Integration
* **Match Winner (H2H):**
  * Probabilities are taken directly from the output of the new `XGBClassifier`:
    * `probs['H'] = clf_probs[0]`
    * `probs['D'] = clf_probs[1]`
    * `probs['A'] = clf_probs[2]`
* **Totals (Over/Under 2.5) and BTTS:**
  * Probabilities are calculated as before using the Poisson probability grid powered by `model_h` and `model_a`'s expected goal predictions (`lambda_h` and `lambda_a`).
* **Kelly Criterion:**
  * Value detection and Quarter Kelly staking (0.25) remain active across all 5 standard markets (Home Win, Draw, Away Win, Over 2.5, BTTS Yes).

---

## 5. Verification & Testing Plan
1. **Out-of-Sample Performance Validation:**
   * Compute multi-class Log Loss and Accuracy for the new outcome classifier on the 2025/2026 validation split.
   * Compare out-of-sample metrics against historical Poisson-based predictions to verify performance gains.
2. **End-to-End Execution Test:**
   * Run the manual predictor (`predict_match.py`) to verify seamless integration of user input, model predictions, and Kelly calculations.
   * Run the automated odds scanner (`main.py`) to verify odds fetching, model application, value detection, and terminal reporting.
