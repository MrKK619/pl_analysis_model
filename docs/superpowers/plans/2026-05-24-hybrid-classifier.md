# Hybrid Sports Betting AI Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transition the Premier League betting model to a hybrid architecture by introducing a direct match outcome classifier (`XGBClassifier`) for 1X2 market predictions while retaining goal regressors for Over/Under and BTTS.

**Architecture:** Train two `XGBRegressor` models for Home and Away Goals respectively, using them with a Poisson grid to calculate Totals (Over/Under 2.5) and BTTS. Concurrently train a single multi-class `XGBClassifier` on the combined pre-match rolling features of both teams to predict 1X2 probabilities directly, bypassing Poisson's independence limitation for winner markets.

**Tech Stack:** Python, Pandas, NumPy, XGBoost, Scikit-Learn

---

### Task 1: Integrate and Validate XGBClassifier in `main.py`

**Files:**
* Modify: `main.py` (lines 40-84)

- [ ] **Step 1: Define Classifier features and target mapping in `main.py`**
  Modify `main.py` to add `features_clf` and prepare target mappings (`H` -> `0`, `D` -> `1`, `A` -> `2`) for both training and validation splits.
  ```python
  # Target mapping
  ftr_map = {'H': 0, 'D': 1, 'A': 2}
  df['FTR_Class'] = df['FTR'].map(ftr_map)

  # Unified features list for the classifier
  features_clf = [
      'HomeID', 'AwayID', 
      'Home_Roll_Goals_Scored', 'Home_Roll_Goals_Conceded', 
      'Home_Roll_Shots', 'Home_Roll_SOT', 'Home_Roll_Corners',
      'Away_Roll_Goals_Scored', 'Away_Roll_Goals_Conceded', 
      'Away_Roll_Shots', 'Away_Roll_SOT', 'Away_Roll_Corners'
  ]
  ```

- [ ] **Step 2: Add validation testing for the Classifier in `main.py`**
  Update the chronological train/val split logic in `main.py` to train an `XGBClassifier` and evaluate out-of-sample Multi-class Log Loss and Accuracy on the 2025/2026 season validation set, comparing it to the existing regressors.
  ```python
  from xgboost import XGBClassifier
  from sklearn.metrics import log_loss, accuracy_score

  X_train_clf, y_train_clf = df[train_mask][features_clf], df[train_mask]['FTR_Class']
  X_val_clf, y_val_clf = df[val_mask][features_clf], df[val_mask]['FTR_Class']

  if len(X_val_h) > 0:
      # Train validation classifier
      val_model_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss').fit(X_train_clf, y_train_clf)
      clf_preds_prob = val_model_clf.predict_proba(X_val_clf)
      clf_preds_class = val_model_clf.predict(X_val_clf)
      
      val_log_loss = log_loss(y_val_clf, clf_preds_prob)
      val_acc = accuracy_score(y_val_clf, clf_preds_class)
      
      print(f"Outcome Classifier Out-of-Sample Performance (Validated on 25/26):")
      print(f"   Log Loss: {val_log_loss:.3f} | Accuracy: {val_acc:.1%}")
  ```

- [ ] **Step 3: Fit Final Classifier on complete dataset**
  Update the final model fitting section of `main.py` to train the final `XGBClassifier` on the full multi-season dataset.
  ```python
  model_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss').fit(df[features_clf], df['FTR_Class'])
  print("Final Direct 1X2 Outcome Classifier trained on complete multi-season form.")
  ```

- [ ] **Step 4: Verify task changes compile and run without errors**
  Run the script to check that training completes successfully.
  Command: `python main.py`
  Expected Output: Output lists the validation performance metrics for both the Goals regressors and the new Outcome Classifier.

---

### Task 2: Update automated scanner H2H prediction logic in `main.py`

**Files:**
* Modify: `main.py` (lines 100-116)

- [ ] **Step 1: Retrieve Direct Classifier probabilities inside the market scanner loop**
  Update the scanner loop inside `main.py` to construct a feature vector matching `features_clf`, pass it to `model_clf.predict_proba`, and use those outputs directly for the H2H probabilities instead of deriving H2H from Poisson.
  ```python
  # Dynamic Features strictly aligned with design
  now = pd.Timestamp.now()
  hs, hsot, hc, hgs, hgc = get_team_rolling_stats(df, home, now, n_games=5)
  as_, asot, ac, ags, agc = get_team_rolling_stats(df, away, now, n_games=5)
  
  # Goals regressors inputs
  input_h = pd.DataFrame([[team_map[home], team_map[away], hgs, agc, hs, as_, hsot, asot, hc, ac]], columns=features_h)
  input_a = pd.DataFrame([[team_map[home], team_map[away], ags, hgc, hs, as_, hsot, asot, hc, ac]], columns=features_a)
  
  # Outcome classifier input
  input_clf = pd.DataFrame([[
      team_map[home], team_map[away],
      hgs, hgc, hs, hsot, hc,
      ags, agc, as_, asot, ac
  ]], columns=features_clf)
  
  lambda_h = max(0.0001, model_h.predict(input_h)[0])
  lambda_a = max(0.0001, model_a.predict(input_a)[0])
  
  # Derive Totals and BTTS from Poisson probability grid
  probs = get_market_probabilities(lambda_h, lambda_a)
  
  # OVERWRITE Home/Draw/Away H2H probabilities using the direct Classifier
  clf_probs = model_clf.predict_proba(input_clf)[0]
  probs['H'] = clf_probs[0]
  probs['D'] = clf_probs[1]
  probs['A'] = clf_probs[2]
  ```

- [ ] **Step 2: Run main.py to verify live scanner output**
  Command: `python main.py`
  Expected Output: A formatted output showing active matchups, bookmaker odds, calculated edges, and Kelly advice. Ensure no `KeyError` or formatting errors occur.

---

### Task 3: Update manual predictor logic in `predict_match.py`

**Files:**
* Modify: `predict_match.py`

- [ ] **Step 1: Integrate XGBClassifier in `predict_match.py`**
  Modify the training segment of `predict_match.py` to import `XGBClassifier`, define `features_clf`, map `FTR` to integer classes, and train `model_clf`.
  ```python
  from xgboost import XGBClassifier
  
  # Target mapping
  ftr_map = {'H': 0, 'D': 1, 'A': 2}
  df['FTR_Class'] = df['FTR'].map(ftr_map)
  
  features_clf = [
      'HomeID', 'AwayID', 
      'Home_Roll_Goals_Scored', 'Home_Roll_Goals_Conceded', 
      'Home_Roll_Shots', 'Home_Roll_SOT', 'Home_Roll_Corners',
      'Away_Roll_Goals_Scored', 'Away_Roll_Goals_Conceded', 
      'Away_Roll_Shots', 'Away_Roll_SOT', 'Away_Roll_Corners'
  ]
  
  # Train dual XGBoost Regressors and Outcome Classifier
  model_h = XGBRegressor(n_estimators=100, random_state=42).fit(df[features_h], df['FTHG'])
  model_a = XGBRegressor(n_estimators=100, random_state=42).fit(df[features_a], df['FTAG'])
  model_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss').fit(df[features_clf], df['FTR_Class'])
  ```

- [ ] **Step 2: Update prediction generation in `predict_match.py`**
  Update the manual inputs and prediction logic in `predict_match.py` to pass features to `model_clf` and overwrite `probs` for `H`, `D`, and `A`.
  ```python
  input_data_clf = [[
      team_to_id[home_in], team_to_id[away_in],
      hgs, hgc, hs, hsot, hc,
      ags, agc, as_, asot, ac
  ]]
  input_row_clf = pd.DataFrame(input_data_clf, columns=features_clf)
  
  lambda_h = max(0.0001, model_h.predict(input_row_h)[0])
  lambda_a = max(0.0001, model_a.predict(input_row_a)[0])
  
  # Base Poisson probs
  probs = get_market_probabilities(lambda_h, lambda_a)
  
  # Overwrite with Direct Classifier H2H probs
  clf_probs = model_clf.predict_proba(input_row_clf)[0]
  probs['H'] = clf_probs[0]
  probs['D'] = clf_probs[1]
  probs['A'] = clf_probs[2]
  ```

- [ ] **Step 3: Run predict_match.py manual test**
  Command: `python predict_match.py`
  * Input Home Team: `Arsenal`
  * Input Away Team: `Chelsea`
  * Input odds: `2.0`, `3.5`, `3.6`, `1.8`, `1.7`
  Expected Output: Success grid displaying correct xG goals, actual direct outcome probabilities from the classifier, edge estimations, and Kelly Staking advice.

---

### Task 4: Complete System Verification

- [ ] **Step 1: End-to-end check of both main entrypoints**
  Run `python main.py` and `python predict_match.py` back-to-back to verify both run smoothly under the hybrid system.
