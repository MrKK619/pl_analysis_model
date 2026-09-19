import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss
from utils import (
    fetch_historical_data, fetch_market_odds, calculate_kelly, 
    normalize_team, add_pre_match_features, get_live_match_features,
    get_market_probabilities, log_prediction, devig_odds_power, MODEL_CACHE,
    compute_dnb_probs, compute_dnb_odds, get_adaptive_market_weight,
    MARKET_PRIOR_WEIGHT, MAX_DIVERGENCE, MIN_ODDS, MAX_ODDS, MIN_EDGE,
    MAX_DRAW_RISK, KELLY_FRACTION, MAX_STAKE_PCT
)

def evaluate_brier_multiclass(y_true, y_probs):
    """Calculates multi-class Brier Score (lower is better, 0.0 is perfect)."""
    n_classes = y_probs.shape[1]
    y_one_hot = np.zeros_like(y_probs)
    for i, val in enumerate(y_true):
        y_one_hot[i, int(val)] = 1.0
    return float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

def expected_calibration_error(y_true, y_probs, n_bins=10):
    """Calculates multi-class Expected Calibration Error (ECE) (Flaw E Fix)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    for c in range(y_probs.shape[1]):
        y_c = (np.array(y_true) == c).astype(float)
        p_c = y_probs[:, c]
        for i in range(n_bins):
            mask = (p_c >= bin_boundaries[i]) & (p_c < bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_acc = y_c[mask].mean()
                bin_conf = p_c[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / (total_samples * y_probs.shape[1]))

def compute_fair_market_probs(row):
    """Converts 1X2 decimal odds into power de-vigged implied probabilities."""
    try:
        h_odd = float(row['B365H'])
        d_odd = float(row['B365D'])
        a_odd = float(row['B365A'])
        if h_odd > 1.0 and d_odd > 1.0 and a_odd > 1.0:
            return devig_odds_power([h_odd, d_odd, a_odd])
    except Exception:
        pass
    return None

def main(force_retrain=False):
    print("=" * 85)
    print("   PREMIER LEAGUE QUANT AI -- PRODUCTION ALPHA & VALUE SCANNER")
    print("=" * 85)
    
    # 1. Fetch & Preprocess Historical Dataset
    df = fetch_historical_data()
    if df is None or len(df) == 0:
        print("Error: Could not retrieve match dataset.")
        return
        
    print("Pre-computing dynamic Elo, venue splits, Dual-Horizon EMAs & Synthetic xG metrics...")
    df, elo_tracker = add_pre_match_features(df)
    
    # 2. Comprehensive Production Feature Set (Tactics, Shot Quality, Momentum & xG/H2H)
    features_clf = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff',
        'Home_Rest_Days', 'Away_Rest_Days',
        'Home_V_Goals_For', 'Home_V_Goals_Against', 'Home_V_Shots', 'Home_V_SOT', 'Home_V_Corners',
        'Away_V_Goals_For', 'Away_V_Goals_Against', 'Away_V_Shots', 'Away_V_SOT', 'Away_V_Corners',
        'Home_All_Goals_For', 'Home_All_Goals_Against', 'Home_All_SOT',
        'Away_All_Goals_For', 'Away_All_Goals_Against', 'Away_All_SOT',
        'Home_Shot_Acc', 'Away_Shot_Acc',
        'Home_Goal_Conv', 'Away_Goal_Conv',
        'Home_Def_Suppression', 'Away_Def_Suppression',
        'Home_Card_Stress', 'Away_Card_Stress',
        'Home_2H_Momentum', 'Away_2H_Momentum',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    
    features_h = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Home_Rest_Days',
        'Home_V_Goals_For', 'Away_V_Goals_Against',
        'Home_V_Shots', 'Away_V_Shots',
        'Home_V_SOT', 'Away_V_SOT',
        'Home_Shot_Acc', 'Home_Goal_Conv', 'Away_Def_Suppression',
        'Home_V_Corners', 'Away_V_Corners',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    
    features_a = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Away_Rest_Days',
        'Away_V_Goals_For', 'Home_V_Goals_Against',
        'Away_V_Shots', 'Home_V_Shots',
        'Away_V_SOT', 'Home_V_SOT',
        'Away_Shot_Acc', 'Away_Goal_Conv', 'Home_Def_Suppression',
        'Away_V_Corners', 'Home_V_Corners',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    
    # 3. Chronological Train/Validation Split (Train <= 24/25, Validate on 25/26)
    val_cutoff = '2025-08-01'
    train_mask = df['Date'] < val_cutoff
    val_mask = df['Date'] >= val_cutoff
    
    X_train_clf = df[train_mask][features_clf]
    y_train_clf = df[train_mask]['FTR_Class']
    w_train = df[train_mask]['Sample_Weight']
    
    X_val_clf = df[val_mask][features_clf]
    y_val_clf = df[val_mask]['FTR_Class']
    
    X_train_h, y_train_h = df[train_mask][features_h], df[train_mask]['FTHG']
    X_train_a, y_train_a = df[train_mask][features_a], df[train_mask]['FTAG']
    X_val_h, y_val_h = df[val_mask][features_h], df[val_mask]['FTHG']
    X_val_a, y_val_a = df[val_mask][features_a], df[val_mask]['FTAG']
    
    print(f"\nDataset Split: {len(X_train_clf)} training matches | {len(X_val_clf)} out-of-sample validation matches")
    
    # 4. Model Training & Sigmoid Calibration
    print("Fitting regularized estimators and probability calibrator...")
    
    base_clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method='sigmoid',
        cv=5
    )
    calibrated_clf.fit(X_train_clf, y_train_clf, sample_weight=w_train)
    
    model_h = XGBRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
        random_state=42
    ).fit(X_train_h, y_train_h, sample_weight=w_train)
    
    model_a = XGBRegressor(
        n_estimators=80, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
        random_state=42
    ).fit(X_train_a, y_train_a, sample_weight=w_train)
    
    # 5. Out-of-Sample Performance Benchmarks vs De-Vigged Market
    if len(X_val_clf) > 0:
        val_probs = calibrated_clf.predict_proba(X_val_clf)
        val_preds = calibrated_clf.predict(X_val_clf)
        
        model_logloss = log_loss(y_val_clf, val_probs)
        model_brier = evaluate_brier_multiclass(y_val_clf.values, val_probs)
        model_acc = np.mean(y_val_clf.values == val_preds)
        
        # Naive Baseline
        class_counts = y_train_clf.value_counts(normalize=True).sort_index().values
        naive_probs = np.tile(class_counts, (len(y_val_clf), 1))
        naive_logloss = log_loss(y_val_clf, naive_probs)
        naive_brier = evaluate_brier_multiclass(y_val_clf.values, naive_probs)
        
        # Fair De-Vigged Market Odds Baseline (Bet365 Closing Odds)
        val_df = df[val_mask].copy()
        market_probs_list = []
        valid_indices = []
        for i, (_, row) in enumerate(val_df.iterrows()):
            mp = compute_fair_market_probs(row)
            if mp is not None:
                market_probs_list.append(mp)
                valid_indices.append(i)
                
        print("\n" + "=" * 85)
        print("  OUT-OF-SAMPLE VALIDATION & BENCHMARKS (2025/2026 Season)")
        print("=" * 85)
        print(f"{'EVALUATION METRIC':<25} | {'CALIBRATED AI':<15} | {'NAIVE BASELINE':<15} | {'FAIR MARKET (B365)':<18}")
        print("-" * 85)
        print(f"{'Accuracy':<25} | {model_acc:<15.1%} | {class_counts.max():<15.1%} | {'~53-56%':<18}")
        print(f"{'Log Loss (Lower is better)':<25} | {model_logloss:<15.4f} | {naive_logloss:<15.4f} | ", end="")
        if len(market_probs_list) > 0:
            m_probs_arr = np.array(market_probs_list)
            y_sub = y_val_clf.values[valid_indices]
            market_logloss = log_loss(y_sub, m_probs_arr)
            market_brier = evaluate_brier_multiclass(y_sub, m_probs_arr)
            val_ece = expected_calibration_error(y_val_clf.values, val_probs)
            market_ece = expected_calibration_error(y_sub, m_probs_arr)
            print(f"{market_logloss:<18.4f}")
            print(f"{'Brier Score (0=Perfect)':<25} | {model_brier:<15.4f} | {naive_brier:<15.4f} | {market_brier:<18.4f}")
            print(f"{'Expected Calib Error (ECE)':<25} | {val_ece:<15.4f} | {'N/A':<15} | {market_ece:<18.4f}")
        else:
            print(f"{'N/A':<18}")
            val_ece = expected_calibration_error(y_val_clf.values, val_probs)
            print(f"{'Brier Score (0=Perfect)':<25} | {model_brier:<15.4f} | {naive_brier:<15.4f} | {'N/A':<18}")
            print(f"{'Expected Calib Error (ECE)':<25} | {val_ece:<15.4f} | {'N/A':<15} | {'N/A':<18}")
            
        preds_h = model_h.predict(X_val_h)
        preds_a = model_a.predict(X_val_a)
        mae_h = mean_absolute_error(y_val_h, preds_h)
        mae_a = mean_absolute_error(y_val_a, preds_a)
        print(f"{'Goals MAE (Home / Away)':<25} | {mae_h:.2f} / {mae_a:.2f}")
        print("=" * 85)
    
    # 6. Fit Production Estimators on All Data with Sample Weights
    w_all = df['Sample_Weight']
    print("\nTraining production calibrated models with Synthetic xG features on full dataset...")
    calibrated_clf.fit(df[features_clf], df['FTR_Class'], sample_weight=w_all)
    model_h.fit(df[features_h], df['FTHG'], sample_weight=w_all)
    model_a.fit(df[features_a], df['FTAG'], sample_weight=w_all)
    
    os.makedirs('data', exist_ok=True)
    joblib.dump({
        'calibrated_clf': calibrated_clf,
        'model_h': model_h,
        'model_a': model_a,
        'elo_tracker': elo_tracker,
        'features_clf': features_clf,
        'features_h': features_h,
        'features_a': features_a
    }, MODEL_CACHE)
    print(f"Production models and dynamic Elo state persisted to {MODEL_CACHE}.")
    
    # 7. Live Market Scanner with De-Vigged Edge & Longshot Filtering
    print("\nScanning live bookmaker markets via The Odds API...")
    live_data = fetch_market_odds()
    bankroll = 1000.0
    
    if not live_data:
        print("No live odds feed returned. (Check ODDS_API_KEY in .env or match schedule).")
        return
        
    print("\n" + "=" * 105)
    print(f"{'MATCHUP':<25} | {'MARKET':<8} | {'SELECTION':<12} | {'ODDS':<6} | {'BOOKIE':<12} | {'DE-VIG EDGE':<12} | {'STAKE'}")
    print("-" * 105)
    
    edges_found = 0
    for game in live_data:
        raw_home, raw_away = game['home_team'], game['away_team']
        home = normalize_team(raw_home)
        away = normalize_team(raw_away)
        
        feat_dict = get_live_match_features(df, elo_tracker, home, away)
        input_clf = pd.DataFrame([[feat_dict[c] for c in features_clf]], columns=features_clf)
        input_h = pd.DataFrame([[feat_dict[c] for c in features_h]], columns=features_h)
        input_a = pd.DataFrame([[feat_dict[c] for c in features_a]], columns=features_a)
        
        lambda_h = max(0.05, float(model_h.predict(input_h)[0]))
        lambda_a = max(0.05, float(model_a.predict(input_a)[0]))
        
        probs = get_market_probabilities(lambda_h, lambda_a)
        clf_probs = calibrated_clf.predict_proba(input_clf)[0]
        probs['H'] = float(clf_probs[0])
        probs['D'] = float(clf_probs[1])
        probs['A'] = float(clf_probs[2])
        
        best_odds = {'H': 0.0, 'D': 0.0, 'A': 0.0, 'Over2.5': 0.0, 'Under2.5': 0.0, 'BTTS_Yes': 0.0, 'BTTS_No': 0.0}
        best_bookie = {k: "" for k in best_odds.keys()}
        
        for bookmaker in game.get('bookmakers', []):
            b_title = bookmaker.get('title', 'Unknown')
            for market in bookmaker.get('markets', []):
                m_key = market.get('key')
                if m_key == 'h2h':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name')
                        price = float(outcome.get('price', 0.0))
                        if name == raw_home and price > best_odds['H']:
                            best_odds['H'] = price
                            best_bookie['H'] = b_title
                        elif name == raw_away and price > best_odds['A']:
                            best_odds['A'] = price
                            best_bookie['A'] = b_title
                        elif name == 'Draw' and price > best_odds['D']:
                            best_odds['D'] = price
                            best_bookie['D'] = b_title
                elif m_key == 'totals':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('point') == 2.5:
                            name = outcome.get('name')
                            price = float(outcome.get('price', 0.0))
                            if name == 'Over' and price > best_odds['Over2.5']:
                                best_odds['Over2.5'] = price
                                best_bookie['Over2.5'] = b_title
                            elif name == 'Under' and price > best_odds['Under2.5']:
                                best_odds['Under2.5'] = price
                                best_bookie['Under2.5'] = b_title
                elif m_key == 'btts':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name')
                        price = float(outcome.get('price', 0.0))
                        if name == 'Yes' and price > best_odds['BTTS_Yes']:
                            best_odds['BTTS_Yes'] = price
                            best_bookie['BTTS_Yes'] = b_title
                        elif name == 'No' and price > best_odds['BTTS_No']:
                            best_odds['BTTS_No'] = price
                            best_bookie['BTTS_No'] = b_title

        # Compute fair de-vigged market probabilities for H2H
        if best_odds['H'] > 1.0 and best_odds['D'] > 1.0 and best_odds['A'] > 1.0:
            fair_1x2 = devig_odds_power([best_odds['H'], best_odds['D'], best_odds['A']])
        else:
            fair_1x2 = [1.0/3.0, 1.0/3.0, 1.0/3.0]

        adaptive_weight = get_adaptive_market_weight(
            match_date=game.get('commence_time'),
            home_team=home,
            away_team=away,
            base_weight=MARKET_PRIOR_WEIGHT
        )
        
        # Draw No Bet synthetic odds and probabilities
        p_h_dnb, p_a_dnb = compute_dnb_probs(probs_1x2=[probs['H'], probs['D'], probs['A']])
        fair_h_dnb, fair_a_dnb = compute_dnb_probs(fair_1x2)
        dnb_h_odds, dnb_a_odds = compute_dnb_odds(best_odds['H'], best_odds['D'], best_odds['A'])
        
        best_odds['H_DNB'] = dnb_h_odds
        best_odds['A_DNB'] = dnb_a_odds
        best_bookie['H_DNB'] = best_bookie['H']
        best_bookie['A_DNB'] = best_bookie['A']

        eval_markets = [
            ('H2H', 'H', probs['H'], f"{home} Win", fair_1x2[0], True, False),
            ('H2H', 'D', probs['D'], "Draw", fair_1x2[1], True, False),
            ('H2H', 'A', probs['A'], f"{away} Win", fair_1x2[2], True, False),
            ('DNB', 'H_DNB', p_h_dnb, f"{home} DNB", fair_h_dnb, False, True),
            ('DNB', 'A_DNB', p_a_dnb, f"{away} DNB", fair_a_dnb, False, True),
        ]
        
        matchup_label = f"{home} vs {away}"
        match_has_edge = False
        p_draw = fair_1x2[1]
        
        for m_cat, key, model_p, sel_name, fair_mkt_p, is_1x2, is_dnb in eval_markets:
            odds = best_odds[key]
            if odds < MIN_ODDS or odds > MAX_ODDS:
                continue
                
            # Shrunk probability using adaptive Empirical Bayes market anchor
            shrunk_p = adaptive_weight * fair_mkt_p + (1.0 - adaptive_weight) * model_p
            edge = shrunk_p - fair_mkt_p
            
            # Adverse selection protection
            if abs(model_p - fair_mkt_p) > MAX_DIVERGENCE:
                continue
                
            # Draw risk protection for match winner bets (DNB bypasses this as draws push)
            if is_1x2 and MAX_DRAW_RISK is not None and p_draw > MAX_DRAW_RISK:
                continue
            
            stake = calculate_kelly(
                prob=shrunk_p, odds=odds, bankroll=bankroll,
                fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
                min_odds=MIN_ODDS, max_odds=MAX_ODDS, min_edge=MIN_EDGE,
                max_divergence=MAX_DIVERGENCE,
                fair_market_prob=fair_mkt_p,
                max_draw_risk=MAX_DRAW_RISK,
                draw_prob=p_draw,
                is_dnb=is_dnb
            )
            
            if stake > 0.0:
                match_has_edge = True
                edges_found += 1
                b_name = best_bookie[key]
                print(f"{matchup_label:<25} | {m_cat:<8} | {sel_name:<12} | {odds:<6.2f} | {b_name:<12} | {edge:+.1%}        | Bet ${stake:.2f} (Score: {probs['Score']})")
                
                log_prediction(
                    home=home, away=away, market=m_cat, selection=sel_name,
                    model_prob=model_p, market_odds=odds, edge=edge, bookie=b_name,
                    stake=stake, kickoff=game.get('commence_time', 'UPCOMING')
                )
                
        if not match_has_edge:
            print(f"{matchup_label:<25} | No statistically validated edge found.")
            
    print("-" * 105)
    print(f"Scan Complete: {edges_found} high-discipline opportunities detected across {len(live_data)} matches.")
    print("=" * 105)

if __name__ == "__main__":
    main()