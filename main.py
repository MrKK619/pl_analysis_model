import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from utils import (
    fetch_historical_data, fetch_market_odds, calculate_kelly, 
    normalize_team, get_team_rolling_stats, add_pre_match_rolling_features,
    get_market_probabilities
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

def main():
    print("PL Betting AI - Multi-Bookmaker Goals & Value Scanner")
    
    # 1. Fetch & Preprocess
    df = fetch_historical_data()
    if df is None: return
    
    print("Pre-computing pre-match rolling stats...")
    df = add_pre_match_rolling_features(df)
    
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_map = {t: i for i, t in enumerate(teams)}
    df['HomeID'] = df['HomeTeam'].map(team_map)
    df['AwayID'] = df['AwayTeam'].map(team_map)
    # 1X2 target mapping for classifier
    ftr_map = {'H': 0, 'D': 1, 'A': 2}
    df['FTR_Class'] = df['FTR'].map(ftr_map)
    
    # Unified feature list for the multi-class outcome classifier
    features_clf = [
        'HomeID', 'AwayID', 
        'Home_Roll_Goals_Scored', 'Home_Roll_Goals_Conceded', 
        'Home_Roll_Shots', 'Home_Roll_SOT', 'Home_Roll_Corners',
        'Away_Roll_Goals_Scored', 'Away_Roll_Goals_Conceded', 
        'Away_Roll_Shots', 'Away_Roll_SOT', 'Away_Roll_Corners'
    ]

    # Training Features lists strictly aligned with design
    features_h = [
        'HomeID', 'AwayID', 
        'Home_Roll_Goals_Scored', 'Away_Roll_Goals_Conceded', 
        'Home_Roll_Shots', 'Away_Roll_Shots', 
        'Home_Roll_SOT', 'Away_Roll_SOT', 
        'Home_Roll_Corners', 'Away_Roll_Corners'
    ]
    features_a = [
        'HomeID', 'AwayID', 
        'Away_Roll_Goals_Scored', 'Home_Roll_Goals_Conceded', 
        'Home_Roll_Shots', 'Away_Roll_Shots', 
        'Home_Roll_SOT', 'Away_Roll_SOT', 
        'Home_Roll_Corners', 'Away_Roll_Corners'
    ]
    
    # 2. Chronological Train/Val Split (24/25 vs 25/26)
    train_mask = df['Date'] < '2025-08-01'
    val_mask = df['Date'] >= '2025-08-01'
    
    X_train_h, y_train_h = df[train_mask][features_h], df[train_mask]['FTHG']
    X_train_a, y_train_a = df[train_mask][features_a], df[train_mask]['FTAG']
    X_val_h, y_val_h = df[val_mask][features_h], df[val_mask]['FTHG']
    X_val_a, y_val_a = df[val_mask][features_a], df[val_mask]['FTAG']
    
    # New Classifier data splits
    X_train_clf, y_train_clf = df[train_mask][features_clf], df[train_mask]['FTR_Class']
    X_val_clf, y_val_clf = df[val_mask][features_clf], df[val_mask]['FTR_Class']
    
    if len(X_val_h) > 0:
        val_model_h = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(X_train_h, y_train_h)
        val_model_a = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(X_train_a, y_train_a)
        val_model_clf = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='mlogloss').fit(X_train_clf, y_train_clf)
        
        preds_h = val_model_h.predict(X_val_h)
        preds_a = val_model_a.predict(X_val_a)
        
        clf_preds_prob = val_model_clf.predict_proba(X_val_clf)
        clf_preds_class = val_model_clf.predict(X_val_clf)
        
        # Expected Goals metrics
        mae_h = mean_absolute_error(y_val_h, preds_h)
        mse_h = mean_squared_error(y_val_h, preds_h)
        
        # Classifier metrics
        val_log_loss = log_loss(y_val_clf, clf_preds_prob)
        val_acc = accuracy_score(y_val_clf, clf_preds_class)
        
        # Over/Under 2.5 goals validation
        correct_totals = 0
        for idx, row in df[val_mask].reset_index(drop=True).iterrows():
            l_h = preds_h[idx]
            l_a = preds_a[idx]
            probs = get_market_probabilities(l_h, l_a)
            pred_over = probs['Over2.5'] > 0.5
            actual_over = (row['FTHG'] + row['FTAG']) > 2.5
            if pred_over == actual_over:
                correct_totals += 1
        
        totals_accuracy = correct_totals / len(y_val_h)
        print(f"Goals Out-of-Sample Performance (Validated on 25/26):")
        print(f"   Home Goals MAE: {mae_h:.3f} | MSE: {mse_h:.3f}")
        print(f"   Over/Under 2.5 Goals Accuracy: {totals_accuracy:.1%}")
        print(f"Outcome Classifier Out-of-Sample Performance (Validated on 25/26):")
        print(f"   Log Loss: {val_log_loss:.3f} | Accuracy: {val_acc:.1%}")
    else:
        print("Validation set empty, skipping split check.")
        
    # Fit Final expected goals regression models and outcome classifier on complete dataset
    model_h = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(df[features_h], df['FTHG'])
    model_a = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(df[features_a], df['FTAG'])
    model_clf = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='mlogloss').fit(df[features_clf], df['FTR_Class'])
    print("Final Expected Goals (xG) Regressors & Direct 1X2 Outcome Classifier trained.")

    # 3. Automated Multi-Market Market Scan
    live_data = fetch_market_odds()
    bankroll = 1000  # Adjust based on your budget

    print("\n" + "="*95)
    print(f"{'MATCHUP':<25} | {'MARKET':<10} | {'BET ON':<12} | {'ODDS':<6} | {'BOOKIE':<12} | {'EDGE'}")
    print("-" * 95)

    for game in live_data:
        raw_home, raw_away = game['home_team'], game['away_team']
        home = normalize_team(raw_home)
        away = normalize_team(raw_away)
        
        if home not in team_map or away not in team_map: 
            print(f"{home + ' vs ' + away:<25} | Unknown Teams")
            continue

        # Dynamic Features strictly aligned with design
        now = pd.Timestamp.now()
        hs, hsot, hc, hgs, hgc = get_team_rolling_stats(df, home, now, n_games=5)
        as_, asot, ac, ags, agc = get_team_rolling_stats(df, away, now, n_games=5)
        
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
        
        # Derive all market probabilities from Poisson Probability Score Matrix
        probs = get_market_probabilities(lambda_h, lambda_a)
        
        # Overwrite H2H probabilities using the direct Classifier
        clf_probs = model_clf.predict_proba(input_clf)[0]
        probs['H'] = clf_probs[0]
        probs['D'] = clf_probs[1]
        probs['A'] = clf_probs[2]
        
        # Scan bookmaker feeds for optimal odds
        best_odds = {'H': 0, 'D': 0, 'A': 0, 'Over2.5': 0, 'Under2.5': 0, 'BTTS_Yes': 0, 'BTTS_No': 0}
        best_bookie = {k: "" for k in best_odds.keys()}
        
        if not game.get('bookmakers'):
            print(f"{home + ' vs ' + away:<25} | No Odds available")
            continue

        for bookmaker in game['bookmakers']:
            if not bookmaker.get('markets'): continue
            for market in bookmaker['markets']:
                m_key = market['key']
                
                # 1. H2H Winner
                if m_key == 'h2h':
                    for outcome in market['outcomes']:
                        name = outcome['name']
                        price = outcome['price']
                        if name == raw_home and price > best_odds['H']:
                            best_odds['H'] = price
                            best_bookie['H'] = bookmaker['title']
                        elif name == raw_away and price > best_odds['A']:
                            best_odds['A'] = price
                            best_bookie['A'] = bookmaker['title']
                        elif name == 'Draw' and price > best_odds['D']:
                            best_odds['D'] = price
                            best_bookie['D'] = bookmaker['title']
                
                # 2. Over/Under 2.5 Goals (totals)
                elif m_key == 'totals':
                    for outcome in market['outcomes']:
                        point = outcome.get('point')
                        if point != 2.5: continue
                        name = outcome['name']
                        price = outcome['price']
                        
                        if name == 'Over' and price > best_odds['Over2.5']:
                            best_odds['Over2.5'] = price
                            best_bookie['Over2.5'] = bookmaker['title']
                        elif name == 'Under' and price > best_odds['Under2.5']:
                            best_odds['Under2.5'] = price
                            best_bookie['Under2.5'] = bookmaker['title']
                            
                # 3. Both Teams to Score (btts)
                elif m_key == 'btts':
                    for outcome in market['outcomes']:
                        name = outcome['name']
                        price = outcome['price']
                        
                        if name == 'Yes' and price > best_odds['BTTS_Yes']:
                            best_odds['BTTS_Yes'] = price
                            best_bookie['BTTS_Yes'] = bookmaker['title']
                        elif name == 'No' and price > best_odds['BTTS_No']:
                            best_odds['BTTS_No'] = price
                            best_bookie['BTTS_No'] = bookmaker['title']

        # Evaluate and stake each market
        outcomes_eval = [
            ('H2H', 'H', probs['H'], home),
            ('H2H', 'D', probs['D'], 'Draw'),
            ('H2H', 'A', probs['A'], away),
            ('Totals', 'Over2.5', probs['Over2.5'], 'Over 2.5'),
            ('Totals', 'Under2.5', probs['Under2.5'], 'Under 2.5'),
            ('BTTS', 'BTTS_Yes', probs['BTTS_Yes'], 'BTTS Yes'),
            ('BTTS', 'BTTS_No', probs['BTTS_No'], 'BTTS No'),
        ]
        
        found_edge = False
        for m_name, key, prob, bet_name in outcomes_eval:
            odds = best_odds[key]
            if odds == 0: continue
            
            implied_prob = 1 / odds
            edge = prob - implied_prob
            
            if edge > 0.05:
                  found_edge = True
                  stake = calculate_kelly(prob, odds, bankroll)
                  print(f"{home + ' vs ' + away:<25} | {m_name:<10} | {bet_name:<12} | {odds:<6.2f} | {best_bookie[key]:<12} | {edge:+.1%}")
                  print(f"   ADVICE: Bet ${stake:.2f} on {bet_name} (Expected Score: {probs['Score']})")
                  
        if not found_edge:
            print(f"{home + ' vs ' + away:<25} | No Value detected")

if __name__ == "__main__":
    main()