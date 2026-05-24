import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from utils import (
    fetch_historical_data, calculate_kelly, log_prediction, 
    get_team_rolling_stats, add_pre_match_rolling_features,
    get_market_probabilities
)

def individual_prediction():
    print("PREMIER LEAGUE AI: EXPECTED GOALS MANUAL PREDICTOR")
    print("-" * 55)
    
    # 1. Load and Train Goals models
    print("Fetching multi-season data and training models...")
    df = fetch_historical_data()
    if df is None: 
        print("Could not connect to data source. Check internet connection.")
        return
    
    print("Pre-computing pre-match rolling stats...")
    df = add_pre_match_rolling_features(df)
    
    # Setup Mappings
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    team_to_id = {t: i for i, t in enumerate(teams)}
    
    df['HomeID'] = df['HomeTeam'].map(team_to_id)
    df['AwayID'] = df['AwayTeam'].map(team_to_id)
    
    # Target mapping for multi-class classifier
    ftr_map = {'H': 0, 'D': 1, 'A': 2}
    df['FTR_Class'] = df['FTR'].map(ftr_map)
    
    # Feature selection matching main.py
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
    features_clf = [
        'HomeID', 'AwayID', 
        'Home_Roll_Goals_Scored', 'Home_Roll_Goals_Conceded', 
        'Home_Roll_Shots', 'Home_Roll_SOT', 'Home_Roll_Corners',
        'Away_Roll_Goals_Scored', 'Away_Roll_Goals_Conceded', 
        'Away_Roll_Shots', 'Away_Roll_SOT', 'Away_Roll_Corners'
    ]
    
    # Train dual XGBoost Regressors and the multi-class Classifier
    model_h = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(df[features_h], df['FTHG'])
    model_a = XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42).fit(df[features_a], df['FTAG'])
    model_clf = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='mlogloss').fit(df[features_clf], df['FTR_Class'])
    print("AI Brain Ready.")

    # 2. User Input Section
    print("\n--- Match Details ---")
    print(f"Example Teams: {', '.join(teams[:8])}...")
    
    home_in = input("Enter Home Team: ").strip()
    away_in = input("Enter Away Team: ").strip()
    
    if home_in not in team_to_id or away_in not in team_to_id:
        print("Error: Team name not recognized. Check README_PREDICT.md.")
        return

    # Odds Input
    try:
        h_odds = float(input(f"Decimal Odds for {home_in} Win: "))
        d_odds = float(input(f"Decimal Odds for Draw: "))
        a_odds = float(input(f"Decimal Odds for {away_in} Win: "))
        over_odds = float(input(f"Decimal Odds for Over 2.5 Goals: "))
        btts_odds = float(input(f"Decimal Odds for BTTS (Yes): "))
    except ValueError:
        print("Error: Please enter numbers (decimals) for the odds.")
        return

    # 3. Generate Prediction
    now = pd.Timestamp.now()
    hs, hsot, hc, hgs, hgc = get_team_rolling_stats(df, home_in, now, n_games=5)
    as_, asot, ac, ags, agc = get_team_rolling_stats(df, away_in, now, n_games=5)
    
    input_data_h = [[team_to_id[home_in], team_to_id[away_in], hgs, agc, hs, as_, hsot, asot, hc, ac]]
    input_data_a = [[team_to_id[home_in], team_to_id[away_in], ags, hgc, hs, as_, hsot, asot, hc, ac]]
    input_data_clf = [[
        team_to_id[home_in], team_to_id[away_in],
        hgs, hgc, hs, hsot, hc,
        ags, agc, as_, asot, ac
    ]]
    
    input_row_h = pd.DataFrame(input_data_h, columns=features_h)
    input_row_a = pd.DataFrame(input_data_a, columns=features_a)
    input_row_clf = pd.DataFrame(input_data_clf, columns=features_clf)
    
    lambda_h = max(0.0001, model_h.predict(input_row_h)[0])
    lambda_a = max(0.0001, model_a.predict(input_row_a)[0])
    
    # Base Poisson probabilities (for Totals and BTTS)
    probs = get_market_probabilities(lambda_h, lambda_a)
    
    # Overwrite H2H probabilities using the direct Classifier
    clf_probs = model_clf.predict_proba(input_row_clf)[0]
    probs['H'] = clf_probs[0]
    probs['D'] = clf_probs[1]
    probs['A'] = clf_probs[2]
    
    # 4. Final Output & Staking Analysis
    print("\n" + "="*65)
    print(f"MATCHUP: {home_in} vs {away_in} (Expected Score: {probs['Score']})")
    print(f"Expected Goals (xG): {home_in} {lambda_h:.2f} - {lambda_a:.2f} {away_in}")
    print("="*65)
    print(f"{'MARKET SELECTION':<20} | {'AI PROB':<10} | {'BOOKIE PROB':<12} | {'EDGE'}")
    print("-" * 65)
    print(f"{home_in + ' Win':<20} | {probs['H']:.1%}   | {1/h_odds:.1%}      | {probs['H'] - 1/h_odds:+.1%}")
    print(f"{'Draw':<20} | {probs['D']:.1%}   | {1/d_odds:.1%}      | {probs['D'] - 1/d_odds:+.1%}")
    print(f"{away_in + ' Win':<20} | {probs['A']:.1%}   | {1/a_odds:.1%}      | {probs['A'] - 1/a_odds:+.1%}")
    print(f"{'Over 2.5 Goals':<20} | {probs['Over2.5']:.1%}   | {1/over_odds:.1%}      | {probs['Over2.5'] - 1/over_odds:+.1%}")
    print(f"{'BTTS (Yes)':<20} | {probs['BTTS_Yes']:.1%}   | {1/btts_odds:.1%}      | {probs['BTTS_Yes'] - 1/btts_odds:+.1%}")
    print("-" * 65)
    
    # Check Kelly bets
    bets = [
        (probs['H'] - 1/h_odds, probs['H'], h_odds, home_in + " Win"),
        (probs['A'] - 1/a_odds, probs['A'], a_odds, away_in + " Win"),
        (probs['Over2.5'] - 1/over_odds, probs['Over2.5'], over_odds, "Over 2.5 Goals"),
        (probs['BTTS_Yes'] - 1/btts_odds, probs['BTTS_Yes'], btts_odds, "BTTS (Yes)"),
    ]
    
    has_value = False
    for edge, prob, odds, label in bets:
        if edge > 0.05:
            has_value = True
            stake = calculate_kelly(prob, odds, bankroll=1000)
            print(f"VALUE DETECTED! Suggested Stake: ${stake:.2f} on {label}")
            
    if not has_value:
        print("NO MATHEMATICAL EDGE. The bookmaker's prices are too accurate.")
    print("="*65)

    # Log winner prob for simple backward compatibility
    log_prediction(home_in, away_in, probs['H'], probs['D'], probs['A'], h_odds, d_odds, a_odds, "Manual")
    print(f"\nPrediction logged to data/prediction_log.csv")

if __name__ == "__main__":
    individual_prediction()