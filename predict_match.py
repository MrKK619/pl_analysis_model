import os
import joblib
import pandas as pd
import numpy as np
from utils import (
    fetch_historical_data, add_pre_match_features, get_live_match_features,
    get_market_probabilities, calculate_kelly, log_prediction,
    devig_odds_power, normalize_team, MODEL_CACHE
)

def load_or_train_models():
    """Loads persisted models from disk or triggers a training run if not cached."""
    if os.path.exists(MODEL_CACHE):
        try:
            return joblib.load(MODEL_CACHE)
        except Exception:
            pass
            
    import main
    main.main()
    return joblib.load(MODEL_CACHE)

def individual_prediction():
    print("=" * 80)
    print("  PREMIER LEAGUE AI: CALIBRATED MATCH PREDICTOR & FAIR VALUE EVALUATOR")
    print("=" * 80)
    
    print("Loading historical match database and calibrated models...")
    df = fetch_historical_data()
    if df is None:
        print("Error: Could not access match database.")
        return
        
    bundle = load_or_train_models()
    calibrated_clf = bundle['calibrated_clf']
    model_h = bundle['model_h']
    model_a = bundle['model_a']
    elo_tracker = bundle['elo_tracker']
    features_clf = bundle['features_clf']
    features_h = bundle['features_h']
    features_a = bundle['features_a']
    
    recent_teams = sorted(list(set(df['HomeTeam'].tail(200).unique())))
    print(f"\nSupported EPL Teams ({len(recent_teams)}):")
    print(", ".join(recent_teams[:10]) + ", ...")
    
    # 1. Match Details Input
    print("\n--- 1. Matchup Input ---")
    home_raw = input("Enter Home Team: ").strip()
    away_raw = input("Enter Away Team: ").strip()
    
    home_in = normalize_team(home_raw)
    away_in = normalize_team(away_raw)
    
    if home_in not in elo_tracker.ratings and home_in not in recent_teams:
        print(f"Error: Home team '{home_raw}' not recognized.")
        return
    if away_in not in elo_tracker.ratings and away_in not in recent_teams:
        print(f"Error: Away team '{away_raw}' not recognized.")
        return
    if home_in == away_in:
        print("Error: Home and Away teams must be different.")
        return

    # 2. Market Odds Input
    print("\n--- 2. Bookmaker Odds Input (Decimal Format e.g. 2.10, 3.40) ---")
    try:
        h_odds = float(input(f"Decimal Odds for {home_in} Win: "))
        d_odds = float(input(f"Decimal Odds for Draw: "))
        a_odds = float(input(f"Decimal Odds for {away_in} Win: "))
        over_odds = float(input(f"Decimal Odds for Over 2.5 Goals: "))
        btts_odds = float(input(f"Decimal Odds for BTTS (Yes): "))
    except ValueError:
        print("Error: Odds must be valid decimal numbers greater than 1.0.")
        return

    # 3. Generate Predictions with Live Pre-Match Features
    feat_dict = get_live_match_features(df, elo_tracker, home_in, away_in)
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
    
    # Compute true de-vigged fair market probabilities
    fair_1x2 = devig_odds_power([h_odds, d_odds, a_odds])
    
    # 4. Display Analysis
    bankroll = 1000.0
    r_h = elo_tracker.get_rating(home_in)
    r_a = elo_tracker.get_rating(away_in)
    
    print("\n" + "=" * 90)
    print(f"MATCHUP: {home_in} (Elo: {r_h:.0f}) vs {away_in} (Elo: {r_a:.0f})")
    print(f"Expected Goals (xG): {home_in} {lambda_h:.2f} - {lambda_a:.2f} {away_in} | Most Likely Score: {probs['Score']}")
    print(f"Shot Accuracy (SAR): {home_in} {feat_dict['Home_Shot_Acc']:.1%} | {away_in} {feat_dict['Away_Shot_Acc']:.1%}")
    print(f"Finishing Conv (FCE): {home_in} {feat_dict['Home_Goal_Conv']:.1%} | {away_in} {feat_dict['Away_Goal_Conv']:.1%}")
    print("=" * 90)
    print(f"{'MARKET SELECTION':<18} | {'AI PROB':<9} | {'RAW BOOKIE':<10} | {'FAIR MKT':<9} | {'EDGE':<8} | {'SUGGESTED STAKE'}")
    print("-" * 90)
    
    markets = [
        ('H2H', 'H', probs['H'], h_odds, fair_1x2[0], f"{home_in} Win", True),
        ('H2H', 'D', probs['D'], d_odds, fair_1x2[1], "Draw", True),
        ('H2H', 'A', probs['A'], a_odds, fair_1x2[2], f"{away_in} Win", True),
        ('Totals', 'Over2.5', probs['Over2.5'], over_odds, 1.0/over_odds, "Over 2.5 Goals", False),
        ('BTTS', 'BTTS_Yes', probs['BTTS_Yes'], btts_odds, 1.0/btts_odds, "BTTS (Yes)", False)
    ]
    
    edges_detected = False
    for m_cat, key, model_p, odds, fair_p, label, is_1x2 in markets:
        # Shrunk edge against fair de-vigged market
        shrunk_p = 0.70 * model_p + 0.30 * fair_p
        edge = shrunk_p - fair_p
        
        stake = calculate_kelly(
            prob=model_p, odds=odds, bankroll=bankroll,
            fraction=0.10, max_stake_pct=0.025, shrinkage=0.30,
            min_prob=0.20 if is_1x2 else 0.40,
            max_odds=3.80 if is_1x2 else 2.50
        )
        
        stake_str = f"Bet ${stake:.2f}" if stake > 0 else "Pass"
        if stake > 0:
            edges_detected = True
            
        print(f"{label:<18} | {model_p:<9.1%} | {1.0/odds:<10.1%} | {fair_p:<9.1%} | {edge:<+8.1%} | {stake_str}")
        
    print("-" * 90)
    if edges_detected:
        print(">> STATISTICAL VALUE DETECTED. Bet sizing calculated with Tenth-Kelly (max 2.5%).")
    else:
        print(">> NO MATHEMATICAL EDGE. Bookmaker prices are efficient for this fixture.")
    print("=" * 90)

    # 5. Log Predictions
    for m_cat, key, model_p, odds, fair_p, label, is_1x2 in markets:
        stake = calculate_kelly(
            prob=model_p, odds=odds, bankroll=bankroll,
            fraction=0.10, max_stake_pct=0.025, shrinkage=0.30,
            min_prob=0.20 if is_1x2 else 0.40,
            max_odds=3.80 if is_1x2 else 2.50
        )
        if stake > 0:
            shrunk_p = 0.70 * model_p + 0.30 * fair_p
            edge = shrunk_p - fair_p
            log_prediction(
                home=home_in, away=away_in, market=m_cat, selection=label,
                model_prob=model_p, market_odds=odds, edge=edge,
                bookie="Manual", stake=stake, kickoff="MANUAL"
            )
            print(f"[OK] Logged bet opportunity: {label} to data/prediction_log.csv")

if __name__ == "__main__":
    individual_prediction()