import os
import joblib
import pandas as pd
import numpy as np
from utils import (
    fetch_historical_data, add_pre_match_features, get_live_match_features,
    get_market_probabilities, calculate_kelly, log_prediction,
    devig_odds_power, normalize_team, MODEL_CACHE,
    compute_dnb_probs, compute_dnb_odds, get_adaptive_market_weight,
    MARKET_PRIOR_WEIGHT, MAX_DIVERGENCE, MIN_ODDS, MAX_ODDS, MIN_EDGE,
    MAX_DRAW_RISK, KELLY_FRACTION, MAX_STAKE_PCT
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
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    match_date_raw = input(f"Enter Match Date (YYYY-MM-DD, default: {today_str}): ").strip()
    match_date = match_date_raw if match_date_raw else today_str
    
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
    bookie_raw = input("Enter Bookmaker Name (default: Manual): ").strip()
    bookie_name = bookie_raw if bookie_raw else "Manual"
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
    
    # Compute true de-vigged fair market probabilities
    fair_1x2 = devig_odds_power([h_odds, d_odds, a_odds])
    
    # Adaptive Empirical Bayes Posterior Probability
    adaptive_weight = get_adaptive_market_weight(
        match_date=match_date,
        home_team=home_in,
        away_team=away_in,
        base_weight=MARKET_PRIOR_WEIGHT
    )
    posterior_1x2 = adaptive_weight * np.array(fair_1x2) + (1.0 - adaptive_weight) * np.array(clf_probs)
    posterior_1x2 /= posterior_1x2.sum()
    probs['H'] = float(posterior_1x2[0])
    probs['D'] = float(posterior_1x2[1])
    probs['A'] = float(posterior_1x2[2])
    
    # Draw No Bet (DNB / Asian Handicap 0) calculations
    p_h_dnb, p_a_dnb = compute_dnb_probs(posterior_1x2)
    fair_h_dnb, fair_a_dnb = compute_dnb_probs(fair_1x2)
    dnb_h_odds, dnb_a_odds = compute_dnb_odds(h_odds, d_odds, a_odds)

    # 4. Display Analysis
    bankroll = 1000.0
    r_h = elo_tracker.get_rating(home_in)
    r_a = elo_tracker.get_rating(away_in)
    
    print("\n" + "=" * 95)
    print(f"MATCHUP: {home_in} (Elo: {r_h:.0f}) vs {away_in} (Elo: {r_a:.0f})")
    print(f"Expected Goals (xG): {home_in} {lambda_h:.2f} - {lambda_a:.2f} {away_in} | Synthetic xG Diff: {feat_dict['Expected_Goals_Diff']:+.2f} | H2H Dominance: {feat_dict['H2H_Dominance']:+.2f}")
    print(f"Shot Accuracy (SAR): {home_in} {feat_dict['Home_Shot_Acc']:.1%} | {away_in} {feat_dict['Away_Shot_Acc']:.1%}")
    print(f"Finishing Conv (FCE): {home_in} {feat_dict['Home_Goal_Conv']:.1%} | {away_in} {feat_dict['Away_Goal_Conv']:.1%}")
    print(f"Adaptive Market Shrinkage: {adaptive_weight:.1%} Market / {1.0 - adaptive_weight:.1%} Tactical ML")
    print("=" * 95)
    print(f"{'MARKET SELECTION':<20} | {'AI POSTERIOR':<12} | {'RAW BOOKIE':<10} | {'FAIR MKT':<9} | {'EDGE':<8} | {'SUGGESTED STAKE'}")
    print("-" * 95)
    
    markets = [
        ('1X2', 'H', probs['H'], h_odds, fair_1x2[0], f"{home_in} Win", False),
        ('1X2', 'D', probs['D'], d_odds, fair_1x2[1], "Draw", False),
        ('1X2', 'A', probs['A'], a_odds, fair_1x2[2], f"{away_in} Win", False),
        ('DNB', 'H_DNB', p_h_dnb, dnb_h_odds, fair_h_dnb, f"{home_in} DNB", True),
        ('DNB', 'A_DNB', p_a_dnb, dnb_a_odds, fair_a_dnb, f"{away_in} DNB", True),
    ]
    
    edges_detected = False
    p_draw = fair_1x2[1]
    
    for m_cat, key, model_p, odds, fair_p, label, is_dnb in markets:
        # Edge of empirical Bayes posterior against fair market baseline
        edge = model_p - fair_p
        raw_divergence = abs(model_p - fair_p)
        
        stake = calculate_kelly(
            prob=model_p, odds=odds, bankroll=bankroll,
            fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS, min_edge=MIN_EDGE,
            max_divergence=MAX_DIVERGENCE,
            fair_market_prob=fair_p,
            max_draw_risk=MAX_DRAW_RISK,
            draw_prob=p_draw,
            is_dnb=is_dnb
        )
        
        effective_min_odds = 1.40 if is_dnb else MIN_ODDS
        if not is_dnb and MAX_DRAW_RISK is not None and p_draw > MAX_DRAW_RISK:
            stake_str = f"Pass (Draw Risk {p_draw:.1%} > {MAX_DRAW_RISK:.0%} - High Variance)"
        elif raw_divergence > MAX_DIVERGENCE:
            stake_str = f"Pass (Divergence {raw_divergence:.1%} > {MAX_DIVERGENCE:.0%} - Adverse Selection Risk)"
        elif odds < effective_min_odds or odds > MAX_ODDS:
            stake_str = f"Pass (Odds {odds:.2f} out of [{effective_min_odds:.2f}, {MAX_ODDS:.2f}])"
        elif stake > 0:
            stake_str = f"Bet ${stake:.2f}"
            edges_detected = True
        else:
            stake_str = f"Pass (< {MIN_EDGE:.1%} Edge)"
            
        print(f"{label:<20} | {model_p:<12.1%} | {1.0/odds:<10.1%} | {fair_p:<9.1%} | {edge:<+8.1%} | {stake_str}")
        
    print("-" * 95)
    if edges_detected:
        print(f">> STATISTICAL VALUE DETECTED. Bet sizing calculated with Tenth-Kelly (max {MAX_STAKE_PCT:.1%}).")
        if p_draw > MAX_DRAW_RISK:
            print(">> NOTE: 1X2 selections with elevated draw risk were filtered to DNB alternatives (draws refund stake).")
    else:
        print(">> NO MATHEMATICAL EDGE. Bookmaker prices are efficient for this fixture.")
    print("=" * 95)

    # 5. Log Predictions
    for m_cat, key, model_p, odds, fair_p, label, is_dnb in markets:
        edge = model_p - fair_p
        stake = calculate_kelly(
            prob=model_p, odds=odds, bankroll=bankroll,
            fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
            min_odds=MIN_ODDS, max_odds=MAX_ODDS, min_edge=MIN_EDGE,
            max_divergence=MAX_DIVERGENCE,
            fair_market_prob=fair_p,
            max_draw_risk=MAX_DRAW_RISK,
            draw_prob=p_draw,
            is_dnb=is_dnb
        )
        if stake > 0:
            log_prediction(
                home=home_in, away=away_in, market=m_cat, selection=label,
                model_prob=model_p, market_odds=odds, edge=edge,
                bookie=bookie_name, stake=stake, kickoff=match_date
            )
            print(f"[OK] Logged bet opportunity: {label} to data/prediction_log.csv")

if __name__ == "__main__":
    individual_prediction()