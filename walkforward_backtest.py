import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, mean_absolute_error
from utils import (
    fetch_historical_data, add_pre_match_features, calculate_kelly,
    get_market_probabilities, devig_odds_power
)

def evaluate_brier_multiclass(y_true, y_probs):
    """Calculates multi-class Brier Score (lower is better, 0.0 is perfect)."""
    y_one_hot = np.zeros_like(y_probs)
    for i, val in enumerate(y_true):
        y_one_hot[i, int(val)] = 1.0
    return float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

def run_walk_forward_backtest(start_test_season='2122', end_test_season='2425', initial_bankroll=1000.0):
    print("=" * 100)
    print("  PREMIER LEAGUE QUANT AI: ADVANCED WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
    print("  (Synthetic xG Features + Power De-Vigging + Longshot/Draw Filters)")
    print("=" * 100)
    
    df = fetch_historical_data()
    if df is None:
        print("Error: Could not retrieve dataset.")
        return
        
    print("Computing dynamic Elo, venue splits, and Synthetic xG metrics across all seasons...")
    df, _ = add_pre_match_features(df)
    
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
        'Home_2H_Momentum', 'Away_2H_Momentum'
    ]
    
    features_h = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Home_Rest_Days',
        'Home_V_Goals_For', 'Away_V_Goals_Against',
        'Home_V_Shots', 'Away_V_Shots',
        'Home_V_SOT', 'Away_V_SOT',
        'Home_Shot_Acc', 'Home_Goal_Conv', 'Away_Def_Suppression'
    ]
    features_a = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Away_Rest_Days',
        'Away_V_Goals_For', 'Home_V_Goals_Against',
        'Away_V_Shots', 'Home_V_Shots',
        'Away_V_SOT', 'Home_V_SOT',
        'Away_Shot_Acc', 'Away_Goal_Conv', 'Home_Def_Suppression'
    ]
    
    seasons = sorted(df['Season'].unique())
    test_seasons = [s for s in seasons if s >= start_test_season and s <= end_test_season]
    
    print(f"Evaluating Out-of-Sample Test Seasons: {', '.join(test_seasons)}")
    print("-" * 100)
    
    overall_results = []
    current_bankroll = initial_bankroll
    
    print(f"{'SEASON':<8} | {'MATCHES':<7} | {'AI BRIER':<9} | {'MKT BRIER':<9} | {'BETS':<5} | {'WIN %':<7} | {'TOTAL STAKED':<13} | {'NET P&L':<11} | {'ROI'}")
    print("-" * 100)
    
    for s_idx, test_season in enumerate(test_seasons):
        train_df = df[df['Season'] < test_season]
        test_df = df[df['Season'] == test_season].copy()
        
        if len(train_df) < 500 or len(test_df) == 0:
            continue
            
        w_train = train_df['Sample_Weight']
        
        # Fit models
        base_clf = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
            random_state=42, eval_metric='mlogloss'
        )
        calibrated_model = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
        calibrated_model.fit(train_df[features_clf], train_df['FTR_Class'], sample_weight=w_train)
        
        reg_h = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42).fit(train_df[features_h], train_df['FTHG'], sample_weight=w_train)
        reg_a = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42).fit(train_df[features_a], train_df['FTAG'], sample_weight=w_train)
        
        # Out of sample predictions
        y_probs = calibrated_model.predict_proba(test_df[features_clf])
        lh = reg_h.predict(test_df[features_h])
        la = reg_a.predict(test_df[features_a])
        
        ai_brier = evaluate_brier_multiclass(test_df['FTR_Class'].values, y_probs)
        
        # Market Brier score
        mkt_probs = []
        valid_mask = []
        for idx, row in test_df.iterrows():
            h, d, a = float(row.get('B365H', 0)), float(row.get('B365D', 0)), float(row.get('B365A', 0))
            if h > 1.0 and d > 1.0 and a > 1.0:
                mkt_probs.append(devig_odds_power([h, d, a]))
                valid_mask.append(True)
            else:
                valid_mask.append(False)
                
        mkt_brier = evaluate_brier_multiclass(test_df['FTR_Class'].values[valid_mask], np.array(mkt_probs)) if sum(valid_mask) > 0 else float('nan')
            
        season_bets = 0
        season_wins = 0
        season_staked = 0.0
        season_pnl = 0.0
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            actual_1x2 = int(row['FTR_Class'])
            fthg = float(row['FTHG'])
            ftag = float(row['FTAG'])
            total_goals = fthg + ftag
            
            odds_1x2 = [float(row.get('B365H', 0.0)), float(row.get('B365D', 0.0)), float(row.get('B365A', 0.0))]
            odd_over = float(row.get('B365>2.5', row.get('BbAv>2.5', 0.0)))
            odd_under = float(row.get('B365<2.5', row.get('BbAv<2.5', 0.0)))
            
            grid = get_market_probabilities(lh[i], la[i])
            
            # De-vig 1X2 market odds
            if all(o > 1.05 for o in odds_1x2):
                fair_1x2 = devig_odds_power(odds_1x2)
            else:
                fair_1x2 = [1/3, 1/3, 1/3]
                
            candidate_bets = []
            
            # 1X2 Markets (With Longshot and Draw Filters)
            for out_idx in [0, 1, 2]:
                odd = odds_1x2[out_idx]
                if odd <= 1.05 or odd > 3.80:  # Longshot guard: reject odds > 3.80
                    continue
                p_model = float(y_probs[i][out_idx])
                p_fair = fair_1x2[out_idx]
                
                # Check edge against fair market
                shrunk_p = 0.70 * p_model + 0.30 * p_fair
                edge = shrunk_p - p_fair
                
                if p_model >= 0.22 and edge >= 0.035:
                    stake = calculate_kelly(
                        prob=p_model, odds=odd, bankroll=current_bankroll,
                        fraction=0.10, max_stake_pct=0.025, shrinkage=0.30,
                        min_prob=0.22, max_odds=3.80
                    )
                    if stake > 0:
                        candidate_bets.append(('1X2', out_idx, odd, edge, stake, actual_1x2 == out_idx))
                        
            # Totals Markets (Over/Under 2.5)
            if odd_over > 1.40 and odd_over < 2.40:
                p_over = grid['Over2.5']
                p_mkt_o = 1.0 / odd_over
                shrunk_o = 0.75 * p_over + 0.25 * p_mkt_o
                edge_o = shrunk_o - p_mkt_o
                if p_over >= 0.45 and edge_o >= 0.03:
                    stake_o = calculate_kelly(p_over, odd_over, bankroll=current_bankroll, fraction=0.10, shrinkage=0.25, min_prob=0.45, max_odds=2.40)
                    if stake_o > 0:
                        candidate_bets.append(('Totals', 'Over', odd_over, edge_o, stake_o, total_goals > 2.5))
                        
            if odd_under > 1.40 and odd_under < 2.40:
                p_under = grid['Under2.5']
                p_mkt_u = 1.0 / odd_under
                shrunk_u = 0.75 * p_under + 0.25 * p_mkt_u
                edge_u = shrunk_u - p_mkt_u
                if p_under >= 0.45 and edge_u >= 0.03:
                    stake_u = calculate_kelly(p_under, odd_under, bankroll=current_bankroll, fraction=0.10, shrinkage=0.25, min_prob=0.45, max_odds=2.40)
                    if stake_u > 0:
                        candidate_bets.append(('Totals', 'Under', odd_under, edge_u, stake_u, total_goals < 2.5))
                        
            # Pick only the highest edge bet per fixture (strict discipline)
            if len(candidate_bets) > 0:
                candidate_bets.sort(key=lambda x: x[3], reverse=True)
                best_bet = candidate_bets[0]
                _, _, b_odd, _, b_stake, won = best_bet
                
                season_bets += 1
                season_staked += b_stake
                if won:
                    season_wins += 1
                    pnl = b_stake * (b_odd - 1.0)
                    season_pnl += pnl
                    current_bankroll += pnl
                else:
                    season_pnl -= b_stake
                    current_bankroll -= b_stake
                    
        win_rate = (season_wins / season_bets) if season_bets > 0 else 0.0
        season_roi = (season_pnl / season_staked) if season_staked > 0 else 0.0
        
        overall_results.append({
            'Season': test_season, 'Matches': len(test_df),
            'AI_Brier': ai_brier, 'Mkt_Brier': mkt_brier,
            'Bets': season_bets, 'Win_Rate': win_rate,
            'Staked': season_staked, 'PnL': season_pnl, 'ROI': season_roi
        })
        print(f"{test_season:<8} | {len(test_df):<7} | {ai_brier:<9.4f} | {mkt_brier:<9.4f} | {season_bets:<5} | {win_rate:<7.1%} | ${season_staked:<12,.2f} | ${season_pnl:<+10,.2f} | {season_roi:+.1%}")

    res_df = pd.DataFrame(overall_results)
    total_staked = res_df['Staked'].sum()
    total_pnl = res_df['PnL'].sum()
    agg_roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
    agg_brier = res_df['AI_Brier'].mean()
    agg_mkt_brier = res_df['Mkt_Brier'].mean()
    total_bets = res_df['Bets'].sum()
    agg_win_rate = (res_df['Win_Rate'] * res_df['Bets']).sum() / total_bets if total_bets > 0 else 0.0

    print("=" * 100)
    print("  AGGREGATE ADVANCED WALK-FORWARD BACKTEST RESULTS")
    print("=" * 100)
    print(f"Total Out-of-Sample Matches:       {res_df['Matches'].sum()}")
    print(f"Disciplined Bets Triggered:        {total_bets} (Selective: ~{total_bets/res_df['Matches'].sum():.1%} of matches)")
    print(f"Aggregate Win Rate:                {agg_win_rate:.1%}")
    print(f"Average Calibrated AI Brier:       {agg_brier:.4f}")
    print(f"Average De-Vigged Market Brier:    {agg_mkt_brier:.4f}")
    print(f"Total Amount Staked:               ${total_staked:,.2f}")
    print(f"Net Realized Profit/Loss:          ${total_pnl:+,.2f}")
    print(f"Aggregate Return on Investment:    {agg_roi:+.2%}")
    print(f"Final Simulated Bankroll:          ${current_bankroll:,.2f} (from ${initial_bankroll:,.2f})")
    print("=" * 100)

if __name__ == "__main__":
    run_walk_forward_backtest()
