import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, mean_absolute_error
from utils import (
    fetch_historical_data, add_pre_match_features, calculate_kelly,
    get_market_probabilities, devig_odds_power,
    compute_dnb_probs, compute_dnb_odds, get_adaptive_market_weight,
    MARKET_PRIOR_WEIGHT, MAX_DIVERGENCE, MIN_ODDS, MAX_ODDS, MIN_EDGE,
    MAX_DRAW_RISK, KELLY_FRACTION, MAX_STAKE_PCT
)

def evaluate_brier_multiclass(y_true, y_probs):
    """Calculates multi-class Brier Score (lower is better, 0.0 is perfect)."""
    y_one_hot = np.zeros_like(y_probs)
    for i, val in enumerate(y_true):
        y_one_hot[i, int(val)] = 1.0
    return float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

def expected_calibration_error(y_true, y_probs, n_bins=10):
    """
    Calculates multi-class Expected Calibration Error (ECE) (Flaw E Fix).
    Weights absolute difference between confidence and empirical accuracy across probability bins.
    """
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

def generate_calibration_plot(all_y_true, all_y_probs, save_path='data/calibration_curve.png'):
    """Generates reliability diagram across all out-of-sample predictions (Flaw E Fix)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = ['Home Win (H)', 'Draw (D)', 'Away Win (A)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for c in range(3):
            y_c = (np.array(all_y_true) == c).astype(int)
            p_c = np.array(all_y_probs)[:, c]
            prob_true, prob_pred = calibration_curve(y_c, p_c, n_bins=10, strategy='uniform')
            
            ax = axes[c]
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            ax.plot(prob_pred, prob_true, marker='s', color=colors[c], linewidth=2, label=f'Model ({labels[c]})')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Empirical Win Rate')
            ax.set_title(f'Reliability Curve: {labels[c]}')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Reliability diagram successfully generated and saved to {save_path}.")
    except Exception as e:
        print(f"Note: Could not generate calibration plot: {e}")

def run_walk_forward_backtest(start_test_season='2122', end_test_season='2425', initial_bankroll=1000.0):
    print("=" * 115)
    print("  PREMIER LEAGUE QUANT AI: ADVANCED WALK-FORWARD OUT-OF-SAMPLE BACKTEST")
    print("  (Pinnacle CLV Tracking + Adverse Selection Filter + Drawdown Circuit Breaker + Bootstrap CI)")
    print("=" * 115)
    
    df = fetch_historical_data()
    if df is None:
        print("Error: Could not retrieve dataset.")
        return
        
    print("Computing dynamic Elo, venue splits, Dual-Horizon EMAs, Synthetic xG & H2H metrics...")
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
        'Home_2H_Momentum', 'Away_2H_Momentum',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    
    features_h = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Home_Rest_Days',
        'Home_V_Goals_For', 'Away_V_Goals_Against',
        'Home_V_Shots', 'Away_V_Shots',
        'Home_V_SOT', 'Away_V_SOT',
        'Home_Shot_Acc', 'Home_Goal_Conv', 'Away_Def_Suppression',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    features_a = [
        'Home_Elo', 'Away_Elo', 'Elo_Diff', 'Away_Rest_Days',
        'Away_V_Goals_For', 'Home_V_Goals_Against',
        'Away_V_Shots', 'Home_V_Shots',
        'Away_V_SOT', 'Home_V_SOT',
        'Away_Shot_Acc', 'Away_Goal_Conv', 'Home_Def_Suppression',
        'Expected_Goals_Diff', 'H2H_Dominance'
    ]
    
    seasons = sorted(df['Season'].unique())
    test_seasons = [s for s in seasons if s >= start_test_season and s <= end_test_season]
    
    print(f"Evaluating Out-of-Sample Test Seasons: {', '.join(test_seasons)}")
    print(f"Unified Strategy Parameters: Market Prior Weight = {MARKET_PRIOR_WEIGHT:.0%}, "
          f"Odds Band = [{MIN_ODDS:.2f}, {MAX_ODDS:.2f}], Min Edge = {MIN_EDGE:.1%}, "
          f"Max Divergence = {MAX_DIVERGENCE:.0%}, Max Draw Risk = {MAX_DRAW_RISK:.0%}")
    print("-" * 115)
    
    overall_results = []
    current_bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown_pct = 0.0
    all_bet_records = []
    all_y_true = []
    all_y_probs = []
    
    print(f"{'SEASON':<8} | {'MATCHES':<7} | {'AI BRIER':<9} | {'MKT BRIER':<9} | {'AI ECE':<8} | {'BETS':<5} | {'WIN %':<7} | {'AVG CLV':<8} | {'TOTAL STAKED':<13} | {'NET P&L':<11} | {'ROI'}")
    print("-" * 115)
    
    for s_idx, test_season in enumerate(test_seasons):
        train_df = df[df['Season'] < test_season]
        test_df = df[df['Season'] == test_season].copy()
        
        if len(train_df) < 400 or len(test_df) == 0:
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
        y_raw_probs = calibrated_model.predict_proba(test_df[features_clf])
        lh = reg_h.predict(test_df[features_h])
        la = reg_a.predict(test_df[features_a])
        
        # Empirical Bayes Market-Anchor Model with Adaptive Seasonality & Uncertainty Shrinkage
        mkt_probs = []
        valid_mask = []
        y_blended_list = []
        for idx, (_, row) in enumerate(test_df.iterrows()):
            h, d, a = float(row.get('B365H', 0)), float(row.get('B365D', 0)), float(row.get('B365A', 0))
            if h > 1.0 and d > 1.0 and a > 1.0:
                p_mkt = devig_odds_power([h, d, a])
                mkt_probs.append(p_mkt)
                valid_mask.append(True)
                # Adaptive Empirical Bayes anchor (decays early season noise & accounts for promoted teams)
                w_mkt = get_adaptive_market_weight(row['Date'], home_team=row['HomeTeam'], away_team=row['AwayTeam'])
                blend = w_mkt * np.array(p_mkt) + (1.0 - w_mkt) * y_raw_probs[idx]
                blend /= blend.sum()
                y_blended_list.append(blend)
            else:
                mkt_probs.append([1/3, 1/3, 1/3])
                valid_mask.append(False)
                y_blended_list.append(y_raw_probs[idx])
                
        y_probs = np.array(y_blended_list)
        all_y_true.extend(test_df['FTR_Class'].values)
        all_y_probs.extend(y_probs)
        
        ai_brier = evaluate_brier_multiclass(test_df['FTR_Class'].values, y_probs)
        mkt_brier = evaluate_brier_multiclass(test_df['FTR_Class'].values[valid_mask], np.array([mkt_probs[j] for j in range(len(mkt_probs)) if valid_mask[j]])) if sum(valid_mask) > 0 else float('nan')
        ece_score = expected_calibration_error(test_df['FTR_Class'].values, y_probs)
            
        season_bets = 0
        season_wins = 0
        season_staked = 0.0
        season_pnl = 0.0
        season_clv_list = []
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            actual_1x2 = int(row['FTR_Class'])
            fthg = float(row['FTHG'])
            ftag = float(row['FTAG'])
            
            b365_1x2 = [float(row.get('B365H', 0.0)), float(row.get('B365D', 0.0)), float(row.get('B365A', 0.0))]
            max_h = float(row.get('MaxH', 0.0))
            max_d = float(row.get('MaxD', 0.0))
            max_a = float(row.get('MaxA', 0.0))
            exec_odds_1x2 = [
                max_h if max_h > 1.0 else b365_1x2[0],
                max_d if max_d > 1.0 else b365_1x2[1],
                max_a if max_a > 1.0 else b365_1x2[2]
            ]
            
            # Pinnacle closing odds for authentic CLV benchmarking
            psch = float(row.get('PSCH', 0.0))
            pscd = float(row.get('PSCD', 0.0))
            psca = float(row.get('PSCA', 0.0))
            pinnacle_closing = [
                psch if psch > 1.0 else exec_odds_1x2[0],
                pscd if pscd > 1.0 else exec_odds_1x2[1],
                psca if psca > 1.0 else exec_odds_1x2[2]
            ]
            
            # De-vig 1X2 market odds using Bet365 baseline
            if all(o > 1.05 for o in b365_1x2):
                fair_1x2 = devig_odds_power(b365_1x2)
            else:
                fair_1x2 = [1/3, 1/3, 1/3]
                
            p_draw = fair_1x2[1]
            candidate_bets = []
            
            # Drawdown Circuit Breaker: If in > 10% drawdown from peak, reduce Kelly fraction by 50%
            current_dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
            max_drawdown_pct = max(max_drawdown_pct, current_dd)
            dd_factor = 0.50 if current_dd > 0.10 else 1.0
            
            # 1X2 Markets with unified thresholds, adverse selection filter, draw risk protection, and line-shopped odds
            for out_idx in [0, 1, 2]:
                odd = exec_odds_1x2[out_idx]
                if odd < MIN_ODDS or odd > MAX_ODDS:
                    continue
                p_fair = fair_1x2[out_idx]
                
                # Shrunk posterior probability using adaptive market weight
                posterior_p = float(y_probs[i][out_idx])
                edge = posterior_p - p_fair
                
                # Adverse Selection Filter (Reject massive divergence where market has injury/lineup news)
                raw_divergence = abs(float(y_raw_probs[i][out_idx]) - p_fair)
                if raw_divergence > MAX_DIVERGENCE:
                    continue
                
                # Draw Risk Filter (Reject 1X2 match outcome bets in high-draw-variance fixtures)
                if MAX_DRAW_RISK is not None and p_draw > MAX_DRAW_RISK:
                    continue
                
                # Minimum edge hurdle post-shrinkage
                if edge >= MIN_EDGE:
                    stake = calculate_kelly(
                        prob=posterior_p, odds=odd, bankroll=current_bankroll,
                        fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
                        min_odds=MIN_ODDS, max_odds=MAX_ODDS, min_edge=MIN_EDGE,
                        max_divergence=MAX_DIVERGENCE, drawdown_factor=dd_factor,
                        fair_market_prob=p_fair, max_draw_risk=MAX_DRAW_RISK, draw_prob=p_draw
                    )
                    if stake > 0:
                        p_close = pinnacle_closing[out_idx]
                        bet_clv = (odd / p_close - 1.0) if p_close > 1.0 else 0.0
                        candidate_bets.append(('1X2', out_idx, odd, edge, stake, actual_1x2 == out_idx, False, bet_clv, p_close))

            # Draw No Bet (DNB) evaluation for high-draw-risk fixtures (bypasses draw risk via push protection)
            if p_draw > MAX_DRAW_RISK:
                fair_dnb = compute_dnb_probs(fair_1x2)
                post_dnb = compute_dnb_probs(y_probs[i])
                exec_dnb = compute_dnb_odds(exec_odds_1x2[0], exec_odds_1x2[1], exec_odds_1x2[2])
                close_dnb = compute_dnb_odds(pinnacle_closing[0], pinnacle_closing[1], pinnacle_closing[2])
                raw_dnb = compute_dnb_probs(y_raw_probs[i])
                
                for dnb_idx in [0, 1]:
                    odd_dnb = exec_dnb[dnb_idx]
                    p_f_dnb = fair_dnb[dnb_idx]
                    p_p_dnb = post_dnb[dnb_idx]
                    edge_dnb = p_p_dnb - p_f_dnb
                    raw_div_dnb = abs(raw_dnb[dnb_idx] - p_f_dnb)
                    
                    # Strict DNB hurdle (requiring >= 5.0% edge to clear the synthetic draw vig)
                    if raw_div_dnb <= MAX_DIVERGENCE and edge_dnb >= 0.050 and 1.60 <= odd_dnb <= 2.60:
                        stake_dnb = calculate_kelly(
                            prob=p_p_dnb, odds=odd_dnb, bankroll=current_bankroll,
                            fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
                            min_odds=1.60, max_odds=2.60, min_edge=0.050,
                            max_divergence=MAX_DIVERGENCE, drawdown_factor=dd_factor,
                            fair_market_prob=p_f_dnb, is_dnb=True
                        )
                        if stake_dnb > 0:
                            p_close_dnb = close_dnb[dnb_idx]
                            bet_clv_dnb = (odd_dnb / p_close_dnb - 1.0) if p_close_dnb > 1.0 else 0.0
                            is_push = (actual_1x2 == 1)
                            is_win = (dnb_idx == 0 and actual_1x2 == 0) or (dnb_idx == 1 and actual_1x2 == 2)
                            dnb_sel = f"Home DNB ({row['HomeTeam']})" if dnb_idx == 0 else f"Away DNB ({row['AwayTeam']})"
                            candidate_bets.append(('DNB', dnb_sel, odd_dnb, edge_dnb, stake_dnb, is_win, is_push, bet_clv_dnb, p_close_dnb))
                        
            # Pick only the highest edge bet per fixture (strict discipline)
            if len(candidate_bets) > 0:
                candidate_bets.sort(key=lambda x: x[3], reverse=True)
                best_bet = candidate_bets[0]
                b_market, b_out, b_odd, b_edge, b_stake, won, is_push, b_clv, b_close = best_bet
                
                season_bets += 1
                season_staked += b_stake
                season_clv_list.append(b_clv)
                
                if is_push:
                    pnl = 0.0
                    ret_pct = 0.0
                elif won:
                    season_wins += 1
                    pnl = b_stake * (b_odd - 1.0)
                    ret_pct = b_odd - 1.0
                    season_pnl += pnl
                    current_bankroll += pnl
                else:
                    pnl = -b_stake
                    ret_pct = -1.0
                    season_pnl -= b_stake
                    current_bankroll -= b_stake
                    
                peak_bankroll = max(peak_bankroll, current_bankroll)
                
                all_bet_records.append({
                    'Season': test_season,
                    'Home': row['HomeTeam'], 'Away': row['AwayTeam'],
                    'Market': b_market,
                    'Selection': b_out, 'Odds': b_odd, 'Pinnacle_Closing': b_close,
                    'CLV': b_clv, 'Stake': b_stake,
                    'Outcome': 'PUSH' if is_push else ('WON' if won else 'LOST'),
                    'PnL': pnl,
                    'Return_Pct': ret_pct
                })
                    
        win_rate = (season_wins / season_bets) if season_bets > 0 else 0.0
        season_roi = (season_pnl / season_staked) if season_staked > 0 else 0.0
        avg_clv = np.mean(season_clv_list) if season_clv_list else 0.0
        
        overall_results.append({
            'Season': test_season, 'Matches': len(test_df),
            'AI_Brier': ai_brier, 'Mkt_Brier': mkt_brier,
            'ECE': ece_score,
            'Bets': season_bets, 'Win_Rate': win_rate, 'Avg_CLV': avg_clv,
            'Staked': season_staked, 'PnL': season_pnl, 'ROI': season_roi
        })
        print(f"{test_season:<8} | {len(test_df):<7} | {ai_brier:<9.4f} | {mkt_brier:<9.4f} | {ece_score:<8.4f} | {season_bets:<5} | {win_rate:<7.1%} | {avg_clv:<+8.1%} | ${season_staked:<12,.2f} | ${season_pnl:<+10,.2f} | {season_roi:+.1%}")

    res_df = pd.DataFrame(overall_results)
    total_staked = res_df['Staked'].sum()
    total_pnl = res_df['PnL'].sum()
    agg_roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
    agg_brier = res_df['AI_Brier'].mean()
    agg_mkt_brier = res_df['Mkt_Brier'].mean()
    agg_ece = res_df['ECE'].mean()
    total_bets = res_df['Bets'].sum()
    agg_win_rate = (res_df['Win_Rate'] * res_df['Bets']).sum() / total_bets if total_bets > 0 else 0.0
    
    # Closing Line Value Aggregate Analysis
    all_clvs = [r['CLV'] for r in all_bet_records]
    agg_clv = np.mean(all_clvs) if all_clvs else 0.0
    positive_clv_pct = np.mean([c > 0.0 for c in all_clvs]) if all_clvs else 0.0

    # 10,000-iteration Bootstrap Resampling for Statistical Significance
    if len(all_bet_records) > 0:
        bet_returns = np.array([r['Return_Pct'] for r in all_bet_records])
        boot_rois = []
        np.random.seed(42)
        for _ in range(10000):
            sample = np.random.choice(bet_returns, size=len(bet_returns), replace=True)
            boot_rois.append(np.mean(sample))
        ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])
        p_positive = np.mean(np.array(boot_rois) > 0.0)
    else:
        ci_low, ci_high, p_positive = 0.0, 0.0, 0.0

    print("=" * 115)
    print("  AGGREGATE ADVANCED WALK-FORWARD BACKTEST & STATISTICAL AUDIT")
    print("=" * 115)
    print(f"Total Out-of-Sample Matches:       {res_df['Matches'].sum():,}")
    print(f"Disciplined Bets Triggered:        {total_bets} (Selective: ~{total_bets/res_df['Matches'].sum():.1%} of matches)")
    print(f"Aggregate Win Rate:                {agg_win_rate:.1%}")
    print(f"Average Calibrated AI Brier:       {agg_brier:.4f}")
    print(f"Average De-Vigged Market Brier:    {agg_mkt_brier:.4f}")
    print(f"Average Expected Calibration Error: {agg_ece:.4f} (Target: < 0.0500)")
    print(f"Average CLV vs Pinnacle Closing:   {agg_clv:+.2%} (Target: >= +2.0%) [Positive CLV Rate: {positive_clv_pct:.1%}]")
    print(f"Total Amount Staked:               ${total_staked:,.2f}")
    print(f"Net Realized Profit/Loss:          ${total_pnl:+,.2f}")
    print(f"Aggregate Return on Investment:    {agg_roi:+.2%}")
    print(f"95% Bootstrap Confidence Interval: [{ci_low:+.2%}, {ci_high:+.2%}]")
    print(f"Empirical Probability ROI > 0:     {p_positive:.1%}")
    print(f"Maximum Bankroll Drawdown:         {max_drawdown_pct:.1%}")
    print(f"Final Simulated Bankroll:          ${current_bankroll:,.2f} (from ${initial_bankroll:,.2f})")
    print("=" * 115)
    
    # Generate and save reliability curve
    if len(all_y_true) > 0 and len(all_y_probs) > 0:
        generate_calibration_plot(all_y_true, all_y_probs)
    
    return res_df, all_bet_records

if __name__ == "__main__":
    run_walk_forward_backtest()
