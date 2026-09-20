import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    devig_odds_power, devig_odds_proportional, compute_ema,
    calculate_kelly, normalize_team, compute_dnb_probs, compute_dnb_odds,
    blend_probabilities_logit, get_adaptive_market_weight,
    MARKET_PRIOR_WEIGHT, MAX_DIVERGENCE, MIN_ODDS, MAX_ODDS, MIN_EDGE,
    MAX_DRAW_RISK, KELLY_FRACTION, MAX_STAKE_PCT
)
from walkforward_backtest import expected_calibration_error

def test_devig_odds_power_valid_distribution():
    """Verify power de-vigged probabilities sum strictly to 1.0 and are positive."""
    odds_cases = [
        [2.10, 3.40, 3.60],
        [1.30, 6.00, 11.00],
        [2.80, 3.10, 2.70],
        [1.95, 3.50, 4.20]
    ]
    for odds in odds_cases:
        probs = devig_odds_power(odds)
        assert len(probs) == 3
        assert np.isclose(sum(probs), 1.0, atol=1e-4)
        assert all(p > 0.0 for p in probs)

def test_devig_odds_power_favorite_longshot_bias():
    """
    Power de-vigging should adjust longshots downward more aggressively
    than proportional de-vigging to counteract the favorite-longshot bias.
    """
    odds = [1.25, 6.50, 12.00]
    power_probs = devig_odds_power(odds)
    prop_probs = devig_odds_proportional(odds)
    
    # In power de-vigging, favorite prob should be higher than proportional, and longshot lower
    assert power_probs[0] > prop_probs[0]
    assert power_probs[2] < prop_probs[2]

def test_compute_ema():
    """Verify exponential moving average weighting and edge cases."""
    assert compute_ema([], alpha=0.25, default=1.5) == 1.5
    assert compute_ema([4.0], alpha=0.25) == 4.0
    
    # With alpha=0.5, two points [2.0, 4.0]: weight on 4.0 is 1.0, on 2.0 is 0.5 -> (2*0.5 + 4*1.0) / 1.5 = 3.333
    vals = [2.0, 4.0]
    expected = (2.0 * 0.5 + 4.0 * 1.0) / (0.5 + 1.0)
    assert np.isclose(compute_ema(vals, alpha=0.5), expected, atol=1e-3)

def test_kelly_odds_band_filter():
    """Bets outside [MIN_ODDS, MAX_ODDS] must return zero stake."""
    prob = 0.52
    fair_p = 0.45
    # Below min_odds
    assert calculate_kelly(prob=prob, odds=1.50, bankroll=1000, fair_market_prob=fair_p) == 0.0
    # Above max_odds
    assert calculate_kelly(prob=prob, odds=3.50, bankroll=1000, fair_market_prob=fair_p) == 0.0
    # Inside valid band
    assert calculate_kelly(prob=prob, odds=2.20, bankroll=1000, fair_market_prob=fair_p) > 0.0

def test_kelly_min_edge_hurdle():
    """Bets with edge < MIN_EDGE (3.0%) must return zero stake."""
    odds = 2.20
    fair_p = 0.45
    # Edge is 1.5% (< 3.0%) -> pass
    assert calculate_kelly(prob=0.465, odds=odds, bankroll=1000, fair_market_prob=fair_p) == 0.0
    # Edge is 5.0% (>= 3.0%) -> bet
    assert calculate_kelly(prob=0.50, odds=odds, bankroll=1000, fair_market_prob=fair_p) > 0.0

def test_kelly_adverse_selection_filter():
    """Massive divergence (> MAX_DIVERGENCE=10%) indicates unmodeled lineup/injury news and must be rejected."""
    odds = 2.00
    fair_p = 0.50
    # Model claims 65% on a 50% market (+15% divergence > 10% MAX_DIVERGENCE)
    assert calculate_kelly(prob=0.65, odds=odds, bankroll=1000, max_divergence=MAX_DIVERGENCE, fair_market_prob=fair_p) == 0.0
    # Moderate divergence (e.g. 5%) with positive expectation should pass
    assert calculate_kelly(prob=0.55, odds=odds, bankroll=1000, max_divergence=MAX_DIVERGENCE, fair_market_prob=fair_p) > 0.0

def test_kelly_draw_risk_filter():
    """Fixtures with elevated market draw probability (> MAX_DRAW_RISK=28%) must be filtered out."""
    prob = 0.48
    odds = 2.30
    fair_p = 0.42
    # High draw risk (32% > 28%) -> pass
    assert calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, draw_prob=0.32) == 0.0
    # Acceptable draw risk (24% <= 28%) -> bet
    assert calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, draw_prob=0.24) > 0.0

def test_expected_calibration_error_metric():
    """Verify Expected Calibration Error calculation on known synthetic cases."""
    # Perfectly calibrated case
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.1, 0.9, 0.0]])
    ece = expected_calibration_error(y_true, y_probs, n_bins=5)
    assert ece >= 0.0
    assert ece < 0.20

def test_kelly_drawdown_scaling():
    """Circuit breaker should scale stake down during drawdown."""
    prob = 0.52
    odds = 2.20
    fair_p = 0.45
    full_stake = calculate_kelly(prob=prob, odds=odds, bankroll=1000, drawdown_factor=1.0, fair_market_prob=fair_p)
    half_stake = calculate_kelly(prob=prob, odds=odds, bankroll=1000, drawdown_factor=0.5, fair_market_prob=fair_p)
    assert full_stake > 0.0
    assert np.isclose(half_stake, round(full_stake * 0.5, 2), atol=0.05)

def test_kelly_max_stake_cap():
    """Individual bet stake must never exceed MAX_STAKE_PCT (2.5% of bankroll)."""
    bankroll = 1000.0
    # Huge edge that would otherwise produce massive Kelly fraction
    stake = calculate_kelly(prob=0.90, odds=2.50, bankroll=bankroll, fair_market_prob=0.30, max_divergence=1.0)
    assert stake <= bankroll * MAX_STAKE_PCT + 1e-4

def test_team_normalization():
    """Verify common team name variants map to standard conventions."""
    assert normalize_team("Manchester City") == "Man City"
    assert normalize_team("Tottenham Hotspur") == "Tottenham"
    assert normalize_team("Brighton & Hove Albion") == "Brighton"
    assert normalize_team("Nottingham Forest") == "Nott'm Forest"

def test_clv_calculation():
    """Verify Closing Line Value mathematics: (odds_taken / odds_closing) - 1.0."""
    executed_odds = 2.20
    closing_sharp_odds = 2.00
    clv = (executed_odds / closing_sharp_odds) - 1.0
    assert np.isclose(clv, 0.10) # +10% CLV

def test_dnb_odds_and_probs_computation():
    """Verify Draw No Bet conditional probability and synthetic odds mathematics."""
    probs_1x2 = [0.45, 0.25, 0.30]
    dnb_probs = compute_dnb_probs(probs_1x2)
    assert np.isclose(sum(dnb_probs), 1.0)
    assert np.isclose(dnb_probs[0], 0.45 / 0.75) # 0.60
    assert np.isclose(dnb_probs[1], 0.30 / 0.75) # 0.40

    # Synthetic odds: Odds_DNB_Home = O_H * (O_D - 1) / O_D
    h_odd, d_odd, a_odd = 2.10, 3.40, 3.60
    home_dnb, away_dnb = compute_dnb_odds(h_odd, d_odd, a_odd)
    expected_home_dnb = 2.10 * (3.40 - 1.0) / 3.40
    assert np.isclose(home_dnb, expected_home_dnb, atol=1e-3)
    assert home_dnb > 1.0 and away_dnb > 1.0

def test_logit_space_bayesian_blending():
    """Verify Dirichlet/Log-odds probability updating."""
    mkt = [0.50, 0.25, 0.25]
    ml = [0.60, 0.20, 0.20]
    blended = blend_probabilities_logit(mkt, ml, prior_weight=0.50)
    assert np.isclose(sum(blended), 1.0)
    assert all(p > 0.0 for p in blended)
    # ML model favors Home, so Home probability should be higher than market prior
    assert blended[0] > mkt[0]

    # Neutral ML model equals market prior
    class_prior = [0.45, 0.25, 0.30]
    neutral_blend = blend_probabilities_logit(mkt, class_prior, prior_weight=0.50, class_prior=class_prior)
    assert np.allclose(neutral_blend, mkt, atol=1e-3)

def test_adaptive_shrinkage_seasonality():
    """Verify early season uses higher market weight than late season and handles tz-aware dates."""
    early_date = "2024-08-20"
    late_date = "2025-03-15"
    w_early = get_adaptive_market_weight(early_date)
    w_late = get_adaptive_market_weight(late_date)
    assert w_early > w_late

    # Promoted team penalty
    w_promoted = get_adaptive_market_weight(early_date, home_team="Ipswich", away_team="Arsenal")
    assert w_promoted >= w_early

    # Timezone-aware date string (e.g. from Odds API)
    tz_aware_date = "2024-08-20T15:00:00Z"
    w_tz = get_adaptive_market_weight(tz_aware_date)
    assert np.isclose(w_tz, w_early)

def test_concurrent_kelly_dampener():
    """Simultaneous kickoff scaling reduces stake by 1 / sqrt(K)."""
    prob = 0.52
    odds = 2.20
    fair_p = 0.45
    single_stake = calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, concurrent_bets=1)
    quad_stake = calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, concurrent_bets=4)
    assert single_stake > 0.0
    assert np.isclose(quad_stake, round(single_stake * 0.50, 2), atol=0.05)

def test_dnb_kelly_bypasses_draw_risk():
    """DNB wagers should bypass the MAX_DRAW_RISK cap because draws result in a push (refund)."""
    prob = 0.65
    odds = 1.70
    fair_p = 0.55
    high_draw = 0.35 # > MAX_DRAW_RISK=0.28
    # 1X2 bet is rejected due to draw risk
    stake_1x2 = calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, draw_prob=high_draw, is_dnb=False)
    assert stake_1x2 == 0.0
    # DNB bet is allowed
    stake_dnb = calculate_kelly(prob=prob, odds=odds, bankroll=1000, fair_market_prob=fair_p, draw_prob=high_draw, is_dnb=True)
    assert stake_dnb > 0.0

def test_evaluate_bet_outcome_dnb_and_teams():
    """Verify DNB evaluation pushes on draw and resolves wins/losses correctly with named teams."""
    from update_results import evaluate_bet_outcome
    # Draw -> PUSH
    assert evaluate_bet_outcome("Arsenal DNB", "DNB", 1, 1, 'D', home_team="Arsenal", away_team="Chelsea") == 'PUSH'
    # Arsenal Win -> WON for Arsenal DNB
    assert evaluate_bet_outcome("Arsenal DNB", "DNB", 2, 1, 'H', home_team="Arsenal", away_team="Chelsea") == 'WON'
    # Chelsea Win -> LOST for Arsenal DNB
    assert evaluate_bet_outcome("Arsenal DNB", "DNB", 0, 1, 'A', home_team="Arsenal", away_team="Chelsea") == 'LOST'
    # Chelsea Win -> WON for Chelsea DNB
    assert evaluate_bet_outcome("Chelsea DNB", "DNB", 0, 1, 'A', home_team="Arsenal", away_team="Chelsea") == 'WON'
    # 1X2 Home Win
    assert evaluate_bet_outcome("Arsenal Win", "H2H", 2, 1, 'H', home_team="Arsenal", away_team="Chelsea") == 'WON'
    assert evaluate_bet_outcome("Chelsea Win", "H2H", 2, 1, 'H', home_team="Arsenal", away_team="Chelsea") == 'LOST'

def test_prediction_log_schema_alignment(tmp_path, monkeypatch):
    """Verify log_prediction outputs identical 15 columns for automated and manual predictions."""
    import utils
    from utils import log_prediction
    
    test_log = tmp_path / "test_prediction_log.csv"
    monkeypatch.setattr(utils, "LOG_FILE", str(test_log))
    
    # Automated scanner log
    log_prediction(
        home="Arsenal", away="Chelsea", market="1X2", selection="Arsenal Win",
        model_prob=0.55, market_odds=2.10, edge=0.05, bookie="Betfair",
        stake=10.0, kickoff="2026-10-10T14:00:00Z"
    )
    
    # Manual prediction log
    log_prediction(
        home="Bournemouth", away="Liverpool", market="DNB", selection="Bournemouth DNB",
        model_prob=0.48, market_odds=2.30, edge=0.04, bookie="Manual",
        stake=6.50, kickoff="2026-09-20"
    )
    
    df = pd.read_csv(str(test_log))
    expected_cols = [
        'Timestamp', 'Kickoff', 'Home', 'Away', 'Market', 'Selection', 
        'Model_Prob', 'Market_Odds', 'Edge', 'Bookmaker', 'Stake', 'Outcome', 'Closing_Odds', 'CLV', 'PnL'
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2
    assert df.iloc[0]['Home'] == 'Arsenal'
    assert df.iloc[0]['Market'] == '1X2'
    assert df.iloc[1]['Home'] == 'Bournemouth'
    assert df.iloc[1]['Kickoff'] == '2026-09-20'
    assert df.iloc[1]['Bookmaker'] == 'Manual'

def test_upcoming_matches_remain_pending(tmp_path, monkeypatch):
    """Verify that matches scheduled in the future or not yet played remain PENDING."""
    import utils
    import update_results
    
    test_log = tmp_path / "test_pending_log.csv"
    monkeypatch.setattr(utils, "LOG_FILE", str(test_log))
    monkeypatch.setattr(update_results, "LOG_FILE", str(test_log))
    
    # Log an upcoming fixture
    utils.log_prediction(
        home="Bournemouth", away="Liverpool", market="DNB", selection="Bournemouth DNB",
        model_prob=0.48, market_odds=2.30, edge=0.04, bookie="Manual",
        stake=6.50, kickoff="2026-10-25"
    )
    
    # Run results update
    update_results.update()
    
    # Verify outcome remains PENDING and was not resolved against past historical matches
    df = pd.read_csv(str(test_log))
    assert len(df) == 1
    assert df.iloc[0]['Outcome'] == 'PENDING'
    assert df.iloc[0]['PnL'] == 0.0
    assert df.iloc[0]['CLV'] == 'PENDING'




