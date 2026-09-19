import pandas as pd
import numpy as np
import requests
import io
import os
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION & QUANT HYPERPARAMETERS ---
API_KEY = os.getenv("ODDS_API_KEY") 
SPORT_KEY = "soccer_epl"   # Premier League
REGION = "uk"              # Options: us, uk, eu, au
MARKET = "h2h,totals"      # 1X2 and Over/Under Goals (totals)
LOG_FILE = "data/prediction_log.csv"
DATA_CACHE = "data/historical_data.csv"
MODEL_CACHE = "data/models.joblib"

# Unified Strategy Parameters (aligned across backtest, live scanner, and predictor)
MARKET_PRIOR_WEIGHT = 0.50       # Empirical Bayes shrinkage towards market (50% market, 50% tactical ML)
MAX_DIVERGENCE = 0.10            # Adverse selection protection threshold (10% max allowable divergence)
MIN_ODDS = 1.90                  # Lower bound of verified profitable odds band
MAX_ODDS = 2.80                  # Upper bound of verified profitable odds band
MIN_EDGE = 0.030                 # Minimum edge hurdle (3.0%) post-shrinkage
MAX_DRAW_RISK = 0.28             # Maximum allowable market draw probability for 1X2 bets
KELLY_FRACTION = 0.10            # Tenth-Kelly conservative sizing
MAX_STAKE_PCT = 0.025            # Maximum single-bet risk (2.5% of bankroll)


# --- TEAM MAPPING (Odds API / Alternate Names -> Football-Data Standard) ---
TEAM_MAPPING = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham',
    'Brighton and Hove Albion': 'Brighton',
    'Brighton & Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest': "Nott'm Forest",
    'Leeds United': 'Leeds',
    'Sheffield United': 'Sheffield United',
    'Luton Town': 'Luton',
    'Leicester City': 'Leicester',
    'Ipswich Town': 'Ipswich',
    'Southampton': 'Southampton',
    'Bournemouth': 'Bournemouth',
    'AFC Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford',
    'Aston Villa': 'Aston Villa',
    'Everton': 'Everton',
    'Chelsea': 'Chelsea',
    'Arsenal': 'Arsenal',
    'Liverpool': 'Liverpool',
    'Fulham': 'Fulham',
    'Crystal Palace': 'Crystal Palace',
    'Burnley': 'Burnley',
    'Sunderland': 'Sunderland'
}

def normalize_team(name):
    """Normalizes team names to the consistent Football-Data convention."""
    if not isinstance(name, str):
        return name
    clean = name.strip()
    return TEAM_MAPPING.get(clean, clean)


# --- 1. MULTI-SEASON HISTORICAL DATA INGESTION & CACHING ---

def fetch_historical_data(start_year=16, end_year=25, force_refresh=False):
    """
    Downloads and combines 10 seasons of Premier League match stats (2016/17 to 2025/26).
    Ingests match results, halftime goals, card discipline, and fouls to support synthetic xG.
    """
    os.makedirs('data', exist_ok=True)
    
    essential_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG', 'HTR',
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
        'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
    ]
    odds_cols = [
        'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'BbAv>2.5', 'BbAv<2.5',
        'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA', 'BbMxH', 'BbMxD', 'BbMxA',
        'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'MaxCH', 'MaxCD', 'MaxCA',
        'AvgCH', 'AvgCD', 'AvgCA', 'B365CH', 'B365CD', 'B365CA'
    ]
    all_desired_cols = essential_cols + odds_cols
    current_season_code = f"{end_year:02d}{end_year+1:02d}"
    
    # Check if cached data exists and contains the expanded columns including Max and Pinnacle closing odds
    if os.path.exists(DATA_CACHE) and not force_refresh:
        try:
            cached_df = pd.read_csv(DATA_CACHE)
            # Verify cache has required halftime, card, line shopping, and Pinnacle closing columns
            if 'HTHG' in cached_df.columns and 'HY' in cached_df.columns and 'MaxH' in cached_df.columns and 'PSCH' in cached_df.columns:
                cached_df['Date'] = pd.to_datetime(cached_df['Date'])
                cached_df['Season'] = cached_df['Season'].astype(str)
                
                # Fetch only the active current season to append latest games
                url_current = f"https://www.football-data.co.uk/mmz4281/{current_season_code}/E0.csv"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                try:
                    resp = requests.get(url_current, headers=headers, timeout=10)
                    if resp.status_code == 200 and len(resp.text) > 100:
                        curr_df = pd.read_csv(io.StringIO(resp.text))
                        avail_cols = [c for c in all_desired_cols if c in curr_df.columns]
                        curr_df = curr_df[avail_cols].copy()
                        curr_df['Date'] = pd.to_datetime(curr_df['Date'], dayfirst=True)
                        curr_df['Season'] = str(current_season_code)
                        curr_df['HomeTeam'] = curr_df['HomeTeam'].map(normalize_team)
                        curr_df['AwayTeam'] = curr_df['AwayTeam'].map(normalize_team)
                        curr_df = _standardize_max_odds(curr_df)
                        
                        historical_prior = cached_df[cached_df['Season'] != str(current_season_code)]
                        combined = pd.concat([historical_prior, curr_df], ignore_index=True)
                        combined = combined.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
                        combined = combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
                        combined = combined.sort_values('Date').reset_index(drop=True)
                        combined = _standardize_max_odds(combined)
                        combined.to_csv(DATA_CACHE, index=False)
                        return _add_sample_weights(combined)
                    else:
                        cached_df = cached_df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
                        cached_df = _standardize_max_odds(cached_df)
                        return _add_sample_weights(cached_df)
                except Exception as e:
                    cached_df = cached_df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
                    cached_df = _standardize_max_odds(cached_df)
                    return _add_sample_weights(cached_df)
        except Exception:
            pass  # Fall through to re-download
            
    # Download all 10 seasons with expanded feature, line-shopping, and closing odds columns
    print(f"Downloading historical Premier League seasons with closing odds (20{start_year}/20{start_year+1} to 20{end_year}/20{end_year+1})...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    dfs = []
    for y in range(start_year, end_year + 1):
        season_code = f"{y:02d}{y+1:02d}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                season_df = pd.read_csv(io.StringIO(resp.text))
                avail_cols = [c for c in all_desired_cols if c in season_df.columns]
                season_df = season_df[avail_cols].copy()
                season_df['Date'] = pd.to_datetime(season_df['Date'], dayfirst=True)
                season_df['Season'] = str(season_code)
                season_df['HomeTeam'] = season_df['HomeTeam'].map(normalize_team)
                season_df['AwayTeam'] = season_df['AwayTeam'].map(normalize_team)
                season_df = _standardize_max_odds(season_df)
                dfs.append(season_df)
                print(f"  [OK] Season {season_code} loaded: {len(season_df)} matches")
            else:
                print(f"  [!] HTTP {resp.status_code} loading season {season_code}")
        except Exception as e:
            print(f"  [!] Could not load season {season_code}: {e}")
            
    if not dfs:
        print("Error: No data could be fetched.")
        return None
        
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    df_combined = df_combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    df_combined = _standardize_max_odds(df_combined)
    
    # Save cache
    df_combined.to_csv(DATA_CACHE, index=False)
    print(f"Dataset cached to {DATA_CACHE} with closing odds (Total: {len(df_combined)} matches).")
    
    return _add_sample_weights(df_combined)

def _standardize_max_odds(df):
    """Ensures MaxH, MaxD, MaxA and Pinnacle Closing PSCH, PSCD, PSCA exist with proper fallbacks."""
    # 1. Line shopping Max odds
    for col, bbmx_col, b365_col in [('MaxH', 'BbMxH', 'B365H'), ('MaxD', 'BbMxD', 'B365D'), ('MaxA', 'BbMxA', 'B365A')]:
        if col not in df.columns:
            df[col] = df.get(bbmx_col, df.get(b365_col, np.nan))
        else:
            df[col] = df[col].fillna(df.get(bbmx_col, df.get(b365_col, np.nan)))
            
    # 2. Pinnacle closing odds (gold standard benchmark for CLV)
    for col, maxc_col, b365c_col, open_col in [
        ('PSCH', 'MaxCH', 'B365CH', 'PSH'),
        ('PSCD', 'MaxCD', 'B365CD', 'PSD'),
        ('PSCA', 'MaxCA', 'B365CA', 'PSA')
    ]:
        fallback = df.get(maxc_col, df.get(b365c_col, df.get(open_col, df.get(open_col.replace('PS', 'B365'), np.nan))))
        if col not in df.columns:
            df[col] = fallback
        else:
            df[col] = df[col].fillna(fallback)
    return df

def _add_sample_weights(df, half_life_days=500):
    """Applies exponential time-decay sample weights so recent form has higher influence."""
    df = df.copy()
    max_date = df['Date'].max()
    days_ago = (max_date - df['Date']).dt.total_seconds() / 86400.0
    df['Sample_Weight'] = np.exp(-np.log(2) * days_ago / half_life_days)
    return df


# --- 2. ODDS DE-VIGGING (ELIMINATING THE FAVORITE-LONGSHOT BIAS) ---

def devig_odds_power(odds_list):
    """
    Computes fair, vig-free probabilities using the Power method.
    Solves for exponent k such that sum((1/odds_i)^k) = 1.0.
    Accurately accounts for the favorite-longshot bias where bookies overcharge on draws/underdogs.
    """
    raw_probs = [1.0 / o if o > 1.0 else 0.0 for o in odds_list]
    total_raw = sum(raw_probs)
    if total_raw <= 0.0:
        return [1.0 / len(odds_list)] * len(odds_list)
        
    if abs(total_raw - 1.0) < 0.001:
        return raw_probs
        
    # Binary search for k in range [0.5, 3.0]
    low, high = 0.5, 3.0
    for _ in range(20):
        mid = (low + high) / 2.0
        val = sum(p ** mid for p in raw_probs)
        if val > 1.0:
            low = mid
        else:
            high = mid
            
    k = (low + high) / 2.0
    fair_probs = [p ** k for p in raw_probs]
    fair_sum = sum(fair_probs)
    return [p / fair_sum for p in fair_probs]

def devig_odds_proportional(odds_list):
    """Simple proportional normalization (removes margin uniformly)."""
    raw_probs = [1.0 / o if o > 1.0 else 0.0 for o in odds_list]
    s = sum(raw_probs)
    return [p / s for p in raw_probs] if s > 0 else raw_probs


# --- 2b. DRAW NO BET (DNB) & ADVANCED BAYESIAN PROBABILITY ENGINE ---

def compute_dnb_probs(probs_1x2):
    """
    Computes Draw No Bet (DNB / Asian Handicap 0) conditional probabilities from 1X2 distribution.
    Conditioned on a non-draw outcome:
    P(Home DNB) = P(H) / (P(H) + P(A)), P(Away DNB) = P(A) / (P(H) + P(A)).
    """
    p_h = float(probs_1x2[0])
    p_a = float(probs_1x2[2])
    denom = p_h + p_a
    if denom <= 0.0:
        return [0.5, 0.5]
    return [p_h / denom, p_a / denom]

def compute_dnb_odds(h_odd, d_odd, a_odd):
    """
    Computes synthetic Draw No Bet (DNB / Asian Handicap 0) decimal odds from 1X2 prices.
    By hedging a fraction on Draw to refund the stake if the match draws:
    Odds_DNB_Home = O_H * (1 - 1 / O_D) = O_H * (O_D - 1) / O_D
    Odds_DNB_Away = O_A * (1 - 1 / O_D) = O_A * (O_D - 1) / O_D
    """
    if h_odd <= 1.0 or d_odd <= 1.0 or a_odd <= 1.0:
        return [0.0, 0.0]
    home_dnb = h_odd * (d_odd - 1.0) / d_odd
    away_dnb = a_odd * (d_odd - 1.0) / d_odd
    return [round(home_dnb, 3), round(away_dnb, 3)]

def blend_probabilities_logit(p_mkt, p_ml, prior_weight=0.50, class_prior=None):
    """
    Combines market prior with ML prediction in Dirichlet / Log-Odds space:
    z_i = log(p_mkt_i) + (1 - prior_weight) * log(p_ml_i / class_prior_i)
    Softmax normalizes back to a valid probability simplex.
    Preserves calibration at distribution extremes without linear flattening.
    """
    if class_prior is None:
        class_prior = [0.45, 0.25, 0.30]  # Empirical Premier League class prior [H, D, A]
        
    p_mkt = np.clip(np.array(p_mkt, dtype=float), 1e-6, 1.0)
    p_ml = np.clip(np.array(p_ml, dtype=float), 1e-6, 1.0)
    class_prior = np.clip(np.array(class_prior, dtype=float), 1e-6, 1.0)
    
    alpha = 1.0 - prior_weight
    # Log-ratio update relative to base rate
    z = np.log(p_mkt) + alpha * np.log(p_ml / class_prior)
    # Numerical stability shift
    z -= np.max(z)
    exp_z = np.exp(z)
    return list(exp_z / np.sum(exp_z))

def get_adaptive_market_weight(match_date, season_start_date=None, home_team=None, away_team=None, base_weight=MARKET_PRIOR_WEIGHT):
    """
    Dynamically scales market prior weight based on gameweek seasonality and team uncertainty:
    - Early season (GW 1-6, days < 45): Higher market weight (0.60 - 0.70) due to low current-season sample size.
    - Mid/Late season (GW 20-38, days > 120): Lower market weight (0.45 - 0.50) as team EMAs stabilize.
    - Promoted team penalty: +0.08 market weight when facing clubs newly promoted to the Premier League.
    """
    if match_date is None or pd.isna(match_date):
        match_dt = pd.Timestamp.now()
    else:
        match_dt = pd.to_datetime(match_date)
        if pd.isna(match_dt):
            match_dt = pd.Timestamp.now()
            
    if season_start_date is None:
        year = int(match_dt.year if match_dt.month >= 8 else match_dt.year - 1)
        season_start_date = pd.Timestamp(year=year, month=8, day=12)
    else:
        season_start_date = pd.to_datetime(season_start_date)
        
    days_into_season = max(0, (match_dt - season_start_date).days)
    
    # Seasonality decay: starts at +0.15 in August and decays across first 150 days
    seasonality_adjustment = max(-0.05, 0.15 - (days_into_season / 150.0) * 0.20)
    
    promoted_by_season = {
        '2122': {'Brentford', 'Norwich', 'Watford'},
        '2223': {'Fulham', 'Bournemouth', "Nott'm Forest"},
        '2324': {'Burnley', 'Sheffield United', 'Luton'},
        '2425': {'Leicester', 'Ipswich', 'Southampton'},
        '2526': {'Leeds', 'Sunderland', 'Burnley'}
    }
    
    promoted_penalty = 0.0
    h_norm = normalize_team(home_team) if home_team else None
    a_norm = normalize_team(away_team) if away_team else None
    
    for s_code, p_teams in promoted_by_season.items():
        if (h_norm in p_teams or a_norm in p_teams) and str(match_dt.year)[-2:] in s_code:
            promoted_penalty = 0.08
            break
            
    adaptive_weight = base_weight + seasonality_adjustment + promoted_penalty
    return float(np.clip(adaptive_weight, 0.35, 0.75))



# --- 3. DYNAMIC ELO RATING SYSTEM ---

class FootballEloTracker:
    """
    Maintains dynamic pre-match ratings for Premier League teams.
    Updates ratings match-by-match with goal difference multipliers and home advantage.
    """
    def __init__(self, k_factor=20.0, home_advantage=65.0, default_rating=1500.0):
        self.k = k_factor
        self.home_adv = home_advantage
        self.default_rating = default_rating
        self.ratings = {}
        
        self.priors = {
            'Man City': 1750.0, 'Liverpool': 1700.0, 'Arsenal': 1680.0,
            'Chelsea': 1650.0, 'Tottenham': 1620.0, 'Man United': 1620.0,
            'Newcastle': 1580.0, 'Aston Villa': 1550.0, 'Brighton': 1520.0,
            'West Ham': 1510.0, 'Everton': 1480.0, 'Crystal Palace': 1480.0,
            'Wolves': 1480.0, 'Bournemouth': 1470.0, 'Brentford': 1470.0,
            'Fulham': 1460.0, "Nott'm Forest": 1450.0, 'Leicester': 1460.0,
            'Southampton': 1420.0, 'Ipswich': 1410.0, 'Leeds': 1440.0,
            'Burnley': 1430.0, 'Luton': 1400.0, 'Sheffield United': 1390.0,
            'Sunderland': 1400.0
        }

    def get_rating(self, team):
        norm = normalize_team(team)
        if norm not in self.ratings:
            self.ratings[norm] = self.priors.get(norm, self.default_rating)
        return self.ratings[norm]

    def expected_outcome(self, r_home, r_away):
        diff = (r_home + self.home_adv) - r_away
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(self, home_team, away_team, fthg, ftag, ftr):
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)
        e_h = self.expected_outcome(r_h, r_a)
        
        s_h = 1.0 if ftr == 'H' else (0.5 if ftr == 'D' else 0.0)
        s_a = 1.0 - s_h
        
        goal_diff = abs(fthg - ftag)
        if goal_diff <= 1:
            multiplier = 1.0
        elif goal_diff == 2:
            multiplier = 1.5
        elif goal_diff == 3:
            multiplier = 1.75
        else:
            multiplier = 1.75 + (goal_diff - 3) / 8.0

        delta_h = self.k * multiplier * (s_h - e_h)
        delta_a = self.k * multiplier * (s_a - (1.0 - e_h))
        
        self.ratings[normalize_team(home_team)] = r_h + delta_h
        self.ratings[normalize_team(away_team)] = r_a + delta_a
        return delta_h, delta_a


# --- 4. ADVANCED FEATURE ENGINEERING (SYNTHETIC xG, FORM & DISCIPLINE) ---

def compute_ema(values, alpha=0.25, default=0.0):
    """
    Computes exponential moving average over past chronological values.
    alpha = 0.25 models ~8-match tactical form horizon.
    alpha = 0.08 models ~25-match team baseline ability horizon.
    """
    if not values:
        return default
    weights = [(1.0 - alpha) ** (len(values) - 1 - i) for i in range(len(values))]
    total_w = sum(weights)
    if total_w <= 0.0:
        return default
    return sum(v * w for v, w in zip(values, weights)) / total_w

def add_pre_match_features(df, n_games=5):
    """
    Computes pre-match features sequentially without future leakage using Dual-Horizon EMAs:
    - Pre-match Elo ratings and differentials
    - Venue-specific form (Home form at home, Away form away via venue EMA)
    - Synthetic xG / Shot Quality: Shot Accuracy (HST/HS), Conversion Efficiency (Goals/HST),
      Defensive Target Shot Suppression
    - Discipline pressure (yellow/red card index via tactical EMA)
    - Second-half momentum ratio
    - Calendar rest days (capped at 14)
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    elo_tracker = FootballEloTracker()
    
    # Elo
    home_elo_list, away_elo_list, elo_diff_list = [], [], []
    
    # Venue stats
    home_v_gf, home_v_ga, home_v_shots, home_v_sot, home_v_corners = [], [], [], [], []
    away_v_gf, away_v_ga, away_v_shots, away_v_sot, away_v_corners = [], [], [], [], []
    
    # Overall form
    home_all_gf, home_all_ga, home_all_sot = [], [], []
    away_all_gf, away_all_ga, away_all_sot = [], [], []
    
    # Synthetic xG & Shot Quality metrics
    home_shot_acc, away_shot_acc = [], []
    home_goal_conv, away_goal_conv = [], []
    home_def_suppress, away_def_suppress = [], []
    home_card_stress, away_card_stress = [], []
    home_2h_momentum, away_2h_momentum = [], []
    xg_diff_list, h2h_dom_list = [], []
    
    # Rest days
    home_rest_list, away_rest_list = [], []
    
    team_history = {}
    h2h_history = {}
    
    for idx, row in df.iterrows():
        h = normalize_team(row['HomeTeam'])
        a = normalize_team(row['AwayTeam'])
        date = row['Date']
        
        # 1. Elo
        r_h = elo_tracker.get_rating(h)
        r_a = elo_tracker.get_rating(a)
        elo_diff = (r_h + elo_tracker.home_adv) - r_a
        
        home_elo_list.append(r_h)
        away_elo_list.append(r_a)
        elo_diff_list.append(elo_diff)
        
        h_past = team_history.get(h, [])
        a_past = team_history.get(a, [])
        
        # Rest Days
        h_rest = min(14.0, (date - h_past[-1]['date']).total_seconds() / 86400.0) if len(h_past) > 0 else 7.0
        a_rest = min(14.0, (date - a_past[-1]['date']).total_seconds() / 86400.0) if len(a_past) > 0 else 7.0
        home_rest_list.append(round(h_rest, 1))
        away_rest_list.append(round(a_rest, 1))
        
        # Venue-Specific Stats (decaying across historical venue games with alpha=0.20)
        h_home = [m for m in h_past if m['is_home']]
        if len(h_home) > 0:
            hvgf = compute_ema([m['gf'] for m in h_home], alpha=0.20, default=1.5)
            hvga = compute_ema([m['ga'] for m in h_home], alpha=0.20, default=1.2)
            hvs = compute_ema([m['shots'] for m in h_home], alpha=0.20, default=12.5)
            hvsot = compute_ema([m['sot'] for m in h_home], alpha=0.20, default=4.2)
            hvc = compute_ema([m['corners'] for m in h_home], alpha=0.20, default=5.5)
        else:
            hvgf, hvga, hvs, hvsot, hvc = 1.5, 1.2, 12.5, 4.2, 5.5
            
        home_v_gf.append(round(hvgf, 2))
        home_v_ga.append(round(hvga, 2))
        home_v_shots.append(round(hvs, 2))
        home_v_sot.append(round(hvsot, 2))
        home_v_corners.append(round(hvc, 2))
        
        a_away = [m for m in a_past if not m['is_home']]
        if len(a_away) > 0:
            avgf = compute_ema([m['gf'] for m in a_away], alpha=0.20, default=1.1)
            avga = compute_ema([m['ga'] for m in a_away], alpha=0.20, default=1.6)
            avs = compute_ema([m['shots'] for m in a_away], alpha=0.20, default=9.5)
            avsot = compute_ema([m['sot'] for m in a_away], alpha=0.20, default=3.2)
            avc = compute_ema([m['corners'] for m in a_away], alpha=0.20, default=4.2)
        else:
            avgf, avga, avs, avsot, avc = 1.1, 1.6, 9.5, 3.2, 4.2
            
        away_v_gf.append(round(avgf, 2))
        away_v_ga.append(round(avga, 2))
        away_v_shots.append(round(avs, 2))
        away_v_sot.append(round(avsot, 2))
        away_v_corners.append(round(avc, 2))
        
        # Overall Dual-Horizon Form (8-match tactical EMA, alpha=0.25)
        home_all_gf.append(round(compute_ema([m['gf'] for m in h_past], alpha=0.25, default=1.4), 2))
        home_all_ga.append(round(compute_ema([m['ga'] for m in h_past], alpha=0.25, default=1.3), 2))
        home_all_sot.append(round(compute_ema([m['sot'] for m in h_past], alpha=0.25, default=4.0), 2))
        
        away_all_gf.append(round(compute_ema([m['gf'] for m in a_past], alpha=0.25, default=1.3), 2))
        away_all_ga.append(round(compute_ema([m['ga'] for m in a_past], alpha=0.25, default=1.4), 2))
        away_all_sot.append(round(compute_ema([m['sot'] for m in a_past], alpha=0.25, default=3.8), 2))
        
        # Synthetic xG & Shot Quality via Dual-Horizon EMA
        if len(h_past) > 0:
            h_tot_shots = compute_ema([m['shots'] for m in h_past], alpha=0.15, default=11.0)
            h_tot_sot = compute_ema([m['sot'] for m in h_past], alpha=0.15, default=4.0)
            h_tot_gf = compute_ema([m['gf'] for m in h_past], alpha=0.15, default=1.4)
            h_tot_opp_shots = compute_ema([m['opp_shots'] for m in h_past], alpha=0.15, default=11.0)
            h_tot_opp_sot = compute_ema([m['opp_sot'] for m in h_past], alpha=0.15, default=4.0)
            h_card_idx = compute_ema([m['cards'] for m in h_past], alpha=0.25, default=1.8)
            h_2h_goals = compute_ema([m['2h_gf'] for m in h_past], alpha=0.25, default=0.7)
            
            h_acc = h_tot_sot / max(1.0, h_tot_shots)
            h_conv = h_tot_gf / max(1.0, h_tot_sot)
            h_supp = h_tot_opp_sot / max(1.0, h_tot_opp_shots)
            h_2h_ratio = h_2h_goals / max(0.5, h_tot_gf)
        else:
            h_acc, h_conv, h_supp, h_card_idx, h_2h_ratio = 0.33, 0.30, 0.33, 1.8, 0.55
            
        home_shot_acc.append(round(h_acc, 3))
        home_goal_conv.append(round(h_conv, 3))
        home_def_suppress.append(round(h_supp, 3))
        home_card_stress.append(round(h_card_idx, 2))
        home_2h_momentum.append(round(min(1.0, h_2h_ratio), 3))
        
        if len(a_past) > 0:
            a_tot_shots = compute_ema([m['shots'] for m in a_past], alpha=0.15, default=10.0)
            a_tot_sot = compute_ema([m['sot'] for m in a_past], alpha=0.15, default=3.5)
            a_tot_gf = compute_ema([m['gf'] for m in a_past], alpha=0.15, default=1.2)
            a_tot_opp_shots = compute_ema([m['opp_shots'] for m in a_past], alpha=0.15, default=11.0)
            a_tot_opp_sot = compute_ema([m['opp_sot'] for m in a_past], alpha=0.15, default=4.0)
            a_card_idx = compute_ema([m['cards'] for m in a_past], alpha=0.25, default=1.8)
            a_2h_goals = compute_ema([m['2h_gf'] for m in a_past], alpha=0.25, default=0.6)
            
            a_acc = a_tot_sot / max(1.0, a_tot_shots)
            a_conv = a_tot_gf / max(1.0, a_tot_sot)
            a_supp = a_tot_opp_sot / max(1.0, a_tot_opp_shots)
            a_2h_ratio = a_2h_goals / max(0.5, a_tot_gf)
        else:
            a_acc, a_conv, a_supp, a_card_idx, a_2h_ratio = 0.33, 0.30, 0.33, 1.8, 0.55
            
        away_shot_acc.append(round(a_acc, 3))
        away_goal_conv.append(round(a_conv, 3))
        away_def_suppress.append(round(a_supp, 3))
        away_card_stress.append(round(a_card_idx, 2))
        away_2h_momentum.append(round(min(1.0, a_2h_ratio), 3))
        
        # Synthetic xG differential (attacking volume * conversion * opp suppression)
        h_xg = round(hvs * h_acc * h_conv * (a_supp / 0.33), 2)
        a_xg = round(avs * a_acc * a_conv * (h_supp / 0.33), 2)
        xg_diff_list.append(round(h_xg - a_xg, 2))
        
        # Head-to-Head historical matchup dominance
        h2h_key = tuple(sorted([h, a]))
        past_meetings = h2h_history.get(h2h_key, [])
        if len(past_meetings) > 0:
            h2h_diffs = [m['diff'] if m['home'] == h else -m['diff'] for m in past_meetings]
            h2h_dom = compute_ema(h2h_diffs, alpha=0.35, default=0.0)
        else:
            h2h_dom = 0.0
        h2h_dom_list.append(round(h2h_dom, 2))
        
        # Post-match stats extraction
        fthg = float(row['FTHG'])
        ftag = float(row['FTAG'])
        ftr = row['FTR']
        hthg = float(row['HTHG']) if pd.notna(row.get('HTHG')) else 0.0
        htag = float(row['HTAG']) if pd.notna(row.get('HTAG')) else 0.0
        
        hs = float(row['HS']) if pd.notna(row.get('HS')) else 11.0
        as_ = float(row['AS']) if pd.notna(row.get('AS')) else 10.0
        hst = float(row['HST']) if pd.notna(row.get('HST')) else 4.0
        ast = float(row['AST']) if pd.notna(row.get('AST')) else 3.5
        hc = float(row['HC']) if pd.notna(row.get('HC')) else 5.0
        ac = float(row['AC']) if pd.notna(row.get('AC')) else 4.5
        
        hy = float(row['HY']) if pd.notna(row.get('HY')) else 1.5
        ay = float(row['AY']) if pd.notna(row.get('AY')) else 1.5
        hr = float(row['HR']) if pd.notna(row.get('HR')) else 0.0
        ar = float(row['AR']) if pd.notna(row.get('AR')) else 0.0
        
        elo_tracker.update(h, a, fthg, ftag, ftr)
        
        team_history.setdefault(h, []).append({
            'date': date, 'is_home': True, 'gf': fthg, 'ga': ftag,
            'shots': hs, 'sot': hst, 'opp_shots': as_, 'opp_sot': ast,
            'corners': hc, 'cards': hy + 2.0 * hr, '2h_gf': max(0.0, fthg - hthg)
        })
        team_history.setdefault(a, []).append({
            'date': date, 'is_home': False, 'gf': ftag, 'ga': fthg,
            'shots': as_, 'sot': ast, 'opp_shots': hs, 'opp_sot': hst,
            'corners': ac, 'cards': ay + 2.0 * ar, '2h_gf': max(0.0, ftag - htag)
        })
        h2h_history.setdefault(h2h_key, []).append({
            'date': date, 'home': h, 'diff': fthg - ftag
        })

    # Attach all feature columns
    df['Home_Elo'] = home_elo_list
    df['Away_Elo'] = away_elo_list
    df['Elo_Diff'] = elo_diff_list
    
    df['Home_Rest_Days'] = home_rest_list
    df['Away_Rest_Days'] = away_rest_list
    
    df['Home_V_Goals_For'] = home_v_gf
    df['Home_V_Goals_Against'] = home_v_ga
    df['Home_V_Shots'] = home_v_shots
    df['Home_V_SOT'] = home_v_sot
    df['Home_V_Corners'] = home_v_corners
    
    df['Away_V_Goals_For'] = away_v_gf
    df['Away_V_Goals_Against'] = away_v_ga
    df['Away_V_Shots'] = away_v_shots
    df['Away_V_SOT'] = away_v_sot
    df['Away_V_Corners'] = away_v_corners
    
    df['Home_All_Goals_For'] = home_all_gf
    df['Home_All_Goals_Against'] = home_all_ga
    df['Home_All_SOT'] = home_all_sot
    
    df['Away_All_Goals_For'] = away_all_gf
    df['Away_All_Goals_Against'] = away_all_ga
    df['Away_All_SOT'] = away_all_sot
    
    # Synthetic xG
    df['Home_Shot_Acc'] = home_shot_acc
    df['Away_Shot_Acc'] = away_shot_acc
    df['Home_Goal_Conv'] = home_goal_conv
    df['Away_Goal_Conv'] = away_goal_conv
    df['Home_Def_Suppression'] = home_def_suppress
    df['Away_Def_Suppression'] = away_def_suppress
    df['Home_Card_Stress'] = home_card_stress
    df['Away_Card_Stress'] = away_card_stress
    df['Home_2H_Momentum'] = home_2h_momentum
    df['Away_2H_Momentum'] = away_2h_momentum
    df['Expected_Goals_Diff'] = xg_diff_list
    df['H2H_Dominance'] = h2h_dom_list
    
    # 1X2 Target
    ftr_map = {'H': 0, 'D': 1, 'A': 2}
    df['FTR_Class'] = df['FTR'].map(ftr_map)
    
    return df, elo_tracker

def get_live_match_features(df, elo_tracker, home_team, away_team, match_date=None, **kwargs):
    """Computes real-time pre-match feature vectors using Dual-Horizon EMAs."""
    h = normalize_team(home_team)
    a = normalize_team(away_team)
    match_date = pd.Timestamp.now() if match_date is None else pd.to_datetime(match_date)
        
    r_h = elo_tracker.get_rating(h)
    r_a = elo_tracker.get_rating(a)
    elo_diff = (r_h + elo_tracker.home_adv) - r_a
    
    h_matches = df[((df['HomeTeam'] == h) | (df['AwayTeam'] == h)) & (df['Date'] < match_date)].sort_values('Date')
    a_matches = df[((df['HomeTeam'] == a) | (df['AwayTeam'] == a)) & (df['Date'] < match_date)].sort_values('Date')
    
    h_rest = 7.0 if len(h_matches) == 0 else min(14.0, (match_date - h_matches.iloc[-1]['Date']).total_seconds() / 86400.0)
    a_rest = 7.0 if len(a_matches) == 0 else min(14.0, (match_date - a_matches.iloc[-1]['Date']).total_seconds() / 86400.0)
    
    # Venue stats via EMA (alpha=0.20)
    h_home = df[(df['HomeTeam'] == h) & (df['Date'] < match_date)].sort_values('Date')
    hv_gf = compute_ema(h_home['FTHG'].tolist(), alpha=0.20, default=1.5)
    hv_ga = compute_ema(h_home['FTAG'].tolist(), alpha=0.20, default=1.2)
    hv_s = compute_ema(h_home['HS'].tolist(), alpha=0.20, default=12.5) if 'HS' in h_home else 12.5
    hv_sot = compute_ema(h_home['HST'].tolist(), alpha=0.20, default=4.2) if 'HST' in h_home else 4.2
    hv_c = compute_ema(h_home['HC'].tolist(), alpha=0.20, default=5.5) if 'HC' in h_home else 5.5
    
    a_away = df[(df['AwayTeam'] == a) & (df['Date'] < match_date)].sort_values('Date')
    av_gf = compute_ema(a_away['FTAG'].tolist(), alpha=0.20, default=1.1)
    av_ga = compute_ema(a_away['FTHG'].tolist(), alpha=0.20, default=1.6)
    av_s = compute_ema(a_away['AS'].tolist(), alpha=0.20, default=9.5) if 'AS' in a_away else 9.5
    av_sot = compute_ema(a_away['AST'].tolist(), alpha=0.20, default=3.2) if 'AST' in a_away else 3.2
    av_c = compute_ema(a_away['AC'].tolist(), alpha=0.20, default=4.2) if 'AC' in a_away else 4.2
    
    # Overall tactical form via EMA (alpha=0.25)
    h_gf_vals = [r['FTHG'] if r['HomeTeam'] == h else r['FTAG'] for _, r in h_matches.iterrows()]
    h_ga_vals = [r['FTAG'] if r['HomeTeam'] == h else r['FTHG'] for _, r in h_matches.iterrows()]
    h_sot_vals = [r['HST'] if r['HomeTeam'] == h else r['AST'] for _, r in h_matches.iterrows()]
    h_s_vals = [r['HS'] if r['HomeTeam'] == h else r['AS'] for _, r in h_matches.iterrows()] if 'HS' in h_matches else []
    h_opp_s_vals = [r['AS'] if r['HomeTeam'] == h else r['HS'] for _, r in h_matches.iterrows()] if 'AS' in h_matches else []
    h_opp_sot_vals = [r['AST'] if r['HomeTeam'] == h else r['HST'] for _, r in h_matches.iterrows()] if 'AST' in h_matches else []
    
    h_gf = compute_ema(h_gf_vals, alpha=0.25, default=1.4)
    h_ga = compute_ema(h_ga_vals, alpha=0.25, default=1.3)
    h_sot = compute_ema(h_sot_vals, alpha=0.25, default=4.0)
    
    a_gf_vals = [r['FTAG'] if r['AwayTeam'] == a else r['FTHG'] for _, r in a_matches.iterrows()]
    a_ga_vals = [r['FTHG'] if r['AwayTeam'] == a else r['FTAG'] for _, r in a_matches.iterrows()]
    a_sot_vals = [r['AST'] if r['AwayTeam'] == a else r['HST'] for _, r in a_matches.iterrows()]
    a_s_vals = [r['AS'] if r['AwayTeam'] == a else r['HS'] for _, r in a_matches.iterrows()] if 'AS' in a_matches else []
    a_opp_s_vals = [r['HS'] if r['AwayTeam'] == a else r['AS'] for _, r in a_matches.iterrows()] if 'HS' in a_matches else []
    a_opp_sot_vals = [r['HST'] if r['AwayTeam'] == a else r['AST'] for _, r in a_matches.iterrows()] if 'HST' in a_matches else []
    
    a_gf = compute_ema(a_gf_vals, alpha=0.25, default=1.3)
    a_ga = compute_ema(a_ga_vals, alpha=0.25, default=1.4)
    a_sot = compute_ema(a_sot_vals, alpha=0.25, default=3.8)
    
    # Synthetic xG & Shot Quality via EMA
    h_tot_s = compute_ema(h_s_vals, alpha=0.15, default=11.0)
    h_tot_sot = compute_ema(h_sot_vals, alpha=0.15, default=4.0)
    h_tot_gf = compute_ema(h_gf_vals, alpha=0.15, default=1.4)
    h_opp_s = compute_ema(h_opp_s_vals, alpha=0.15, default=11.0)
    h_opp_sot = compute_ema(h_opp_sot_vals, alpha=0.15, default=4.0)
    
    a_tot_s = compute_ema(a_s_vals, alpha=0.15, default=10.0)
    a_tot_sot = compute_ema(a_sot_vals, alpha=0.15, default=3.5)
    a_tot_gf = compute_ema(a_gf_vals, alpha=0.15, default=1.2)
    a_opp_s = compute_ema(a_opp_s_vals, alpha=0.15, default=11.0)
    a_opp_sot = compute_ema(a_opp_sot_vals, alpha=0.15, default=4.0)
    
    h_acc = h_tot_sot / max(1.0, h_tot_s)
    h_conv = h_tot_gf / max(1.0, h_tot_sot)
    h_supp = h_opp_sot / max(1.0, h_opp_s)
    
    a_acc = a_tot_sot / max(1.0, a_tot_s)
    a_conv = a_tot_gf / max(1.0, a_tot_sot)
    a_supp = a_opp_sot / max(1.0, a_opp_s)
    
    # Synthetic xG differential
    h_xg = hv_s * h_acc * h_conv * (a_supp / 0.33)
    a_xg = av_s * a_acc * a_conv * (h_supp / 0.33)
    xg_diff = round(h_xg - a_xg, 2)
    
    # Head-to-head historical matchup dominance
    h2h_past = df[(((df['HomeTeam'] == h) & (df['AwayTeam'] == a)) | ((df['HomeTeam'] == a) & (df['AwayTeam'] == h))) & (df['Date'] < match_date)].sort_values('Date')
    if len(h2h_past) > 0:
        h2h_diffs = [
            (float(r['FTHG']) - float(r['FTAG'])) if r['HomeTeam'] == h else (float(r['FTAG']) - float(r['FTHG']))
            for _, r in h2h_past.iterrows()
        ]
        h2h_dom = compute_ema(h2h_diffs, alpha=0.35, default=0.0)
    else:
        h2h_dom = 0.0
        
    return {
        'Home_Elo': r_h, 'Away_Elo': r_a, 'Elo_Diff': elo_diff,
        'Home_Rest_Days': round(h_rest, 1), 'Away_Rest_Days': round(a_rest, 1),
        'Home_V_Goals_For': round(hv_gf, 2), 'Home_V_Goals_Against': round(hv_ga, 2),
        'Home_V_Shots': round(hv_s, 2), 'Home_V_SOT': round(hv_sot, 2), 'Home_V_Corners': round(hv_c, 2),
        'Away_V_Goals_For': round(av_gf, 2), 'Away_V_Goals_Against': round(av_ga, 2),
        'Away_V_Shots': round(av_s, 2), 'Away_V_SOT': round(av_sot, 2), 'Away_V_Corners': round(av_c, 2),
        'Home_All_Goals_For': round(h_gf, 2), 'Home_All_Goals_Against': round(h_ga, 2), 'Home_All_SOT': round(h_sot, 2),
        'Away_All_Goals_For': round(a_gf, 2), 'Away_All_Goals_Against': round(a_ga, 2), 'Away_All_SOT': round(a_sot, 2),
        'Home_Shot_Acc': round(h_acc, 3), 'Away_Shot_Acc': round(a_acc, 3),
        'Home_Goal_Conv': round(h_conv, 3), 'Away_Goal_Conv': round(a_conv, 3),
        'Home_Def_Suppression': round(h_supp, 3), 'Away_Def_Suppression': round(a_supp, 3),
        'Home_Card_Stress': 1.8, 'Away_Card_Stress': 1.8,
        'Home_2H_Momentum': 0.55, 'Away_2H_Momentum': 0.55,
        'Expected_Goals_Diff': xg_diff,
        'H2H_Dominance': round(h2h_dom, 2)
    }


# --- 5. DIXON-COLES BIVARIATE SCORE MATRIX ---

def dixon_coles_tau(x, y, lambda_h, lambda_a, rho=-0.13):
    """Dixon & Coles (1997) bivariate correction factor for low-scoring states (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1.0 - (lambda_h * lambda_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + (lambda_h * rho)
    elif x == 1 and y == 0:
        return 1.0 + (lambda_a * rho)
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def calculate_dixon_coles_grid(lambda_h, lambda_a, max_goals=10, rho=-0.13):
    """Computes an adjusted bivariate Poisson probability grid."""
    lambda_h = max(0.01, float(lambda_h))
    lambda_a = max(0.01, float(lambda_a))
    grid = np.zeros((max_goals, max_goals))
    
    for x in range(max_goals):
        p_x = (math.exp(-lambda_h) * (lambda_h ** x)) / math.factorial(x)
        for y in range(max_goals):
            p_y = (math.exp(-lambda_a) * (lambda_a ** y)) / math.factorial(y)
            tau = dixon_coles_tau(x, y, lambda_h, lambda_a, rho)
            grid[x, y] = max(0.0, p_x * p_y * tau)
            
    grid /= np.sum(grid)
    return grid

def get_market_probabilities(lambda_h, lambda_a, rho=-0.13):
    """Derives market probabilities (Totals, BTTS, Exact Score) from the bivariate Dixon-Coles grid."""
    grid = calculate_dixon_coles_grid(lambda_h, lambda_a, max_goals=10, rho=rho)
    
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    under_25_prob = 0.0
    btts_yes_prob = 0.0
    
    for x in range(10):
        for y in range(10):
            p = grid[x, y]
            if x > y: home_win_prob += p
            elif x == y: draw_prob += p
            else: away_win_prob += p
            if (x + y) < 2.5: under_25_prob += p
            if x >= 1 and y >= 1: btts_yes_prob += p
                
    over_25_prob = 1.0 - under_25_prob
    btts_no_prob = 1.0 - btts_yes_prob
    
    max_idx = np.unravel_index(np.argmax(grid), grid.shape)
    correct_score = f"{max_idx[0]}-{max_idx[1]}"
    
    return {
        'H': float(home_win_prob), 'D': float(draw_prob), 'A': float(away_win_prob),
        'Over2.5': float(over_25_prob), 'Under2.5': float(under_25_prob),
        'BTTS_Yes': float(btts_yes_prob), 'BTTS_No': float(btts_no_prob),
        'Score': correct_score
    }


# --- 6. SAFE PRODUCTION STAKING & LONGSHOT FILTERING ---

def calculate_kelly(prob, odds, bankroll=1000, fraction=KELLY_FRACTION, max_stake_pct=MAX_STAKE_PCT,
                    min_odds=MIN_ODDS, max_odds=MAX_ODDS, min_edge=MIN_EDGE,
                    max_divergence=MAX_DIVERGENCE, drawdown_factor=1.0,
                    fair_market_prob=None, max_draw_risk=MAX_DRAW_RISK, draw_prob=None,
                    is_dnb=False, concurrent_bets=1, **kwargs):
    """
    Calculates disciplined bet size with Tier 1/2 quantitative parameters:
    1. Rejects toxic odds outside profitable band [min_odds, max_odds] (default [1.90, 2.80] for 1X2, [1.40, 2.80] for DNB).
    2. Requires a minimum edge of min_edge (default +3.0%).
    3. Adverse selection filter: Rejects if divergence > max_divergence (default 10%).
    4. Draw risk protection: Rejects 1X2 bets when match draw probability > max_draw_risk (default 28%).
       Ignored for DNB because draws result in stake refund (push).
    5. Computes clean Tenth-Kelly sizing with drawdown circuit breaker.
    6. Applies simultaneous concurrent bet dampener: stake * (1 / sqrt(concurrent_bets)).
    7. Capped at max_stake_pct (default 2.5% of bankroll).
    """
    effective_min_odds = 1.40 if is_dnb else min_odds
    if odds < effective_min_odds or odds > max_odds or prob <= 0.0:
        return 0.0
        
    benchmark_prob = fair_market_prob if (fair_market_prob is not None and fair_market_prob > 0) else (1.0 / odds)
    
    # Adverse selection protection: If model disagrees massively with market, market likely has injury/lineup news
    if max_divergence is not None and round(abs(prob - benchmark_prob), 6) > max_divergence:
        return 0.0
        
    # Draw risk protection: High draw volatility causes catastrophic losses on 1X2 match betting.
    # For DNB bets, draw pushes/refunds stake, so max_draw_risk filter is bypassed.
    if not is_dnb and max_draw_risk is not None and draw_prob is not None and draw_prob > max_draw_risk:
        return 0.0
        
    edge = prob - benchmark_prob
    if edge < min_edge:
        return 0.0
        
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    
    if f_star <= 0.0:
        return 0.0
        
    # Simultaneous portfolio dampener for concurrent matches
    concurrent_dampener = 1.0 / math.sqrt(max(1, concurrent_bets))
    suggested_stake = f_star * fraction * drawdown_factor * concurrent_dampener * bankroll
    max_stake = bankroll * max_stake_pct
    return round(float(min(suggested_stake, max_stake)), 2)


# --- 7. ODDS API & AUDITED LOGGING WITH CLV ---

def fetch_market_odds():
    """Fetches live odds from multiple bookmakers via The Odds API."""
    if not API_KEY:
        print("Note: No ODDS_API_KEY found in .env. Live odds scanning unavailable.")
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        'apiKey': API_KEY, 'regions': REGION,
        'markets': MARKET, 'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json() if response.status_code == 200 else []
    except Exception:
        return []

def log_prediction(home, away, market, selection, model_prob, market_odds, edge, bookie, stake=0.0, kickoff="MANUAL", closing_odds=None):
    """
    Appends predictions to CSV with a complete schema supporting Closing Line Value (CLV).
    """
    os.makedirs('data', exist_ok=True)
    
    headers = [
        'Timestamp', 'Kickoff', 'Home', 'Away', 'Market', 'Selection', 
        'Model_Prob', 'Market_Odds', 'Edge', 'Bookmaker', 'Stake', 'Outcome', 'Closing_Odds', 'CLV', 'PnL'
    ]
    
    clv_val = round((market_odds / closing_odds - 1.0) * 100.0, 2) if (closing_odds and closing_odds > 1.0) else 0.0
    
    entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Kickoff': kickoff,
        'Home': normalize_team(home),
        'Away': normalize_team(away),
        'Market': market,
        'Selection': selection,
        'Model_Prob': round(float(model_prob), 3),
        'Market_Odds': round(float(market_odds), 2),
        'Edge': f"{edge:+.1%}",
        'Bookmaker': bookie,
        'Stake': round(float(stake), 2),
        'Outcome': 'PENDING',
        'Closing_Odds': round(float(closing_odds), 2) if closing_odds else 0.0,
        'CLV': f"{clv_val:+.1f}%" if closing_odds else "PENDING",
        'PnL': 0.0
    }
    
    df = pd.DataFrame([entry], columns=headers)
    write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    
    try:
        df.to_csv(LOG_FILE, mode='a', index=False, header=write_header)
    except PermissionError:
        backup = f"data/backup_log_{datetime.now().strftime('%H%M%S')}.csv"
        df.to_csv(backup, index=False, header=write_header)