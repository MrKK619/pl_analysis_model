import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("ODDS_API_KEY") 
SPORT_KEY = "soccer_epl"   # Premier League
REGION = "uk"              # Options: us, uk, eu, au
MARKET = "h2h,totals"             # 1X2 and Over/Under Goals (totals)
LOG_FILE = "data/prediction_log.csv"

# --- TEAM MAPPING (Odds API -> Football Data) ---
TEAM_MAPPING = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham',
    'Brighton and Hove Albion': 'Brighton',
    'Wolverhampton Wanderers': 'Wolves',
    'Nottingham Forest': "Nott'm Forest",
    'Leeds United': 'Leeds',
    'Sheffield United': 'Sheffield United',
    'Luton Town': 'Luton',
    'Leicester City': 'Leicester',
    'Ipswich Town': 'Ipswich',
    'Southampton': 'Southampton'
}

def normalize_team(name):
    return TEAM_MAPPING.get(name, name)


def fetch_historical_data():
    """Downloads latest 2024/2025 and 2025/2026 season stats for model training."""
    url_2425 = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    url_2526 = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    
    essential_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC'
    ]
    
    try:
        print("Downloading 2024/2025 season...")
        df_2425 = pd.read_csv(url_2425)[essential_cols].copy()
        df_2425['Date'] = pd.to_datetime(df_2425['Date'], dayfirst=True)
        
        print("Downloading 2025/2026 season...")
        df_2526 = pd.read_csv(url_2526)[essential_cols].copy()
        df_2526['Date'] = pd.to_datetime(df_2526['Date'], dayfirst=True)
        
        # Combine and sort chronologically
        df_combined = pd.concat([df_2425, df_2526], ignore_index=True)
        df_combined = df_combined.sort_values('Date').dropna(subset=['FTR'])
        return df_combined
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None

def get_team_rolling_stats(df, team, before_date, n_games=5):
    """Calculates a team's rolling averages for their last N games strictly before before_date."""
    # Filter for games played by team prior to before_date
    team_games = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < before_date)].tail(n_games)
    
    if len(team_games) == 0:
        return 10.0, 4.0, 5.0, 1.35, 1.35  # Safe defaults: shots, SOT, corners, goals scored, goals conceded
    
    shots = team_games.apply(lambda x: x['HS'] if x['HomeTeam'] == team else x['AS'], axis=1).mean()
    shots_on_target = team_games.apply(lambda x: x['HST'] if x['HomeTeam'] == team else x['AST'], axis=1).mean()
    corners = team_games.apply(lambda x: x['HC'] if x['HomeTeam'] == team else x['AC'], axis=1).mean()
    
    # Dynamic Goals Scored & Conceded
    goals_scored = team_games.apply(lambda x: x['FTHG'] if x['HomeTeam'] == team else x['FTAG'], axis=1).mean()
    goals_conceded = team_games.apply(lambda x: x['FTAG'] if x['HomeTeam'] == team else x['FTHG'], axis=1).mean()
    
    return round(shots, 2), round(shots_on_target, 2), round(corners, 2), round(goals_scored, 2), round(goals_conceded, 2)

def add_pre_match_rolling_features(df, n_games=5):
    """Computes and appends pre-match rolling averages as features to the historical dataframe."""
    home_shots, home_sot, home_corners, home_goals_scored, home_goals_conceded = [], [], [] ,[], []
    away_shots, away_sot, away_corners, away_goals_scored, away_goals_conceded = [], [], [] ,[], []
    
    for idx, row in df.iterrows():
        date = row['Date']
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        
        # Rolling stats for Home Team
        hs, hsot, hc, hgs, hgc = get_team_rolling_stats(df, h_team, date, n_games)
        home_shots.append(hs)
        home_sot.append(hsot)
        home_corners.append(hc)
        home_goals_scored.append(hgs)
        home_goals_conceded.append(hgc)
        
        # Rolling stats for Away Team
        as_, asot, ac, ags, agc = get_team_rolling_stats(df, a_team, date, n_games)
        away_shots.append(as_)
        away_sot.append(asot)
        away_corners.append(ac)
        away_goals_scored.append(ags)
        away_goals_conceded.append(agc)
        
    df['Home_Roll_Shots'] = home_shots
    df['Home_Roll_SOT'] = home_sot
    df['Home_Roll_Corners'] = home_corners
    df['Home_Roll_Goals_Scored'] = home_goals_scored
    df['Home_Roll_Goals_Conceded'] = home_goals_conceded
    
    df['Away_Roll_Shots'] = away_shots
    df['Away_Roll_SOT'] = away_sot
    df['Away_Roll_Corners'] = away_corners
    df['Away_Roll_Goals_Scored'] = away_goals_scored
    df['Away_Roll_Goals_Conceded'] = away_goals_conceded
    return df


def fetch_market_odds():
    """Fetches live odds from multiple bookmakers via The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        'apiKey': API_KEY,
        'regions': REGION,
        'markets': MARKET,
        'oddsFormat': 'decimal'
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Odds API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

def american_to_decimal(american_odds):
    """Converts American odds (+150, -110) to Decimal (2.50, 1.91)."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

def calculate_kelly(prob, odds, bankroll=1000, fraction=0.25):
    """Calculates optimal bet size using Quarter Kelly (0.25)."""
    if odds <= 1.0 or prob <= 0: return 0
    # Kelly Formula: f = (bp - q) / b
    b = odds - 1
    q = 1 - prob
    f_star = (b * prob - q) / b
    return round(max(0, f_star * fraction * bankroll), 2)

def log_prediction(home, away, h_prob, d_prob, a_prob, h_odds, d_odds, a_odds, bookie, kickoff="MANUAL"):
    """Saves predictions to CSV with a safety net for PermissionErrors (Excel)."""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Kickoff': kickoff,
        'Home': home,
        'Away': away,
        'Model_H_Prob': round(h_prob, 3),
        'Model_D_Prob': round(d_prob, 3),
        'Model_A_Prob': round(a_prob, 3),
        'Market_H_Odds': h_odds,
        'Bookmaker': bookie,
        'Outcome': 'PENDING'
    }
    
    df = pd.DataFrame([entry])
    header = not os.path.exists(LOG_FILE)
    
    try:
        # Try to save to the main log
        df.to_csv(LOG_FILE, mode='a', index=False, header=header)
    except PermissionError:
        # If file is open in Excel, save to a backup instead of crashing
        print("\nPERMISSION DENIED: 'prediction_log.csv' is open in another program.")
        backup_name = f"data/backup_log_{datetime.now().strftime('%H%M%S')}.csv"
        df.to_csv(backup_name, index=False)
        print(f"Backup created: {backup_name}. Close Excel to merge later.")

import math

def poisson_probability(lmbda, k):
    """Calculates the Poisson probability for k events given rate lambda."""
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)

def calculate_poisson_grid(lmbda_h, lmbda_a):
    """Generates a 10x10 score grid of probabilities for home and away goals."""
    grid = np.zeros((10, 10))
    for x in range(10):
        p_h = poisson_probability(lmbda_h, x)
        for y in range(10):
            p_a = poisson_probability(lmbda_a, y)
            grid[x, y] = p_h * p_a
    return grid

def get_market_probabilities(lmbda_h, lmbda_a):
    """Derives multiple betting market probabilities from predicted expected goals."""
    grid = calculate_poisson_grid(lmbda_h, lmbda_a)
    
    # 1. 1X2 Winner
    home_win_prob = np.sum(np.triu(grid, 1).T)  # x > y
    draw_prob = np.sum(np.diag(grid))            # x == y
    away_win_prob = np.sum(np.tril(grid, -1).T)  # x < y
    
    # 2. Over/Under 2.5 Goals
    under_25_prob = grid[0,0] + grid[1,0] + grid[0,1] + grid[1,1] + grid[2,0] + grid[0,2]
    over_25_prob = 1.0 - under_25_prob
    
    # 3. Both Teams to Score (BTTS)
    btts_yes_prob = (1.0 - math.exp(-max(0.0001, lmbda_h))) * (1.0 - math.exp(-max(0.0001, lmbda_a)))
    btts_no_prob = 1.0 - btts_yes_prob
    
    # 4. Correct Score
    max_idx = np.unravel_index(np.argmax(grid), grid.shape)
    correct_score = f"{max_idx[0]}-{max_idx[1]}"
    
    return {
        'H': home_win_prob,
        'D': draw_prob,
        'A': away_win_prob,
        'Over2.5': over_25_prob,
        'Under2.5': under_25_prob,
        'BTTS_Yes': btts_yes_prob,
        'BTTS_No': btts_no_prob,
        'Score': correct_score
    }