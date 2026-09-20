import pandas as pd
import numpy as np
import os
from utils import fetch_historical_data, normalize_team, LOG_FILE

def evaluate_bet_outcome(selection, market, fthg, ftag, ftr, home_team=None, away_team=None):
    """Evaluates whether a specific betting market selection won, lost, or pushed."""
    market = str(market).upper()
    sel = str(selection).strip()
    
    is_away = False
    if 'Away' in sel or (away_team and normalize_team(away_team) in sel):
        is_away = True
    elif 'Home' in sel or (home_team and normalize_team(home_team) in sel):
        is_away = False
    elif 'Win' in sel and not ('Away' in sel):
        is_away = False

    if market in ('H2H', '1X2'):
        if 'Draw' in sel:
            return 'WON' if ftr == 'D' else 'LOST'
        elif is_away:
            return 'WON' if ftr == 'A' else 'LOST'
        else:
            return 'WON' if ftr == 'H' else 'LOST'
            
    elif market == 'TOTALS':
        total_goals = fthg + ftag
        if 'Over' in sel:
            return 'WON' if total_goals > 2.5 else 'LOST'
        elif 'Under' in sel:
            return 'WON' if total_goals < 2.5 else 'LOST'
            
    elif market == 'BTTS':
        btts_actual = (fthg > 0 and ftag > 0)
        if 'Yes' in sel:
            return 'WON' if btts_actual else 'LOST'
        elif 'No' in sel:
            return 'WON' if not btts_actual else 'LOST'
            
    elif market == 'DNB':
        if ftr == 'D':
            return 'PUSH'
        elif is_away:
            return 'WON' if ftr == 'A' else 'LOST'
        else:
            return 'WON' if ftr == 'H' else 'LOST'
            
    return 'PENDING'

def get_closing_odds_from_match(match_row, market, selection, home_team=None, away_team=None):
    """Extracts the sharp Pinnacle closing odds (with market fallbacks) for CLV auditing."""
    market = str(market).upper()
    sel = str(selection).strip()
    is_away = 'Away' in sel or (away_team and normalize_team(away_team) in sel)
    try:
        if market in ('H2H', '1X2'):
            if 'Draw' in sel:
                val = float(match_row.get('PSCD', match_row.get('MaxCD', match_row.get('B365D', 0.0))))
                return val if val > 1.0 else float(match_row.get('B365D', 0.0))
            elif is_away:
                val = float(match_row.get('PSCA', match_row.get('MaxCA', match_row.get('B365A', 0.0))))
                return val if val > 1.0 else float(match_row.get('B365A', 0.0))
            else:
                val = float(match_row.get('PSCH', match_row.get('MaxCH', match_row.get('B365H', 0.0))))
                return val if val > 1.0 else float(match_row.get('B365H', 0.0))
        elif market == 'TOTALS':
            if 'Over' in sel:
                return float(match_row.get('MaxC>2.5', match_row.get('B365>2.5', match_row.get('BbAv>2.5', 0.0))))
            elif 'Under' in sel:
                return float(match_row.get('MaxC<2.5', match_row.get('B365<2.5', match_row.get('BbAv<2.5', 0.0))))
        elif market == 'DNB':
            psch = float(match_row.get('PSCH', match_row.get('MaxCH', match_row.get('B365H', 0.0))))
            pscd = float(match_row.get('PSCD', match_row.get('MaxCD', match_row.get('B365D', 0.0))))
            psca = float(match_row.get('PSCA', match_row.get('MaxCA', match_row.get('B365A', 0.0))))
            from utils import compute_dnb_odds
            close_dnb = compute_dnb_odds(psch, pscd, psca)
            return close_dnb[1] if is_away else close_dnb[0]
    except Exception:
        pass
    return 0.0

def update():
    print("=" * 70)
    print("  PREMIER LEAGUE AI: AUDITED RESULTS SYNC & CLOSING LINE VALUE (CLV)")
    print("=" * 70)
    
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        print("No prediction log found in data/prediction_log.csv.")
        return
        
    logs = pd.read_csv(LOG_FILE)
    if 'Outcome' not in logs.columns:
        print("Log file format is unrecognized.")
        return
        
    actuals = fetch_historical_data()
    if actuals is None:
        print("Could not fetch match results.")
        return
        
    actuals['HomeNorm'] = actuals['HomeTeam'].map(normalize_team)
    actuals['AwayNorm'] = actuals['AwayTeam'].map(normalize_team)
    
    pending_count = 0
    resolved_count = 0
    
    for idx, row in logs.iterrows():
        if row['Outcome'] == 'PENDING':
            pending_count += 1
            h = normalize_team(row.get('Home'))
            a = normalize_team(row.get('Away'))
            
            matches = actuals[(actuals['HomeNorm'] == h) & (actuals['AwayNorm'] == a)]
            if not matches.empty:
                kickoff_val = row.get('Kickoff')
                time_to_parse = kickoff_val if (pd.notna(kickoff_val) and str(kickoff_val).upper() not in ('MANUAL', 'UPCOMING', '')) else row.get('Timestamp', '2000-01-01')
                pred_time = pd.to_datetime(time_to_parse, errors='coerce')
                if pd.isna(pred_time):
                    continue
                if getattr(pred_time, 'tzinfo', None) is not None:
                    pred_time = pred_time.tz_localize(None)
                    
                pred_date = pred_time.normalize()
                now_date = pd.Timestamp.now().normalize()
                
                # If scheduled kickoff date is in the future, fixture has not been played yet
                if pred_date > now_date:
                    continue
                    
                # Search for a finished match within a tight calendar window around the scheduled date
                window_start = pred_date - pd.Timedelta(days=2)
                window_end = pred_date + pd.Timedelta(days=3)
                candidate_matches = matches[(matches['Date'] >= window_start) & (matches['Date'] <= window_end)]
                
                # If no match exists in this window, do NOT resolve against historical past matches!
                if candidate_matches.empty:
                    continue
                    
                match_row = candidate_matches.iloc[0]
                if pd.isna(match_row.get('FTR')) or pd.isna(match_row.get('FTHG')) or pd.isna(match_row.get('FTAG')):
                    continue
                    
                fthg = float(match_row['FTHG'])
                ftag = float(match_row['FTAG'])
                ftr = str(match_row['FTR']).strip()
                
                market = str(row.get('Market', 'H2H'))
                selection = str(row.get('Selection', ''))
                odds = float(row.get('Market_Odds', 1.0))
                stake = float(row.get('Stake', 0.0))
                
                outcome = evaluate_bet_outcome(selection, market, fthg, ftag, ftr, home_team=h, away_team=a)
                
                if outcome in ('WON', 'LOST', 'PUSH'):
                    resolved_count += 1
                    logs.at[idx, 'Outcome'] = outcome
                    if outcome == 'PUSH':
                        pnl = 0.0
                    else:
                        pnl = round(stake * (odds - 1.0), 2) if outcome == 'WON' else round(-stake, 2)
                    logs.at[idx, 'PnL'] = pnl
                    
                    # Extract Closing Odds and CLV
                    closing_odd = get_closing_odds_from_match(match_row, market, selection, home_team=h, away_team=a)
                    if closing_odd > 1.0:
                        clv_val = round(((odds / closing_odd) - 1.0) * 100.0, 2)
                        logs.at[idx, 'Closing_Odds'] = closing_odd
                        logs.at[idx, 'CLV'] = f"{clv_val:+.1f}%"
                    else:
                        logs.at[idx, 'CLV'] = "N/A"
                        
                    print(f"[RESOLVED] {h} vs {a} ({int(fthg)}-{int(ftag)}): {selection} -> {outcome} (P&L: ${pnl:+.2f})")
                    
    logs.to_csv(LOG_FILE, index=False)
    print("-" * 70)
    print(f"Results Sync Complete: {resolved_count} bets resolved (Pending remaining: {pending_count - resolved_count})")
    
    # Audit Performance & CLV Metrics
    resolved_bets = logs[logs['Outcome'].isin(['WON', 'LOST', 'PUSH'])]
    decisive_bets = logs[logs['Outcome'].isin(['WON', 'LOST'])]
    if len(resolved_bets) > 0:
        total_staked = resolved_bets['Stake'].sum()
        total_pnl = resolved_bets['PnL'].sum()
        roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
        win_rate = np.mean(decisive_bets['Outcome'] == 'WON') if len(decisive_bets) > 0 else 0.0
        
        # Calculate Average CLV
        valid_clvs = []
        for val in resolved_bets['CLV'].dropna():
            if isinstance(val, str) and '%' in val:
                try:
                    valid_clvs.append(float(val.replace('%', '')))
                except Exception:
                    pass
        avg_clv = np.mean(valid_clvs) if len(valid_clvs) > 0 else 0.0
        
        print("\n" + "=" * 50)
        print("         BANKROLL AUDIT SUMMARY")
        print("=" * 50)
        print(f"Total Bets Graded:         {len(resolved_bets)}")
        print(f"Win Rate:                  {win_rate:.1%}")
        print(f"Total Amount Staked:       ${total_staked:,.2f}")
        print(f"Net Realized Profit/Loss:  ${total_pnl:+,.2f}")
        print(f"Return on Investment (ROI):{roi:+.2%}")
        print(f"Average Closing Line Value:{avg_clv:+.2f}%")
        print("=" * 50)
        if avg_clv > 0.0:
            print(">> POSITIVE CLV CONFIRMED: Your wagers consistently beat the market closing price!")
        elif len(valid_clvs) > 0:
            print(">> NEGATIVE CLV: Your odds were worse than the closing line; tighter filtering advised.")

if __name__ == "__main__":
    update()