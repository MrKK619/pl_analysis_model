import pandas as pd
import numpy as np
import os
from utils import fetch_historical_data, normalize_team, LOG_FILE

def evaluate_bet_outcome(selection, market, fthg, ftag, ftr):
    """Evaluates whether a specific betting market selection won or lost."""
    if market == 'H2H':
        if 'Draw' in selection:
            return 'WON' if ftr == 'D' else 'LOST'
        elif 'Win' in selection:
            if ftr == 'H':
                return 'WON' if not ('Away' in selection) else 'LOST'
            elif ftr == 'A':
                return 'WON' if ('Away' in selection) else 'LOST'
            else:
                return 'LOST'
        return 'LOST'
        
    elif market == 'Totals':
        total_goals = fthg + ftag
        if 'Over' in selection:
            return 'WON' if total_goals > 2.5 else 'LOST'
        elif 'Under' in selection:
            return 'WON' if total_goals < 2.5 else 'LOST'
            
    elif market == 'BTTS':
        btts_actual = (fthg > 0 and ftag > 0)
        if 'Yes' in selection:
            return 'WON' if btts_actual else 'LOST'
        elif 'No' in selection:
            return 'WON' if not btts_actual else 'LOST'
            
    return 'PENDING'

def get_closing_odds_from_match(match_row, market, selection):
    """Extracts the bookmaker closing odds for the specific market selection."""
    try:
        if market == 'H2H':
            if 'Draw' in selection:
                return float(match_row.get('B365D', 0.0))
            elif 'Away' in selection:
                return float(match_row.get('B365A', 0.0))
            else:
                return float(match_row.get('B365H', 0.0))
        elif market == 'Totals':
            if 'Over' in selection:
                return float(match_row.get('B365>2.5', match_row.get('BbAv>2.5', 0.0)))
            elif 'Under' in selection:
                return float(match_row.get('B365<2.5', match_row.get('BbAv<2.5', 0.0)))
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
                pred_time = pd.to_datetime(row.get('Timestamp', '2000-01-01'))
                post_matches = matches[matches['Date'] >= pred_time - pd.Timedelta(days=2)]
                match_row = post_matches.iloc[0] if not post_matches.empty else matches.iloc[-1]
                
                fthg = float(match_row['FTHG'])
                ftag = float(match_row['FTAG'])
                ftr = match_row['FTR']
                
                market = str(row.get('Market', 'H2H'))
                selection = str(row.get('Selection', ''))
                odds = float(row.get('Market_Odds', 1.0))
                stake = float(row.get('Stake', 0.0))
                
                outcome = evaluate_bet_outcome(selection, market, fthg, ftag, ftr)
                
                if outcome in ('WON', 'LOST'):
                    resolved_count += 1
                    logs.at[idx, 'Outcome'] = outcome
                    pnl = round(stake * (odds - 1.0), 2) if outcome == 'WON' else round(-stake, 2)
                    logs.at[idx, 'PnL'] = pnl
                    
                    # Extract Closing Odds and CLV
                    closing_odd = get_closing_odds_from_match(match_row, market, selection)
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
    resolved_bets = logs[logs['Outcome'].isin(['WON', 'LOST'])]
    if len(resolved_bets) > 0:
        total_staked = resolved_bets['Stake'].sum()
        total_pnl = resolved_bets['PnL'].sum()
        roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
        win_rate = np.mean(resolved_bets['Outcome'] == 'WON')
        
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