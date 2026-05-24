import pandas as pd
from utils import fetch_historical_data, LOG_FILE
import os

def update():
    if not os.path.exists(LOG_FILE):
        print("❌ No logs found.")
        return
    logs = pd.read_csv(LOG_FILE)
    actuals = fetch_historical_data()
    
    for idx, row in logs.iterrows():
        if row['Outcome'] == 'PENDING':
            match = actuals[(actuals['HomeTeam'] == row['Home']) & (actuals['AwayTeam'] == row['Away'])]
            if not match.empty:
                logs.at[idx, 'Outcome'] = match.iloc[0]['FTR']
    
    logs.to_csv(LOG_FILE, index=False)
    print("✅ Results Synchronized.")

if __name__ == "__main__":
    update()