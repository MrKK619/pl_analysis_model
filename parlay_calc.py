def calculate_parlay():
    print("🔗 Multi-Leg Parlay Calculator")
    legs = int(input("How many legs in the parlay? "))
    
    total_odds = 1.0
    total_prob = 1.0
    
    for i in range(legs):
        print(f"\n--- Leg {i+1} ---")
        prob = float(input("AI Win Probability (e.g., 0.60): "))
        odds = float(input("Bookie Decimal Odds (e.g., 1.91): "))
        total_odds *= odds
        total_prob *= prob
    
    print("\n" + "="*40)
    print(f"📊 PARLAY SUMMARY")
    print(f"Total Combined Odds: {total_odds:.2f}")
    print(f"AI Success Prob:    {total_prob:.1%}")
    print(f"Implied Market Prob: {1/total_odds:.1%}")
    print(f"Expected Return:     ${(total_prob * total_odds):.2f} per $1 bet")
    print("="*40)

if __name__ == "__main__":
    calculate_parlay()