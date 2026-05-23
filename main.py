import os
import requests
import json
from datetime import datetime, timezone

# Fetch API Key from GitHub Secrets
API_KEY = os.getenv("ODDS_API_KEY")

if not API_KEY:
    raise ValueError("Error: ODDS_API_KEY is missing in GitHub Secrets!")

def get_todays_odds():
    # Target major football (soccer) leagues
    target_leagues = [
        "soccer_epl",                 # Premier League
        "soccer_spain_la_liga",       # La Liga
        "soccer_italy_serie_a",       # Serie A
        "soccer_germany_bundesliga",  # Bundesliga
        "soccer_france_ligue_one",    # Ligue 1
        "soccer_uefa_champs_league"   # Champions League
    ]
    
    # Get today's date in UTC format (required by The Odds API)
    today_date = datetime.now(timezone.utc).date()
    print(f"Fetching matches and odds for {today_date}...\n")

    important_matches = []

    for sport_key in target_leagues:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        
        # Parameters: European bookmakers, Decimal odds (1.50, 2.10), Head-to-Head & Over/Under
        params = {
            "apiKey": API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "commenceTimeFrom": f"{today_date}T00:00:00Z",
            "commenceTimeTo": f"{today_date}T23:59:59Z"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            matches = response.json()
            
            for match in matches:
                home_team = match.get("home_team")
                away_team = match.get("away_team")
                start_time = match.get("commence_time")
                
                # Extract odds from the first available bookmaker
                bookmakers = match.get("bookmakers", [])
                if not bookmakers:
                    continue
                    
                first_bookie = bookmakers[0]
                bookie_name = first_bookie.get("title")
                
                h2h_odds = None
                totals_odds = None
                
                for market in first_bookie.get("markets", []):
                    if market["key"] == "h2h":
                        h2h_odds = market["outcomes"]
                    elif market["key"] == "totals":
                        # Filter to only get Over/Under 2.5 goals
                        totals_odds = [o for o in market["outcomes"] if o.get("point") == 2.5]
                
                important_matches.append({
                    "league": sport_key,
                    "home_team": home_team,
                    "away_team": away_team,
                    "start_time": start_time,
                    "bookmaker": bookie_name,
                    "odds_h2h": h2h_odds,
                    "odds_totals_2.5": totals_odds
                })
        else:
            print(f"Error fetching {sport_key}: HTTP {response.status_code}")

    return important_matches

if __name__ == "__main__":
    matches = get_todays_odds()
    print(f"Found {len(matches)} important matches today:\n")
    print(json.dumps(matches, indent=4))
