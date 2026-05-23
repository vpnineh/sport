import os
import requests
from google import genai

# Load API Keys from GitHub Secrets
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not ODDS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Error: API Keys are missing in GitHub Secrets! Make sure they are set in Settings -> Secrets AND in the bot.yml env section.")

def get_upcoming_events():
    print("Fetching top upcoming events across all sports...")
    
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Odds API Error: {response.status_code}")
        return []

def analyze_with_gemini(events):
    if not events:
        return "No upcoming events found to analyze."

    prompt_data = "Raw Data of upcoming sports events and their decimal odds:\n\n"
    
    for i, event in enumerate(events[:5]):
        sport = event.get("sport_title", "Unknown Sport")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: continue
        
        outcomes = bookmakers[0].get("markets", [])[0].get("outcomes", [])
        odds_str = " | ".join([f"{o['name']}: {o['price']}" for o in outcomes])
        
        prompt_data += f"Match {i+1}:\nSport: {sport}\nTeams: {home} vs {away}\nOdds (Decimal): {odds_str}\n---\n"

    prompt_instructions = """
    You are an elite sports betting analyst managing a VIP Telegram channel.
    Analyze the provided upcoming matches and their decimal odds. 
    Use the odds to determine the clear favorite and the underdog.
    
    RULES:
    - Write the analysis in English.
    - Provide a structured, engaging post.
    - DO NOT hallucinate statistics; rely purely on the logic of the provided odds.
    - Format EACH match exactly like the template below, using emojis.

    FORMAT TEMPLATE:
    🏆 Sport: [Sport Name]
    ⚔️ Match: [Home Team] vs [Away Team]
    📊 Odds Logic: [1 sentence explaining what the bookmaker odds imply]
    🎯 AI Prediction: [Your logical pick based on the odds]
    🔥 Risk Level: [Low/Medium/High]
    
    DATA TO ANALYZE:
    """

    print("Sending structured prompt to Gemini for analysis...")
    
    # Using the new Google GenAI SDK syntax
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt_instructions + prompt_data,
    )
    
    return response.text

if __name__ == "__main__":
    events = get_upcoming_events()
    ai_prediction_post = analyze_with_gemini(events)
    
    print("\n" + "="*40)
    print("🚀 AI GENERATED TELEGRAM POST:")
    print("="*40 + "\n")
    print(ai_prediction_post)
