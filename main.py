import os
import requests
import google.generativeai as genai

# Load API Keys from GitHub Secrets
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not ODDS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Error: API Keys are missing in GitHub Secrets!")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

def get_upcoming_events():
    print("Fetching top upcoming events across all sports...")
    
    # The 'upcoming' endpoint gets the next 8 active events globally
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h", # Head-to-Head (Win/Loss) odds
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

    # 1. Structure the raw data for the AI
    prompt_data = "Raw Data of upcoming sports events and their decimal odds:\n\n"
    
    # Limit to 5 matches per post to keep the Telegram message clean
    for i, event in enumerate(events[:5]):
        sport = event.get("sport_title", "Unknown Sport")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: continue
        
        # Get the first bookmaker's H2H odds
        outcomes = bookmakers[0].get("markets", [])[0].get("outcomes", [])
        odds_str = " | ".join([f"{o['name']}: {o['price']}" for o in outcomes])
        
        prompt_data += f"Match {i+1}:\nSport: {sport}\nTeams: {home} vs {away}\nOdds (Decimal): {odds_str}\n---\n"

    # 2. The Prompt Engineering section (Visual Coding for Text)
    # This forces the AI to output exactly the format we want.
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
    
    # Using the fast and free gemini-1.5-flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt_instructions + prompt_data)
    
    return response.text

if __name__ == "__main__":
    events = get_upcoming_events()
    ai_prediction_post = analyze_with_gemini(events)
    
    print("\n" + "="*40)
    print("🚀 AI GENERATED TELEGRAM POST:")
    print("="*40 + "\n")
    print(ai_prediction_post)
