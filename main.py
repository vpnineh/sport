import os
import requests

# Load API Keys from GitHub Secrets
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not ODDS_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Error: ODDS and GEMINI API Keys are missing in GitHub Secrets!")

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

    print("Sending direct REST API request to Gemini 2.0 Flash...")
    
    # Updated to gemini-2.0-flash to fix the 404 error
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt_instructions + prompt_data}]
        }]
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error: Unexpected response structure from Gemini API."
    else:
        print(f"Gemini API Error: {response.text}")
        return f"Failed to get AI prediction. Status Code: {response.status_code}"

def send_to_telegram(message_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Outputting to console only.")
        return

    print("Sending analysis to Telegram channel...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message_text
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Successfully posted to Telegram!")
    else:
        print(f"❌ Telegram Error: {response.text}")

if __name__ == "__main__":
    events = get_upcoming_events()
    ai_prediction_post = analyze_with_gemini(events)
    
    print("\n" + "="*40)
    print("🚀 AI GENERATED TELEGRAM POST:")
    print("="*40 + "\n")
    print(ai_prediction_post)
    
    # Trigger Telegram broadcast
    if not ai_prediction_post.startswith("Failed to get AI prediction") and not ai_prediction_post.startswith("Error:"):
        send_to_telegram(ai_prediction_post)
