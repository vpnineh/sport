import os
import requests

# Load API Keys from GitHub Secrets
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not ODDS_API_KEY or not GROQ_API_KEY:
    raise ValueError("Error: ODDS and GROQ API Keys are missing in GitHub Secrets!")

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

def analyze_with_groq(events):
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
    """

    print("Sending request to Groq API (Llama 3 70B)...")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": prompt_data}
        ],
        "temperature": 0.3
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Groq API Error: {response.text}")
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
    ai_prediction_post = analyze_with_groq(events)
    
    print("\n" + "="*40)
    print("🚀 AI GENERATED TELEGRAM POST:")
    print("="*40 + "\n")
    print(ai_prediction_post)
    
    if not ai_prediction_post.startswith("Failed to get AI prediction"):
        send_to_telegram(ai_prediction_post)
