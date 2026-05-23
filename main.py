import os
import requests
from datetime import datetime, timedelta, timezone

# Load API Keys from GitHub Secrets
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not ODDS_API_KEY or not GROQ_API_KEY:
    raise ValueError("Error: ODDS and GROQ API Keys are missing in GitHub Secrets!")

def get_next_2_hours_events():
    # 1. Get current time in UTC and define the 2-hour window
    now_utc = datetime.now(timezone.utc)
    end_window_utc = now_utc + timedelta(hours=2)
    
    print(f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scanning for matches starting between now and {end_window_utc.strftime('%H:%M:%S')} UTC...")
    
    url = "https://api.the-odds-api.com/v4/sports/upcoming/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Odds API Error: {response.status_code}")
        return []
        
    all_events = response.json()
    filtered_events = []
    
    # 2. Filter matches that start EXACTLY in the next 2 hours
    for event in all_events:
        commence_time_str = event.get("commence_time")
        if not commence_time_str:
            continue
            
        # Convert ISO timestamp (e.g., 2026-05-23T23:00:00Z) to datetime object
        match_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        
        if now_utc <= match_time <= end_window_utc:
            filtered_events.append(event)
            
    return filtered_events

def analyze_with_groq(events):
    if not events:
        return "No matches starting in the next 2 hours."

    prompt_data = "Raw Data of matches starting in the next 2 hours:\n\n"
    
    # Process all matches found in this 2-hour window (Llama 3.3 can easily handle up to 30 matches)
    for i, event in enumerate(events):
        sport = event.get("sport_title", "Unknown Sport")
        home = event.get("home_team", "Unknown")
        away = event.get("away_team", "Unknown")
        
        bookmakers = event.get("bookmakers", [])
        if not bookmakers: continue
        first_bookie = bookmakers[0]
        h2h_str = "N/A"
        totals_str = "N/A"
        
        for market in first_bookie.get("markets", []):
            if market["key"] == "h2h":
                outcomes = market.get("outcomes", [])
                h2h_str = " | ".join([f"{o['name']}: {o['price']}" for o in outcomes])
            elif market["key"] == "totals":
                outcomes = market.get("outcomes", [])
                totals_str = " | ".join([f"{o['name']} {o.get('point', '')}: {o['price']}" for o in outcomes])
        
        prompt_data += f"Match {i+1}:\nSport: {sport}\nTeams: {home} vs {away}\nH2H Odds: {h2h_str}\nTotals Odds: {totals_str}\n---\n"

    prompt_instructions = """
    You are an elite, minimalist VIP sports betting tipster.
    Analyze the provided matches and odds across various global sports. 
    
    STRICT RULES:
    1. ZERO fluff. No introductory or concluding sentences.
    2. Write completely in English.
    3. Output EXACTLY in the format below, nothing else.
    4. Keep the "Logic" section to ONE short, punchy sentence based strictly on the provided odds.

    FORMAT TEMPLATE FOR EACH MATCH:
    🏆 Sport: [Sport Name]
    ⚔️ Match: [Home Team] vs [Away Team]
    🎯 Winner Pick: [Team Name]
    ⚽ Goals/Points Pick: [Over/Under X.X] (Write N/A if totals odds are unavailable)
    💡 Logic: [E.g., Low odds of 1.45 heavily favor Home, while high totals odds suggest a tight game.]
    🔥 Risk Level: [Low/Medium/High]
    """

    print(f"Sending {len(events)} matches to Groq API (Llama 3.3 70B)...")
    
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
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Groq API Error: {response.text}")
        return f"Failed to get AI prediction. Status Code: {response.status_code}"

def send_to_telegram(message_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
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
    events = get_next_2_hours_events()
    
    if not events:
        print("No matches starting in the next 2 hours. Script shutting down cleanly.")
    else:
        print(f"Found {len(events)} matches starting soon.")
        ai_prediction_post = analyze_with_groq(events)
        
        print("\n" + "="*40)
        print("🚀 AI GENERATED TELEGRAM POST:")
        print("="*40 + "\n")
        print(ai_prediction_post)
        
        if not ai_prediction_post.startswith("Failed to get AI prediction"):
            send_to_telegram(ai_prediction_post)
