# =========================================================
# ZBET90 ENGINE v7.2 | AI Judge | Multi-Sport | All Bugs Fixed
# =========================================================
import os,sys,time,json,re,random,logging,html as html_lib
import hashlib,asyncio,aiohttp,requests,numpy as np,pandas as pd
import pickle,warnings,threading,difflib
from io import StringIO
from functools import wraps
from datetime import datetime,timedelta,timezone
from pathlib import Path
from dataclasses import dataclass,field
from typing import Optional,List,Dict,Tuple,Any
from collections import defaultdict,deque
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import scipy.stats as stats_scipy
from scipy.optimize import brentq

try:
    import statsapi as mlb_statsapi
    HAS_STATSAPI=True
except ImportError:
    HAS_STATSAPI=False

# =========================================================
# 1. CONFIG
# =========================================================
@dataclass
class Config:
    CACHE_DIR:Path=Path("api_cache")
    LOG_DIR:Path=Path("log")
    HISTORICAL_DIR:Path=Path("api_cache/historical")
    ML_DIR:Path=Path("api_cache/ml_models")
    HISTORY_FILE:Path=Path("api_cache/sent_history.json")
    TEAM_ID_CACHE_FILE:Path=Path("api_cache/team_id_cache.json")
    ODDS_CACHE_FILE:Path=Path("api_cache/odds_cache.json")
    API_USAGE_FILE:Path=Path("api_cache/api_usage_tracker.json")
    PERFORMANCE_FILE:Path=Path("api_cache/performance_tracker.json")
    LOG_FILE:Path=Path("api_cache/execution_logs.log")
    MATCH_WINDOW_HOURS:float=6.0
    TELEGRAM_SLEEP_BETWEEN:float=3.0
    ODDS_API_MARKETS:List[str]=field(default_factory=lambda:["h2h","totals"])
    ODDS_API_REGIONS:str="eu,us,uk,au"
    TTL_ODDS_CACHE_MINUTES:float=6.0
    TTL_SENT_HISTORY:float=48.0
    TTL_TEAM_FORM:float=6.0
    TTL_GITHUB_DATA:float=12.0
    H2H_MIN_ODDS:float=1.40
    H2H_MIN_EV:float=0.020
    TOTALS_MIN_ODDS:float=1.50
    TOTALS_MIN_EV:float=0.022
    MAX_REALISTIC_EV:float=0.25
    MATH_MIN_EV_TO_ANALYZE:float=0.010
    MARKET_EXPECTED_OUTCOMES:Dict[str,Any]=field(default_factory=lambda:{"h2h":{"min":2,"max":3},"totals":{"min":2,"max":2}})
    MAX_VALID_IMPLIED_SUM:float=1.15
    MIN_VALID_IMPLIED_SUM:float=0.75
    KELLY_FRACTION:float=0.25
    MAX_KELLY_PCT:float=5.0
    MIN_MATH_SCORE_TO_CALL_AI:int=45
    MIN_CONFIDENCE_TO_SEND:int=65
    HIGH_CONFIDENCE:int=78
    MEDIUM_CONFIDENCE:int=65
    AI_IS_FINAL_JUDGE:bool=True
    AI_WEIGHT:float=0.70
    MATH_WEIGHT:float=0.30
    MAX_AI_BOOST:int=12
    MAX_AI_PENALTY:int=8
    AI_MODEL_ANALYST:str="gemini-2.0-flash"
    AI_MAX_TOKENS:int=3000
    AI_TEMPERATURE:float=0.05
    TELEGRAM_ID:str="@zBET90"
    SHARP_BOOKMAKERS:List[str]=field(default_factory=lambda:["pinnacle","betfair_ex_eu","matchbook","betfair_ex_uk","sport888","betsson"])
    # BUG-15: Higher threshold for sport=other
    MIN_MATH_SCORE_OTHER_SPORT:int=60
    GITHUB_SOURCES:Dict[str,Any]=field(default_factory=lambda:{
        "atp":"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        "wta":"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        "atp_rankings":"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
        "wta_rankings":"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
        "football_eu":"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
        "club_elo":"http://api.clubelo.com/{team}",
    })
    FOOTBALL_DATA_UK_LEAGUES:Dict[str,str]=field(default_factory=lambda:{
        "E0":"Premier League","E1":"Championship","D1":"Bundesliga","SP1":"La Liga",
        "I1":"Serie A","F1":"Ligue 1","N1":"Eredivisie","P1":"Liga Portugal","T1":"Super Lig","B1":"Jupiler League",
    })
    # BUG-17: Added 2526 season
    FOOTBALL_DATA_UK_SEASONS:List[str]=field(default_factory=lambda:["2324","2425","2526"])

CFG=Config()

# =========================================================
# 2. LOGGING
# =========================================================
DEBUG_MODE=os.getenv("DEBUG_MODE","false").lower()=="true"
for d in [CFG.CACHE_DIR,CFG.LOG_DIR,CFG.HISTORICAL_DIR,CFG.ML_DIR]:
    d.mkdir(parents=True,exist_ok=True)
logger=logging.getLogger("ZBET90_ENGINE")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_fmt=logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s","%Y-%m-%d %H:%M:%S")
_ch=logging.StreamHandler(sys.stdout);_ch.setFormatter(_fmt);logger.addHandler(_ch)
_fh=logging.FileHandler(CFG.LOG_FILE,mode="a",encoding="utf-8");_fh.setFormatter(_fmt);logger.addHandler(_fh)

# =========================================================
# 3. HYBRID AI MANAGER
# =========================================================
import google.genai as genai
from google.genai import types
try:
    from groq import Groq
    HAS_GROQ=True
except ImportError:
    HAS_GROQ=False

GROQ_STRICT_SYSTEM="""You are a STRICT sports betting risk analyst. Your PRIMARY job is to PROTECT bankroll.

RULES:
- SKIP at least 60% of bets
- Only output BET when: confidence >= 70 AND EV > 3% AND multiple data sources confirm edge
- Low Kelly (<1.5%) = always SKIP
- MLB/NHL with only standings data (no recent game logs) = max confidence 58 → SKIP
- When data is limited or conflicting = SKIP
- Never give BET just because EV is positive

You must output ONLY valid JSON: {"decision":"BET" or "SKIP","confidence":<int 0-100>,"sport_emoji":"<emoji>","risk_level":"Low" or "Medium" or "High","key_factors":["fact1","fact2"],"logic":"2-3 sentences","red_flags":["flag1"]}"""

class HybridAIManager:
    _instance:Optional["HybridAIManager"]=None
    _lock=threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance=super().__new__(cls)
                cls._instance._initialized=False
            return cls._instance

    def __init__(self):
        if self._initialized:return
        gem_keys=[k.strip() for k in [os.getenv("GEMINI",""),os.getenv("GEMINI1",""),os.getenv("GEMINI2",""),os.getenv("GEMINI3","")] if k.strip()]
        self.gemini_clients=[genai.Client(api_key=k) for k in gem_keys] if gem_keys else []
        self._safety=[types.SafetySetting(category=c,threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in [
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT]]
        groq_keys=[k.strip() for k in [os.getenv("GROQ_API_KEY",""),os.getenv("GROQ1",""),os.getenv("GROQ2","")] if k.strip()]
        self.groq_clients=[Groq(api_key=k) for k in groq_keys] if (groq_keys and HAS_GROQ) else []
        self._last_provider="none"
        # BUG-12: Store timestamp per failed key instead of just a set
        self._gemini_failed:Dict[int,float]={}
        self._last_call_time:float=0.0
        self._rate_lock=threading.Lock()
        self._initialized=True
        logger.info("✅ [AI MANAGER] Loaded %d Gemini keys, %d Groq keys",len(self.gemini_clients),len(self.groq_clients))

    def _is_key_failed(self,idx:int)->bool:
        """BUG-12: Keys auto-recover after 15 minutes."""
        ft=self._gemini_failed.get(idx)
        if ft is None:return False
        if time.time()-ft>900:
            del self._gemini_failed[idx];return False
        return True

    def generate(self,prompt:str,system_instruction:str=None,is_groq_strict:bool=False)->Optional[dict]:
        with self._rate_lock:
            elapsed=time.time()-self._last_call_time
            if elapsed<3.0:time.sleep(3.0-elapsed)
            self._last_call_time=time.time()

        if self.gemini_clients:
            cfg_kw=dict(temperature=CFG.AI_TEMPERATURE,max_output_tokens=CFG.AI_MAX_TOKENS,
                        response_mime_type="application/json",safety_settings=self._safety)
            if system_instruction:cfg_kw["system_instruction"]=system_instruction
            gen_cfg=types.GenerateContentConfig(**cfg_kw)
            available=[i for i in range(len(self.gemini_clients)) if not self._is_key_failed(i)]
            if not available:
                self._gemini_failed.clear()
                available=list(range(len(self.gemini_clients)))
            for _ in range(2):
                if not available:break
                idx=random.choice(available)
                client=self.gemini_clients[idx]
                try:
                    resp=client.models.generate_content(model=CFG.AI_MODEL_ANALYST,contents=prompt,config=gen_cfg)
                    if getattr(resp,"prompt_feedback",None) and resp.prompt_feedback.block_reason:
                        logger.warning("[GEMINI] Blocked → Groq");break
                    raw=resp.text
                    if raw:
                        self._last_provider="gemini"
                        try:return json.loads(raw)
                        except:return robust_json_extractor(raw)
                except Exception as e:
                    es=str(e)
                    # BUG-12: Sanitize key from error message
                    sanitized=re.sub(r'key["\s:=]+[A-Za-z0-9_\-]{10,}',"key=***",es,flags=re.IGNORECASE)
                    logger.warning("[GEMINI ERROR] Key %d: %s",idx,sanitized[:100])
                    if "429" in es or "quota" in es.lower():
                        self._gemini_failed[idx]=time.time();available.remove(idx);continue
                    break

        if self.groq_clients:
            logger.info("🔄 [AI MANAGER] Switching to Groq Fallback...")
            groq_sys=GROQ_STRICT_SYSTEM if is_groq_strict else (system_instruction or GROQ_STRICT_SYSTEM)
            messages=[{"role":"system","content":groq_sys},{"role":"user","content":prompt}]
            for attempt in range(2):
                try:
                    client=random.choice(self.groq_clients)
                    cc=client.chat.completions.create(
                        messages=messages,
                        # Selected qwen/qwen3-32b for better reasoning vs gpt-oss-120b
                        model="qwen/qwen3-32b",
                        temperature=CFG.AI_TEMPERATURE,max_completion_tokens=CFG.AI_MAX_TOKENS,
                        top_p=1,stream=False,stop=None,
                        response_format={"type":"json_object"})
                    raw=cc.choices[0].message.content
                    if raw:
                        self._last_provider="groq"
                        try:return json.loads(raw)
                        except:return robust_json_extractor(raw)
                except Exception as e:
                    logger.warning("[GROQ ERROR] Attempt %d: %s",attempt+1,str(e)[:100])
                    time.sleep(3)
        logger.error("❌ [AI MANAGER] Both failed.")
        return None

ai_manager=HybridAIManager()

# =========================================================
# 4. API KEY MANAGER
# =========================================================
class OddsAPIKeyManager:
    def __init__(self):
        self.keys:List[Dict]=[]
        self._lock=threading.Lock()
        for env,label in [("ODDS_API_KEY","primary"),("ODDS_API_KEY2","backup_1"),("ODDS_API_KEY3","backup_2")]:
            k=os.getenv(env,"").strip()
            if k:
                self.keys.append({"key":k,"label":label,"env":env,"failed":False,"fail_reason":None,"fail_time":None})
                logger.info("🔑 [API KEY] %s: Loaded ✓",label)
        if not self.keys:logger.critical("FATAL: No ODDS_API_KEY!");sys.exit(1)
        self.usage=self._load_usage()

    def _load_usage(self)->dict:
        try:
            if CFG.API_USAGE_FILE.exists():
                d=json.loads(CFG.API_USAGE_FILE.read_text())
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if d.get("date")==today:return d
        except:pass
        return {"date":datetime.now(timezone.utc).strftime("%Y-%m-%d"),"keys":{}}

    def _save_usage(self):
        try:CFG.API_USAGE_FILE.write_text(json.dumps(self.usage,indent=2))
        except:pass

    def record_usage(self,label:str,used:int=0,remaining:int=-1):
        # BUG-3: All usage access inside lock
        with self._lock:
            self.usage["keys"].setdefault(label,{"calls":0,"remaining":-1,"last_used":None})
            self.usage["keys"][label]["calls"]+=1
            self.usage["keys"][label]["last_used"]=datetime.now(timezone.utc).isoformat()
            if remaining>=0:self.usage["keys"][label]["remaining"]=remaining
            self._save_usage()

    def mark_failed(self,idx:int,reason:str):
        with self._lock:
            if 0<=idx<len(self.keys):
                self.keys[idx].update({"failed":True,"fail_reason":reason,"fail_time":datetime.now(timezone.utc).isoformat()})
                logger.warning("🔑❌ %s FAILED: %s",self.keys[idx]["label"],reason)

    def get_active_keys(self)->List[Dict]:
        now=datetime.now(timezone.utc);active=[]
        for k in self.keys:
            if not k["failed"]:active.append(k)
            elif k.get("fail_time"):
                try:
                    ft=datetime.fromisoformat(k["fail_time"])
                    if ft.tzinfo is None:ft=ft.replace(tzinfo=timezone.utc)
                    if now-ft>timedelta(minutes=30):k["failed"]=False;active.append(k)
                except:pass
        if not active:
            for k in self.keys:k["failed"]=False
            active=list(self.keys)
        return active

    def get_usage_summary(self)->str:
        # BUG-3: Read usage inside lock
        with self._lock:
            usage_snapshot=json.loads(json.dumps(self.usage))
        parts=[]
        for k in self.keys:
            u=usage_snapshot.get("keys",{}).get(k["label"],{})
            parts.append(f"{'❌' if k['failed'] else '✅'} {k['label']}: {u.get('calls',0)} calls (rem:{u.get('remaining','?')})")
        return " | ".join(parts)

GEMINI_API_KEY=os.getenv("GEMINI","").strip()
TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID","").strip()
if not all([GEMINI_API_KEY,TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID]):
    logger.critical("FATAL: Missing env vars");sys.exit(1)
odds_key_manager=OddsAPIKeyManager()

# =========================================================
# 5. NATIONALITY FLAGS
# =========================================================
NATIONALITY_FLAGS:dict={
    "alcaraz":"ES","nadal":"ES","djokovic":"RS","sinner":"IT","zverev":"DE",
    "tiafoe":"US","fritz":"US","paul":"US","medvedev":"RU","rublev":"RU",
    "tsitsipas":"GR","ruud":"NO","rune":"DK","hurkacz":"PL","swiatek":"PL",
    "auger-aliassime":"CA","shapovalov":"CA","kyrgios":"AU","de minaur":"AU",
    "sabalenka":"BY","gauff":"US","rybakina":"KZ","jabeur":"TN",
    "real madrid":"ES","barcelona":"ES","bayern":"DE","dortmund":"DE",
    "manchester city":"GB","liverpool":"GB","arsenal":"GB","chelsea":"GB",
    "juventus":"IT","milan":"IT","inter":"IT","napoli":"IT",
    "psg":"FR","ajax":"NL","porto":"PT","benfica":"PT",
    "lakers":"US","celtics":"US","warriors":"US","bulls":"US",
    "flamengo":"BR","palmeiras":"BR","river plate":"AR","boca juniors":"AR",
    # BUG-14: Additional common teams/players
    "tottenham":"GB","manchester united":"GB","atletico madrid":"ES","sevilla":"ES",
    "valencia":"ES","villarreal":"ES","real sociedad":"ES","athletic bilbao":"ES",
    "borussia":"DE","leverkusen":"DE","frankfurt":"DE","hoffenheim":"DE",
    "roma":"IT","lazio":"IT","atalanta":"IT","fiorentina":"IT","torino":"IT",
    "marseille":"FR","lyon":"FR","monaco":"FR","nice":"FR","lille":"FR",
    "ajax":"NL","psv":"NL","feyenoord":"NL",
    "sporting":"PT","braga":"PT",
    "galatasaray":"TR","fenerbahce":"TR","besiktas":"TR",
    "anderlecht":"BE","club brugge":"BE",
    "celtic":"GB","rangers":"GB",
    "shakhtar":"UA","dynamo kyiv":"UA",
    "slavia prague":"CZ","sparta prague":"CZ",
    "murray":"GB","djokovic":"RS","federer":"CH","wawrinka":"CH",
    "bencic":"CH","vondrousova":"CZ","krejcikova":"CZ","muchova":"CZ",
    "pegula":"US","keys":"US","navarro":"ES","badosa":"ES",
    "halep":"RO","simona":"RO",
    "miami heat":"US","brooklyn nets":"US","golden state":"US","boston":"US",
    "milwaukee":"US","phoenix":"US","denver":"US","dallas":"US",
    "new york":"US","toronto":"CA","chicago":"US","cleveland":"US",
    "houston":"US","memphis":"US","oklahoma":"US","portland":"US",
    "utah":"US","sacramento":"US","san antonio":"US","indiana":"US",
    "new orleans":"US","charlotte":"US","washington":"US","detroit":"US",
    "atlanta":"US","orlando":"US","minnesota":"US",
    "yankees":"US","red sox":"US","dodgers":"US","cubs":"US","mets":"US",
    "braves":"US","cardinals":"US","giants":"US","astros":"US","rangers":"US",
    "bruins":"US","blackhawks":"US","maple leafs":"CA","canadiens":"CA",
    "penguins":"US","capitals":"US","lightning":"US","rangers":"US",
    "oilers":"CA","flames":"CA","canucks":"CA","senators":"CA","jets":"CA",
}

def _code_to_flag(code:str)->str:
    code=code.upper()
    return chr(ord(code[0])+0x1F1E6-ord('A'))+chr(ord(code[1])+0x1F1E6-ord('A'))

def get_flag_from_name(name:str)->str:
    nl=name.lower()
    # Exact match first
    if nl in NATIONALITY_FLAGS:return _code_to_flag(NATIONALITY_FLAGS[nl])
    # Substring match
    for kw,code in NATIONALITY_FLAGS.items():
        if kw in nl:return _code_to_flag(code)
    # BUG-14: Try word-by-word match for compound names
    words=nl.split()
    for w in words:
        if len(w)>3 and w in NATIONALITY_FLAGS:return _code_to_flag(NATIONALITY_FLAGS[w])
    return "🏆"  # BUG-14: Better fallback than 🏳️

# =========================================================
# 6. CACHE MANAGER
# =========================================================
_cache_lock=threading.Lock()

class CacheManager:
    @staticmethod
    def load(fp:Path)->dict:
        try:
            if fp.exists():return json.loads(fp.read_text(encoding="utf-8"))
        except:pass
        return {}

    @staticmethod
    def save(fp:Path,data:dict):
        # BUG-6 & BUG-16: Atomic write with proper locking
        try:
            fp.parent.mkdir(parents=True,exist_ok=True)
            tmp_name=f"{fp.name}.tmp.{os.getpid()}_{threading.get_ident()}_{int(time.time()*1000)}"
            tmp=fp.with_name(tmp_name)
            content=json.dumps(data,ensure_ascii=False,indent=2,default=str)
            with _cache_lock:
                tmp.write_text(content,encoding="utf-8")
                try:
                    tmp.replace(fp)
                except PermissionError:
                    # Windows fallback
                    if fp.exists():fp.unlink()
                    tmp.rename(fp)
        except Exception as e:
            logger.debug("[CACHE SAVE] Error: %s",e)
            try:tmp.unlink(missing_ok=True)
            except:pass

    @staticmethod
    def is_valid(cache:dict,key:str,ttl_hours:float)->bool:
        e=cache.get(key)
        if not isinstance(e,dict) or "timestamp" not in e:return False
        try:
            t=datetime.fromisoformat(e["timestamp"])
            if t.tzinfo is None:t=t.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc)-t<timedelta(hours=ttl_hours)
        except:return False

    @staticmethod
    def is_valid_minutes(cache:dict,key:str,ttl_min:float)->bool:
        return CacheManager.is_valid(cache,key,ttl_min/60)

    @staticmethod
    def set(cache:dict,key:str,value:Any)->dict:
        cache[key]={"timestamp":datetime.now(timezone.utc).isoformat(),"data":value}
        return cache

    @staticmethod
    def get(cache:dict,key:str)->Any:
        return cache.get(key,{}).get("data")

# =========================================================
# 7. PERFORMANCE TRACKER
# =========================================================
class PerformanceTracker:
    def __init__(self):
        # BUG-11: Add lock for thread safety
        self._lock=threading.Lock()
        self.data=CacheManager.load(CFG.PERFORMANCE_FILE)
        self.data.setdefault("signals",[])
        self.data.setdefault("summary",{})

    def record_signal(self,home,away,pick,market,odds,ev,confidence,prob,sport="other",api_sport_key=""):
        sig={"id":hashlib.md5(f"{home}|{away}|{market}|{datetime.now(timezone.utc).date()}".encode()).hexdigest()[:8],
             "timestamp":datetime.now(timezone.utc).isoformat(),"sport":sport,"api_sport_key":api_sport_key,
             "home":home,"away":away,"pick":pick,"market":market,"odds":odds,"ev":ev,
             "confidence":confidence,"implied_prob":prob,"outcome":None,"profit_loss":None}
        # BUG-11: Thread-safe record
        with self._lock:
            self.data["signals"].append(sig)
            if len(self.data["signals"])>500:self.data["signals"]=self.data["signals"][-500:]
            self._update_summary()
        CacheManager.save(CFG.PERFORMANCE_FILE,self.data)

    def _update_summary(self):
        res=[s for s in self.data["signals"] if s.get("outcome")]
        if not res:return
        wins=[s for s in res if s["outcome"]=="win"]
        pl=sum(s.get("profit_loss",0) or 0 for s in res)
        self.data["summary"]={"total_signals":len(self.data["signals"]),"resolved":len(res),
                               "win_rate":round(len(wins)/len(res),3),"total_profit_loss_units":round(pl,2),
                               "roi_pct":round(pl/len(res)*100,2),"last_updated":datetime.now(timezone.utc).isoformat()}

performance_tracker=PerformanceTracker()

# =========================================================
# 8. SENT HISTORY
# =========================================================
class SentHistory:
    def __init__(self):
        self.history=CacheManager.load(CFG.HISTORY_FILE)
        self._cleanup()

    def _cleanup(self):
        now=datetime.now(timezone.utc);to_del=[]
        for k,v in self.history.items():
            try:
                t=datetime.fromisoformat(v.get("sent_at","2000-01-01T00:00:00+00:00"))
                if t.tzinfo is None:t=t.replace(tzinfo=timezone.utc)
                if now-t>timedelta(hours=CFG.TTL_SENT_HISTORY):to_del.append(k)
            except:to_del.append(k)
        for k in to_del:del self.history[k]

    @staticmethod
    def _key(home,away,market)->str:
        return hashlib.md5(f"{home.lower()}|{away.lower()}|{market.lower()}".encode()).hexdigest()

    def was_sent(self,home,away,market)->bool:
        return self._key(home,away,market) in self.history

    def mark_sent(self,home,away,pick,market):
        self.history[self._key(home,away,market)]={"match":f"{home} vs {away}","pick":pick,"market":market,"sent_at":datetime.now(timezone.utc).isoformat()}
        CacheManager.save(CFG.HISTORY_FILE,self.history)

# =========================================================
# 9. FREE DATA ENGINE
# =========================================================
class FreeDataEngine:
    def __init__(self):
        self.atp_matches:Optional[pd.DataFrame]=None
        self.wta_matches:Optional[pd.DataFrame]=None
        self.atp_rankings:Optional[pd.DataFrame]=None
        self.wta_rankings:Optional[pd.DataFrame]=None
        self.football_data:Dict[str,pd.DataFrame]={}
        self.nba_data:Optional[pd.DataFrame]=None
        self.nhl_data:Optional[pd.DataFrame]=None
        self.mlb_data:Optional[pd.DataFrame]=None
        self.cricket_data:Optional[pd.DataFrame]=None
        self.elo_cache:dict=CacheManager.load(CFG.CACHE_DIR/"elo_cache.json")
        self.us_cache:dict=CacheManager.load(CFG.CACHE_DIR/"us_sports_cache.json")
        self.years=[2022,2023,2024,2025]

    def _download_csv(self,url:str,path:Path,timeout:int=30)->bool:
        if path.exists() and (time.time()-path.stat().st_mtime)/3600<CFG.TTL_GITHUB_DATA:return True
        logger.info("[FREE DATA] Downloading: %s",path.name)
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=timeout+attempt*10,headers={"User-Agent":"Mozilla/5.0 (compatible; ZBET90/7.2)"})
                if r.status_code==200 and len(r.content)>100:
                    path.write_bytes(r.content);return True
                logger.debug("[FREE DATA] HTTP %d for %s",r.status_code,url);break
            except requests.exceptions.Timeout:
                logger.warning("[FREE DATA] Timeout attempt %d: %s",attempt+1,path.name);time.sleep(2*(attempt+1))
            except Exception as e:
                logger.warning("[FREE DATA] %s: %s",path.name,str(e)[:80]);break
        return False

    def load_tennis_data(self):
        COLS=["tourney_date","tourney_name","surface","draw_size","tourney_level","round",
              "winner_id","winner_name","winner_rank","winner_rank_points","winner_age","winner_ht","winner_ioc",
              "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
              "loser_id","loser_name","loser_rank","loser_rank_points","loser_age","loser_ht","loser_ioc",
              "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
              "score","best_of","minutes"]
        atp_dfs,wta_dfs=[],[]
        for year in self.years:
            for tour,lst,key in [("atp",atp_dfs,"atp"),("wta",wta_dfs,"wta")]:
                url=CFG.GITHUB_SOURCES[key].format(year=year)
                path=CFG.HISTORICAL_DIR/f"{key}_{year}.csv"
                if self._download_csv(url,path):
                    try:
                        df=pd.read_csv(path,low_memory=False,encoding="utf-8",encoding_errors="replace")
                        sub=df[[c for c in COLS if c in df.columns]].copy()
                        if "tourney_date" in sub.columns:sub["tourney_date"]=pd.to_numeric(sub["tourney_date"],errors="coerce")
                        lst.append(sub)
                    except Exception as e:logger.error("[TENNIS] %s %s: %s",tour.upper(),year,e)
        for lst,attr,name in [(atp_dfs,"atp_matches","ATP"),(wta_dfs,"wta_matches","WTA")]:
            if lst:
                df=pd.concat(lst,ignore_index=True)
                if "tourney_date" in df.columns:df=df.sort_values("tourney_date").reset_index(drop=True)
                setattr(self,attr,df);logger.info("✅ [TENNIS] %s: %d matches",name,len(df))
        for tour,key,attr in [("atp","atp_rankings","atp_rankings"),("wta","wta_rankings","wta_rankings")]:
            path=CFG.HISTORICAL_DIR/f"{key}.csv"
            if self._download_csv(CFG.GITHUB_SOURCES[key],path):
                try:setattr(self,attr,pd.read_csv(path,low_memory=False));logger.info("✅ [RANKINGS] %s loaded",tour.upper())
                except Exception as e:logger.error("[RANKINGS] %s: %s",tour,e)

    def get_player_ranking(self,name:str,is_wta:bool=False)->Optional[int]:
        df=self.wta_rankings if is_wta else self.atp_rankings
        if df is None or df.empty:return None
        nc=next((c for c in ["player","name","player_name"] if c in df.columns),None)
        if not nc:return None
        name_lower=name.lower().strip()
        col_lower=df[nc].astype(str).str.lower()
        exact=df[col_lower==name_lower]
        if not exact.empty:m=exact
        else:
            parts=name.split()
            if len(parts)>1:
                last_two=" ".join(parts[-2:]).lower()
                m=df[col_lower.str.contains(re.escape(last_two),na=False)]
                if m.empty and "-" in name:
                    after_hyphen=name.split("-")[-1].lower()
                    m=df[col_lower.str.contains(re.escape(after_hyphen),na=False)]
                if m.empty:
                    m=df[col_lower.str.contains(re.escape(parts[-1].lower()),na=False)]
            else:
                m=df[col_lower.str.contains(re.escape(name_lower),na=False)]
        if not m.empty:
            rc=next((c for c in ["rank","ranking","player_rank"] if c in m.columns),None)
            if rc:
                v=m.iloc[0][rc]
                return int(v) if pd.notna(v) else None
        return None

    def _player_rolling(self,df:pd.DataFrame,clean:str,n:int=20)->dict:
        wins=df[df["winner_name"].str.lower().str.contains(re.escape(clean),na=False)]
        losses=df[df["loser_name"].str.lower().str.contains(re.escape(clean),na=False)]
        total=len(wins)+len(losses)
        if total==0:return {}
        all_r=([(r.get("tourney_date",0),"W",r) for _,r in wins.iterrows()]+
               [(r.get("tourney_date",0),"L",r) for _,r in losses.iterrows()])
        all_r.sort(key=lambda x:x[0] if pd.notna(x[0]) else 0,reverse=True)
        recent=all_r[:n];rw=sum(1 for x in recent if x[1]=="W")
        result={"total_matches":total,"win_rate_overall":round(len(wins)/total,3),
                "recent_form":"".join(x[1] for x in recent[:10]),
                "recent_win_rate":round(rw/len(recent),3) if recent else 0}
        rw_df=wins.tail(n//2)
        for stat,col in [("aces_per_match","w_ace"),("df_per_match","w_df"),("svpt_per_match","w_svpt")]:
            if col in rw_df.columns:
                v=rw_df[col].dropna()
                if len(v):result[stat]=round(float(v.mean()),2)
        if all(c in rw_df.columns for c in ["w_1stIn","w_svpt"]):
            sv=rw_df["w_svpt"].dropna().mean()
            if sv:result["first_serve_in_pct"]=round(float(rw_df["w_1stIn"].dropna().mean()/sv),3)
        if all(c in rw_df.columns for c in ["w_1stWon","w_1stIn"]):
            i1=rw_df["w_1stIn"].dropna().mean()
            if i1:result["first_serve_win_pct"]=round(float(rw_df["w_1stWon"].dropna().mean()/i1),3)
        if all(c in rw_df.columns for c in ["w_bpSaved","w_bpFaced"]):
            bpf=rw_df["w_bpFaced"].dropna().mean()
            if bpf:result["bp_saved_pct"]=round(float(rw_df["w_bpSaved"].dropna().mean()/bpf),3)
        ss={}
        for surf in ["Hard","Clay","Grass"]:
            if "surface" in wins.columns:
                sw=wins[wins["surface"].str.lower()==surf.lower()]
                sl=losses[losses["surface"].str.lower()==surf.lower()] if "surface" in losses.columns else pd.DataFrame()
                st=len(sw)+len(sl)
                if st>=5:ss[surf]={"win_rate":round(len(sw)/st,3),"matches":st}
        if ss:result["surface_stats"]=ss
        return result

    def get_tennis_stats(self,pa:str,pb:str,is_wta:bool=False)->dict:
        df=self.wta_matches if is_wta else self.atp_matches
        if df is None or df.empty:return {}
        def cl(n):
            n=n.strip();parts=n.split()
            if len(parts)>=2:
                candidate=" ".join(parts[-2:]).lower()
                if df is not None:
                    wn=df["winner_name"].astype(str).str.lower()
                    ln=df["loser_name"].astype(str).str.lower()
                    if any(wn.str.contains(re.escape(candidate),na=False)) or any(ln.str.contains(re.escape(candidate),na=False)):
                        return candidate
            return parts[-1].lower()
        ca,cb=cl(pa),cl(pb)
        stats={"player_a":{"name":pa},"player_b":{"name":pb},"h2h":{}}
        for p_c,key,p_f,is_w in [(ca,"player_a",pa,is_wta),(cb,"player_b",pb,is_wta)]:
            s=self._player_rolling(df,p_c)
            if s:
                stats[key].update(s)
                r=self.get_player_ranking(p_f,is_w)
                if r:stats[key]["current_ranking"]=r
                # ── NEW: add data quality flag ──
                stats[key]["data_quality"]=(
                    "good" if s.get("total_matches",0)>=20
                    else "limited" if s.get("total_matches",0)>=5
                    else "poor"
                )
        h2h_a=df[df["winner_name"].str.lower().str.contains(ca,na=False)&df["loser_name"].str.lower().str.contains(cb,na=False)]
        h2h_b=df[df["winner_name"].str.lower().str.contains(cb,na=False)&df["loser_name"].str.lower().str.contains(ca,na=False)]
        t=len(h2h_a)+len(h2h_b)
        if t:
            stats["h2h"]={"total":t,f"{pa}_wins":len(h2h_a),f"{pb}_wins":len(h2h_b),
                          "dominance":(f"{pa}_dominant" if len(h2h_a)>len(h2h_b)*2 else f"{pb}_dominant" if len(h2h_b)>len(h2h_a)*2 else "balanced")}
            if "surface" in h2h_a.columns:
                bs={}
                for surf in ["Hard","Clay","Grass"]:
                    sa=h2h_a[h2h_a["surface"].str.lower()==surf.lower()]
                    sb=h2h_b[h2h_b["surface"].str.lower()==surf.lower()]
                    if len(sa)+len(sb):bs[surf]={f"{pa}_wins":len(sa),f"{pb}_wins":len(sb)}
                if bs:stats["h2h"]["by_surface"]=bs
            logger.info("✅ [H2H TENNIS] %s vs %s: %d matches",pa,pb,t)
        # ── NEW: overall data quality summary ──
        qa=stats["player_a"].get("data_quality","poor")
        qb=stats["player_b"].get("data_quality","poor")
        stats["data_quality_summary"]={
            "player_a":qa,"player_b":qb,
            "h2h_matches":t,
            "overall":"good" if qa=="good" and qb=="good" and t>=3
                      else "limited" if qa!="poor" or qb!="poor"
                      else "poor"
        }
        return stats

    def load_football_data(self):
        COLS=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HTHG","HTAG","HTR","HS","AS","HST","AST",
              "HC","AC","HF","AF","HY","AY","HR","AR","B365H","B365D","B365A","BbMxH","BbMxD","BbMxA",
              "BbAvH","BbAvD","BbAvA","BbMx>2.5","BbAv>2.5","BbMx<2.5","BbAv<2.5"]
        all_dfs=[]
        for season in CFG.FOOTBALL_DATA_UK_SEASONS:
            for code,name in CFG.FOOTBALL_DATA_UK_LEAGUES.items():
                url=CFG.GITHUB_SOURCES["football_eu"].format(season=season,league=code)
                path=CFG.HISTORICAL_DIR/f"football_{code}_{season}.csv"
                if self._download_csv(url,path):
                    try:
                        df=pd.read_csv(path,low_memory=False,encoding="latin-1")
                        avail=[c for c in COLS if c in df.columns]
                        if len(avail)<5:continue
                        sub=df[avail].copy();sub["League"]=name;sub["Season"]=season
                        if "Date" in sub.columns:sub["Date"]=pd.to_datetime(sub["Date"],format="mixed",dayfirst=True,errors="coerce")
                        if "HomeTeam" in sub.columns:sub=sub.dropna(subset=["HomeTeam","AwayTeam"])
                        all_dfs.append(sub)
                    except Exception as e:logger.debug("[FOOTBALL] %s: %s",path.name,e)
        if all_dfs:
            comb=pd.concat(all_dfs,ignore_index=True)
            if "Date" in comb.columns:comb=comb.sort_values("Date").reset_index(drop=True)
            self.football_data["all"]=comb;logger.info("✅ [FOOTBALL] %d matches",len(comb))

    def _fuzzy(self,team:str,col:pd.Series)->pd.Series:
        clean=team.lower().strip();m=col.str.lower().str.strip()==clean
        if m.any():return m
        # try word by word
        for p in clean.split():
            if len(p)>3:
                m2=col.str.lower().str.contains(re.escape(p),na=False)
                if m2.sum()<=15:  # avoid overly broad matches
                    if m2.any():return m2
        return pd.Series([False]*len(col),index=col.index)

    def get_football_stats(self,home:str,away:str)->dict:
        df=self.football_data.get("all")
        if df is None or df.empty:return {}
        stats:dict={"home":{},"away":{},"h2h":{}}
        for team,key,is_home in [(home,"home",True),(away,"away",False)]:
            hm=self._fuzzy(team,df["HomeTeam"]);am=self._fuzzy(team,df["AwayTeam"])
            th=df[hm];ta=df[am];all_r=[]
            for _,row in th.iterrows():
                hg=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr=row.get("FTR","")
                all_r.append({"date":row.get("Date"),"result":"W" if ftr=="H" else("D" if ftr=="D" else "L"),
                               "scored":hg,"conceded":ag,"venue":"home",
                               "shots":int(row["HS"]) if pd.notna(row.get("HS")) else 0,
                               "shots_target":int(row["HST"]) if pd.notna(row.get("HST")) else 0,
                               "corners":int(row["HC"]) if pd.notna(row.get("HC")) else 0,
                               "yellows":int(row["HY"]) if pd.notna(row.get("HY")) else 0})
            for _,row in ta.iterrows():
                hg=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                ag=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                ftr=row.get("FTR","")
                all_r.append({"date":row.get("Date"),"result":"W" if ftr=="A" else("D" if ftr=="D" else "L"),
                               "scored":ag,"conceded":hg,"venue":"away",
                               "shots":int(row["AS"]) if pd.notna(row.get("AS")) else 0,
                               "shots_target":int(row["AST"]) if pd.notna(row.get("AST")) else 0,
                               "corners":int(row["AC"]) if pd.notna(row.get("AC")) else 0,
                               "yellows":int(row["AY"]) if pd.notna(row.get("AY")) else 0})
            all_r.sort(key=lambda x:(x["date"] if isinstance(x["date"],pd.Timestamp) else pd.Timestamp.min),reverse=True)
            recent=all_r[:10]
            if not recent:continue
            n=len(recent);sc=[r["scored"] for r in recent];cn=[r["conceded"] for r in recent]
            sh=[r["shots"] for r in recent];avg_sh=float(np.mean(sh)) if sh else 1.;avg_sh=max(avg_sh,1.)
            wts=np.array([1/(i+1) for i in range(n)]);wts/=wts.sum()
            rpts=np.array([3 if r["result"]=="W" else(1 if r["result"]=="D" else 0) for r in recent],dtype=np.float64)
            stats[key]={"name":team,"form_string":"".join(r["result"] for r in recent),
                        "win_rate":round(sum(1 for r in recent if r["result"]=="W")/n,3),
                        "draw_rate":round(sum(1 for r in recent if r["result"]=="D")/n,3),
                        "loss_rate":round(sum(1 for r in recent if r["result"]=="L")/n,3),
                        "avg_scored":round(float(np.mean(sc)),2),"avg_conceded":round(float(np.mean(cn)),2),
                        "std_scored":round(float(np.std(sc)),2),
                        "btts_rate":round(sum(1 for r in recent if r["scored"]>0 and r["conceded"]>0)/n,3),
                        "over25_rate":round(sum(1 for r in recent if r["scored"]+r["conceded"]>2.5)/n,3),
                        "over35_rate":round(sum(1 for r in recent if r["scored"]+r["conceded"]>3.5)/n,3),
                        "clean_sheet_rate":round(sum(1 for r in recent if r["conceded"]==0)/n,3),
                        "avg_shots":round(float(np.mean(sh)),1),
                        "avg_shots_target":round(float(np.mean([r["shots_target"] for r in recent])),1),
                        "avg_corners":round(float(np.mean([r["corners"] for r in recent])),1),
                        "shot_conversion":round(float(np.mean(sc))/avg_sh,3),
                        "weighted_form_points":round(float(np.dot(wts,rpts)),3),
                        "matches_analyzed":n,"total_historical":len(all_r),
                        # ── NEW: data quality ──
                        "data_quality":(
                            "good" if len(all_r)>=20
                            else "limited" if len(all_r)>=8
                            else "poor"
                        )}
            vk="home" if is_home else "away"
            vm=[r for r in all_r[:20] if r["venue"]==vk]
            if len(vm)>=3:
                vn=len(vm)
                stats[key]["venue_win_rate"]=round(sum(1 for r in vm if r["result"]=="W")/vn,3)
                stats[key]["venue_avg_goals"]=round(float(np.mean([r["scored"]+r["conceded"] for r in vm])),2)
                stats[key]["venue_btts"]=round(sum(1 for r in vm if r["scored"]>0 and r["conceded"]>0)/vn,3)
        if "HomeTeam" in df.columns:
            hm2=self._fuzzy(home,df["HomeTeam"]);am2=self._fuzzy(away,df["AwayTeam"])
            hm3=self._fuzzy(away,df["HomeTeam"]);am3=self._fuzzy(home,df["AwayTeam"])
            h2h_df=df[(hm2&am2)|(hm3&am3)]
            if len(h2h_df):
                h2hr=[]
                for _,row in h2h_df.iterrows():
                    hg=int(row["FTHG"]) if pd.notna(row.get("FTHG")) else 0
                    ag=int(row["FTAG"]) if pd.notna(row.get("FTAG")) else 0
                    h2hr.append({"total_goals":hg+ag,"btts":hg>0 and ag>0,"over25":hg+ag>2.5,"over35":hg+ag>3.5})
                hn=len(h2hr);gl=[r["total_goals"] for r in h2hr]
                stats["h2h"]={"total_matches":hn,"avg_goals":round(float(np.mean(gl)),2),
                               "btts_rate":round(sum(1 for r in h2hr if r["btts"])/hn,3),
                               "over25_rate":round(sum(1 for r in h2hr if r["over25"])/hn,3),
                               "over35_rate":round(sum(1 for r in h2hr if r["over35"])/hn,3),
                               "std_goals":round(float(np.std(gl)),2)}
                logger.info("✅ [FOOTBALL H2H] %s vs %s: %d matches",home,away,hn)
        return stats

    def get_club_elo(self,team:str)->Optional[float]:
        ck=f"elo_{team.lower()}"
        if CacheManager.is_valid(self.elo_cache,ck,CFG.TTL_TEAM_FORM):return CacheManager.get(self.elo_cache,ck)
        clean=re.sub(r"[^a-zA-Z]","",team).replace("FC","").strip()
        if not clean:return None
        try:
            r=requests.get(CFG.GITHUB_SOURCES["club_elo"].format(team=clean),timeout=8,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200 and r.text.strip():
                lines=[l for l in r.text.strip().split("\n") if l.strip()]
                if len(lines)>1:
                    parts=lines[-1].split(",")
                    if len(parts)>=5:
                        elo=float(parts[4])
                        self.elo_cache=CacheManager.set(self.elo_cache,ck,elo)
                        CacheManager.save(CFG.CACHE_DIR/"elo_cache.json",self.elo_cache)
                        return elo
        except:pass
        return None

    def get_elo_delta(self,home:str,away:str)->Optional[dict]:
        he=self.get_club_elo(home);ae=self.get_club_elo(away)
        if not(he and ae):return None
        delta=he-ae;hp=min(0.95,1/(1+10**(-delta/400))+0.03)
        return {"home_elo":round(he,1),"away_elo":round(ae,1),"delta":round(delta,1),
                "home_win_prob_elo":round(hp,3),"away_win_prob_elo":round(1-hp,3),
                "elo_confidence":"high" if abs(delta)>150 else "medium" if abs(delta)>75 else "low"}

    def load_nba_data(self):
        cache_path=CFG.HISTORICAL_DIR/"nba_standings.json"
        if cache_path.exists() and (time.time()-cache_path.stat().st_mtime)/3600<12:
            try:
                data=json.loads(cache_path.read_text())
                if data:self.nba_data=pd.DataFrame(data);logger.info("✅ [NBA] %d teams from cache",len(self.nba_data));return
            except:pass
        try:
            from nba_api.stats.endpoints import leaguestandings
            standings=leaguestandings.LeagueStandings(season="2024-25",season_type="Regular Season",league_id="00",
                headers={"User-Agent":"Mozilla/5.0","Accept":"application/json","Referer":"https://www.nba.com/","Origin":"https://www.nba.com"},timeout=10)
            df=standings.get_data_frames()[0]
            if df is not None and not df.empty:
                self.nba_data=df;cache_path.write_text(json.dumps(df.to_dict(orient="records"),indent=2))
                logger.info("✅ [NBA] %d teams via nba_api",len(df));return
        except ImportError:logger.warning("[NBA] nba_api not installed")
        except Exception as e:logger.warning("[NBA] nba_api error: %s",str(e)[:80])
        if cache_path.exists():
            try:
                data=json.loads(cache_path.read_text())
                if data:self.nba_data=pd.DataFrame(data);logger.info("✅ [NBA] %d teams from stale cache",len(self.nba_data));return
            except:pass
        logger.warning("[NBA] No data available");self.nba_data=None

    def get_nba_stats(self,team:str)->dict:
        if self.nba_data is None or self.nba_data.empty:return {}
        df=self.nba_data;clean=team.lower().strip();m=pd.DataFrame()
        for col in ["TeamName","TEAM_NAME","TeamCity","TEAM_CITY"]:
            if col in df.columns:
                found=df[df[col].astype(str).str.lower().str.contains(re.escape(clean),na=False)]
                if not found.empty:m=found;break
        if m.empty:return {}
        row=m.iloc[0]
        def si(*cols):
            for c in cols:
                if c in row.index:
                    try:return int(row[c] or 0)
                    except:return 0
            return 0
        def sf(*cols):
            for c in cols:
                if c in row.index:
                    try:return float(row[c] or 0)
                    except:return 0.
            return 0.
        wins=si("WINS","W");losses=si("LOSSES","L");gp=max(wins+losses,1)
        return {"season_record":f"{wins}W-{losses}L","win_pct":round(sf("WinPCT","WIN_PCT","PCT"),3),
                "avg_pts_scored":round(sf("PointsPG","PTS_PG"),1),"avg_pts_allowed":round(sf("OppPointsPG","OPP_PTS_PG"),1),
                "pt_diff":round(sf("PointsPG","PTS_PG")-sf("OppPointsPG","OPP_PTS_PG"),1),
                "last_10":str(row.get("L10",row.get("LAST_TEN",""))),
                "streak":str(row.get("strCurrentStreak",row.get("CurrentStreak",""))),"games_played":gp,"source":"nba_api"}

    def get_nba_matchup(self,home:str,away:str)->dict:
        hs=self.get_nba_stats(home);aw=self.get_nba_stats(away)
        if not hs or not aw:return {}
        h_str=hs.get("win_pct",0.5)*0.6+max(min(hs.get("pt_diff",0)/20,0.3),-0.3)
        a_str=aw.get("win_pct",0.5)*0.6+max(min(aw.get("pt_diff",0)/20,0.3),-0.3)
        total=h_str+a_str;home_prob=min(0.85,max(0.15,(h_str/total)+0.02)) if total>0 else 0.52
        return {"home":hs,"away":aw,"elo_home_win_prob":round(home_prob,3),"elo_away_win_prob":round(1-home_prob,3)}

    def load_nhl_data(self):
        url="https://api-web.nhle.com/v1/standings/now"
        cache_path=CFG.HISTORICAL_DIR/"nhl_standings.json"
        def parse_rows(standings):
            rows=[]
            for team in standings:
                rows.append({"team":team.get("teamName",{}).get("default",""),"teamAbbrev":team.get("teamAbbrev",{}).get("default",""),
                              "wins":team.get("wins",0),"losses":team.get("losses",0),"otLosses":team.get("otLosses",0),
                              "points":team.get("points",0),"goalsFor":team.get("goalFor",0),"goalsAgainst":team.get("goalAgainst",0),
                              "goalsForPctg":team.get("goalsForPctg",0.),"home_wins":team.get("homeWins",0),
                              "home_losses":team.get("homeLosses",0),"road_wins":team.get("roadWins",0),
                              "road_losses":team.get("roadLosses",0),"l10Wins":team.get("l10Wins",0),
                              "l10Losses":team.get("l10Losses",0),"streakCode":team.get("streakCode",""),"streakCount":team.get("streakCount",0)})
            return rows
        try:
            r=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200:
                data=r.json();standings=data.get("standings",[])
                if standings:
                    rows=parse_rows(standings);self.nhl_data=pd.DataFrame(rows)
                    cache_path.write_text(json.dumps(data,indent=2))
                    logger.info("✅ [NHL] %d teams",len(rows));return
        except Exception as e:logger.warning("[NHL] Live API error: %s",str(e)[:80])
        if cache_path.exists():
            try:
                data=json.loads(cache_path.read_text());standings=data.get("standings",[])
                if standings:
                    rows=parse_rows(standings);self.nhl_data=pd.DataFrame(rows)
                    logger.info("✅ [NHL] %d teams from cache",len(rows));return
            except:pass
        logger.warning("[NHL] No data");self.nhl_data=None

    def get_nhl_stats(self,team:str)->dict:
        if self.nhl_data is None or self.nhl_data.empty:return {}
        clean=team.lower().strip()
        m=self.nhl_data[self.nhl_data["team"].str.lower().str.contains(re.escape(clean),na=False)|
                        self.nhl_data["teamAbbrev"].str.lower().str.contains(re.escape(clean),na=False)]
        if m.empty:return {}
        row=m.iloc[0];gp=max(int(row.get("wins",0))+int(row.get("losses",0))+int(row.get("otLosses",0)),1)
        gf=int(row.get("goalsFor",0));ga=int(row.get("goalsAgainst",0))
        return {"wins":int(row.get("wins",0)),"losses":int(row.get("losses",0)),"ot_losses":int(row.get("otLosses",0)),
                "points":int(row.get("points",0)),"games_played":gp,"win_pct":round(int(row.get("wins",0))/gp,3),
                "avg_goals_for":round(gf/gp,2),"avg_goals_against":round(ga/gp,2),"goal_diff_per_game":round((gf-ga)/gp,2),
                "last_10":f"{int(row.get('l10Wins',0))}W-{int(row.get('l10Losses',0))}L",
                "streak":f"{row.get('streakCode','?')}{row.get('streakCount',0)}",
                "home_wins":int(row.get("home_wins",0)),"road_wins":int(row.get("road_wins",0)),"source":"nhl_official_api"}

    def load_mlb_data(self):
        url="https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025&standingsTypes=regularSeason"
        cache_path=CFG.HISTORICAL_DIR/"mlb_standings.json"
        def parse_rows(data):
            rows=[]
            for record in data.get("records",[]):
                for tr in record.get("teamRecords",[]):
                    ti=tr.get("team",{});sr=tr.get("records",{}).get("splitRecords",[])
                    rows.append({"team":ti.get("name",""),"team_id":ti.get("id",0),
                                 "wins":tr.get("wins",0),"losses":tr.get("losses",0),
                                 "win_pct":float(tr.get("winningPercentage",0) or 0),
                                 "runs_scored":tr.get("runsScored",0),"runs_allowed":tr.get("runsAllowed",0),
                                 "home_wins":next((s.get("wins",0) for s in sr if s.get("type")=="home"),0),
                                 "away_wins":next((s.get("wins",0) for s in sr if s.get("type")=="away"),0),
                                 "last10_wins":next((s.get("wins",0) for s in sr if s.get("type")=="lastTen"),0),
                                 "streak":tr.get("streak",{}).get("streakCode","")})
            return rows
        try:
            r=requests.get(url,timeout=15,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200:
                data=r.json();rows=parse_rows(data)
                if rows:self.mlb_data=pd.DataFrame(rows);cache_path.write_text(json.dumps(data,indent=2));logger.info("✅ [MLB] %d teams",len(rows));return
        except Exception as e:logger.warning("[MLB] Live API error: %s",str(e)[:80])
        if cache_path.exists():
            try:
                data=json.loads(cache_path.read_text());rows=parse_rows(data)
                if rows:self.mlb_data=pd.DataFrame(rows);logger.info("✅ [MLB] %d teams from cache",len(rows));return
            except:pass
        logger.warning("[MLB] No data");self.mlb_data=None

    def get_mlb_stats(self,team:str)->dict:
        if HAS_STATSAPI:
            try:
                teams=mlb_statsapi.lookup_team(team)
                if teams:
                    tid=teams[0]["id"];sd=(datetime.now()-timedelta(days=14)).strftime("%Y-%m-%d");ed=datetime.now().strftime("%Y-%m-%d")
                    sched=mlb_statsapi.schedule(team=tid,start_date=sd,end_date=ed)
                    finished=[g for g in sched if g.get("status")=="Final"][-7:]
                    if finished:
                        wins=losses=rs=ra=0
                        for g in finished:
                            ih=g.get("home_id")==tid;hs=g.get("home_score",0) or 0;as_=g.get("away_score",0) or 0
                            ts=hs if ih else as_;os_=as_ if ih else hs
                            if ts>os_:wins+=1
                            else:losses+=1
                            rs+=ts;ra+=os_
                        tg=max(wins+losses,1)
                        return {"recent_form":f"{wins}W-{losses}L","avg_runs_scored":round(rs/tg,1),
                                "avg_runs_allowed":round(ra/tg,1),"run_diff":round((rs-ra)/tg,1),"source":"statsapi_live"}
            except Exception as e:logger.debug("[MLB STATSAPI] %s: %s",team,str(e)[:80])
        if self.mlb_data is None or self.mlb_data.empty:return {}
        clean=team.lower().strip()
        m=self.mlb_data[self.mlb_data["team"].str.lower().str.contains(re.escape(clean),na=False)]
        if m.empty:return {}
        row=m.iloc[0];w=int(row.get("wins",0));l=int(row.get("losses",0));gp=max(w+l,1)
        rs=int(row.get("runs_scored",0));ra=int(row.get("runs_allowed",0))
        return {"season_record":f"{w}W-{l}L","win_pct":round(float(row.get("win_pct",0)),3),
                "avg_runs_scored":round(rs/gp,1),"avg_runs_allowed":round(ra/gp,1),
                "run_diff_per_game":round((rs-ra)/gp,2),"streak":str(row.get("streak","")),"source":"mlb_official_api"}

    def load_cricket_data(self):
        import zipfile,io
        extracted=CFG.HISTORICAL_DIR/"cricket_t20_matches.csv"
        if extracted.exists() and (time.time()-extracted.stat().st_mtime)/3600<CFG.TTL_GITHUB_DATA:
            try:self.cricket_data=pd.read_csv(extracted,low_memory=False);logger.info("✅ [CRICKET] %d records",len(self.cricket_data));return
            except:pass
        try:
            r=requests.get("https://cricsheet.org/downloads/t20s_csv2.zip",timeout=30,headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code==200 and len(r.content)>1000:
                import zipfile,io
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    csv_files=sorted([f for f in z.namelist() if f.endswith(".csv")],key=lambda x:len(x))
                    if csv_files:
                        with z.open(csv_files[0]) as f:extracted.write_bytes(f.read())
                        self.cricket_data=pd.read_csv(extracted,low_memory=False)
                        if "date" in self.cricket_data.columns:self.cricket_data["date"]=pd.to_datetime(self.cricket_data["date"],errors="coerce")
                        logger.info("✅ [CRICKET] %d records",len(self.cricket_data));return
        except Exception as e:logger.debug("[CRICKET] %s",str(e)[:80])
        logger.warning("[CRICKET] No data");self.cricket_data=None

    def get_cricket_stats(self,team:str)->dict:
        if self.cricket_data is None or self.cricket_data.empty:return {}
        df=self.cricket_data;tc=next((c for c in ["batting_team","team1","team"] if c in df.columns),None)
        if not tc:return {}
        m=df[df[tc].str.lower().str.contains(re.escape(team.lower()),na=False)]
        if len(m)<5:return {}
        recent=m.tail(10);run_col=next((c for c in ["runs_off_bat","total_runs","runs"] if c in recent.columns),None)
        result={"matches_found":len(m),"form_sample":len(recent)}
        if run_col:result["avg_runs_recent"]=round(float(recent[run_col].dropna().mean()),1)
        return result

    def get_us_sports_stats(self,sport:str,team:str)->dict:
        sport_lower=sport.lower();ck=f"{sport_lower}_{team.lower().replace(' ','')}"
        if CacheManager.is_valid(self.us_cache,ck,12.):
            cached=CacheManager.get(self.us_cache,ck)
            if cached:return cached
        result={}
        if "basketball" in sport_lower or "nba" in sport_lower:result=self.get_nba_stats(team)
        elif "baseball" in sport_lower or "mlb" in sport_lower:result=self.get_mlb_stats(team)
        elif "hockey" in sport_lower or "nhl" in sport_lower:result=self.get_nhl_stats(team)
        elif "cricket" in sport_lower or "ipl" in sport_lower or "t20" in sport_lower:result=self.get_cricket_stats(team)
        if result:
            self.us_cache=CacheManager.set(self.us_cache,ck,result)
            CacheManager.save(CFG.CACHE_DIR/"us_sports_cache.json",self.us_cache)
        return result

# =========================================================
# 10. ML ENGINE  — BUG-1 FIXED (No Data Leakage)
# =========================================================
class MLPredictionEngine:
    def __init__(self,de:FreeDataEngine):
        self.de=de
        self.football_pipeline:Optional[dict]=None
        self.tennis_pipelines:Dict[str,Optional[dict]]={"atp":None,"wta":None}
        self.is_football_trained=False
        self.is_nba_trained=False
        self._football_team_deques:Dict[str,deque]=defaultdict(lambda:deque(maxlen=10))
        self._football_team_stats:Dict[str,List]={}
        self._rng=np.random.RandomState(42)

    @property
    def is_tennis_trained(self)->bool:
        return any(p is not None for p in self.tennis_pipelines.values())

    def load_or_train_football_model(self):
        path=CFG.ML_DIR/"football_model_v72.pkl"
        if path.exists() and (time.time()-path.stat().st_mtime)/3600<24:
            try:
                d=pickle.loads(path.read_bytes())
                self.football_pipeline=d["pipeline"]
                self._football_team_stats=d["stats"]
                self._football_team_deques=defaultdict(lambda:deque(maxlen=10))
                for k,v in self._football_team_stats.items():
                    self._football_team_deques[k]=deque(v,maxlen=10)
                self.is_football_trained=True;logger.info("⚡ [ML FOOTBALL] Loaded from cache");return
            except:pass
        self._train_football()
        if self.is_football_trained:
            try:
                stats_snapshot={k:list(v) for k,v in self._football_team_deques.items()}
                path.write_bytes(pickle.dumps({"pipeline":self.football_pipeline,"stats":stats_snapshot}))
                logger.info("💾 [ML FOOTBALL] Saved")
            except:pass

    def _train_football(self):
        df=self.de.football_data.get("all")
        if df is None or len(df)<300:logger.warning("[ML FOOTBALL] Insufficient data (%d)",len(df) if df is not None else 0);return
        # BUG-2: Ensure sort before rolling
        if "Date" in df.columns:df=df.sort_values("Date").reset_index(drop=True)
        result_lookup=self._build_fb_rolling(df)
        X,y=self._build_fb_features(result_lookup)
        if len(X)<200 or len(np.unique(y))<2:logger.warning("[ML FOOTBALL] Too few samples");return
        scaler=RobustScaler();Xs=scaler.fit_transform(X)
        model=CalibratedClassifierCV(
            StackingClassifier(estimators=[
                ("gb",GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)),
                ("rf",RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1))],
                final_estimator=LogisticRegression(max_iter=1000,C=0.1,random_state=42),cv=3),cv=3,method="isotonic")
        try:
            model.fit(Xs,y);self.football_pipeline={"model":model,"scaler":scaler}
            self.is_football_trained=True;logger.info("✅ [ML FOOTBALL] Trained on %d samples",len(X))
        except Exception as e:logger.error("[ML FOOTBALL] Training failed: %s",e)

    def _build_fb_rolling(self,df:pd.DataFrame)->dict:
        # BUG-2: df must be sorted by Date before calling this method
        self._football_team_deques=defaultdict(lambda:deque(maxlen=10))
        lookup={}
        for idx,row in df.iterrows():
            ht=str(row.get("HomeTeam","") or "");at=str(row.get("AwayTeam","") or "");ftr=str(row.get("FTR","") or "")
            if not ht or not at or ftr not in ["H","D","A"]:continue
            try:hg=float(row.get("FTHG",0) or 0);ag=float(row.get("FTAG",0) or 0)
            except:continue
            def gs(team):
                h=list(self._football_team_deques[team])
                if len(h)<3:return None
                w=np.array([1/(i+1) for i in range(len(h))][::-1]);w/=w.sum()
                return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),"avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                        "form_pts":float(np.dot(w,[x["pts"] for x in h])),"win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
            hs=gs(ht);aws=gs(at)
            if hs and aws:lookup[idx]={"home_stats":hs,"away_stats":aws,"label":{"H":0,"D":1,"A":2}[ftr]}
            self._football_team_deques[ht].appendleft({"gs":hg,"gc":ag,"pts":3 if ftr=="H" else(1 if ftr=="D" else 0)})
            self._football_team_deques[at].appendleft({"gs":ag,"gc":hg,"pts":3 if ftr=="A" else(1 if ftr=="D" else 0)})
        self._football_team_stats={k:list(v) for k,v in self._football_team_deques.items()}
        return lookup

    def _build_fb_features(self,lookup:dict)->Tuple[np.ndarray,np.ndarray]:
        feats,labels=[],[]
        for _,d in lookup.items():
            hs=d["home_stats"];aws=d["away_stats"]
            if not hs or not aws:continue
            feats.append([hs.get("avg_gs",0),hs.get("avg_gc",0),hs.get("form_pts",0),hs.get("win_rate",0),
                          aws.get("avg_gs",0),aws.get("avg_gc",0),aws.get("form_pts",0),aws.get("win_rate",0),
                          hs.get("avg_gs",0)-aws.get("avg_gc",0),aws.get("avg_gs",0)-hs.get("avg_gc",0)])
            labels.append(d["label"])
        if not feats:return np.array([]),np.array([])
        return np.nan_to_num(np.array(feats,dtype=np.float64)),np.array(labels,dtype=np.int32)

    def predict_football(self,home:str,away:str)->Optional[dict]:
        if not self.is_football_trained:return None
        def ft(team):
            cl=team.lower().strip()
            bm=next((k for k in self._football_team_deques if cl in k.lower() or k.lower() in cl),None)
            if not bm:return None
            h=list(self._football_team_deques[bm])
            if len(h)<3:return None
            w=np.array([1/(i+1) for i in range(len(h))][::-1]);w/=w.sum()
            return {"avg_gs":float(np.dot(w,[x["gs"] for x in h])),"avg_gc":float(np.dot(w,[x["gc"] for x in h])),
                    "form_pts":float(np.dot(w,[x["pts"] for x in h])),"win_rate":sum(1 for x in h if x["pts"]==3)/len(h)}
        hs=ft(home);aws=ft(away)
        if not hs or not aws:return None
        fv=[hs["avg_gs"],hs["avg_gc"],hs["form_pts"],hs["win_rate"],
            aws["avg_gs"],aws["avg_gc"],aws["form_pts"],aws["win_rate"],
            hs["avg_gs"]-aws["avg_gc"],aws["avg_gs"]-hs["avg_gc"]]
        X=np.nan_to_num(np.array([fv],dtype=np.float64));Xs=self.football_pipeline["scaler"].transform(X)
        try:
            probs=self.football_pipeline["model"].predict_proba(Xs)[0]
            classes=self.football_pipeline["model"].classes_
            lm={0:"home_win",1:"draw",2:"away_win"}
            return {lm.get(int(c),f"c{c}"):round(float(p),4) for c,p in zip(classes,probs)}
        except Exception as e:logger.warning("[ML FOOTBALL] Predict error: %s",e);return None

    def load_or_train_tennis_model(self,is_wta:bool=False):
        tour="wta" if is_wta else "atp"
        path=CFG.ML_DIR/f"tennis_model_{tour}_v72.pkl"
        if path.exists() and (time.time()-path.stat().st_mtime)/3600<24:
            try:
                d=pickle.loads(path.read_bytes())
                self.tennis_pipelines[tour]=d["pipeline"]
                logger.info("⚡ [ML TENNIS %s] Loaded",tour.upper());return
            except:pass
        self._train_tennis(is_wta)
        if self.tennis_pipelines[tour]:
            try:
                path.write_bytes(pickle.dumps({"pipeline":self.tennis_pipelines[tour]}))
                logger.info("💾 [ML TENNIS %s] Saved",tour.upper())
            except:pass

    # -------------------------------------------------------
    # BUG-1 FIX: Build player aggregate stats BEFORE each match
    # Uses rolling history up to (but NOT including) the current row.
    # NO in-match stats (w_ace, w_1stWon etc.) are used as features.
    # -------------------------------------------------------
    def _build_tennis_features_leak_free(self,df:pd.DataFrame)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Leak-free feature builder for tennis.
        For each match we compute rolling aggregate stats for BOTH players
        from their PREVIOUS matches only (no current-match stats used).
        Features: ranking gap, surface dummies, best_of,
                  rolling win_rate, rolling aces/svpt, rolling 1stWon/1stIn,
                  rolling bp_saved for each player.
        """
        # Sort by date ascending to ensure chronological order
        df=df.sort_values("tourney_date").reset_index(drop=True)

        # Per-player rolling stats accumulator
        # key: player_id (winner_id or loser_id), value: list of historical match stat dicts
        from collections import defaultdict as _dd
        player_history:Dict[Any,List[dict]]=_dd(list)

        feats,labels,wts=[],[],[]

        def _agg(history:list,n:int=20)->dict:
            """Aggregate last n matches for a player."""
            recent=history[-n:] if len(history)>=n else history
            if not recent:return {}
            total=len(recent)
            wins=sum(1 for h in recent if h["won"])
            ace_sum=sum(h.get("ace",0) for h in recent)
            svpt_sum=sum(h.get("svpt",1) for h in recent)
            in1_sum=sum(h.get("1stIn",0) for h in recent)
            won1_sum=sum(h.get("1stWon",0) for h in recent)
            bps_sum=sum(h.get("bpSaved",0) for h in recent)
            bpf_sum=sum(h.get("bpFaced",1) for h in recent)
            svpt_safe=max(svpt_sum,1);in1_safe=max(in1_sum,1);bpf_safe=max(bpf_sum,1)
            return {
                "win_rate":wins/total,
                "ace_rate":ace_sum/svpt_safe,
                "first_in_rate":in1_sum/svpt_safe,
                "first_won_rate":won1_sum/in1_safe,
                "bp_saved_rate":bps_sum/bpf_safe,
                "n":total,
            }

        def _sf(v,d=0.):
            try:return float(v or d)
            except:return d

        for _,row in df.iterrows():
            wid=row.get("winner_id");lid=row.get("loser_id")
            wr=_sf(row.get("winner_rank",0));lr=_sf(row.get("loser_rank",0))
            if wr<=0 or lr<=0:
                # Still update history even if we skip this sample
                pass
            else:
                surf=str(row.get("surface","Hard") or "Hard").lower()
                bo=_sf(row.get("best_of",3))
                td=_sf(row.get("tourney_date"),20200101)

                # --- Get PREVIOUS stats for winner and loser ---
                w_hist=player_history.get(wid,[])
                l_hist=player_history.get(lid,[])
                w_agg=_agg(w_hist)
                l_agg=_agg(l_hist)

                # Only add sample if both players have at least 3 prior matches
                if w_agg.get("n",0)>=3 and l_agg.get("n",0)>=3:
                    # Anchor: lower rank number = better = P1
                    is_winner_p1=(wr<lr)
                    if wr==lr:
                        is_winner_p1=(hash(str(wid)+str(lid))%2==0)

                    p1r,p2r=(wr,lr) if is_winner_p1 else(lr,wr)
                    p1a,p2a=(w_agg,l_agg) if is_winner_p1 else(l_agg,w_agg)

                    # label=1 means P1 won (better-ranked player won)
                    label=1 if is_winner_p1 else 0

                    fv=[
                        p1r,p2r,
                        p2r-p1r,            # rank gap (positive = P1 better)
                        p2r/max(p1r,1.),    # rank ratio
                        1. if surf=="hard" else 0.,
                        1. if surf=="clay" else 0.,
                        1. if surf=="grass" else 0.,
                        bo,
                        # P1 rolling aggregate features
                        p1a.get("win_rate",0.5),
                        p1a.get("ace_rate",0.05),
                        p1a.get("first_in_rate",0.6),
                        p1a.get("first_won_rate",0.7),
                        p1a.get("bp_saved_rate",0.6),
                        float(p1a.get("n",0)),
                        # P2 rolling aggregate features
                        p2a.get("win_rate",0.5),
                        p2a.get("ace_rate",0.05),
                        p2a.get("first_in_rate",0.6),
                        p2a.get("first_won_rate",0.7),
                        p2a.get("bp_saved_rate",0.6),
                        float(p2a.get("n",0)),
                        # Differentials
                        p1a.get("win_rate",0.5)-p2a.get("win_rate",0.5),
                        p1a.get("first_won_rate",0.7)-p2a.get("first_won_rate",0.7),
                        p1a.get("bp_saved_rate",0.6)-p2a.get("bp_saved_rate",0.6),
                    ]
                    feats.append(fv);labels.append(label)
                    wts.append(float(np.clip(0.5+0.5*(td-20200101)/max(20260101-20200101,1),0.5,1.)))

            # --- Update player history AFTER extracting features ---
            # Winner stats from this match
            w_match={
                "won":True,
                "ace":_sf(row.get("w_ace")),
                "svpt":max(_sf(row.get("w_svpt",50)),1.),
                "1stIn":_sf(row.get("w_1stIn")),
                "1stWon":_sf(row.get("w_1stWon")),
                "bpSaved":_sf(row.get("w_bpSaved")),
                "bpFaced":max(_sf(row.get("w_bpFaced")),1.),
            }
            l_match={
                "won":False,
                "ace":_sf(row.get("l_ace")),
                "svpt":max(_sf(row.get("l_svpt",50)),1.),
                "1stIn":_sf(row.get("l_1stIn")),
                "1stWon":_sf(row.get("l_1stWon")),
                "bpSaved":_sf(row.get("l_bpSaved")),
                "bpFaced":max(_sf(row.get("l_bpFaced")),1.),
            }
            if wid is not None:player_history[wid].append(w_match)
            if lid is not None:player_history[lid].append(l_match)

        if not feats:return np.array([]),np.array([]),np.array([])
        return (np.nan_to_num(np.array(feats,dtype=np.float64)),
                np.array(labels,dtype=np.int32),
                np.array(wts,dtype=np.float64))

    def _train_tennis(self,is_wta:bool=False):
        df=self.de.wta_matches if is_wta else self.de.atp_matches
        tour="wta" if is_wta else "atp"
        if df is None or len(df)<500:logger.warning("[ML TENNIS %s] Insufficient data",tour.upper());return
        # BUG-1: Use leak-free feature builder
        X,y,sw=self._build_tennis_features_leak_free(df)
        if len(X)<200 or len(np.unique(y))<2:logger.warning("[ML TENNIS %s] Too few samples (%d)",tour.upper(),len(X));return
        scaler=RobustScaler();Xs=scaler.fit_transform(X)
        try:
            gb=GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42,subsample=0.8)
            cal=CalibratedClassifierCV(estimator=gb,cv=3,method="isotonic");cal.fit(Xs,y,sample_weight=sw if len(sw)==len(X) else None)
            self.tennis_pipelines[tour]={"model":cal,"scaler":scaler}
            logger.info("✅ [ML TENNIS %s] Leak-free training on %d samples",tour.upper(),len(X))
        except Exception as e:logger.error("[ML TENNIS %s] Training failed: %s",tour.upper(),e)

    def predict_tennis(self,pa:str,pb:str,stats:dict,surface:str="hard")->Optional[dict]:
        # BUG-9: Correctly detect tour from stats dict
        tour="wta" if stats.get("tour","").lower()=="wta" else "atp"
        pipeline=self.tennis_pipelines.get(tour)
        if not pipeline:
            # Fallback to whichever is available
            for t in ["atp","wta"]:
                if self.tennis_pipelines.get(t):pipeline=self.tennis_pipelines[t];break
        if not pipeline:return None

        pas=stats.get("player_a",{});pbs=stats.get("player_b",{})
        ra=float(pas.get("current_ranking",100) or 100)
        rb=float(pbs.get("current_ranking",100) or 100)

        # Rolling aggregate features at inference time (from historical stats)
        def gs(p):
            wr=float(p.get("recent_win_rate",0.5) or 0.5)
            svpt=max(float(p.get("svpt_per_match",50) or 50),1.)
            ace=float(p.get("aces_per_match",5) or 5)
            return {
                "win_rate":wr,
                "ace_rate":ace/svpt,
                "first_in_rate":float(p.get("first_serve_in_pct",0.6) or 0.6),
                "first_won_rate":float(p.get("first_serve_win_pct",0.7) or 0.7),
                "bp_saved_rate":float(p.get("bp_saved_pct",0.6) or 0.6),
                "n":float(p.get("total_matches",10) or 10),
            }

        is_pa_p1=(ra<=rb)
        if ra==rb:is_pa_p1=(hash(pa+pb)%2==0)
        p1r,p2r=(ra,rb) if is_pa_p1 else(rb,ra)
        p1a,p2a=(gs(pas),gs(pbs)) if is_pa_p1 else(gs(pbs),gs(pas))

        fv=[
            p1r,p2r,p2r-p1r,p2r/max(p1r,1.),
            1. if surface=="hard" else 0.,1. if surface=="clay" else 0.,1. if surface=="grass" else 0.,
            3.,
            p1a["win_rate"],p1a["ace_rate"],p1a["first_in_rate"],p1a["first_won_rate"],p1a["bp_saved_rate"],p1a["n"],
            p2a["win_rate"],p2a["ace_rate"],p2a["first_in_rate"],p2a["first_won_rate"],p2a["bp_saved_rate"],p2a["n"],
            p1a["win_rate"]-p2a["win_rate"],
            p1a["first_won_rate"]-p2a["first_won_rate"],
            p1a["bp_saved_rate"]-p2a["bp_saved_rate"],
        ]
        try:
            X=np.nan_to_num(np.array([fv],dtype=np.float64));Xs=pipeline["scaler"].transform(X)
            probs=pipeline["model"].predict_proba(Xs)[0]
            pm={int(c):float(p) for c,p in zip(pipeline["model"].classes_,probs)}
            p1_win_prob=pm.get(1,0.5)
            pa_p=p1_win_prob if is_pa_p1 else(1-p1_win_prob)
            return {f"{pa}_win_prob":round(pa_p,4),f"{pb}_win_prob":round(1-pa_p,4)}
        except Exception as e:logger.warning("[ML TENNIS] Predict error: %s",e);return None

    def load_or_train_nba_model(self):
        logger.info("[ML NBA] Using standings-based matchup prediction");self.is_nba_trained=False

    def predict_nba(self,home_stats:dict,away_stats:dict)->Optional[dict]:
        if not home_stats or not away_stats:return None
        h_str=home_stats.get("win_pct",0.5)*0.6+max(min(home_stats.get("pt_diff",0)/20,0.3),-0.3)
        a_str=away_stats.get("win_pct",0.5)*0.6+max(min(away_stats.get("pt_diff",0)/20,0.3),-0.3)
        total=h_str+a_str
        if total<=0:return None
        home_prob=min(0.85,max(0.15,(h_str/total)+0.02))
        return {"home_win_prob":round(home_prob,4),"away_win_prob":round(1-home_prob,4)}

# =========================================================
# 11. POISSON ENGINE
# =========================================================
class PoissonEngine:
    @staticmethod
    def calculate_match_probabilities(home:str,away:str,df:Optional[pd.DataFrame])->dict:
        if df is None or df.empty:return {}
        req={"HomeTeam","AwayTeam","FTHG","FTAG"}
        if not req.issubset(df.columns):return {}
        rec=df.dropna(subset=["FTHG","FTAG"]).tail(1500).copy()
        if len(rec)<50:return {}
        la_home=rec["FTHG"].astype(float).mean();la_away=rec["FTAG"].astype(float).mean()
        if pd.isna(la_home) or la_home==0:return {}
        def fz(t,col):
            cl=t.lower().strip();m=col.str.lower().str.strip()==cl
            if m.any():return m
            for p in cl.split():
                if len(p)>3:
                    m2=col.str.lower().str.contains(re.escape(p),na=False)
                    if m2.any():return m2
            return pd.Series([False]*len(col),index=col.index)
        hm=rec[fz(home,rec["HomeTeam"])];am=rec[fz(away,rec["AwayTeam"])]
        if len(hm)<5 or len(am)<5:return {}
        ha=hm["FTHG"].astype(float).mean()/la_home;hd=hm["FTAG"].astype(float).mean()/la_away
        aa=am["FTAG"].astype(float).mean()/la_away;ad=am["FTHG"].astype(float).mean()/la_home
        if any(pd.isna(v) or v==0 for v in [ha,hd,aa,ad]):return {}
        hxg=float(np.clip(ha*ad*la_home,0.1,8.));axg=float(np.clip(aa*hd*la_away,0.1,8.))
        mg=6;pm=np.zeros((mg+1,mg+1));rho=-0.1
        for x in range(mg+1):
            for y in range(mg+1):
                base=stats_scipy.poisson.pmf(x,hxg)*stats_scipy.poisson.pmf(y,axg)
                adj=(1-hxg*axg*rho if x==0 and y==0 else(1+hxg*rho if x==0 and y==1 else(1+axg*rho if x==1 and y==0 else(1-rho if x==1 and y==1 else 1.))))
                pm[x,y]=base*max(0.,adj)
        t=pm.sum()
        if t==0:return {}
        pm/=t
        return {"home_xg":round(hxg,2),"away_xg":round(axg,2),
                "home_win_prob_poisson":round(float(np.sum(np.tril(pm,-1))),3),
                "draw_prob_poisson":round(float(np.sum(np.diag(pm))),3),
                "away_win_prob_poisson":round(float(np.sum(np.triu(pm,1))),3)}

# =========================================================
# 12. EV ENGINE
# =========================================================
class EVEngine:
    @staticmethod
    def remove_vig_power(odds_list:List[float])->List[float]:
        implied=[1/o for o in odds_list if o>1.]
        if not implied:return []
        total=sum(implied)
        if abs(total-1.)<0.001:return implied
        def f(k):return sum(p**k for p in implied)-1.
        try:
            fa,fb=f(0.5),f(3.0)
            if fa*fb>=0:return [p/total for p in implied]
            k=brentq(f,0.5,3.0,xtol=1e-6);tp=[p**k for p in implied];s=sum(tp)
            return [p/s for p in tp] if s>0 else [p/total for p in implied]
        except:return [p/total for p in implied]

    @staticmethod
    def remove_vig_shin(odds_list:List[float])->List[float]:
        implied=[1/o for o in odds_list if o>1.]
        if not implied:return []
        n=len(implied);total=sum(implied)
        if total==0:return implied
        z=max(0.,min((total-1)/max(total-1/n,1e-10),0.2));tp=[]
        for p in implied:
            denom=2*(1-n*z)
            if abs(denom)<1e-10:tp.append(p/total);continue
            inner=z**2+4*(1-z)*(p**2/total)
            if inner<0:tp.append(p/total);continue
            tp.append((-z+inner**0.5)/denom)
        s=sum(tp)
        return [p/s for p in tp] if s>0 else [p/total for p in implied]

    @staticmethod
    def kelly(prob:float,odds:float)->float:
        b=odds-1.
        if b<=0 or prob<=0 or prob>=1:return 0.
        k=max(0.,(prob*b-(1-prob))/b)
        return round(min(k*CFG.KELLY_FRACTION,CFG.MAX_KELLY_PCT/100),4)

def calculate_sharp_ev_advanced(markets_data:dict)->list:
    best_per_market:dict={}
    for mk,ml in markets_data.items():
        if not isinstance(ml,list):continue
        # BUG-10: Use composite key (name, point) to avoid outcome collision
        sharp_all:Dict[Tuple,List[float]]=defaultdict(list)
        soft_all:Dict[Tuple,List[float]]=defaultdict(list)
        best_mkt:Dict[Tuple,Tuple[float,str,str]]={}  # key→(price, bookmaker, display_name)
        for entry in ml:
            if not isinstance(entry,dict):continue
            bk=entry.get("bookmaker_key","");bk_name=entry.get("bookmaker",bk)
            is_sharp=bk in CFG.SHARP_BOOKMAKERS
            for o in entry.get("outcomes",[]):
                if not isinstance(o,dict):continue
                raw_name=o.get("name","")
                if not raw_name:continue
                point=o.get("point")
                # BUG-10: Composite key avoids collision between e.g. Over 2.5 and Over 3.5
                comp_key=(raw_name,point)
                display_name=(f"{raw_name} {point}" if point is not None else raw_name)
                try:price=float(o["price"])
                except:continue
                if price<=1.:continue
                (sharp_all if is_sharp else soft_all)[comp_key].append(price)
                if comp_key not in best_mkt or price>best_mkt[comp_key][0]:
                    best_mkt[comp_key]=(price,bk_name,display_name)
        if not best_mkt:continue
        sharp_best={k:max(p) for k,p in sharp_all.items() if p}
        has_sharp=bool(sharp_best)
        if not sharp_best:sharp_best={k:max(p) for k,p in soft_all.items() if p}
        if not sharp_best:continue
        comp_keys=list(sharp_best.keys());odds_list=[sharp_best[k] for k in comp_keys]
        impl_sum=sum(1/o for o in odds_list if o>0)
        if not(CFG.MIN_VALID_IMPLIED_SUM<=impl_sum<=CFG.MAX_VALID_IMPLIED_SUM):continue
        if len(comp_keys)<CFG.MARKET_EXPECTED_OUTCOMES.get(mk,{}).get("min",2):continue
        try:
            tp_pw=EVEngine.remove_vig_power(odds_list);tp_sh=EVEngine.remove_vig_shin(odds_list)
            if len(tp_pw)!=len(comp_keys) or len(tp_sh)!=len(comp_keys):raise ValueError()
            tp={comp_keys[i]:0.6*tp_pw[i]+0.4*tp_sh[i] for i in range(len(comp_keys))}
        except:tp={comp_keys[i]:(1/odds_list[i])/max(impl_sum,1e-10) for i in range(len(comp_keys))}
        min_odds=CFG.H2H_MIN_ODDS if mk=="h2h" else CFG.TOTALS_MIN_ODDS
        min_ev=(CFG.H2H_MIN_EV if mk=="h2h" else CFG.TOTALS_MIN_EV)*(1. if has_sharp else 1.5)
        best_opp=None
        for ck in comp_keys:
            true_p=tp.get(ck,0)
            if true_p<=0 or true_p>=1:continue
            bp,bbm,disp_name=best_mkt.get(ck,(0,"?","?"))
            if bp<=1.:continue
            ev=true_p*bp-1.
            if ev<min_ev or ev>CFG.MAX_REALISTIC_EV or bp<min_odds:continue
            kelly_p=EVEngine.kelly(true_p,bp)
            sp=sharp_best.get(ck,bp);clv=(bp/sp-1)*100 if sp>0 else 0.
            opp={"pick":disp_name,"market":mk,"market_label":get_market_label(mk),
                 "prob":round(true_p,4),"odds":round(bp,3),"bookmaker":bbm,
                 "ev":round(ev,4),"edge_pct":round(ev*100,2),"kelly_pct":round(kelly_p*100,2),
                 "clv_pct":round(clv,2),"has_sharp_line":has_sharp,"devigging_method":"power_shin_weighted","steam_pct":0.}
            if best_opp is None or opp["ev"]>best_opp["ev"]:best_opp=opp
        if best_opp:best_per_market[mk]=best_opp
    return sorted(best_per_market.values(),key=lambda x:x["ev"],reverse=True)

# =========================================================
# 13. CONFIDENCE ENGINE
# =========================================================
class ConfidenceEngine:
    W={"base":42,"ev_high":15,"ev_medium":10,"ev_low":5,"sharp_line":8,
       "football_stats":5,"elo_high":8,"elo_medium":4,"ml_strong":8,"ml_medium":4,
       "poisson_confirm":5,"smart_money":8,"kelly_high":5,"kelly_medium":3,
       "data_quality_good":4,"data_quality_poor":-10}

    @classmethod
    def calculate_math_score(cls,opp:dict,stats:dict,market:str,
                              ml_pred:Optional[dict]=None,
                              poisson_pred:Optional[dict]=None)->int:
        s=cls.W["base"]
        ev=opp.get("ev",0)*100
        s+=(cls.W["ev_high"] if ev>5 else cls.W["ev_medium"] if ev>3 else cls.W["ev_low"] if ev>1 else 0)
        if ev<2.0:s-=5
        if opp.get("has_sharp_line"):s+=cls.W["sharp_line"]
        kelly=opp.get("kelly_pct",0)
        s+=(cls.W["kelly_high"] if kelly>2 else cls.W["kelly_medium"] if kelly>1 else 0)

        # ── football stats + data quality ──
        if stats.get("football_stats"):
            fs=stats["football_stats"]
            s+=cls.W["football_stats"]
            hq=fs.get("home",{}).get("data_quality","poor")
            aq=fs.get("away",{}).get("data_quality","poor")
            if hq=="good" and aq=="good":s+=cls.W["data_quality_good"]
            elif hq=="poor" or aq=="poor":s+=cls.W["data_quality_poor"]

        # ── tennis data quality ──
        if stats.get("historical_data"):
            dq=stats["historical_data"].get("data_quality_summary",{})
            overall=dq.get("overall","poor")
            if overall=="good":s+=cls.W["data_quality_good"]
            elif overall=="poor":s+=cls.W["data_quality_poor"]

        delta=abs(stats.get("elo_data",{}).get("delta",0))
        s+=(cls.W["elo_high"] if delta>150 else cls.W["elo_medium"] if delta>75 else 0)
        if ml_pred:
            mx=max((v for v in ml_pred.values() if isinstance(v,float) and 0<v<=1),default=0)
            s+=(cls.W["ml_strong"] if mx>0.65 else cls.W["ml_medium"] if mx>0.55 else 0)
        if poisson_pred:s+=cls.W["poisson_confirm"]
        steam=opp.get("steam_pct")
        if steam is not None and steam!=0.:
            if steam>=3:s+=cls.W["smart_money"]
            elif steam<=-5:s-=cls.W["smart_money"]
        us=stats.get("us_sports",{});home_data=us.get("home",{})
        if home_data.get("source")=="mlb_official_api":s-=8
        elif home_data.get("source")=="nhl_official_api":s-=6
        elif home_data.get("source")=="statsapi_live":s+=5
        if kelly<1.5 and ev<3.:s-=5
        return int(np.clip(s,0,100))

# =========================================================
# 14. UTILITIES
# =========================================================
def robust_json_extractor(raw:str)->Optional[dict]:
    if not raw:return None
    clean=re.sub(r"<think>[\s\S]*?</think>","",raw,flags=re.IGNORECASE)
    clean=re.sub(r"```(?:json)?","",clean).strip().rstrip("`").strip()
    try:return json.loads(clean)
    except:pass
    for m in reversed(list(re.finditer(r"\{[^{}]*\}",clean))):
        try:
            r=json.loads(m.group(0))
            if isinstance(r,dict) and r:return r
        except:continue
    try:
        m=re.search(r"\{[\s\S]*\}",clean)
        if m:return json.loads(m.group(0))
    except:pass
    return None

def clean_team_name(name:str)->str:
    return re.sub(r"\s*\([^)]*\)","",str(name or "")).strip()

def normalize_sport_key(sport_title:str)->str:
    lower=(sport_title or "").lower()
    if any(k in lower for k in ["tennis","atp","wta"]):return "tennis"
    if any(k in lower for k in ["soccer","football","premier league","la liga","bundesliga","serie a","ligue 1","champions","brasileirao","liga mx"]):return "football"
    if any(k in lower for k in ["basketball","nba","euroleague"]):return "basketball"
    if any(k in lower for k in ["baseball","mlb"]):return "baseball"
    if any(k in lower for k in ["hockey","nhl"]):return "hockey"
    if any(k in lower for k in ["cricket","ipl","t20","odi"]):return "cricket"
    return "other"

def get_countdown_str(ct:str,now:datetime)->str:
    try:
        mt=datetime.fromisoformat(ct.replace("Z","+00:00"))
        if mt.tzinfo is None:mt=mt.replace(tzinfo=timezone.utc)
        mins=int((mt-now).total_seconds()/60)
        if mins>60:return f"{mins//60}h {mins%60}m"
        if mins>0:return f"{mins}m"
        return "LIVE"
    except:return "N/A"

def get_market_label(mk:str)->str:
    return {"h2h":"Match Winner","totals":"Over/Under","spreads":"Point Spread"}.get(mk,mk.replace("_"," ").title())

def _get_sport_emoji(sk:str)->str:
    return {"tennis":"🎾","football":"⚽","basketball":"🏀","baseball":"⚾","hockey":"🏒","cricket":"🏏"}.get(sk,"🏆")

def get_sport_name(sport_key:str,sport_title:str)->str:
    names={"tennis":"🎾 Tennis","football":"⚽ Football","basketball":"🏀 Basketball",
           "baseball":"⚾ Baseball","hockey":"🏒 Ice Hockey","cricket":"🏏 Cricket"}
    return names.get(sport_key,f"🏆 {sport_title}")

def get_sport_name_fa(sport_key:str,sport_title:str)->str:
    names={"tennis":"🎾 تنیس","football":"⚽ فوتبال","basketball":"🏀 بسکتبال",
           "baseball":"⚾ بیسبال","hockey":"🏒 هاکی روی یخ","cricket":"🏏 کریکت"}
    return names.get(sport_key,f"🏆 {sport_title}")

def get_confidence_text(fc:int)->str:
    if fc>=78:return "خیلی قوی 🔥🔥"
    elif fc>=70:return "قوی 🔥"
    elif fc>=65:return "متوسط ✅"
    else:return "استاندارد ⚡"

def get_risk_text(risk:str)->str:
    return {"Low":"Low 🟢","Medium":"Medium 🟠","High":"High 🔴"}.get(risk,"Medium 🟠")

def get_risk_text_fa(risk:str)->str:
    return {"Low":"پایین 🟢","Medium":"متوسط 🟠","High":"بالا 🔴"}.get(risk,"متوسط 🟠")

# =========================================================
# 15. LINE MOVEMENT TRACKER — BUG-4 FIXED
# =========================================================
class LineMovementTracker:
    def __init__(self):
        self._path=CFG.CACHE_DIR/"line_movement.json";self._lock=threading.Lock()
        self.data=CacheManager.load(self._path);self._cleanup()

    def _cleanup(self):
        now=datetime.now(timezone.utc);to_del=[]
        for k,v in self.data.items():
            if not isinstance(v,dict):to_del.append(k);continue
            try:
                t=datetime.fromisoformat(v.get("timestamp",""))
                if t.tzinfo is None:t=t.replace(tzinfo=timezone.utc)
                if now-t>timedelta(hours=48):to_del.append(k)
            except:to_del.append(k)
        for k in to_del:self.data.pop(k,None)

    def record_and_get_movement(self,home:str,away:str,market:str,outcome:str,odds:float)->Optional[float]:
        """
        BUG-4: Returns None on first observation (no movement data yet).
        Returns float steam% on subsequent calls.
        Positive = odds dropped = money came in (steam).
        Negative = odds drifted up = money left (fade).
        """
        if odds<=1.:return None
        mk=hashlib.md5(f"{home}|{away}|{market}|{outcome}".encode()).hexdigest()
        with self._lock:
            now=datetime.now(timezone.utc).isoformat()
            if mk not in self.data:
                # BUG-4: First observation - store and return None (not 0)
                self.data[mk]={"initial_odds":odds,"current_odds":odds,"timestamp":now,"first_seen":True}
                CacheManager.save(self._path,self.data);return None
            init=self.data[mk].get("initial_odds",odds)
            self.data[mk].update({"current_odds":odds,"timestamp":now,"first_seen":False})
            CacheManager.save(self._path,self.data)
        return round((init/odds-1)*100,2) if init>0 else 0.

line_movement_tracker=LineMovementTracker()

# =========================================================
# 16. PICK TRANSLATOR — BUG-13 FIXED
# =========================================================
def translate_pick_for_public(pick:str,market:str,home:str,away:str,odds:float,prob:float)->str:
    pick_lower=pick.lower().strip()
    market_lower=market.lower().strip()
    home_sim=difflib.SequenceMatcher(None,home.lower(),pick_lower).ratio()
    away_sim=difflib.SequenceMatcher(None,away.lower(),pick_lower).ratio()
    is_home=home_sim>away_sim and home_sim>0.3
    is_away=away_sim>home_sim and away_sim>0.3
    is_draw="draw" in pick_lower or "tie" in pick_lower
    action=pick.title()
    if "lay" in market_lower or "lay" in pick_lower:
        if is_home:action=f"{away} or Draw (Double Chance)"
        elif is_away:action=f"{home} or Draw (Double Chance)"
        elif is_draw:action=f"{home} or {away} (Any Team to Win)"
        else:action=f"Lay / Bet Against {pick.title()}"
    elif market_lower=="h2h":
        if is_home:action=f"{home} to Win"
        elif is_away:action=f"{away} to Win"
        elif is_draw:action="Match to end in a Draw"
    elif "total" in market_lower:
        # BUG-13: Anchored regex - look for over/under keyword first, then the number
        # Pattern: optional team name words, then over/under, then the line number
        m=re.search(r"\b(over|under)\b\s+([\d.]+)",pick_lower)
        if not m:
            # Try reversed: number then over/under
            m2=re.search(r"([\d.]+)\s+\b(over|under)\b",pick_lower)
            if m2:
                line=m2.group(1);direction=m2.group(2)
            else:
                # Last resort: just find a decimal number that looks like a line (e.g. 2.5, 47.5)
                nums=re.findall(r"\b(\d+\.5|\d+\.0|\d{1,3})\b",pick_lower)
                line=nums[-1] if nums else "?"
                direction="over" if "over" in pick_lower else "under" if "under" in pick_lower else ""
        else:
            direction=m.group(1);line=m.group(2)
        if "over" in (direction if isinstance(direction,str) else ""):
            action=f"Over {line} Goals/Points"
        elif "under" in (direction if isinstance(direction,str) else ""):
            action=f"Under {line} Goals/Points"
        else:
            action=pick.title()
    elif "spread" in market_lower or "handicap" in market_lower:
        team_name=home if is_home else(away if is_away else pick.title())
        m=re.search(r"([+-]?[\d.]+)",pick_lower)
        handicap=m.group(1) if m else ""
        action=f"{team_name} {handicap} Handicap"
    elif "btts" in market_lower:
        if "yes" in pick_lower:action="Both Teams To Score: YES"
        elif "no" in pick_lower:action="Both Teams To Score: NO"
    return action

def format_odds_nice(odds:float)->str:
    if odds>=2.0:return f"{odds:.2f} 🔥"
    elif odds>=1.7:return f"{odds:.2f} ✅"
    else:return f"{odds:.2f}"

# =========================================================
# 17. AI DECISION ENGINE
# =========================================================
def generate_ai_decision(home:str,away:str,sport:str,sport_key:str,opp:dict,stats:dict,math_score:int,
                          ml_pred:Optional[dict]=None,poisson_pred:Optional[dict]=None)->dict:
    default={"sport_emoji":_get_sport_emoji(sport_key),"decision":"SKIP","ai_confidence":math_score,
             "math_confidence":math_score,"final_confidence":math_score,"risk_level":"High",
             "logic":"Insufficient data for AI decision.","key_factors":[],"red_flags":[]}
    if math_score<CFG.MIN_MATH_SCORE_TO_CALL_AI:
        return {**default,"logic":f"Math score {math_score} below threshold."}
    us=stats.get("us_sports",{});home_data=us.get("home",{})
    if sport_key in ["baseball","hockey"] and home_data.get("source") in ["mlb_official_api","nhl_official_api"]:
        if math_score<55:
            logger.info("⏭️ SKIP: %s - standings-only data insufficient (math=%d)",sport_key,math_score)
            return {**default,"decision":"SKIP","logic":f"Standings-only data for {sport_key}. Need game logs for reliable signal."}

    parts=[]
    steam_val=opp.get("steam_pct")
    steam_str=f"{steam_val:.1f}%" if steam_val is not None else "N/A (first seen)"

    # ── data quality warning ──
    dq_warnings=[]
    if stats.get("historical_data"):
        dq=stats["historical_data"].get("data_quality_summary",{})
        if dq.get("overall")=="poor":
            dq_warnings.append("⚠️ TENNIS: Poor historical data coverage for one or both players")
        elif dq.get("overall")=="limited":
            dq_warnings.append("⚠️ TENNIS: Limited historical data - lower confidence")
    if stats.get("football_stats"):
        hq=stats["football_stats"].get("home",{}).get("data_quality","poor")
        aq=stats["football_stats"].get("away",{}).get("data_quality","poor")
        if hq=="poor":dq_warnings.append(f"⚠️ FOOTBALL: Poor data for {home}")
        if aq=="poor":dq_warnings.append(f"⚠️ FOOTBALL: Poor data for {away}")
    if dq_warnings:
        parts.append("=== DATA QUALITY WARNINGS ===\n"+"\n".join(dq_warnings))

    parts.append(f"=== MARKET ===\nPick:{opp['pick']} | Market:{opp['market_label']} | "
                 f"Odds:{opp['odds']} | TrueProb:{opp['prob']*100:.1f}% | EV:{opp['edge_pct']:+.2f}% | "
                 f"Kelly:{opp.get('kelly_pct',0):.1f}% | SharpLine:{opp.get('has_sharp_line',False)} | "
                 f"CLV:{opp.get('clv_pct',0):+.1f}% | Steam:{steam_str} | MathScore:{math_score}/100")

    if stats.get("historical_data"):
        pa=stats["historical_data"].get("player_a",{});pb=stats["historical_data"].get("player_b",{})
        h2h=stats["historical_data"].get("h2h",{})
        dq=stats["historical_data"].get("data_quality_summary",{})
        parts.append(f"=== TENNIS ===\n{home}: Rank={pa.get('current_ranking','N/A')} Form={pa.get('recent_form','N/A')} "
                     f"WR={pa.get('recent_win_rate',0)*100:.1f}% Matches={pa.get('total_matches',0)} Quality={pa.get('data_quality','?')}\n"
                     f"{away}: Rank={pb.get('current_ranking','N/A')} Form={pb.get('recent_form','N/A')} "
                     f"WR={pb.get('recent_win_rate',0)*100:.1f}% Matches={pb.get('total_matches',0)} Quality={pb.get('data_quality','?')}\n"
                     f"H2H:{h2h.get('total',0)} Dominance:{h2h.get('dominance','balanced')} OverallQuality:{dq.get('overall','?')}")
    if stats.get("football_stats"):
        hm=stats["football_stats"].get("home",{});aw=stats["football_stats"].get("away",{})
        h2h=stats["football_stats"].get("h2h",{})
        parts.append(f"=== FOOTBALL ===\n{home}(H): Form={hm.get('form_string','N/A')} "
                     f"GS={hm.get('avg_scored',0):.2f} GC={hm.get('avg_conceded',0):.2f} "
                     f"WR={hm.get('win_rate',0)*100:.1f}% Quality={hm.get('data_quality','?')}\n"
                     f"{away}(A): Form={aw.get('form_string','N/A')} "
                     f"GS={aw.get('avg_scored',0):.2f} GC={aw.get('avg_conceded',0):.2f} "
                     f"WR={aw.get('win_rate',0)*100:.1f}% Quality={aw.get('data_quality','?')}\n"
                     f"H2H({h2h.get('total_matches',0)}): AvgG={h2h.get('avg_goals',0):.2f} Over25={h2h.get('over25_rate',0)*100:.1f}%")
    if stats.get("elo_data"):
        e=stats["elo_data"]
        parts.append(f"=== ELO ===\n{home}:{e.get('home_elo')} {away}:{e.get('away_elo')} "
                     f"Delta:{e.get('delta')}({e.get('elo_confidence','?')}) "
                     f"HP:{e.get('home_win_prob_elo',0)*100:.1f}% AP:{e.get('away_win_prob_elo',0)*100:.1f}%")
    if ml_pred:parts.append(f"=== ML MODEL ===\n{json.dumps(ml_pred)}")
    if poisson_pred:
        parts.append(f"=== POISSON ===\nhome_xg:{poisson_pred.get('home_xg')} away_xg:{poisson_pred.get('away_xg')} "
                     f"H:{poisson_pred.get('home_win_prob_poisson',0)*100:.1f}% D:{poisson_pred.get('draw_prob_poisson',0)*100:.1f}% "
                     f"A:{poisson_pred.get('away_win_prob_poisson',0)*100:.1f}%")
    if stats.get("us_sports"):
        us2=stats["us_sports"]
        parts.append(f"=== US SPORTS ===\n{home}:{json.dumps(us2.get('home',{}))} {away}:{json.dumps(us2.get('away',{}))}")

    sys_inst=(
        "You are an elite sports betting analyst. Analyze ALL data and make a BET/SKIP decision.\n\n"
        "CRITICAL RULES:\n"
        "- If DATA QUALITY WARNINGS present → be extra conservative\n"
        "- If data_quality=poor for either team → max confidence 62\n"
        "- BET when: EV>2.5% AND (sharp line OR strong historical edge) AND models agree AND data quality ≥ limited\n"
        "- SKIP when: EV<2.5% OR conflicting signals OR poor data quality OR Kelly<1.5%\n\n"
        "Confidence: 78-100=Strong BET | 65-77=Moderate BET | 50-64=Weak (SKIP) | 0-49=NO BET\n\n"
        'Output ONLY valid JSON:\n'
        '{"decision":"BET" or "SKIP","confidence":<int 0-100>,"sport_emoji":"<emoji>",'
        '"risk_level":"Low" or "Medium" or "High","key_factors":["fact1","fact2"],"logic":"2-3 sentences","red_flags":["flag1"]}'
    )
    prompt=f"MATCH:{home} vs {away} | SPORT:{sport} | PICK:{opp['pick']} | MARKET:{opp['market_label']}\n\n"+"\n\n".join(parts)
    ai_data=ai_manager.generate(prompt=prompt,system_instruction=sys_inst,is_groq_strict=True)
    if not ai_data or not isinstance(ai_data,dict):
        logger.warning("[AI JUDGE] No response → math fallback")
        return {**default,"decision":"BET" if math_score>=60 else "SKIP","logic":"AI unavailable - math models only."}
    decision=str(ai_data.get("decision","SKIP")).upper().strip()
    if decision not in ["BET","SKIP"]:decision="SKIP"
    try:ai_conf=int(np.clip(float(ai_data.get("confidence",math_score)),0,100))
    except:ai_conf=math_score

    # ── poor data quality cap ──
    has_poor_data=False
    if stats.get("historical_data"):
        if stats["historical_data"].get("data_quality_summary",{}).get("overall")=="poor":
            has_poor_data=True
    if stats.get("football_stats"):
        hq=stats["football_stats"].get("home",{}).get("data_quality","poor")
        aq=stats["football_stats"].get("away",{}).get("data_quality","poor")
        if hq=="poor" or aq=="poor":has_poor_data=True
    if has_poor_data and ai_conf>62:
        ai_conf=62
        logger.warning("[AI] Poor data quality → capped confidence at 62")

    if ai_manager._last_provider=="groq" and ai_conf>=70 and opp.get("ev",0)*100<3.:
        ai_conf=min(ai_conf-10,62)
        logger.warning("[GROQ PENALTY] Confidence adjusted to %d",ai_conf)
    hybrid=ai_conf*CFG.AI_WEIGHT+math_score*CFG.MATH_WEIGHT
    ai_delta=hybrid-math_score
    if ai_delta>CFG.MAX_AI_BOOST:hybrid=math_score+CFG.MAX_AI_BOOST
    elif ai_delta<-CFG.MAX_AI_PENALTY:hybrid=math_score-CFG.MAX_AI_PENALTY
    final=int(np.clip(hybrid,0,100))
    if decision=="BET" and ai_conf<55:decision="SKIP";logger.warning("[AI] Inconsistent BET+conf%d→SKIP",ai_conf)
    kf_raw=ai_data.get("key_factors",[]);kf=[str(f)[:120] for f in kf_raw[:5]] if isinstance(kf_raw,list) else []
    rf_raw=ai_data.get("red_flags",[]);rf=[str(f)[:120] for f in rf_raw[:3]] if isinstance(rf_raw,list) else []
    rl=str(ai_data.get("risk_level","Medium"));rl=rl if rl in ["Low","Medium","High"] else "Medium"
    se_raw=ai_data.get("sport_emoji","");se=str(se_raw).strip() if se_raw else _get_sport_emoji(sport_key)
    logic=str(ai_data.get("logic",default["logic"]))[:600]
    logger.info("[AI JUDGE] %s vs %s | %s | AI:%d Math:%d Final:%d | Flags:%s",
                home,away,decision,ai_conf,math_score,final,rf if rf else "none")
    return {"sport_emoji":se,"decision":decision,"ai_confidence":ai_conf,"math_confidence":math_score,
            "final_confidence":final,"risk_level":rl,"logic":logic,"key_factors":kf,"red_flags":rf}

# =========================================================
# 18. TELEGRAM
# =========================================================
def send_telegram(msg:str)->bool:
    MAX=4000;chunks=[]
    if len(msg)<=MAX:chunks=[msg]
    else:
        cur=""
        for line in msg.split("\n"):
            if len(cur)+len(line)+1>MAX:
                if cur:chunks.append(cur.strip())
                cur=line+"\n"
            else:cur+=line+"\n"
        if cur.strip():chunks.append(cur.strip())
    ok=True
    for chunk in chunks:
        try:
            r=requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                            json={"chat_id":TELEGRAM_CHAT_ID,"text":chunk,"parse_mode":"HTML","disable_web_page_preview":True},timeout=15)
            if not r.ok:logger.error("Telegram [%d]: %s",r.status_code,r.text[:200]);ok=False
        except requests.RequestException as e:logger.error("Telegram: %s",e);ok=False
    return ok

def build_signal_message(home,away,sport,sport_key,opp,ai_data,stats,math_score,ml_pred,poisson_pred,now_utc,commence_time)->str:
    fc=ai_data["final_confidence"]
    he=html_lib.escape(home);ae=html_lib.escape(away)
    sport_fa=get_sport_name_fa(sport_key,sport)
    public_pick=translate_pick_for_public(opp["pick"],opp["market"],home,away,opp["odds"],opp["prob"])
    public_pick_escaped=html_lib.escape(public_pick)
    conf_text=get_confidence_text(fc)
    risk_text=get_risk_text_fa(ai_data["risk_level"])
    countdown=get_countdown_str(commence_time,now_utc)
    badges=[]
    if stats.get("historical_data"):badges.append("📚 آمار تاریخی")
    if stats.get("football_stats"):badges.append("⚽ آمار بازی")
    if stats.get("elo_data"):badges.append("📊 رتبه‌بندی Elo")
    if ml_pred:badges.append("🤖 هوش مصنوعی")
    if poisson_pred:badges.append("📐 مدل آماری")
    if stats.get("nba_matchup"):badges.append("🏀 آمار NBA")
    if stats.get("us_sports"):badges.append("🇺🇸 آمار تیم")
    sources=" | ".join(badges) if badges else "اطلاعات عمومی"
    logic_escaped=html_lib.escape(str(ai_data.get("logic",""))[:400])
    kf=ai_data.get("key_factors",[])
    kf_lines=""
    if kf:
        kf_lines="\n\n📌 <b>دلایل اصلی:</b>\n"+"".join(f"  • {html_lib.escape(str(f)[:100])}\n" for f in kf[:3])
    rf=ai_data.get("red_flags",[])
    rf_line=f"\n⚠️ <b>نکته احتیاطی:</b> <i>"+" | ".join(html_lib.escape(str(f)[:80]) for f in rf)+"</i>" if rf else ""
    ev_pct=opp["edge_pct"]
    ev_text=f"{ev_pct:.1f}% مازاد" if ev_pct>0 else f"{ev_pct:.1f}%"
    kelly_text=f"{opp.get('kelly_pct',0):.1f}% از کل سرمایه"
    msg=(
        f"{ai_data.get('sport_emoji','🏆')} <b>{sport_fa}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⚔️ <b>{he}</b>  vs  <b>{ae}</b>\n"
        f"⏰ <b>شروع بازی:</b> {countdown} دیگه\n\n"
        f"🎯 <b>پیش‌بینی ما:</b>\n"
        f"<code>{public_pick_escaped}</code>\n\n"
        f"💰 <b>چقدر شرط ببندم؟</b> {kelly_text}\n"
        f"📈 <b>مزیت ما نسبت به بازار:</b> {ev_text}\n\n"
        f"{'🔥🔥' if fc>=78 else '🔥' if fc>=70 else '✅'} <b>قدرت سیگنال:</b> {conf_text} ({fc}%)\n"
        f"⚖️ <b>ریسک:</b> {risk_text}\n\n"
        f"🔬 <b>تحلیل سیستم:</b>\n"
        f"<blockquote>{logic_escaped}</blockquote>"
        f"{kf_lines}{rf_line}\n\n"
        f"📋 <b>منابع داده:</b> {sources}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 <i>کانال {html_lib.escape(CFG.TELEGRAM_ID)} | سیستم هوشمند v7.2</i>"
    )
    return msg

# =========================================================
# 19. ODDS FETCHER
# =========================================================
class SmartOddsCache:
    def __init__(self):self.cache=CacheManager.load(CFG.ODDS_CACHE_FILE)
    def _key(self,markets,wh):
        raw=f"{','.join(sorted(markets))}|{wh}|{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
        return hashlib.md5(raw.encode()).hexdigest()
    def get_cached(self,markets,wh):
        k=self._key(markets,wh)
        if CacheManager.is_valid_minutes(self.cache,k,CFG.TTL_ODDS_CACHE_MINUTES):
            d=CacheManager.get(self.cache,k)
            if d:logger.info("💾 [ODDS CACHE] HIT %d events",len(d));return d
        return None
    def save_cached(self,markets,wh,events):
        k=self._key(markets,wh);self.cache=CacheManager.set(self.cache,k,events)
        CacheManager.save(CFG.ODDS_CACHE_FILE,self.cache);logger.info("💾 [ODDS CACHE] SAVED %d events",len(events))
    def get_stale(self,markets,wh,max_ttl=2.):
        k=self._key(markets,wh)
        if CacheManager.is_valid(self.cache,k,max_ttl):return CacheManager.get(self.cache,k)
        return None

odds_cache=SmartOddsCache()

async def fetch_market(session,market,now_utc,api_key,label):
    end=now_utc+timedelta(hours=CFG.MATCH_WINDOW_HOURS)
    params={"apiKey":api_key,"regions":CFG.ODDS_API_REGIONS,"markets":market,"oddsFormat":"decimal","dateFormat":"iso"}
    try:
        async with session.get("https://api.the-odds-api.com/v4/sports/upcoming/odds",params=params,
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            rem=int(r.headers.get("x-requests-remaining",-1));used=int(r.headers.get("x-requests-used",0))
            if r.status==200:
                events=await r.json(content_type=None);odds_key_manager.record_usage(label,used,rem);valid=[]
                for e in events:
                    if not isinstance(e,dict):continue
                    try:
                        mt=datetime.fromisoformat(e.get("commence_time","").replace("Z","+00:00"))
                        if mt.tzinfo is None:mt=mt.replace(tzinfo=timezone.utc)
                        if now_utc<=mt<=end:valid.append(e)
                    except:continue
                logger.info("🔑 [%s] OK rem:%d market:%s events:%d",label,rem,market,len(valid));return valid,200,None
            err=await r.text();reasons={401:"Invalid key",402:"Quota exhausted",429:"Rate limited",422:"Invalid params"}
            return [],r.status,reasons.get(r.status,f"HTTP {r.status}:{err[:80]}")
    except asyncio.TimeoutError:return [],0,"Timeout"
    except Exception as e:return [],0,str(e)[:80]

async def fetch_all_odds_async()->list:
    now=datetime.now(timezone.utc)
    cached=odds_cache.get_cached(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS)
    if cached:return cached
    logger.info("💾 [ODDS CACHE] MISS - Calling API...")
    all_events:Dict[str,dict]={};pending_markets=list(CFG.ODDS_API_MARKETS)
    for ki in odds_key_manager.get_active_keys():
        if not pending_markets:break
        ak=ki["key"];label=ki["label"]
        logger.info("🔑 [TRYING] %s for %s",label,pending_markets)
        conn=aiohttp.TCPConnector(limit=10,ssl=False)
        async with aiohttp.ClientSession(connector=conn) as sess:
            tasks=[fetch_market(sess,m,now,ak,label) for m in pending_markets]
            results=await asyncio.gather(*tasks,return_exceptions=True)
        failed_markets=[];hard_fail=None
        for i,res in enumerate(results):
            m=pending_markets[i]
            if isinstance(res,Exception):
                logger.error("🔑❌ [%s] %s failed: %s",label,m,res);failed_markets.append(m);continue
            events,status,err=res
            if status==200:
                for e in events:
                    eid=e.get("id")
                    if not eid:continue
                    if eid not in all_events:
                        all_events[eid]={"id":eid,"sport_key":e.get("sport_key",""),"sport_title":e.get("sport_title",""),
                                         "commence_time":e.get("commence_time",""),"home_team":e.get("home_team",""),
                                         "away_team":e.get("away_team",""),"_markets_data":{}}
                    for bm in e.get("bookmakers",[]):
                        bk=bm.get("key","");bt=bm.get("title",bk)
                        for md in bm.get("markets",[]):
                            mk=md.get("key","")
                            if not mk:continue
                            all_events[eid]["_markets_data"].setdefault(mk,[]).append(
                                {"bookmaker":bt,"bookmaker_key":bk,"outcomes":md.get("outcomes",[])})
            else:
                logger.warning("🔑⚠️ [%s] %s error: %s",label,m,err);failed_markets.append(m)
                if status in [401,402,429]:hard_fail=status
        if hard_fail:
            idx=next((i for i,k in enumerate(odds_key_manager.keys) if k["label"]==label),-1)
            if idx>=0:odds_key_manager.mark_failed(idx,f"HTTP {hard_fail}")
        pending_markets=failed_markets
    if pending_markets:
        logger.error("🔑❌ Failed to fetch markets: %s",pending_markets)
        if not all_events:
            stale=odds_cache.get_stale(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS,2.)
            if stale:logger.warning("💾 [STALE] %d events",len(stale));return stale
            return []
    final=list(all_events.values())
    if final:odds_cache.save_cached(CFG.ODDS_API_MARKETS,CFG.MATCH_WINDOW_HOURS,final)
    logger.info("📊 [API USAGE] %s",odds_key_manager.get_usage_summary())
    return final

# =========================================================
# 20. MAIN PIPELINE
# =========================================================
async def async_main():
    logger.info("="*65)
    logger.info("  ZBET90 ENGINE v7.2 | Multi-Sport | AI 70%% + Math 30%%")
    logger.info("="*65)
    logger.info("🔑 %s",odds_key_manager.get_usage_summary())
    sent=SentHistory();now=datetime.now(timezone.utc)
    logger.info("📥 [PHASE 1] Loading data...")
    de=FreeDataEngine()
    de.load_tennis_data();de.load_football_data();de.load_nba_data()
    de.load_nhl_data();de.load_mlb_data();de.load_cricket_data()
    logger.info("🧠 [PHASE 2] ML models...")
    ml=MLPredictionEngine(de)
    ml.load_or_train_football_model()
    ml.load_or_train_tennis_model(is_wta=False)
    ml.load_or_train_tennis_model(is_wta=True)  # BUG-9: Train WTA model too
    ml.load_or_train_nba_model()
    logger.info("📡 [PHASE 3] Fetching odds (%.1fh window)...",CFG.MATCH_WINDOW_HOURS)
    events=await fetch_all_odds_async()
    if not events:
        logger.info("❌ No events in window.");logger.info("📊 %s",odds_key_manager.get_usage_summary());return
    logger.info("🔍 [PHASE 4] Analyzing %d events...",len(events))
    events.sort(key=lambda x:x.get("commence_time",""))
    total_sent=total_analyzed=0
    skip_no_opp=skip_ev=skip_sent=skip_math=skip_ai=skip_conf=0
    for event in events:
        home=clean_team_name(event.get("home_team",""));away=clean_team_name(event.get("away_team",""))
        sport=event.get("sport_title","Unknown");sport_key=normalize_sport_key(sport)
        if not home or not away:continue
        markets_data=event.get("_markets_data",{});opps=calculate_sharp_ev_advanced(markets_data)
        if not opps:
            logger.info("⏭️ NO_OPP: %s vs %s (no EV opportunity)",home,away);skip_no_opp+=1;continue
        opp=opps[0];total_analyzed+=1
        if opp["ev"]<CFG.MATH_MIN_EV_TO_ANALYZE:
            skip_ev+=1;logger.info("⏭️ LOW_EV: %s vs %s EV=%.2f%%",home,away,opp["edge_pct"]);continue
        if sent.was_sent(home,away,opp["market"]):
            logger.info("⏭️ ALREADY_SENT: %s vs %s [%s]",home,away,opp["market"]);skip_sent+=1;continue
        # BUG-4: steam_pct is now Optional[float]; None = first seen
        opp["steam_pct"]=line_movement_tracker.record_and_get_movement(home,away,opp["market"],opp["pick"],opp["odds"])
        stats:dict={};ml_pred=None;poisson_pred=None
        if sport_key=="tennis":
            is_wta="wta" in sport.lower()
            ts=de.get_tennis_stats(home,away,is_wta)
            if ts:
                # BUG-9: Set tour in stats so ML can pick correct model
                ts["tour"]="wta" if is_wta else "atp"
                stats["historical_data"]=ts
            if ml.is_tennis_trained and ts:
                # BUG-7: Complete surface detection
                sport_lower=sport.lower()
                if any(k in sport_lower for k in ["wimbledon","queens","halle","grass"]):
                    surf="grass"
                elif any(k in sport_lower for k in ["french open","roland garros","monte carlo","madrid open","rome","clay"]):
                    surf="clay"
                elif any(k in sport_lower for k in ["us open","australian open","indian wells","miami","hard","indoor"]):
                    surf="hard"
                else:
                    surf="hard"  # default
                ml_pred=ml.predict_tennis(home,away,ts,surf)
                if ml_pred:stats["ml_prediction"]=ml_pred
        elif sport_key=="football":
            fs=de.get_football_stats(home,away)
            if fs:stats["football_stats"]=fs
            ed=de.get_elo_delta(home,away)
            if ed:stats["elo_data"]=ed
            if ml.is_football_trained:
                ml_pred=ml.predict_football(home,away)
                if ml_pred:stats["ml_prediction"]=ml_pred
            poisson_pred=PoissonEngine.calculate_match_probabilities(home,away,de.football_data.get("all"))
            if poisson_pred:stats["poisson_prediction"]=poisson_pred
        elif sport_key=="basketball":
            hs=de.get_nba_stats(home);aws=de.get_nba_stats(away)
            nb_matchup=de.get_nba_matchup(home,away)
            if nb_matchup:stats["nba_matchup"]=nb_matchup
            if hs or aws:stats["us_sports"]={"home":hs,"away":aws}
            if hs and aws:
                nba_pred=ml.predict_nba(hs,aws)
                if nba_pred:ml_pred=nba_pred;stats["ml_prediction"]=ml_pred
        elif sport_key in ["baseball","hockey","cricket"]:
            hs=de.get_us_sports_stats(sport,home);aws=de.get_us_sports_stats(sport,away)
            if hs or aws:stats["us_sports"]={"home":hs,"away":aws}
            h_source=hs.get("source","") if hs else ""
            if h_source in ["mlb_official_api","nhl_official_api"] and not hs.get("recent_form"):
                logger.info("⏭️ SKIP: %s vs %s standings-only",home,away);skip_math+=1;continue
        elif sport_key=="other":
            # BUG-15: For unknown sports, we have no stats - use higher threshold
            pass
        math_score=ConfidenceEngine.calculate_math_score(opp,stats,opp["market"],ml_pred,poisson_pred)
        # BUG-15: Higher threshold for sport=other (no stats available)
        min_math=(CFG.MIN_MATH_SCORE_OTHER_SPORT if sport_key=="other" else CFG.MIN_MATH_SCORE_TO_CALL_AI)
        if math_score<min_math:
            skip_math+=1;logger.info("⏭️ SKIP(math:%d<%d) %s vs %s EV=%.2f%%",
                                      math_score,min_math,home,away,opp["edge_pct"]);continue
        ai=generate_ai_decision(home,away,sport,sport_key,opp,stats,math_score,ml_pred,poisson_pred)
        fc=ai["final_confidence"]
        if ai.get("decision")=="SKIP":
            skip_ai+=1;logger.info("⏭️ AI SKIP: %s vs %s Math:%d AI:%d Final:%d Flags:%s",
                                    home,away,math_score,ai["ai_confidence"],fc,ai.get("red_flags",[]));continue
        if fc<CFG.MIN_CONFIDENCE_TO_SEND:
            skip_conf+=1;logger.info("⏭️ SKIP(conf:%d<%d) %s vs %s",fc,CFG.MIN_CONFIDENCE_TO_SEND,home,away);continue
        logger.info("✅ APPROVED %s vs %s | Math:%d AI:%d Final:%d EV=%.2f%%",
                    home,away,math_score,ai["ai_confidence"],fc,opp["edge_pct"])
        msg=build_signal_message(home,away,sport,sport_key,opp,ai,stats,math_score,ml_pred,poisson_pred,now,event.get("commence_time",""))
        if send_telegram(msg):
            sent.mark_sent(home,away,opp["pick"],opp["market"])
            performance_tracker.record_signal(home,away,opp["pick"],opp["market"],opp["odds"],opp["ev"],fc,opp["prob"],sport_key,event.get("sport_key",""))
            total_sent+=1;logger.info("📤 SENT: %s vs %s EV=%.2f%% Conf=%d%%",home,away,opp["edge_pct"],fc)
        else:logger.error("❌ Telegram failed: %s vs %s",home,away)
        await asyncio.sleep(CFG.TELEGRAM_SLEEP_BETWEEN)
    logger.info("="*65)
    logger.info("📊 SUMMARY | Events:%d Analyzed:%d Sent:%d | Skip(no_opp):%d Skip(ev):%d Skip(sent):%d Skip(math):%d Skip(AI):%d Skip(conf):%d",
                len(events),total_analyzed,total_sent,skip_no_opp,skip_ev,skip_sent,skip_math,skip_ai,skip_conf)
    logger.info("📊 %s",odds_key_manager.get_usage_summary())
    perf=performance_tracker.data.get("summary",{})
    if perf.get("resolved",0)>0:
        logger.info("📈 WR=%.1f%% ROI=%.1f%% Signals=%d",perf["win_rate"]*100,perf["roi_pct"],perf["total_signals"])
    logger.info("="*65)

if __name__=="__main__":
    try:asyncio.run(async_main())
    except KeyboardInterrupt:logger.info("Stopped.")
    except Exception as e:logger.critical("SYSTEM FAILURE: %s",e,exc_info=True);sys.exit(1)
