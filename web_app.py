"""
Scrapbot Web API and UI - Light theme with ScrapIt teal branding.
"""
import os
import sys
import json
import asyncio
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from chatbot import chatbot_response
from db.database import get_db
from db.crud import log_interaction
from context.context_manager import context_manager

app = FastAPI(title="Scrapbot API", description="Chat API for Scrapbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str = ""

class ChatResponse(BaseModel):
    reply: str
    suggestions: list[str] = []
    error: Optional[str] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def index():
    return _fallback_html()

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, db: AsyncSession = Depends(get_db)):
    msg = (body.message or "").strip()
    if not msg:
        return ChatResponse(reply="Please type a message.", suggestions=[])
    try:
        loop = asyncio.get_event_loop()
        reply, suggestions, meta = await loop.run_in_executor(None, chatbot_response, msg)
        domain = context_manager.get_domain() or "unknown"
        entities = meta.get("groq_enrichment", {}).get("entities") or {}
        await log_interaction(db, domain, msg, entities)
        return ChatResponse(reply=reply or "I didn't understand that.", suggestions=list(suggestions)[:8] if suggestions else [])
    except Exception as e:
        traceback.print_exc()
        return ChatResponse(reply="Something went wrong. Please try again.", suggestions=[], error=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.send_message(websocket, {
            "reply": "GREETING",
            "suggestions": ["Food", "Jobs", "Travel", "Automobiles", "E-Commerce", "Real Estate"],
            "status": "connected"
        })
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                msg = payload.get("message", "").strip()
            except json.JSONDecodeError:
                msg = data.strip()
            if not msg:
                await manager.send_message(websocket, {"reply": "Please type a message.", "suggestions": [], "status": "error"})
                continue
            await manager.send_message(websocket, {"status": "typing"})
            try:
                loop = asyncio.get_event_loop()
                reply, suggestions, _meta = await loop.run_in_executor(None, chatbot_response, msg)
                await manager.send_message(websocket, {
                    "reply": reply or "I didn't understand that.",
                    "suggestions": list(suggestions)[:8] if suggestions else [],
                    "status": "success"
                })
            except Exception as e:
                traceback.print_exc()
                await manager.send_message(websocket, {"reply": "Something went wrong.", "suggestions": [], "status": "error"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _fallback_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScrapBot</title>
<style>
:root {
  --teal: #0d7377;
  --teal2: #14a0a5;
  --teal3: #1bcccc;
  --teal-light: #e6f7f7;
  --teal-lighter: #f0fbfb;
  --bg: #f5fafa;
  --surface: #ffffff;
  --surface2: #edf7f7;
  --border: #c8e8e8;
  --border2: #9fd4d4;
  --text: #0d2b2b;
  --text2: #2d5f5f;
  --muted: #6aabab;
  --white: #ffffff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }

#splash { position: fixed; inset: 0; background: var(--white); display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; transition: opacity 0.7s ease; }
.splash-logo { width: 90px; height: 90px; background: linear-gradient(135deg, var(--teal), var(--teal2)); border-radius: 28px; display: flex; align-items: center; justify-content: center; margin-bottom: 22px; animation: splashPulse 1.5s ease-in-out infinite; box-shadow: 0 8px 30px rgba(13,115,119,0.25); }
@keyframes splashPulse { 0%,100%{transform:scale(1);box-shadow:0 8px 30px rgba(13,115,119,0.25);} 50%{transform:scale(1.07);box-shadow:0 12px 40px rgba(13,115,119,0.35);} }
.splash-name { font-size: 2rem; font-weight: 800; color: var(--teal); letter-spacing: 1px; margin-bottom: 6px; }
.splash-tag { font-size: 13px; color: var(--muted); margin-bottom: 28px; }
.splash-bar { width: 160px; height: 3px; background: var(--teal-light); border-radius: 3px; overflow: hidden; }
.splash-fill { height: 100%; background: linear-gradient(90deg, var(--teal), var(--teal3)); border-radius: 3px; animation: barSlide 1.6s ease-in-out infinite; }
@keyframes barSlide { 0%{width:0;margin-left:0;} 50%{width:65%;margin-left:0;} 100%{width:0;margin-left:100%;} }

header { display: flex; align-items: center; justify-content: space-between; padding: 13px 20px; background: var(--white); border-bottom: 1px solid var(--border); box-shadow: 0 2px 12px rgba(13,115,119,0.08); }
.hd-left { display: flex; align-items: center; gap: 12px; }
.hd-right { display: flex; align-items: center; gap: 10px; }
.logo-box { width: 44px; height: 44px; background: linear-gradient(135deg, var(--teal), var(--teal2)); border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(13,115,119,0.3); animation: logoPulse 3s ease-in-out infinite; flex-shrink: 0; }
@keyframes logoPulse { 0%,100%{box-shadow:0 4px 15px rgba(13,115,119,0.25);} 50%{box-shadow:0 6px 25px rgba(13,115,119,0.4);} }
.hd-title { font-size: 1.25rem; font-weight: 800; color: var(--teal); letter-spacing: 0.5px; }
.hd-sub { font-size: 11px; color: var(--muted); }
#clear-btn { background: transparent; border: 1px solid var(--border2); color: var(--muted); padding: 5px 12px; border-radius: 20px; cursor: pointer; font-size: 11px; font-family: inherit; transition: all 0.2s; }
#clear-btn:hover { background: var(--teal-light); color: var(--teal); border-color: var(--teal); }
#status { font-size: 11px; padding: 5px 13px; border-radius: 20px; font-weight: 600; }
.connected { background: #e6f7ef; color: #0a6e45; border: 1px solid #9fd4be; }
.disconnected { background: #fde8e8; color: #9b2335; border: 1px solid #f5b8be; }
.connecting { background: #fef6e4; color: #8a6000; border: 1px solid #f5d88a; }

#log { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
#log::-webkit-scrollbar { width: 4px; }
#log::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

.mw { display: flex; flex-direction: column; margin: 10px 0; animation: fadeUp 0.25s ease; }
@keyframes fadeUp { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
.mw.u { align-items: flex-end; }
.mw.b { align-items: flex-start; }
.brow { display: flex; align-items: flex-end; gap: 8px; }
.av { width: 32px; height: 32px; background: linear-gradient(135deg, var(--teal), var(--teal2)); border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.bubble { padding: 11px 15px; border-radius: 16px; max-width: 78%; font-size: 14px; line-height: 1.7; word-break: break-word; position: relative; }
.bubble.u { background: linear-gradient(135deg, var(--teal), var(--teal2)); color: var(--white); font-weight: 500; border-radius: 16px 16px 4px 16px; }
.bubble.b { background: var(--white); border: 1px solid var(--border); color: var(--text); white-space: pre-wrap; border-radius: 16px 16px 16px 4px; box-shadow: 0 1px 6px rgba(13,115,119,0.07); }
.bubble.b a { color: var(--teal); text-decoration: none; font-weight: 500; }
.bubble.b a:hover { text-decoration: underline; }
.copy-btn { position: absolute; top: 8px; right: 8px; background: var(--teal-light); border: 1px solid var(--border); color: var(--muted); padding: 3px 8px; border-radius: 8px; cursor: pointer; font-size: 11px; opacity: 0; transition: all 0.2s; font-family: inherit; }
.bubble.b:hover .copy-btn { opacity: 1; }
.copy-btn:hover { background: var(--teal); color: var(--white); border-color: var(--teal); }
.ts { font-size: 10px; color: var(--muted); margin-top: 4px; padding: 0 5px; }

.typing-bub { background: var(--white); border: 1px solid var(--border); border-radius: 16px 16px 16px 4px; padding: 13px 17px; display: flex; gap: 5px; align-items: center; box-shadow: 0 1px 6px rgba(13,115,119,0.07); }
.dot { width: 7px; height: 7px; border-radius: 50%; animation: bounce 1.3s ease-in-out infinite; }
.d1 { background: var(--teal); }
.d2 { background: var(--teal2); animation-delay: 0.2s; }
.d3 { background: var(--teal3); animation-delay: 0.4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0);opacity:0.4;} 30%{transform:translateY(-7px);opacity:1;} }

.chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 8px; padding-left: 40px; }
.chips button { background: var(--white); color: var(--teal); border: 1px solid var(--border2); padding: 6px 15px; border-radius: 20px; cursor: pointer; font-size: 12.5px; transition: all 0.18s; font-family: inherit; }
.chips button:hover { background: var(--teal); color: var(--white); border-color: var(--teal); }

#row { display: flex; gap: 10px; padding: 14px 20px; background: var(--white); border-top: 1px solid var(--border); }
#inp { flex: 1; padding: 12px 18px; border-radius: 25px; border: 1px solid var(--border2); background: var(--bg); color: var(--text); font-size: 14.5px; font-family: inherit; transition: all 0.2s; }
#inp:focus { outline: none; border-color: var(--teal); box-shadow: 0 0 0 3px rgba(13,115,119,0.12); }
#inp::placeholder { color: var(--muted); }
#inp:disabled { opacity: 0.5; }
#btn { padding: 12px 22px; border-radius: 25px; border: none; background: linear-gradient(135deg, var(--teal), var(--teal2)); color: var(--white); font-weight: 700; cursor: pointer; font-size: 14.5px; font-family: inherit; transition: all 0.2s; }
#btn:hover { box-shadow: 0 4px 20px rgba(13,115,119,0.35); transform: scale(1.02); }
#btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; box-shadow: none; }
.toast { position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: var(--teal); color: white; padding: 8px 20px; border-radius: 20px; font-size: 13px; font-weight: 600; z-index: 1000; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
.toast.show { opacity: 1; }
</style>
</head>
<body>

<div id="splash">
  <div class="splash-logo">
    <svg width="50" height="30" viewBox="0 0 100 60" fill="none">
      <path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="7" stroke-linecap="round" fill="none"/>
      <path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="7" stroke-linecap="round" fill="none"/>
    </svg>
  </div>
  <div class="splash-name">ScrapBot</div>
  <div class="splash-tag">AI-Powered Multi-Domain Assistant</div>
  <div class="splash-bar"><div class="splash-fill"></div></div>
</div>

<div class="toast" id="toast">Copied!</div>

<header>
  <div class="hd-left">
    <div class="logo-box">
      <svg width="26" height="16" viewBox="0 0 100 60" fill="none">
        <path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="8" stroke-linecap="round" fill="none"/>
        <path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="8" stroke-linecap="round" fill="none"/>
      </svg>
    </div>
    <div>
      <div class="hd-title">ScrapBot</div>
      <div class="hd-sub">Powered by ScrapIt</div>
    </div>
  </div>
  <div class="hd-right">
    <button id="clear-btn" onclick="clearChat()">🗑 Clear Chat</button>
    <span id="status" class="connecting">Connecting...</span>
  </div>
</header>

<div id="log"></div>

<div id="row">
  <input id="inp" type="text" placeholder="Ask about jobs, food, travel, cars, properties..." autocomplete="off">
  <button id="btn">Send ➤</button>
</div>

<script>
const log=document.getElementById('log'),inp=document.getElementById('inp'),btn=document.getElementById('btn'),status=document.getElementById('status'),splash=document.getElementById('splash'),toast=document.getElementById('toast');
let ws=null,tDiv=null,attempts=0;

setTimeout(()=>{splash.style.opacity='0';setTimeout(()=>splash.style.display='none',700);},2200);

function greeting(){const h=new Date().getHours();if(h>=5&&h<12)return'🌅 Good Morning';if(h>=12&&h<17)return'☀️ Good Afternoon';if(h>=17&&h<21)return'🌆 Good Evening';return'🌙 Good Night';}
function ts(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});}
function setStatus(cls,txt){status.className=cls;status.textContent=txt;inp.disabled=false;btn.disabled=false;}
function showToast(msg){toast.textContent=msg;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2000);}
function copyText(text){navigator.clipboard.writeText(text).then(()=>showToast('Copied!'));}

function clearChat(){
  log.innerHTML='';
  add(`${greeting()}! Chat cleared. How can I help you?`,'bot');
  addChips(["Food","Jobs","Travel","Automobiles","E-Commerce","Real Estate"]);
  showToast('Chat cleared!');
}

function makeAv(){
  const d=document.createElement('div');d.className='av';
  d.innerHTML=`<svg width="16" height="10" viewBox="0 0 100 60" fill="none"><path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="10" stroke-linecap="round" fill="none"/><path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="10" stroke-linecap="round" fill="none"/></svg>`;
  return d;
}

function add(msg,who){
  removeTy();
  const w=document.createElement('div');w.className='mw '+(who==='user'?'u':'b');
  if(who==='bot'){
    const row=document.createElement('div');row.className='brow';
    row.appendChild(makeAv());
    const b=document.createElement('div');b.className='bubble b';
    let t=msg.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    t=t.replace(/(https?:\/\/[^\s&]+)/g,'<a href="$1" target="_blank">🔗 View Details</a>');
    b.innerHTML=t;
    const cp=document.createElement('button');cp.className='copy-btn';cp.textContent='📋 Copy';cp.onclick=()=>copyText(msg);
    b.appendChild(cp);row.appendChild(b);w.appendChild(row);
  } else {
    const b=document.createElement('div');b.className='bubble u';b.textContent=msg;w.appendChild(b);
  }
  const t=document.createElement('div');t.className='ts';t.textContent=ts();w.appendChild(t);
  log.appendChild(w);log.scrollTop=log.scrollHeight;
}

function showTy(){
  if(tDiv)return;tDiv=document.createElement('div');tDiv.className='mw b';
  const row=document.createElement('div');row.className='brow';row.appendChild(makeAv());
  const b=document.createElement('div');b.className='typing-bub';
  b.innerHTML='<div class="dot d1"></div><div class="dot d2"></div><div class="dot d3"></div>';
  row.appendChild(b);tDiv.appendChild(row);log.appendChild(tDiv);log.scrollTop=log.scrollHeight;
}
function removeTy(){if(tDiv){tDiv.remove();tDiv=null;}}

function addChips(arr){
  if(!arr||!arr.length)return;
  const w=document.createElement('div');w.className='chips';
  arr.forEach(s=>{const b=document.createElement('button');b.textContent=s;b.onclick=()=>{inp.value=s;send();};w.appendChild(b);});
  log.appendChild(w);log.scrollTop=log.scrollHeight;
}

function connect(){
  const p=location.protocol==='https:'?'wss:':'ws:';
  ws=new WebSocket(`${p}//${location.host}/ws`);
  ws.onopen=()=>{setStatus('connected','● Online');attempts=0;inp.focus();};
  ws.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.status==='typing'){showTy();return;}
    removeTy();
    if(d.reply){
      let r=d.reply;
      if(r==='GREETING')r=`${greeting()}! Welcome to ScrapBot\n\nI can help you with:\n🍕 Food - restaurants & delivery\n💼 Jobs - career opportunities\n✈️ Travel - trips & destinations\n🚗 Automobiles - cars & vehicles\n🛍️ E-Commerce - products & deals\n🏠 Real Estate - properties for rent & sale\n\nHow can I help you today?`;
      add(r,'bot');
    }
    if(d.suggestions&&d.suggestions.length)addChips(d.suggestions);
    inp.disabled=false;btn.disabled=false;inp.focus();
  };
  ws.onclose=()=>{
    setStatus('disconnected','● Offline');
    if(attempts<5){attempts++;setStatus('connecting','Reconnecting...');setTimeout(connect,2000*attempts);}
  };
  ws.onerror=()=>setStatus('disconnected','● Error');
}

function send(){
  const msg=inp.value.trim();if(!msg)return;
  inp.value='';add(msg,'user');inp.disabled=true;btn.disabled=true;
  if(ws&&ws.readyState===1)ws.send(JSON.stringify({message:msg}));
  else doREST(msg);
}

async function doREST(msg){
  showTy();
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    const d=await r.json();removeTy();add(d.reply||'No response','bot');addChips(d.suggestions||[]);
  }catch(e){removeTy();add('Network error. Try again.','bot');}
  finally{inp.disabled=false;btn.disabled=false;inp.focus();}
}

btn.onclick=send;
inp.onkeydown=e=>{if(e.key==='Enter')send();};
connect();
</script>
</body>
</html>"""
