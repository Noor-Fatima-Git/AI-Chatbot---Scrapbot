"""
Scrapbot Web API and UI - Final version with ScrapIt branding.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        return ChatResponse(
            reply=reply or "I didn't understand that.",
            suggestions=list(suggestions)[:8] if suggestions else [],
        )
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
                await manager.send_message(websocket, {"reply": "Something went wrong. Please try again.", "suggestions": [], "status": "error"})
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
      --teal-dark: #0d2b2b;
      --teal-mid: #0e3d3d;
      --teal-main: #0d7377;
      --teal-light: #14a0a5;
      --teal-bright: #1bcccc;
      --bg: #071a1a;
      --surface: #0e2c2c;
      --surface2: #133535;
      --border: #1a4a4a;
      --text: #e0f4f4;
      --text-muted: #6aabab;
      --white: #ffffff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }

    #loading { position: fixed; inset: 0; background: var(--bg); display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 9999; transition: opacity 0.6s ease; }
    .loading-logo { width: 90px; height: 90px; background: linear-gradient(135deg, var(--teal-main), var(--teal-light)); border-radius: 28px; display: flex; align-items: center; justify-content: center; margin-bottom: 24px; animation: loadingPulse 1.5s ease-in-out infinite; box-shadow: 0 0 50px #0d737766; }
    @keyframes loadingPulse { 0%, 100% { transform: scale(1); box-shadow: 0 0 30px #0d737766; } 50% { transform: scale(1.08); box-shadow: 0 0 60px #14a0a588; } }
    .infinity-svg { width: 50px; height: 30px; }
    .loading-title { font-size: 2rem; font-weight: 800; color: var(--teal-bright); letter-spacing: 1px; margin-bottom: 6px; }
    .loading-sub { font-size: 13px; color: var(--text-muted); margin-bottom: 24px; }
    .loading-bar { width: 180px; height: 3px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
    .loading-bar-fill { height: 100%; background: linear-gradient(90deg, var(--teal-main), var(--teal-bright)); border-radius: 3px; animation: barSlide 1.5s ease-in-out infinite; }
    @keyframes barSlide { 0% { width: 0; margin-left: 0; } 50% { width: 60%; margin-left: 0; } 100% { width: 0; margin-left: 100%; } }

    header { display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; background: linear-gradient(135deg, var(--teal-dark) 0%, var(--teal-mid) 100%); border-bottom: 1px solid var(--border); box-shadow: 0 2px 20px #0d737733; }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .header-right { display: flex; align-items: center; gap: 10px; }
    .logo-box { width: 44px; height: 44px; background: linear-gradient(135deg, var(--teal-main), var(--teal-light)); border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px #0d737755; animation: logoPulse 3s ease-in-out infinite; flex-shrink: 0; }
    @keyframes logoPulse { 0%, 100% { box-shadow: 0 0 15px #0d737755; } 50% { box-shadow: 0 0 30px #14a0a577; } }
    .logo-inf { width: 26px; height: 16px; }
    .header-title h1 { font-size: 1.3rem; font-weight: 800; color: var(--teal-bright); letter-spacing: 0.5px; }
    .header-subtitle { font-size: 11px; color: var(--text-muted); }

    #clear-btn { background: transparent; border: 1px solid var(--border); color: var(--text-muted); padding: 5px 12px; border-radius: 20px; cursor: pointer; font-size: 11px; font-family: inherit; transition: all 0.2s; }
    #clear-btn:hover { background: var(--surface2); color: var(--teal-bright); border-color: var(--teal-main); }

    #status { font-size: 11px; padding: 5px 12px; border-radius: 20px; font-weight: 600; letter-spacing: 0.3px; }
    .connected { background: #0a2e1a; color: #34d399; border: 1px solid #34d39933; }
    .disconnected { background: #2e0a0a; color: #f87171; border: 1px solid #f8717133; }
    .connecting { background: #2e1e0a; color: #fbbf24; border: 1px solid #fbbf2433; }

    #log { flex: 1; overflow-y: auto; padding: 20px; scroll-behavior: smooth; }
    #log::-webkit-scrollbar { width: 4px; }
    #log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .msg-wrap { display: flex; flex-direction: column; margin: 10px 0; animation: fadeUp 0.3s ease; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .msg-wrap.user-wrap { align-items: flex-end; }
    .msg-wrap.bot-wrap { align-items: flex-start; }
    .bot-row { display: flex; align-items: flex-end; gap: 8px; }
    .bot-avatar { width: 30px; height: 30px; background: linear-gradient(135deg, var(--teal-main), var(--teal-light)); border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .msg { padding: 12px 16px; border-radius: 16px; max-width: 78%; line-height: 1.65; word-break: break-word; font-size: 14.5px; position: relative; }
    .user { background: linear-gradient(135deg, var(--teal-main), var(--teal-light)); color: var(--white); font-weight: 500; border-radius: 16px 16px 4px 16px; }
    .bot { background: var(--surface); border: 1px solid var(--border); white-space: pre-wrap; border-radius: 16px 16px 16px 4px; color: var(--text); }
    .bot a { color: var(--teal-bright); text-decoration: none; font-weight: 500; }
    .bot a:hover { text-decoration: underline; }

    .copy-btn { position: absolute; top: 8px; right: 8px; background: var(--surface2); border: 1px solid var(--border); color: var(--text-muted); padding: 3px 8px; border-radius: 8px; cursor: pointer; font-size: 11px; opacity: 0; transition: all 0.2s; font-family: inherit; }
    .msg.bot:hover .copy-btn { opacity: 1; }
    .copy-btn:hover { background: var(--teal-main); color: var(--white); border-color: var(--teal-main); }

    .timestamp { font-size: 10px; color: var(--text-muted); margin-top: 4px; padding: 0 6px; }

    .typing-bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 16px 16px 16px 4px; padding: 14px 18px; display: flex; align-items: center; gap: 6px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; animation: dotBounce 1.2s ease-in-out infinite; }
    .dot:nth-child(1) { background: var(--teal-main); }
    .dot:nth-child(2) { background: var(--teal-light); animation-delay: 0.2s; }
    .dot:nth-child(3) { background: var(--teal-bright); animation-delay: 0.4s; }
    @keyframes dotBounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.4; } 30% { transform: translateY(-8px); opacity: 1; } }

    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; padding-left: 38px; }
    .suggestions button { background: var(--surface); color: var(--teal-bright); border: 1px solid var(--border); padding: 7px 16px; border-radius: 20px; cursor: pointer; font-size: 13px; transition: all 0.2s; font-family: inherit; }
    .suggestions button:hover { background: var(--teal-main); color: var(--white); border-color: var(--teal-main); }

    #row { display: flex; gap: 10px; padding: 16px 20px; background: var(--teal-dark); border-top: 1px solid var(--border); }
    #input { flex: 1; padding: 13px 18px; border-radius: 25px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-size: 15px; font-family: inherit; transition: all 0.2s; }
    #input:focus { outline: none; border-color: var(--teal-main); box-shadow: 0 0 0 3px #0d737722; }
    #input::placeholder { color: var(--text-muted); }
    #input:disabled { opacity: 0.5; }
    #send { padding: 13px 22px; border-radius: 25px; border: none; background: linear-gradient(135deg, var(--teal-main), var(--teal-light)); color: var(--white); font-weight: 700; cursor: pointer; font-size: 15px; font-family: inherit; transition: all 0.2s; }
    #send:hover { transform: scale(1.03); box-shadow: 0 4px 20px #0d737755; }
    #send:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    .toast { position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: var(--teal-main); color: white; padding: 8px 20px; border-radius: 20px; font-size: 13px; font-weight: 600; z-index: 1000; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>

<div id="loading">
  <div class="loading-logo">
    <svg class="infinity-svg" viewBox="0 0 100 60" fill="none">
      <path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="7" stroke-linecap="round" fill="none"/>
      <path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="7" stroke-linecap="round" fill="none"/>
    </svg>
  </div>
  <div class="loading-title">ScrapBot</div>
  <div class="loading-sub">AI-Powered Multi-Domain Assistant</div>
  <div class="loading-bar"><div class="loading-bar-fill"></div></div>
</div>

<div class="toast" id="toast">Copied!</div>

<header>
  <div class="header-left">
    <div class="logo-box">
      <svg class="logo-inf" viewBox="0 0 100 60" fill="none">
        <path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="8" stroke-linecap="round" fill="none"/>
        <path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="8" stroke-linecap="round" fill="none"/>
      </svg>
    </div>
    <div class="header-title">
      <h1>ScrapBot</h1>
      <span class="header-subtitle">Powered by ScrapIt</span>
    </div>
  </div>
  <div class="header-right">
    <button id="clear-btn" onclick="clearChat()">🗑 Clear Chat</button>
    <span id="status" class="connecting">Connecting...</span>
  </div>
</header>

<div id="log"></div>

<div id="row">
  <input id="input" type="text" placeholder="Ask about jobs, food, travel, cars, properties..." autocomplete="off">
  <button id="send">Send ➤</button>
</div>

<script>
  const log = document.getElementById('log');
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  const status = document.getElementById('status');
  const loading = document.getElementById('loading');
  const toast = document.getElementById('toast');
  let ws = null, typingWrap = null, reconnectAttempts = 0;

  setTimeout(() => {
    loading.style.opacity = '0';
    setTimeout(() => loading.style.display = 'none', 600);
  }, 2200);

  function getGreeting() {
    const h = new Date().getHours();
    if (h >= 5 && h < 12) return '🌅 Good Morning';
    if (h >= 12 && h < 17) return '☀️ Good Afternoon';
    if (h >= 17 && h < 21) return '🌆 Good Evening';
    return '🌙 Good Night';
  }

  function getTimestamp() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function setStatus(state, text) {
    status.className = state;
    status.textContent = text;
    input.disabled = false;
    send.disabled = false;
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2000);
  }

  function clearChat() {
    log.innerHTML = '';
    const greeting = `${getGreeting()}! Chat cleared. How can I help you?`;
    add(greeting, 'bot');
    addSuggestions(["Food", "Jobs", "Travel", "Automobiles", "E-Commerce", "Real Estate"]);
    showToast('Chat cleared!');
  }

  function copyText(text) {
    navigator.clipboard.writeText(text).then(() => showToast('Copied!'));
  }

  function makeInfinityAvatar() {
    const div = document.createElement('div');
    div.className = 'bot-avatar';
    div.innerHTML = `<svg width="18" height="11" viewBox="0 0 100 60" fill="none">
      <path d="M50 30 C50 30 35 5 20 5 C9 5 1 13 1 25 C1 37 9 45 20 45 C35 45 50 30 50 30 Z" stroke="white" stroke-width="9" stroke-linecap="round" fill="none"/>
      <path d="M50 30 C50 30 65 5 80 5 C91 5 99 13 99 25 C99 37 91 45 80 45 C65 45 50 30 50 30 Z" stroke="white" stroke-width="9" stroke-linecap="round" fill="none"/>
    </svg>`;
    return div;
  }

  function add(msg, who) {
    removeTyping();
    const wrap = document.createElement('div');
    wrap.className = `msg-wrap ${who === 'user' ? 'user-wrap' : 'bot-wrap'}`;

    if (who === 'bot') {
      const row = document.createElement('div');
      row.className = 'bot-row';
      row.appendChild(makeInfinityAvatar());
      const bubble = document.createElement('div');
      bubble.className = 'msg bot';
      let text = msg;
      text = text.replace(/(https?:\\/\\/[^\\s<]+)/g, '<a href="$1" target="_blank">🔗 View Details</a>');
      bubble.innerHTML = text;
      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.textContent = '📋 Copy';
      copyBtn.onclick = () => copyText(msg);
      bubble.appendChild(copyBtn);
      row.appendChild(bubble);
      wrap.appendChild(row);
    } else {
      const bubble = document.createElement('div');
      bubble.className = 'msg user';
      bubble.textContent = msg;
      wrap.appendChild(bubble);
    }

    const ts = document.createElement('div');
    ts.className = 'timestamp';
    ts.textContent = getTimestamp();
    wrap.appendChild(ts);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  function showTyping() {
    if (typingWrap) return;
    typingWrap = document.createElement('div');
    typingWrap.className = 'msg-wrap bot-wrap';
    const row = document.createElement('div');
    row.className = 'bot-row';
    row.appendChild(makeInfinityAvatar());
    const bubble = document.createElement('div');
    bubble.className = 'typing-bubble';
    bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    row.appendChild(bubble);
    typingWrap.appendChild(row);
    log.appendChild(typingWrap);
    log.scrollTop = log.scrollHeight;
  }

  function removeTyping() {
    if (typingWrap) { typingWrap.remove(); typingWrap = null; }
  }

  function addSuggestions(suggestions) {
    if (!suggestions || !suggestions.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'suggestions';
    suggestions.forEach(s => {
      const btn = document.createElement('button');
      btn.textContent = s;
      btn.onclick = () => { input.value = s; doSend(); };
      wrap.appendChild(btn);
    });
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.onopen = () => { setStatus('connected', '● Online'); reconnectAttempts = 0; input.focus(); };
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'typing') { showTyping(); return; }
      removeTyping();
      if (data.reply) {
        let reply = data.reply;
        if (reply === 'GREETING') {
          reply = `${getGreeting()}! Welcome to ScrapBot\\n\\nI can help you with:\\n🍕 Food - restaurants & delivery\\n💼 Jobs - career opportunities\\n✈️ Travel - trips & destinations\\n🚗 Automobiles - cars & vehicles\\n🛍️ E-Commerce - products & deals\\n🏠 Real Estate - properties for rent & sale\\n\\nHow can I help you today?`;
        }
        add(reply, 'bot');
      }
      if (data.suggestions && data.suggestions.length) addSuggestions(data.suggestions);
      input.disabled = false; send.disabled = false; input.focus();
    };
    ws.onclose = () => {
      setStatus('disconnected', '● Offline');
      if (reconnectAttempts < 5) {
        reconnectAttempts++;
        setStatus('connecting', 'Reconnecting...');
        setTimeout(connectWebSocket, 2000 * reconnectAttempts);
      }
    };
    ws.onerror = () => setStatus('disconnected', '● Error');
  }

  function doSend() {
    const msg = input.value.trim();
    if (!msg || !ws || ws.readyState !== WebSocket.OPEN) return;
    input.value = ''; add(msg, 'user');
    input.disabled = true; send.disabled = true;
    ws.send(JSON.stringify({ message: msg }));
  }

  async function doSendREST() {
    const msg = input.value.trim();
    if (!msg) return;
    input.value = ''; add(msg, 'user');
    input.disabled = true; send.disabled = true; showTyping();
    try {
      const r = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: msg }) });
      const data = await r.json();
      removeTyping(); add(data.reply || 'No response', 'bot');
      addSuggestions(data.suggestions || []);
    } catch (e) { removeTyping(); add('Network error. Try again.', 'bot'); }
    finally { input.disabled = false; send.disabled = false; input.focus(); }
  }

  send.onclick = () => ws && ws.readyState === WebSocket.OPEN ? doSend() : doSendREST();
  input.onkeydown = (e) => { if (e.key === 'Enter') ws && ws.readyState === WebSocket.OPEN ? doSend() : doSendREST(); };
  connectWebSocket();
</script>
</body>
</html>"""
