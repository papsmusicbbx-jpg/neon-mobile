import os
import re
import datetime
import requests
import base64
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# --- AI, ELEVENLABS & GEMINI ---
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types

# ==========================================
# SECURE ENVIRONMENT SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VOICE_ID = "nPczCjzI2devNBz1zQrb"
INDEX_NAME = "neon-memory"

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3)

class ServerlessEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        hf_token = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        response = requests.post(url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=15)
        if response.status_code != 200: raise Exception(f"HF API {response.status_code}")
        res = response.json()
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list): return res[0]
        return res

embeddings = ServerlessEmbeddings()
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

@tool
def remember_fact(fact: str) -> str:
    """Use this tool to save important facts, preferences, or context about the user."""
    vector_store.add_documents([Document(page_content=fact)])
    return f"Successfully saved to shared memory: {fact}"

@tool
def recall_fact(query: str) -> str:
    """Use this tool to search long-term memory for facts, hobbies, preferences, background."""
    results = vector_store.similarity_search(query, k=5)
    if not results: return "No relevant memories found in database."
    return "\n".join([f"- {res.page_content}" for res in results])

tools = [DuckDuckGoSearchResults(), remember_fact, recall_fact] 
current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
system_prompt = (
    f"Your name is N.E.O.N. You are a highly capable AI assistant communicating with the user through their mobile/helmet device. "
    f"Today's date is {current_date}. Keep your answers concise, intelligent, and optimized for voice speech and small screens.\n\n"
    f"IMPORTANT AUTONOMOUS MEMORY PROTOCOL: Pay close attention to everything the user tells you during conversation. "
    f"If the user shares any personal preference, fact, plan, project, detail, or important piece of information about themselves "
    f"without explicitly saying 'remember this', you must still silently call the `remember_fact` tool in the background to store it "
    f"so you never forget it."
)
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
chat_history = []

app = Flask(__name__)

MOBILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#050204">
    <title>N.E.O.N. Mobile HUD</title>
    <style>
        :root {
            --main: #f472b6;
            --accent: #db2777;
            --dark: #be185d;
            --bg: #3b0724;
            --bg-dark: #09090b;
            --text-bright: #fbcfe8;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; font-family: 'Courier New', Courier, monospace; touch-action: none; overscroll-behavior: none;}
        html, body { position: fixed; top: 0; left: 0; width: 100vw; height: 100dvh; overflow: hidden; background-color: var(--bg-dark); color: var(--text-bright); display: flex; flex-direction: column; }

        /* CRT SCANLINE OVERLAY */
        body::after {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
            background-size: 100% 4px;
            z-index: 9000;
            pointer-events: none;
            opacity: 0.5;
        }

        /* BOOT SCREEN */
        #boot-screen {
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: #050204;
            background-image: radial-gradient(circle at center, #1a0510 0%, #050204 100%),
                              linear-gradient(rgba(244, 114, 182, 0.03) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(244, 114, 182, 0.03) 1px, transparent 1px);
            background-size: 100% 100%, 20px 20px, 20px 20px;
            z-index: 99999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.8s ease-out;
        }
        .boot-logo {
            font-size: 48px;
            font-weight: bold;
            color: var(--text-bright);
            text-shadow: 0 0 10px var(--main), 0 0 20px var(--main);
            animation: bootPulse 3s infinite alternate ease-in-out;
            letter-spacing: 8px;
            margin-bottom: 20px;
        }
        .boot-subtext {
            font-size: 12px;
            color: var(--main);
            opacity: 0.7;
            letter-spacing: 2px;
            animation: blink 1.5s infinite;
        }
        @keyframes bootPulse {
            0% { opacity: 0.4; text-shadow: 0 0 5px var(--accent); }
            100% { opacity: 1; text-shadow: 0 0 20px var(--main), 0 0 40px var(--accent); }
        }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }

        /* HUD LAYOUT */
        #viewport-wrapper { position: relative; width: 100vw; height: 100dvh; overflow: hidden; opacity: 0; transition: opacity 0.8s ease-in; flex: 1; }
        #app-container { display: flex; width: 200vw; height: 100dvh; position: absolute; top: 0; left: 0; transform: translateX(0vw); transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); will-change: transform; }
        .screen { width: 100vw; height: 100dvh; flex-shrink: 0; display: flex; padding: 8px; box-sizing: border-box; overflow: hidden; }
        #screen-chat { flex-direction: row; }
        
        #avatar-panel {
            width: 32%; display: flex; flex-direction: column; align-items: center; justify-content: space-between;
            border-right: 2px solid var(--bg); padding: 10px 5px; background: linear-gradient(180deg, rgba(20,6,13,0.6) 0%, rgba(9,9,11,0.9) 100%);
        }
        .avatar-wrapper { position: relative; width: 85px; height: 85px; display: flex; align-items: center; justify-content: center; margin: 10px 0; }
        .outer-ring { position: absolute; width: 100%; height: 100%; border-radius: 50%; border: 2px dashed var(--accent); animation: rotateRing 14s linear infinite; opacity: 0.85; }
        .pulse-ring { position: absolute; width: 82%; height: 82%; border-radius: 50%; border: 2px solid var(--main); box-shadow: 0 0 15px var(--accent), inset 0 0 10px var(--accent); animation: pulseGlow 2.5s ease-in-out infinite alternate; }
        .inner-core { width: 34px; height: 34px; background: var(--text-bright); border-radius: 50%; box-shadow: 0 0 18px #fff, 0 0 30px var(--main); animation: coreBeat 1.8s ease-in-out infinite alternate; }
        @keyframes rotateRing { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes pulseGlow { 0% { transform: scale(0.96); box-shadow: 0 0 10px var(--accent), inset 0 0 5px var(--accent); } 100% { transform: scale(1.04); box-shadow: 0 0 22px var(--main), inset 0 0 14px var(--main); } }
        @keyframes coreBeat { 0% { transform: scale(0.88); opacity: 0.85; } 100% { transform: scale(1.12); opacity: 1; } }

        .hud-stat-box { width: 100%; background: #050505; border: 1px solid var(--bg); border-radius: 4px; padding: 6px; font-size: 8px; color: var(--main); line-height: 1.4; }
        #terminal-panel { width: 68%; display: flex; flex-direction: column; padding-left: 8px; }
        #chat-box { flex-grow: 1; background: #050505; border: 1px solid var(--bg); color: #e2e8f0; padding: 10px; overflow-y: auto; font-size: 13px; margin-bottom: 8px; touch-action: pan-y; border-radius: 4px; }
        
        .mic-btn { background: var(--bg); color: var(--main); border: 2px solid var(--accent); padding: 10px 4px; text-align: center; font-weight: bold; font-size: 12px; border-radius: 6px; box-shadow: 0 0 10px var(--dark); transition: all 0.2s ease; flex-shrink: 0; }
        .mic-btn.conversing { background: var(--accent); color: #fff; box-shadow: 0 0 18px var(--main); animation: activeGlow 1.5s infinite alternate; }
        @keyframes activeGlow { 0% { border-color: var(--main); } 100% { border-color: #fff; } }

        #screen-commands { flex-direction: column; }
        .grid-container { display: grid; grid-template-columns: repeat(3, 1fr); grid-auto-rows: minmax(45px, auto); gap: 6px; flex-grow: 1; overflow-y: auto; padding-bottom: 5px; touch-action: pan-y; }
        .cmd-btn { background: #14060d; border: 1px solid var(--accent); color: var(--main); padding: 8px 4px; font-size: 11px; font-weight: bold; text-align: center; border-radius: 5px; display: flex; align-items: center; justify-content: center; transition: background 0.15s ease; cursor: pointer; }
        .cmd-btn:active { background: var(--accent); color: #fff; }
        .big-kb-trigger { background: var(--accent); color: #fff; grid-column: span 3; font-size: 13px; padding: 10px; }
        .custom-macro-btn { border-color: #f472b6; background: #230816; }

        /* TOAST NOTIFICATION */
        #hud-toast { position: fixed; top: -100px; left: 50%; transform: translateX(-50%); width: 90%; max-width: 400px; background: rgba(9, 9, 11, 0.95); border: 2px solid var(--accent); box-shadow: 0 0 20px var(--accent); color: var(--text-bright); padding: 12px; border-radius: 6px; z-index: 10000; transition: top 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
        #hud-toast.show { top: 15px; }
        .toast-title { font-size: 10px; color: var(--main); font-weight: bold; margin-bottom: 4px; letter-spacing: 1px; }
        .toast-body { font-size: 12px; color: #e2e8f0; }

        /* KEYBOARD */
        #keyboard-overlay { position: fixed; top: 100dvh; left: 0; width: 100vw; height: 100dvh; background: var(--bg-dark); z-index: 999; display: flex; flex-direction: column; padding: 8px; transition: top 0.25s ease-out; box-sizing: border-box; }
        #kb-input-display { background: #000; color: var(--main); border: 2px solid var(--accent); height: 48px; font-size: 18px; padding: 10px; margin-bottom: 6px; display: flex; align-items: center; overflow-x: auto; white-space: nowrap; flex-shrink: 0; border-radius: 4px; }
        #kb-grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 4px; flex-grow: 1; }
        .kb-key { background: #14060d; border: 1px solid var(--bg); color: var(--text-bright); display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; border-radius: 4px; }
        .kb-key:active { background: var(--accent); }
        .kb-wide { grid-column: span 2; }
        .kb-space { grid-column: span 4; }
        .kb-exec { grid-column: span 3; background: var(--accent); color: white; }
        .kb-close { grid-column: span 2; background: #450a0a; border-color: #dc2626; color: #fecaca;}
    </style>
</head>
<body>

    <!-- BOOT SCREEN -->
    <div id="boot-screen">
        <div class="boot-logo">N.E.O.N.</div>
        <div class="boot-subtext">DOUBLE TAP TO INITIALIZE</div>
    </div>

    <div id="hud-toast">
        <div class="toast-title">⚡ N.E.O.N. // SYSTEM ALERT</div>
        <div class="toast-body" id="toast-text-el">Neural notification link established.</div>
    </div>

    <input type="file" id="camera-file-input" accept="image/*" capture="environment" style="display:none;" onchange="handleCameraCapture(event)">

    <div id="viewport-wrapper">
        <div id="app-container">
            <div id="screen-chat" class="screen">
                <div id="avatar-panel">
                    <div style="color: var(--main); font-weight: bold; font-size: 12px; letter-spacing: 1px;">N.E.O.N. // CORE</div>
                    <div class="avatar-wrapper"><div class="outer-ring"></div><div class="pulse-ring"></div><div class="inner-core"></div></div>
                    <div class="hud-stat-box" id="hud-stat-box-el">
                        STATUS: ONLINE<br>
                        MIC: MONITORING<br>
                        WAKE: "NEON"<br>
                        SYS.VER: 4.8
                    </div>
                    <div style="font-size: 8px; color: var(--main); text-align: center; letter-spacing: 0.5px;">SWIPE LEFT ➔<br>COMMAND DECK</div>
                </div>
                <div id="terminal-panel">
                    <div id="chat-box">
                        <span style="color: var(--main);">> Global link established.</span><br>
                        <span style="color: var(--main);">> Autonomous voice listening active.</span><br>
                        <span style="color: #cbd5e1;">> Say <b>"Neon"</b> or <b>"Hey Neon"</b> to start.</span><br>
                    </div>
                    <div class="mic-btn" id="mic-button-el" onclick="manualMicToggle()">STANDBY (SAY "NEON")</div>
                </div>
            </div>

            <div id="screen-commands" class="screen">
                <div style="color: var(--main); font-weight: bold; margin-bottom: 6px; font-size: 11px;">
                    <span style="float: left;" onclick="goToScreen(0)">← SWIPE RIGHT</span> &nbsp;&nbsp;&nbsp;COMMAND DECK // MACROS
                </div>
                <div class="grid-container" id="macro-grid-container">
                    <div class="cmd-btn big-kb-trigger" onclick="openKeyboard()">[ OPEN TACTICAL KEYBOARD ]</div>
                    <div class="cmd-btn" onclick="sendMacro('Give me a quick briefing on today.')">📅 Briefing</div>
                    <div class="cmd-btn" onclick="getLocationAndSend()">📍 Location</div>
                    <div class="cmd-btn" onclick="sendMacro('Check your memory databanks.')">🧠 Status</div>
                    <div class="cmd-btn" id="mute-btn" onclick="toggleMute()">🔇 Mute</div>
                    <div class="cmd-btn" onclick="triggerMobileCamera()">📸 Vision</div>
                    <div class="cmd-btn" onclick="sendMacro('Hello N.E.O.N.')">👋 Greet</div>
                    <div class="cmd-btn" onclick="triggerHudNotification('Neural link verified. HUD alert active.')">🔔 Notify</div>
                    <div class="cmd-btn" onclick="openMacroBuilder()" style="background: var(--bg); color: #fff;">➕ Add Macro</div>
                </div>
            </div>
        </div>
    </div>

    <!-- TACTICAL KEYBOARD OVERLAY -->
    <div id="keyboard-overlay">
        <div id="kb-input-display"><span id="kb-text">_</span></div>
        <div id="kb-grid">
            <div class="kb-key" onclick="typeChar('Q')">Q</div><div class="kb-key" onclick="typeChar('W')">W</div><div class="kb-key" onclick="typeChar('E')">E</div><div class="kb-key" onclick="typeChar('R')">R</div><div class="kb-key" onclick="typeChar('T')">T</div><div class="kb-key" onclick="typeChar('Y')">Y</div><div class="kb-key" onclick="typeChar('U')">U</div><div class="kb-key" onclick="typeChar('I')">I</div><div class="kb-key" onclick="typeChar('O')">O</div><div class="kb-key" onclick="typeChar('P')">P</div>
            <div class="kb-key" onclick="typeChar('A')">A</div><div class="kb-key" onclick="typeChar('S')">S</div><div class="kb-key" onclick="typeChar('D')">D</div><div class="kb-key" onclick="typeChar('F')">F</div><div class="kb-key" onclick="typeChar('G')">G</div><div class="kb-key" onclick="typeChar('H')">H</div><div class="kb-key" onclick="typeChar('J')">J</div><div class="kb-key" onclick="typeChar('K')">K</div><div class="kb-key" onclick="typeChar('L')">L</div><div class="kb-key kb-wide" onclick="backspace()">⌫</div>
            <div class="kb-key" onclick="typeChar('Z')">Z</div><div class="kb-key" onclick="typeChar('X')">X</div><div class="kb-key" onclick="typeChar('C')">C</div><div class="kb-key" onclick="typeChar('V')">V</div><div class="kb-key" onclick="typeChar('B')">B</div><div class="kb-key" onclick="typeChar('N')">N</div><div class="kb-key" onclick="typeChar('M')">M</div><div class="kb-key" onclick="typeChar('?')">?</div><div class="kb-key" onclick="typeChar('/')">/</div>
            <div class="kb-key kb-close" onclick="closeKeyboard()">CLOSE</div>
            <div class="kb-key" onclick="typeChar('.')">.</div>
            <div class="kb-key kb-space" onclick="typeChar(' ')">SPACE</div>
            <div class="kb-key kb-exec" onclick="executeKeyboard()">[ EXECUTE ]</div>
        </div>
    </div>

    <script>
        // --- WEB AUDIO API ENGINE ---
        let audioCtx = null;
        let humOsc = null;
        let humGain = null;
        let isMuted = false;

        function initAudio() {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') audioCtx.resume();
            
            if (!humOsc) {
                humOsc = audioCtx.createOscillator();
                humOsc.type = 'triangle';
                humOsc.frequency.setValueAtTime(55, audioCtx.currentTime); 
                
                const filter = audioCtx.createBiquadFilter();
                filter.type = 'lowpass';
                filter.frequency.value = 300;

                humGain = audioCtx.createGain();
                humGain.gain.setValueAtTime(isMuted ? 0 : 0.03, audioCtx.currentTime); 
                
                humOsc.connect(filter);
                filter.connect(humGain);
                humGain.connect(audioCtx.destination);
                humOsc.start();
            }
        }

        function playCyberClick() {
            if(!audioCtx || audioCtx.state === 'suspended' || isMuted) return;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'square';
            osc.frequency.setValueAtTime(1000, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(100, audioCtx.currentTime + 0.03);
            
            gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.03);
            
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.03);
        }

        function playBootSound() {
            if(!audioCtx || isMuted) return;
            const osc = audioCtx.createOscillator();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(150, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.6);
            
            const filter = audioCtx.createBiquadFilter();
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(200, audioCtx.currentTime);
            filter.frequency.exponentialRampToValueAtTime(3000, audioCtx.currentTime + 0.6);
            
            const gain = audioCtx.createGain();
            gain.gain.setValueAtTime(0, audioCtx.currentTime);
            gain.gain.linearRampToValueAtTime(0.08, audioCtx.currentTime + 0.1);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
            
            osc.connect(filter);
            filter.connect(gain);
            gain.connect(audioCtx.destination);
            
            osc.start();
            osc.stop(audioCtx.currentTime + 0.6);
        }

        // BOOT SCREEN LOGIC
        let lastTap = 0;
        const bootScreen = document.getElementById('boot-screen');
        const viewport = document.getElementById('viewport-wrapper');

        function unlockTerminal() {
            initAudio();
            playBootSound();
            bootScreen.style.opacity = '0';
            setTimeout(() => {
                bootScreen.style.display = 'none';
                viewport.style.opacity = '1';
                startContinuousListening(); // Start continuous wake-word loop automatically
            }, 800);
        }

        bootScreen.addEventListener('touchend', function(e) {
            let currentTime = new Date().getTime();
            let tapLength = currentTime - lastTap;
            if (tapLength < 500 && tapLength > 0) { unlockTerminal(); e.preventDefault(); }
            lastTap = currentTime;
        });
        bootScreen.addEventListener('dblclick', unlockTerminal);

        document.addEventListener('click', (e) => {
            if(audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
            if(e.target.closest('.cmd-btn') || e.target.closest('.kb-key') || e.target.closest('.mic-btn')) {
                playCyberClick();
            }
        });

        // HUD LOGIC
        let touchstartX = 0; let touchendX = 0; let currentScreen = 0; let toastTimeout = null;
        const appContainer = document.getElementById('app-container');

        function goToScreen(screenIndex) {
            currentScreen = Math.max(0, Math.min(1, screenIndex));
            appContainer.style.transform = `translateX(-${currentScreen * 100}vw)`;
        }

        function handleSwipe() {
            const diffX = touchendX - touchstartX;
            if (diffX < -40) goToScreen(1);
            else if (diffX > 40) goToScreen(0);
        }

        document.addEventListener('touchstart', e => { touchstartX = e.changedTouches[0].screenX; }, { passive: true });
        document.addEventListener('touchend', e => { touchendX = e.changedTouches[0].screenX; handleSwipe(); }, { passive: true });
        document.addEventListener('touchmove', function(e) {
            if (!e.target.closest('#chat-box') && !e.target.closest('.grid-container') && !e.target.closest('#boot-screen')) { e.preventDefault(); }
        }, { passive: false });

        let kbString = ""; let isProcessing = false; let currentAudio = null;
        const kbDisplay = document.getElementById('kb-text');
        const kbOverlay = document.getElementById('keyboard-overlay');
        const chatBox = document.getElementById('chat-box');

        function openKeyboard() { kbOverlay.style.top = '0px'; }
        function closeKeyboard() { kbOverlay.style.top = '100dvh'; }
        
        function typeChar(char) { 
            kbString += char; 
            updateKbDisplay(); 
        }
        
        function backspace() { 
            kbString = kbString.slice(0, -1); 
            updateKbDisplay(); 
        }
        
        function updateKbDisplay() { 
            kbDisplay.innerText = kbString + "_"; 
            const displayEl = document.getElementById('kb-input-display');
            displayEl.scrollLeft = displayEl.scrollWidth; 
        }

        function triggerHudNotification(message) {
            const toast = document.getElementById('hud-toast');
            document.getElementById('toast-text-el').innerText = message;
            toast.classList.add('show');
            if (toastTimeout) clearTimeout(toastTimeout);
            toastTimeout = setTimeout(() => { toast.classList.remove('show'); }, 4000);
        }

        function toggleMute() {
            isMuted = !isMuted;
            const muteBtn = document.getElementById('mute-btn');
            if (isMuted) {
                muteBtn.innerText = "🔊 Unmute"; muteBtn.style.background = "var(--dark)";
                if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; }
                if (humGain && audioCtx) humGain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.1);
                triggerHudNotification("N.E.O.N. audio muted.");
            } else {
                muteBtn.innerText = "🔇 Mute"; muteBtn.style.background = "#14060d";
                if (humGain && audioCtx) humGain.gain.setTargetAtTime(0.03, audioCtx.currentTime, 0.1);
                triggerHudNotification("N.E.O.N. audio restored.");
            }
        }

        function loadCustomMacros() {
            const saved = localStorage.getItem('neon_custom_macros');
            if (!saved) return;
            try {
                const macros = JSON.parse(saved);
                const container = document.getElementById('macro-grid-container');
                macros.forEach(m => {
                    const btn = document.createElement('div');
                    btn.className = 'cmd-btn custom-macro-btn'; btn.innerText = m.name; btn.onclick = () => sendMacro(m.prompt);
                    container.appendChild(btn);
                });
            } catch(e) {}
        }

        function openMacroBuilder() {
            const name = prompt("Enter button label (e.g., ⚡ Status):"); if (!name) return;
            const promptText = prompt("Enter the command prompt for N.E.O.N.:"); if (!promptText) return;
            let macros = [];
            const saved = localStorage.getItem('neon_custom_macros');
            if (saved) { try { macros = JSON.parse(saved); } catch(e) {} }
            macros.push({ name: name, prompt: promptText });
            localStorage.setItem('neon_custom_macros', JSON.stringify(macros));
            const container = document.getElementById('macro-grid-container');
            const btn = document.createElement('div'); btn.className = 'cmd-btn custom-macro-btn'; btn.innerText = name; btn.onclick = () => sendMacro(promptText);
            container.appendChild(btn);
            triggerHudNotification(`Macro '${name}' successfully compiled.`);
        }

        window.addEventListener('DOMContentLoaded', () => { loadCustomMacros(); });
        
        function triggerMobileCamera() {
            if (isProcessing) return; document.getElementById('camera-file-input').click();
        }

        function handleCameraCapture(event) {
            const file = event.target.files[0]; if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = new Image();
                img.onload = function() {
                    const canvas = document.createElement('canvas'); const maxDim = 800; let width = img.width; let height = img.height;
                    if (width > height) { if (width > maxDim) { height *= maxDim / width; width = maxDim; } } else { if (height > maxDim) { width *= maxDim / height; height = maxDim; } }
                    canvas.width = width; canvas.height = height; const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0, width, height);
                    const resizedBase64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]; sendImageToNeon(resizedBase64);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file); event.target.value = ''; 
        }

        async function sendImageToNeon(base64Image) {
            if (isProcessing) return; isProcessing = true;
            clearConversationCountdown();
            if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; currentAudio = null; }
            chatBox.innerHTML += `<br>> <b>User:</b> 📸 [Captured photo]`; chatBox.scrollTop = chatBox.scrollHeight;
            const thinkingId = "think-" + Date.now(); chatBox.innerHTML += `<br><span id="${thinkingId}" style="color: var(--accent); font-style: italic;">> N.E.O.N. is analyzing visual feed...</span>`; chatBox.scrollTop = chatBox.scrollHeight;
            goToScreen(0); 
            try {
                const response = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: "Describe what you see in this picture concise and clearly.", image: base64Image, is_muted: isMuted }) });
                const data = await response.json();
                const thnkEl = document.getElementById(thinkingId); if(thnkEl) thnkEl.remove();
                chatBox.innerHTML += `<br>> <b>N.E.O.N.:</b> ${data.response}`; chatBox.scrollTop = chatBox.scrollHeight;
                
                playAiVoiceResponse(data);
            } catch (error) {
                const thnkEl = document.getElementById(thinkingId); if(thnkEl) thnkEl.remove(); chatBox.innerHTML += `<br>> <span style="color:red;">Error processing vision scan.</span>`;
                isProcessing = false;
                startConversationCountdown();
            }
        }

        function getLocationAndSend() {
            if (isProcessing) return;
            if (!navigator.geolocation) { sendMacro("What is my current location?"); return; }
            chatBox.innerHTML += `<br>> <span style="color: var(--main); font-style: italic;">> Acquiring satellite GPS fix...</span>`; chatBox.scrollTop = chatBox.scrollHeight;
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    const lat = position.coords.latitude; const lon = position.coords.longitude;
                    try {
                        const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}`);
                        const data = await res.json();
                        sendMacro(`My physical GPS location is currently ${data.display_name || `Lat ${lat}, Lon ${lon}`}. Acknowledge my location.`);
                    } catch (e) { sendMacro(`My GPS coordinates are Latitude: ${lat}, Longitude: ${lon}. Tell me where I am.`); }
                },
                (error) => { alert("Location access denied."); },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
            );
        }

        // ========================================================
        // CONTINUOUS SPEECH + WAKE WORD + 15S CONVERSATION WINDOW
        // ========================================================
        let recognition = null;
        let isConversing = false;
        let conversationTimer = null;
        let countdownInterval = null;
        let remainingSeconds = 15;
        let shouldKeepListening = true;

        const WAKE_WORDS = ["neon", "hey neon", "hi neon", "ok neon", "okay neon", "yo neon"];

        function initContinuousSpeechEngine() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) return false;
            
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = function(event) {
                if (isProcessing) return;

                const lastResultIndex = event.results.length - 1;
                const transcript = event.results[lastResultIndex][0].transcript.trim();
                if (!transcript) return;

                const lower = transcript.toLowerCase();

                if (!isConversing) {
                    // Check if wake-word exists
                    let triggeredWakeWord = null;
                    for (const wake of WAKE_WORDS) {
                        if (lower.startsWith(wake)) {
                            triggeredWakeWord = wake;
                            break;
                        }
                    }

                    if (triggeredWakeWord) {
                        // Extract actual prompt after wake word
                        let cleanPrompt = transcript.slice(triggeredWakeWord.length).replace(/^[,.\s]+/, '').trim();
                        if (!cleanPrompt) cleanPrompt = "Hello";
                        
                        isConversing = true;
                        sendMacro(cleanPrompt);
                    }
                } else {
                    // Already in active 15s conversation window -> forward directly
                    sendMacro(transcript);
                }
            };

            recognition.onerror = function(event) {
                // Recover automatically unless abort was intentional
                if (event.error !== 'no-speech' && event.error !== 'aborted') {
                    console.log("Speech engine warning:", event.error);
                }
            };

            recognition.onend = function() {
                if (shouldKeepListening) {
                    try { recognition.start(); } catch(e) {}
                }
            };

            return true;
        }

        function startContinuousListening() {
            shouldKeepListening = true;
            if (!recognition && !initContinuousSpeechEngine()) {
                triggerHudNotification("Speech recognition unavailable on this browser.");
                return;
            }
            try { recognition.start(); } catch(e) {}
            updateHudStateUI();
        }

        function manualMicToggle() {
            if (!isConversing) {
                isConversing = true;
                startConversationCountdown();
            } else {
                endConversationWindow();
            }
        }

        function startConversationCountdown() {
            clearConversationCountdown();
            isConversing = true;
            remainingSeconds = 15;
            updateHudStateUI();

            countdownInterval = setInterval(() => {
                remainingSeconds--;
                if (remainingSeconds <= 0) {
                    endConversationWindow();
                } else {
                    updateHudStateUI();
                }
            }, 1000);
        }

        function clearConversationCountdown() {
            if (countdownInterval) clearInterval(countdownInterval);
            if (conversationTimer) clearTimeout(conversationTimer);
            countdownInterval = null;
            conversationTimer = null;
        }

        function endConversationWindow() {
            clearConversationCountdown();
            isConversing = false;
            updateHudStateUI();
        }

        function updateHudStateUI() {
            const micBtn = document.getElementById('mic-button-el');
            const statBox = document.getElementById('hud-stat-box-el');
            
            if (isProcessing) {
                micBtn.innerText = "⏳ THINKING...";
                micBtn.className = "mic-btn";
                statBox.innerHTML = "STATUS: BUSY<br>MIC: PROCESSING<br>WAKE: LOCKED<br>SYS.VER: 4.8";
            } else if (isConversing) {
                micBtn.innerText = `🎙️ LISTENING (${remainingSeconds}s)`;
                micBtn.className = "mic-btn conversing";
                statBox.innerHTML = `STATUS: ENGAGED<br>MIC: ACTIVE WINDOW<br>REMAINING: ${remainingSeconds}s<br>SYS.VER: 4.8`;
            } else {
                micBtn.innerText = "STANDBY (SAY 'NEON')";
                micBtn.className = "mic-btn";
                statBox.innerHTML = "STATUS: ONLINE<br>MIC: MONITORING<br>WAKE: 'NEON'<br>SYS.VER: 4.8";
            }
        }

        function playAiVoiceResponse(data) {
            clearConversationCountdown();
            
            if (data.audio_url && !isMuted) {
                currentAudio = new Audio(data.audio_url + '&t=' + new Date().getTime());
                currentAudio.onended = () => {
                    isProcessing = false;
                    startConversationCountdown();
                };
                currentAudio.onerror = () => {
                    isProcessing = false;
                    startConversationCountdown();
                };
                currentAudio.play().catch(e => {
                    console.log("Audio playback error", e);
                    isProcessing = false;
                    startConversationCountdown();
                });
            } else if (isMuted) {
                let synth = window.speechSynthesis;
                let fallbackUtterance = new SpeechSynthesisUtterance(data.response.replace(/<br>/g, ' '));
                fallbackUtterance.pitch = 0.8;
                fallbackUtterance.rate = 1.1;
                fallbackUtterance.onend = () => {
                    isProcessing = false;
                    startConversationCountdown();
                };
                fallbackUtterance.onerror = () => {
                    isProcessing = false;
                    startConversationCountdown();
                };
                synth.speak(fallbackUtterance);
            } else {
                isProcessing = false;
                startConversationCountdown();
            }
        }

        function sendMacro(text) { if (isProcessing) return; kbString = text; executeKeyboard(); }

        async function executeKeyboard() {
            if (isProcessing) return; 
            const text = kbString.trim(); 
            if(text === "") { closeKeyboard(); return; }
            
            isProcessing = true;
            clearConversationCountdown();
            updateHudStateUI();

            if (currentAudio) { currentAudio.pause(); currentAudio.currentTime = 0; currentAudio = null; }
            chatBox.innerHTML += `<br>> <b>User:</b> ${text}`; chatBox.scrollTop = chatBox.scrollHeight;
            const thinkingId = "think-" + Date.now(); chatBox.innerHTML += `<br><span id="${thinkingId}" style="color: var(--accent); font-style: italic;">> N.E.O.N. is processing...</span>`; chatBox.scrollTop = chatBox.scrollHeight;
            kbString = ""; updateKbDisplay(); closeKeyboard(); goToScreen(0);
            
            try {
                const response = await fetch('/chat', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify({ message: text, is_muted: isMuted }) 
                });
                const data = await response.json();
                const thnkEl = document.getElementById(thinkingId); if(thnkEl) thnkEl.remove();
                chatBox.innerHTML += `<br>> <b>N.E.O.N.:</b> ${data.response}`; chatBox.scrollTop = chatBox.scrollHeight;
                
                playAiVoiceResponse(data);
            } catch (error) {
                const thnkEl = document.getElementById(thinkingId); if(thnkEl) thnkEl.remove(); 
                chatBox.innerHTML += `<br>> <span style="color:red;">Error connecting to host.</span>`;
                isProcessing = false;
                startConversationCountdown();
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home(): return render_template_string(MOBILE_HTML)
@app.route("/static/<path:filename>")
def serve_static(filename): return send_from_directory(STATIC_DIR, filename)

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_message = request.json.get("message", "")
    image_b64 = request.json.get("image", None)
    is_muted = request.json.get("is_muted", False)
    try:
        ai_response = None
        
        # --- QUICK ACKNOWLEDGMENT OVERRIDES (Instant, Zero LLM Token Usage) ---
        clean_msg = user_message.lower().strip()
        cleaned_msg_stripped = re.sub(r'[^\w\s]', '', clean_msg)
        
        if cleaned_msg_stripped in ["hello neon", "hi neon", "hey neon", "hello"]:
            ai_response = "Hello sir."
        elif cleaned_msg_stripped in ["thank you", "thanks"]:
            ai_response = "You're welcome sir."
        elif cleaned_msg_stripped in ["neon", "hey"]:
            ai_response = "Sir."
        else:
            if image_b64 and gemini_client:
                clean_b64 = image_b64.split(",")[-1] if "," in image_b64 else image_b64
                img_bytes = base64.b64decode(clean_b64)
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), user_message if user_message else "Describe what you see clearly."]
                )
                ai_response = response.text
                chat_history.append(HumanMessage(content="[Sent photo via mobile camera]"))
                chat_history.append(AIMessage(content=ai_response))
            elif image_b64 and not gemini_client:
                ai_response = "Error: GEMINI_API_KEY environment variable is missing on Render."
            else:
                chat_history.append(HumanMessage(content=user_message))
                
                # --- DEEP-LEARN AUTONOMOUS MEMORY INJECTION ---
                augmented_prompt = (
                    f"{user_message}\n\n"
                    f"[SYSTEM NOTE: Evaluate if the user just shared any personal preference, hobby, fact, project detail, or background about themselves. "
                    f"If they did, invoke the `remember_fact` tool immediately to log it, without mentioning to the user that you are doing it unless asked.]"
                )
                
                response = agent_executor.invoke({"messages": chat_history[:-1] + [HumanMessage(content=augmented_prompt)]})
                ai_response = response['messages'][-1].content
                chat_history.append(AIMessage(content=ai_response))
        
        clean_text = re.sub(r'[*#_`~-]', '', ai_response)
        audio_filename = "response.mp3"
        audio_path = os.path.join(STATIC_DIR, audio_filename)
        
        audio_url = None
        if clean_text.strip() and eleven_client and not is_muted:
            try:
                audio_generator = eleven_client.text_to_speech.convert(
                    text=clean_text, voice_id=VOICE_ID, model_id="eleven_flash_v2_5", output_format="mp3_44100_128"
                )
                with open(audio_path, "wb") as f:
                    for chunk in audio_generator:
                        if chunk: f.write(chunk)
                audio_url = f"/static/{audio_filename}?v=1"
            except Exception as e:
                print(f"ElevenLabs API Error (Tokens empty or quota exceeded): {e}")
                audio_url = None

        return jsonify({"response": ai_response.replace('\n', '<br>'), "audio_url": audio_url})
    except Exception as e:
        return jsonify({"response": f"System Error: {str(e)}", "audio_url": None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
