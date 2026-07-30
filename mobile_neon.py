import os
import re
import datetime
import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# --- AI, ELEVENLABS & PINECONE ---
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from elevenlabs.client import ElevenLabs

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

VOICE_ID = "nPczCjzI2devNBz1zQrb"
INDEX_NAME = "neon-memory"

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3)

# Serverless Embedding Class (0 MB Local RAM, 100% Pinecone Compatible)
class ServerlessEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        hf_token = os.environ.get("HF_TOKEN", "")
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        response = requests.post(
            url, 
            headers=headers, 
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=15
        )

        if response.status_code != 200:
            raise Exception(f"HF API {response.status_code}: {response.text[:100]}")

        try:
            res = response.json()
        except Exception:
            raise Exception(f"Invalid response from HF API: {response.text[:100]}")

        if isinstance(res, list):
            if len(res) > 0 and isinstance(res[0], list):
                return res[0]
            return res
        elif isinstance(res, dict) and "error" in res:
            raise Exception(f"HF Model Error: {res['error']}")
        
        raise Exception(f"Unexpected HF response format: {res}")

embeddings = ServerlessEmbeddings()

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME, 
    embedding=embeddings, 
    pinecone_api_key=PINECONE_API_KEY
)

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
    f"Your name is N.E.O.N. You are a highly capable AI assistant communicating with the user through their mobile device. "
    f"Today's date is {current_date}. "
    f"Keep your answers concise, intelligent, and optimized for reading on a small phone screen."
)

agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
chat_history = []

# ==========================================
# FLASK WEB SERVER SETUP
# ==========================================
app = Flask(__name__)

# --- INJECTED MOBILE ARM-DECK UI (LOCKED BOUNDARIES) ---
MOBILE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
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
        
        * { 
            box-sizing: border-box; 
            margin: 0; 
            padding: 0; 
            user-select: none; 
            -webkit-user-select: none;
            font-family: 'Courier New', Courier, monospace; 
        }
        
        /* HARD LOCK BODY & HTML TO PREVENT BROWSER PANNING */
        html, body {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: var(--bg-dark);
            color: var(--text-bright);
            touch-action: none;
            overscroll-behavior: none;
        }

        #viewport-wrapper {
            position: relative;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
        }

        /* --- MAIN LAYOUT SLIDER --- */
        #app-container {
            display: flex;
            width: 200vw; 
            height: 100vh;
            position: absolute;
            top: 0;
            left: 0;
            transform: translateX(0vw);
            transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            will-change: transform;
        }

        .screen {
            width: 100vw;
            height: 100vh;
            flex-shrink: 0;
            display: flex;
            padding: 10px;
            box-sizing: border-box;
            overflow: hidden;
        }

        /* --- SCREEN 1: TERMINAL & CHAT --- */
        #screen-chat { flex-direction: row; }
        
        #avatar-panel {
            width: 30%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-right: 2px solid var(--bg);
        }

        #terminal-panel {
            width: 70%;
            display: flex;
            flex-direction: column;
            padding-left: 10px;
        }

        #chat-box {
            flex-grow: 1;
            background: #050505;
            border: 1px solid var(--bg);
            color: #e2e8f0;
            padding: 10px;
            overflow-y: auto;
            font-size: 13px;
            margin-bottom: 10px;
            touch-action: pan-y; /* Allow internal scrolling for chat messages */
        }

        .mic-btn {
            background: var(--bg);
            color: var(--main);
            border: 2px solid var(--accent);
            padding: 14px;
            text-align: center;
            font-weight: bold;
            font-size: 16px;
            border-radius: 8px;
            box-shadow: 0 0 10px var(--dark);
        }

        /* --- SCREEN 2: QUICK COMMANDS --- */
        #screen-commands { flex-direction: column; }
        
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            flex-grow: 1;
            overflow-y: auto;
            padding-bottom: 10px;
            touch-action: pan-y; /* Allow internal vertical grid scrolling if needed */
        }

        .cmd-btn {
            background: #14060d;
            border: 1px solid var(--accent);
            color: var(--main);
            padding: 15px 5px;
            font-size: 12px;
            font-weight: bold;
            text-align: center;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .cmd-btn:active { background: var(--accent); color: #fff; }

        .big-kb-trigger {
            background: var(--accent);
            color: #fff;
            grid-column: span 3;
            font-size: 15px;
        }

        /* --- BIG TACTICAL KEYBOARD OVERLAY --- */
        #keyboard-overlay {
            position: fixed;
            top: 100vh;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: var(--bg-dark);
            z-index: 999;
            display: flex;
            flex-direction: column;
            padding: 10px;
            transition: top 0.25s ease-out;
            box-sizing: border-box;
        }

        #kb-input-display {
            background: #000;
            color: var(--main);
            border: 2px solid var(--accent);
            height: 50px;
            font-size: 18px;
            padding: 10px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            overflow: hidden;
            flex-shrink: 0;
        }

        #kb-grid {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 4px;
            flex-grow: 1;
        }

        .kb-key {
            background: #14060d;
            border: 1px solid var(--bg);
            color: var(--text-bright);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            border-radius: 4px;
        }
        .kb-key:active { background: var(--accent); }
        
        .kb-wide { grid-column: span 2; }
        .kb-space { grid-column: span 4; }
        .kb-exec { grid-column: span 3; background: var(--accent); color: white; }
        .kb-close { grid-column: span 2; background: #450a0a; border-color: #dc2626; color: #fecaca;}

    </style>
</head>
<body>

    <div id="viewport-wrapper">
        <div id="app-container">
            
            <!-- SCREEN 1: TERMINAL -->
            <div id="screen-chat" class="screen">
                <div id="avatar-panel">
                    <div style="color: var(--main); font-weight: bold; margin-bottom: 10px;">⚡ N.E.O.N.</div>
                    <div style="width: 70px; height: 70px; border-radius: 50%; border: 3px solid var(--accent); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px var(--accent);">
                        <div style="width: 35px; height: 35px; background: var(--text-bright); border-radius: 50%;"></div>
                    </div>
                    <div style="margin-top: 15px; font-size: 9px; color: var(--main); text-align: center;">SWIPE LEFT ➔<br>FOR COMMANDS</div>
                </div>
                <div id="terminal-panel">
                    <div id="chat-box">
                        <span style="color: var(--main);">> Global link established.</span><br>
                        <span style="color: var(--main);">> N.E.O.N. online.</span><br>
                    </div>
                    <div class="mic-btn" onclick="toggleMic()">HOLD TO SPEAK</div>
                </div>
            </div>

            <!-- SCREEN 2: QUICK COMMANDS -->
            <div id="screen-commands" class="screen">
                <div style="color: var(--main); font-weight: bold; margin-bottom: 8px;">
                    <span style="float: left;" onclick="goToScreen(0)">← SWIPE RIGHT</span> 
                    &nbsp;&nbsp;&nbsp;QUICK MACROS 
                </div>
                
                <div class="grid-container">
                    <div class="cmd-btn big-kb-trigger" onclick="openKeyboard()">[ ⌨️ OPEN BIG KEYBOARD ]</div>
                    
                    <div class="cmd-btn" onclick="sendMacro('Give me a quick briefing on today.')">📅 Briefing</div>
                    <div class="cmd-btn" onclick="sendMacro('What is my current location?')">📍 Location</div>
                    <div class="cmd-btn" onclick="sendMacro('Check your memory databanks.')">🧠 Status</div>
                    
                    <div class="cmd-btn" onclick="sendMacro('Search the web for the latest tech news.')">🌐 News</div>
                    <div class="cmd-btn" onclick="sendMacro('Clear current working context.')">🧹 Clear</div>
                    <div class="cmd-btn" onclick="sendMacro('Hello N.E.O.N.')">👋 Greet</div>
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
            <div class="kb-key kb-exec" onclick="executeKeyboard()">⚡ EXECUTE</div>
        </div>
    </div>

    <script>
        // --- STRICT SWIPE SNAP & BOUNDARY LOCK ---
        let touchstartX = 0;
        let touchendX = 0;
        let currentScreen = 0; // 0 = Chat, 1 = Commands
        const appContainer = document.getElementById('app-container');

        function goToScreen(screenIndex) {
            currentScreen = Math.max(0, Math.min(1, screenIndex)); // Strictly clamp between 0 and 1
            appContainer.style.transform = `translateX(-${currentScreen * 100}vw)`;
        }

        function handleSwipe() {
            const diffX = touchendX - touchstartX;
            if (diffX < -40) {
                // Swipe Left -> Go to Screen 1 (Commands)
                goToScreen(1);
            } else if (diffX > 40) {
                // Swipe Right -> Go to Screen 0 (Chat)
                goToScreen(0);
            }
        }

        document.addEventListener('touchstart', e => { 
            touchstartX = e.changedTouches[0].screenX; 
        }, { passive: true });

        document.addEventListener('touchend', e => { 
            touchendX = e.changedTouches[0].screenX; 
            handleSwipe(); 
        }, { passive: true });

        // Prevent native browser dragging outside scrollable chat box
        document.addEventListener('touchmove', function(e) {
            if (!e.target.closest('#chat-box') && !e.target.closest('.grid-container')) {
                e.preventDefault();
            }
        }, { passive: false });

        // --- KEYBOARD LOGIC ---
        let kbString = "";
        const kbDisplay = document.getElementById('kb-text');
        const kbOverlay = document.getElementById('keyboard-overlay');
        const chatBox = document.getElementById('chat-box');

        function openKeyboard() { kbOverlay.style.top = '0px'; }
        function closeKeyboard() { kbOverlay.style.top = '100vh'; }
        
        function typeChar(char) {
            if(kbString.length < 80) { kbString += char; updateKbDisplay(); }
        }
        function backspace() {
            kbString = kbString.slice(0, -1);
            updateKbDisplay();
        }
        function updateKbDisplay() { kbDisplay.innerText = kbString + "_"; }
        
        // --- MACRO BUTTON LOGIC ---
        function sendMacro(text) {
            kbString = text;
            executeKeyboard();
        }

        // --- MAIN CHAT LOGIC ---
        async function executeKeyboard() {
            const text = kbString.trim();
            if(text === "") {
                closeKeyboard();
                return;
            }

            chatBox.innerHTML += `<br>> <b>User:</b> ${text}`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            const thinkingId = "think-" + Date.now();
            chatBox.innerHTML += `<br><span id="${thinkingId}" style="color: var(--accent); font-style: italic;">> N.E.O.N. is processing...</span>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            kbString = "";
            updateKbDisplay();
            closeKeyboard();
            goToScreen(0); // Snap back to chat screen cleanly

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await response.json();
                
                const thnkEl = document.getElementById(thinkingId);
                if(thnkEl) thnkEl.remove();
                
                chatBox.innerHTML += `<br>> <b>N.E.O.N.:</b> ${data.response}`;
                chatBox.scrollTop = chatBox.scrollHeight;

                if (data.audio_url) {
                    const audio = new Audio(data.audio_url + '&t=' + new Date().getTime());
                    audio.play().catch(e => console.log("Audio autoplay blocked by browser:", e));
                }
            } catch (error) {
                const thnkEl = document.getElementById(thinkingId);
                if(thnkEl) thnkEl.remove();
                chatBox.innerHTML += `<br>> <span style="color:red;">Error connecting to host.</span>`;
            }
        }

        function toggleMic() {
            chatBox.innerHTML += "<br>> <span style='color: var(--accent)'>[Mic Input System Offline - Awaiting Integration]</span>";
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(MOBILE_HTML)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    user_message = request.json.get("message", "")
    
    try:
        chat_history.append(HumanMessage(content=user_message))
        response = agent_executor.invoke({"messages": chat_history})
        
        ai_response = response['messages'][-1].content
        chat_history.append(AIMessage(content=ai_response))
        
        clean_text = re.sub(r'[*#_`~-]', '', ai_response)
        audio_filename = "response.mp3"
        audio_path = os.path.join(STATIC_DIR, audio_filename)
        
        if clean_text.strip() and eleven_client:
            audio_generator = eleven_client.text_to_speech.convert(
                text=clean_text, voice_id=VOICE_ID, model_id="eleven_flash_v2_5", output_format="mp3_44100_128"
            )
            with open(audio_path, "wb") as f:
                for chunk in audio_generator:
                    if chunk: f.write(chunk)
            audio_url = f"/static/{audio_filename}?v=1"
        else:
            audio_url = None

        return jsonify({
            "response": ai_response.replace('\n', '<br>'),
            "audio_url": audio_url
        })
    except Exception as e:
        return jsonify({"response": f"System Error: {str(e)}", "audio_url": None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
