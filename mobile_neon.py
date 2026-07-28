import os
import re
import datetime
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# --- AI & ELEVENLABS ---
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from elevenlabs.client import ElevenLabs

# ==========================================
# CREDENTIALS & SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ELEVENLABS_API_KEY = "sk_024866df16c46d30259cab9fd01f163ef9dd57a54b63614f"
VOICE_ID = "nPczCjzI2devNBz1zQrb" # Rachel

eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3)
tools = [DuckDuckGoSearchResults()] 

current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
system_prompt = (
    f"Your name is N.E.O.N. You are a highly capable AI assistant communicating with the user through their mobile device. "
    f"Today's date is {current_date}. "
    f"Keep your answers concise, intelligent, and optimized for reading on a small phone screen."
)

agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
chat_history = []

# ==========================================
# WEB SERVER SETUP (FLASK)
# ==========================================
app = Flask(__name__)

MOBILE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>N.E.O.N. Mobile</title>
    <style>
        body {
            background-color: #09090b;
            color: #e2e8f0;
            font-family: 'Courier New', Courier, monospace;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #header {
            background-color: #14060d;
            color: #f472b6;
            padding: 15px;
            text-align: center;
            font-weight: bold;
            border-bottom: 2px solid #db2777;
            box-shadow: 0 4px 10px rgba(219, 39, 119, 0.2);
        }
        #chatbox {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .msg { padding: 10px 14px; border-radius: 8px; max-width: 85%; word-wrap: break-word; }
        .user-msg { background-color: #3b0724; color: #fbcfe8; align-self: flex-end; border: 1px solid #be185d; }
        .ai-msg { background-color: #050505; color: #e2e8f0; align-self: flex-start; border: 1px solid #3b0724; }
        #input-area {
            display: flex;
            padding: 10px;
            background-color: #14060d;
            border-top: 1px solid #3b0724;
        }
        #user-input {
            flex-grow: 1;
            background-color: #050505;
            color: #f8fafc;
            border: 1px solid #be185d;
            padding: 12px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
        }
        #user-input:focus { outline: none; border-color: #f472b6; }
        #send-btn {
            background-color: #db2777;
            color: white;
            border: none;
            padding: 0 20px;
            margin-left: 10px;
            border-radius: 4px;
            font-weight: bold;
            font-family: 'Courier New', Courier, monospace;
        }
        #send-btn:active { background-color: #be185d; }
        .typing { color: #f472b6; font-style: italic; font-size: 0.9em; align-self: flex-start; display: none;}
    </style>
</head>
<body>
    <div id="header">⚡ N.E.O.N. // MOBILE LINK</div>
    <div id="chatbox">
        <div class="msg ai-msg">Mobile link established. N.E.O.N. online.</div>
    </div>
    <div id="typing-indicator" class="typing">N.E.O.N. is processing...</div>
    <div id="input-area">
        <input type="text" id="user-input" placeholder="Enter command..." onkeypress="handleKeyPress(event)">
        <button id="send-btn" onclick="sendMessage()">EXEC</button>
    </div>

    <script>
        const chatbox = document.getElementById('chatbox');
        const input = document.getElementById('user-input');
        const typingIndicator = document.getElementById('typing-indicator');

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;

            chatbox.innerHTML += `<div class="msg user-msg">${text}</div>`;
            input.value = '';
            chatbox.scrollTop = chatbox.scrollHeight;
            typingIndicator.style.display = 'block';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await response.json();
                
                typingIndicator.style.display = 'none';
                chatbox.innerHTML += `<div class="msg ai-msg">${data.response}</div>`;
                chatbox.scrollTop = chatbox.scrollHeight;

                // Play Rachel's voice directly on the phone
                if (data.audio_url) {
                    const audio = new Audio(data.audio_url + '&t=' + new Date().getTime());
                    audio.play().catch(e => console.log("Audio autoplay blocked:", e));
                }
            } catch (error) {
                typingIndicator.style.display = 'none';
                chatbox.innerHTML += `<div class="msg ai-msg" style="color:red;">Error connecting to host.</div>`;
            }
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
        
        # Generate ElevenLabs audio file for the phone
        clean_text = re.sub(r'[*#_`~-]', '', ai_response)
        audio_filename = "response.mp3"
        audio_path = os.path.join(STATIC_DIR, audio_filename)
        
        if clean_text.strip():
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
    print("========================================")
    print(" N.E.O.N. MOBILE SERVER + VOICE ACTIVE")
    print("========================================")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
