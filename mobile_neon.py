import os
import requests
import uuid
from flask import Flask, render_template_string, request, jsonify
from langchain_groq import ChatGroq
from pinecone import Pinecone

app = Flask(__name__)

# Load API Keys from Render Environment Variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone Index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("neon-memory")

# Initialize Groq LLM
llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# Serverless API Embedding function (0 MB local RAM usage)
def get_embedding(text):
    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    response = requests.post(api_url, json={"inputs": text, "options": {"wait_for_model": True}})
    res = response.json()
    if isinstance(res, list) and isinstance(res[0], list):
        return res[0]
    return res

def recall_memory(query_text):
    try:
        vector = get_embedding(query_text)
        results = index.query(vector=vector, top_k=3, include_metadata=True)
        memories = [match['metadata']['text'] for match in results.get('matches', []) if 'metadata' in match]
        return "\n".join(memories) if memories else "No relevant stored memories."
    except Exception as e:
        return f"Memory recall error: {str(e)}"

def store_memory(text_to_remember):
    try:
        vector = get_embedding(text_to_remember)
        index.upsert(vectors=[(str(uuid.uuid4()), vector, {"text": text_to_remember})])
        return "Memory successfully stored in Pinecone."
    except Exception as e:
        return f"Memory storage error: {str(e)}"

# Cyberpunk Mobile HUD Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>N.E.O.N. Mobile</title>
    <style>
        body { background-color: #0d0d0d; color: #00ffcc; font-family: monospace; padding: 15px; margin: 0; }
        #chat { height: 70vh; overflow-y: auto; border: 1px solid #00ffcc; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .msg { margin: 8px 0; }
        .user { color: #ff007f; }
        .agent { color: #00ffcc; }
        input { width: 70%; padding: 10px; background: #1a1a1a; border: 1px solid #00ffcc; color: white; border-radius: 5px; }
        button { width: 25%; padding: 10px; background: #00ffcc; border: none; color: black; font-weight: bold; border-radius: 5px; }
    </style>
</head>
<body>
    <h3>N.E.O.N. MOBILE LINK</h3>
    <div id="chat"></div>
    <input type="text" id="user-input" placeholder="Communicate...">
    <button onclick="sendMessage()">SEND</button>

    <script>
        async function sendMessage() {
            const input = document.getElementById("user-input");
            const chat = document.getElementById("chat");
            const text = input.value.trim();
            if(!text) return;

            chat.innerHTML += `<div class="msg user">> ${text}</div>`;
            input.value = "";
            chat.scrollTop = chat.scrollHeight;

            const res = await fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: text})
            });
            const data = await res.json();
            chat.innerHTML += `<div class="msg agent">NEON: ${data.response}</div>`;
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    
    context = recall_memory(user_msg)
    
    prompt = f"""You are N.E.O.N., a concise cyberpunk AI assistant.
Relevant Memories: {context}

User: {user_msg}
N.E.O.N.:"""

    response = llm.invoke(prompt)
    bot_reply = response.content

    if "remember" in user_msg.lower():
        store_memory(user_msg)

    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
