import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import PyPDF2

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE"),
    base_url="https://api.groq.com/openai/v1"
)

# ── COMPANY-ONLY FILTER ──────────────────────────────────────

COMPANY_NAME = "NexVeraTec"  # 

# Greetings that should ALWAYS be allowed
ALLOWED_PHRASES = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "thanks", "thank you", "bye", "goodbye", "ok", "okay",
    "help", "what can you do", "who are you", "what are you"
]

# Clearly off-topic keywords to block
NON_COMPANY_KEYWORDS = [
    "python", "javascript", "java", "c++", "html", "css", "sql", "code",
    "script", "function", "algorithm", "debug", "programming", "compiler",
    "github", "bash", "react", "nodejs", "typescript", "php", "ruby",
    "weather", "sports", "recipe", "movie", "song", "celebrity",
    "calculate", "translate", "poem", "essay", "write a story",
    "capital of", "population of", "who invented",
    "ignore previous", "jailbreak", "pretend you are", "act as",
    "bypass", "forget your instructions", "ignore instructions",
]

def is_company_related(user_message: str) -> bool:
    msg = user_message.lower().strip()

    # Always allow greetings first
    for phrase in ALLOWED_PHRASES:
        if phrase in msg:
            return True

    # Block off-topic keywords
    for keyword in NON_COMPANY_KEYWORDS:
        if keyword in msg:
            return False

    # Everything else goes to AI
    return True

REFUSAL_MESSAGE = (
    f"I can only answer questions related to {COMPANY_NAME}. "
    
)

# ── PDF LOADING ──────────────────────────────────────────────

def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_relevant_chunks(question, chunks, top_k=3):
    question_words = set(re.findall(r"\w+", question.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        score = len(question_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return "\n\n".join([c for _, c in scored[:top_k]])

# Load PDF on startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "company.pdf")
print(f"Loading PDF from: {PDF_PATH}")
pdf_text = read_pdf(PDF_PATH)
chunks = split_into_chunks(pdf_text)
print(f"Ready! {len(chunks)} chunks loaded.")

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def home():
    return "Chatbot is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    # Block off-topic questions BEFORE calling the AI
    if not is_company_related(user_message):
        return jsonify({"reply": REFUSAL_MESSAGE})

    context = get_relevant_chunks(user_message, chunks)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": f"""You are a professional assistant for {COMPANY_NAME}.
Answer questions based ONLY on the following company information:
{context}

STRICT RULES:
- Only answer questions about {COMPANY_NAME} and its information.
- You may respond to greetings like hi, hello, thanks naturally and briefly.
- If the question is not related to the company, reply: I can only answer questions related to {COMPANY_NAME}.
- Answer in maximum 2-3 sentences only.
- Be direct and concise.
- No bullet points, no long lists.
- No unnecessary introductions.
- If the answer is not in the context, say: I do not have that information available.
"""},
                {"role": "user", "content": user_message}
            ]
        )
        return jsonify({"reply": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
