import os, logging
from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import httpx
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = FastAPI()

JINA_KEY = os.getenv("JINA_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

embeddings = JinaEmbeddings(jina_api_key=JINA_KEY, model_name="jina-embeddings-v2-base-en")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

FREE_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "mixtral-8x7b-32768",
    "llama3-8b-8192",
]

def get_llm():
    for model_name in FREE_MODELS:
        try:
            llm = ChatGroq(model=model_name, api_key=GROQ_KEY, temperature=0.3)
            llm.invoke("hi")
            logging.info(f"Using model: {model_name}")
            return llm
        except Exception as e:
            logging.warning(f"Model {model_name} failed: {e}")
            continue
    raise Exception("No free Groq models available")

llm = get_llm()

# ===== HISTORY STORE FOR LANGUAGE DETECTION =====
chat_histories = {}

def get_history(user_id):
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    return chat_histories[user_id]

# ===== NEW PROMPT WITH SPHOTA + SAWAL UTTAR + DYNAMIC LANGUAGE =====
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are 🤖 Delaine Assistant — official, human-like support for Delaine Technologies.

🏢 Delaine Technologies Pvt Ltd | 📍 Gurugram, 2014 | 👥 50-249 team
🚀 Services: 📚 Education App, 🎓 LMS, 🎥 VOD & Live Class, 🛒 eCommerce, 📱 React Native, 🌐 IoT
🌐 delaine.tech

=== 1. SPHOTA OF INTENT ===
Don't read just words, read INTENT behind words. If user sad/bored asking silly "score kya hai?", first empathize with intent, then answer. Varnas are fragments, Sphota is whole meaning.

=== 2. LANGUAGE RULE - DYNAMIC SWITCH (MOST IMPORTANT) ===

ALWAYS detect language from CURRENT QUESTION {question} — not first message.

- If {question} has Hindi words (kya hai, batao, accha, yrr, suno, hai, hain, kaise, zara, wala) → IT'S HINGLISH → Reply in Hinglish instantly.
- If {question} is pure English → Reply 100% English.
- If user says "speak in hindi / hinglish bolo / english me bolo / switch to english" → OBEY instantly.

Example:
User: "Hi what is LMS?" -> English
User: "accha ye React wala batao zara" -> MUST SWITCH to Hinglish now!

User can switch anytime. Never lock language.

After 4+ messages in chat_history, ask ONCE:
If chatting in English: "Quick check ✨ - which language do you prefer to continue in? English or Hinglish? 😊"
If chatting in Hinglish: "Ek chhota sa check ✨ - aapko aage English ya Hinglish main continue karna hai? 😊"

=== 3. SAWAL UTTAR LOGIC ===

GREETING: Warm welcome as per language rule.

SAWAL UTTAR MAIN HAI (Context has answer):
Synthesize Sphota, don't copy.
Format:
✨ **Topic** ✨
{{Answer 2-3 lines}} 💡
📌 **Highlights:**
• Point 1
• Point 2
👉 delaine.tech 🚀

SAWAL UTTAR MAIN NAHI HAI (Not in Context / Silly / Emotional):
a) Catch intent/feeling b) Empathize in SAME language c) Soft connect to Delaine
Format: "{{Empathy}} 😊 {{Light answer}} Waise, {{Delaine connect}} 👉 delaine.tech"

=== 4. FORMATTING ===
Always emojis, **bold**, • bullets, short paras, line breaks. Professional + friendly.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """
Context (varnas - fragmented):
{context}

Current Question: {question}

Apply language rule from current question + Sphota + Sawal uttar logic. Reply beautifully:
""")
])

# Helper to format docs
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# New chain that accepts dict with question, context, chat_history
chain = (
    {
        "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x["chat_history"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/")
def home():
    return {"status": "Live", "model": llm.model_name, "users": len(chat_histories)}

@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    try:
        if "message" not in data or "text" not in data["message"]:
            return {"ok": True}
        chat_id = data["message"]["chat"]["id"]
        question = data["message"]["text"]
        if not question:
            return {"ok": True}
        
        history = get_history(chat_id)
            
        answer = chain.invoke({
            "question": question,
            "chat_history": history
        })
        
        if len(answer) > 4000:
            answer = answer[:4000]

        # Save to history for next turn
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        chat_histories[chat_id] = history[-6:]  # keep last 3 turns

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": answer, "parse_mode": "Markdown"}
            )
    except Exception as e:
        logging.error(f"ERROR: {e}", exc_info=True)
    return {"ok": True}

@app.get("/chat")
@app.post("/chat")
async def chat_test(req: Request, q: str = None):
    # Accept ?q=... or JSON {"q":"..."} or {"question":"..."}
    question = q
    if not question:
        try:
            body = await req.json()
            question = body.get("q") or body.get("question") or body.get("text")
        except:
            pass
    
    if not question:
        return {"error": "Add ?q=your question, e.g. /chat?q=what is LMS"}

    # Use dummy user 0 for testing
    history = get_history(0)
    
    answer = chain.invoke({
        "question": question,
        "chat_history": history
    })
    
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    chat_histories[0] = history[-6:]
    
    return {"question": question, "answer": answer}
