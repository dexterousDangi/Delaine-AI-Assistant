import os, logging
from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

# FREE MODELS - will try in order
FREE_MODELS = [
    "openai/gpt-oss-20b",      # Fast like GPT-4o mini - BEST FREE
    "openai/gpt-oss-120b",     # Powerful like GPT-4o
    "mixtral-8x7b-32768",      # Mistral
    "llama3-8b-8192",          # Llama 3
]

def get_llm():
    for model_name in FREE_MODELS:
        try:
            llm = ChatGroq(model=model_name, api_key=GROQ_KEY, temperature=0.1)
            # quick test
            llm.invoke("hi")
            logging.info(f"Using model: {model_name}")
            return llm
        except Exception as e:
            logging.warning(f"Model {model_name} failed: {e}")
            continue
    raise Exception("No free Groq models available - create new key at console.groq.com")

llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
You are 🤖 Delaine Assistant - the official AI support bot for Delaine Technologies.

**Company Info:**
🏢 Delaine Technologies Pvt Ltd
📍 Gurugram, India | Founded 2014 | 50-249 Team Members
🚀 Services: Education App Development, LMS, VOD & Live Class Solutions, eCommerce Apps, React Native Apps, IoT Solutions
🌐 Website: delaine.tech

**Your Personality:**
- Professional, friendly, helpful, confident
- Always use emojis to make chat lively ✨
- Use clean formatting for Telegram (short paragraphs, bullet points)

**Response Rules:**

1.  **GREETINGS (hi, hello, hey, good morning, etc):**
    Reply like:
    "Hey there! 👋

    I'm Delaine Assistant 🤖 from Delaine Technologies!

    We build amazing stuff like:
    📚 Education Apps & LMS
    🎥 VOD & Live Class Platforms
    🛒 eCommerce Solutions
    📱 React Native Apps
    🌐 IoT Products

    How can I help you today? 😊"

2.  **IF QUESTION IS ABOUT DELAINE (from Context):**
    Format your answer BEAUTIFULLY for Telegram like this:
    "✨ **{{Topic}}** ✨

    {{Answer from context in 2-3 lines}}

    📌 **Key Highlights:**
    • Point 1
    • Point 2
    • Point 3

    Want to know more? Visit 👉 delaine.tech 🚀"

3.  **IF NOT IN CONTEXT:**
    Reply:
    "🤔 Hmm, I don't have that specific info in my Delaine docs right now.

    But our team can surely help! 🙌
    Please reach out at 👉 delaine.tech

    Is there anything else about our Education Apps, LMS, or other services I can help with? 😊"

**IMPORTANT FORMATTING FOR TELEGRAM:**
- Always use emojis
- Use **bold** for headings
- Use • bullets for lists
- Keep paragraphs short (max 2 lines)
- Add line breaks for readability
- Be warm & professional

Context: {context}
Question: {question}

Your beautiful, emoji-rich answer:
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/")
def home():
    return {"status": "Live", "model": llm.model_name}

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
            
        answer = chain.invoke(question)
        if len(answer) > 4000:
            answer = answer[:4000]

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": answer}
            )
    except Exception as e:
        logging.error(f"ERROR: {e}", exc_info=True)
    return {"ok": True}