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
You are Delaine Technologies support assistant - friendly, helpful.

RULES:
1. If user says hi/hello/hey/thanks/bye or general chat: Reply friendly, introduce yourself as Delaine bot from Gurugram (since 2014), ask how you can help with LMS, Education App, etc.
2. If user asks about Delaine services, use Context to answer.
3. If question NOT in Context and NOT greeting: Say "I don't have this in Delaine docs, please contact team at delaine.tech"

Context: {context}
Question: {question}
Answer in friendly tone:
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