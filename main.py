import os
import logging
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

print(f"JINA loaded: {bool(JINA_KEY)}, GROQ loaded: {bool(GROQ_KEY)}, TELEGRAM loaded: {bool(TELEGRAM_TOKEN)}")

# Load DB
embeddings = JinaEmbeddings(jina_api_key=JINA_KEY, model_name="jina-embeddings-v2-base-en")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

## llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_KEY, temperature=0.1)
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_KEY, temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are Delaine Technologies support bot. Gurugram, 2014, 50-249 team. Services: Education App, LMS, VOD Live Class, eCommerce, React Native, IoT.
Answer ONLY from context. If not found, say: I don't have this in Delaine docs, contact delaine.tech.
Context: {context}
Question: {question}
Answer:
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/")
def home():
    return {"status": "Live", "bot": "Delaine Bot Running"}

@app.get("/health")
def health():
    return {"status": "ok", "faiss": True}

@app.get("/ask")
def ask(question: str):
    return {"answer": chain.invoke(question)}

@app.post("/telegram-webhook")
async def telegram_webhook(req: Request):
    data = await req.json()
    logging.info(f"Incoming: {data}")
    try:
        if "message" not in data or "text" not in data["message"]:
            return {"ok": True}
            
        chat_id = data["message"]["chat"]["id"]
        question = data["message"]["text"]
        
        if not question or question.startswith("/"):
            question = "What is Delaine Technologies?"
            
        logging.info(f"Question from {chat_id}: {question}")
        
        answer = chain.invoke(question)
        logging.info(f"Answer: {answer[:100]}")
        
        # Telegram limit 4096 chars
        if len(answer) > 4000:
            answer = answer[:4000]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": answer}
            )
            logging.info(f"Telegram send status: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        logging.error(f"ERROR in webhook: {e}", exc_info=True)
        
    return {"ok": True}