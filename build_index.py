import os, asyncio
from playwright.async_api import async_playwright
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv

load_dotenv()

URLS =[
  "https://delaine.tech/",
  "https://delaine.tech/ai-district",
  "https://delaine.tech/blog",
  "https://delaine.tech/case-study",
  "https://delaine.tech/company",
  "https://delaine.tech/contact-us",
  "https://delaine.tech/culture",
  "https://delaine.tech/industries",
  "https://delaine.tech/privacy-policy",
  "https://delaine.tech/services",
  "https://delaine.tech/terms-and-conditions",
  "https://delaine.tech/services/ai-and-machine-learning",
  "https://delaine.tech/services/cloud-solutions",
  "https://delaine.tech/services/digital-transformation",
  "https://delaine.tech/services/e-commerce-solutions",
  "https://delaine.tech/services/mobile-app-development",
  "https://delaine.tech/services/ui-ux-design",
  "https://delaine.tech/services/web-development",
  "https://delaine.tech/industries/automotive",
  "https://delaine.tech/industries/e-commerce-solutions",
  "https://delaine.tech/industries/educations",
  "https://delaine.tech/industries/energy-and-utility",
  "https://delaine.tech/industries/finance",
  "https://delaine.tech/industries/food-and-restaurant",
  "https://delaine.tech/industries/logistics-and-transport",
  "https://delaine.tech/industries/media-and-entertainment",
  "https://delaine.tech/industries/middle-east-startups",
  "https://delaine.tech/industries/on-demand-delivery-apps",
  "https://delaine.tech/industries/real-estate",
  "https://delaine.tech/industries/sports-and-fitness",
  "https://delaine.tech/industries/travel-and-hospitality",
  "https://delaine.tech/case-study/artune",
  "https://delaine.tech/case-study/daeem",
  "https://delaine.tech/case-study/das",
  "https://delaine.tech/case-study/betting-matrix",
  "https://delaine.tech/case-study/dhurina",
  "https://delaine.tech/case-study/eva-social",
  "https://delaine.tech/case-study/galinia",
  "https://delaine.tech/case-study/ved-guru",
  "https://delaine.tech/case-study/fencing-club",
  "https://delaine.tech/case-study/menonmove",
  "https://delaine.tech/case-study/private-crcl",
  "https://delaine.tech/case-study/robomark",
  "https://delaine.tech/case-study/mentalwell",
  "https://delaine.tech/case-study/omniya",
  "https://delaine.tech/case-study/vomely",
  "https://delaine.tech/blog/reasons-app-launches-get-delayed",
  "https://delaine.tech/blog/ai-tools-ultimate-guide",
  "https://delaine.tech/blog/business-website-cost-2026",
  "https://delaine.tech/blog/cloud-computing-how-it-works-which-platform-to-use"
]


async def scrape():
    texts = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for url in URLS:
            try:
                print(f"Scraping {url}...")
                await page.goto(url, timeout=60000, wait_until="networkidle")
                await page.wait_for_timeout(4000)
                content = await page.inner_text('body')
                print(f" -> Got {len(content)} chars")
                if len(content) > 500:  # only keep if not 404
                    texts.append({"content": content, "url": url})
                else:
                    print(f" -> SKIPPED, too small")
            except Exception as e:
                print(f" -> FAILED {url}: {e}")
        await browser.close()
    return texts

def build():
    docs = asyncio.run(scrape())
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = []
    for d in docs:
        for chunk in splitter.split_text(d["content"]):
            if len(chunk.strip()) > 100:
                chunks.append(f"Source: {d['url']}\n{chunk}")
    
    print(f"Total chunks: {len(chunks)}")
    if len(chunks) < 50:
        print("WARNING: Still low, check logs above for FAILED pages")
        return
        
    embeddings = JinaEmbeddings(jina_api_key=os.getenv("JINA_API_KEY"), model_name="jina-embeddings-v2-base-en")
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local("faiss_index")
    print("DONE - faiss_index folder created")

if __name__ == "__main__":
    build()