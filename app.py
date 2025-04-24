from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
import os
import requests
import json
import logging
from typing import Optional, List
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
import pinecone
import time

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pdf-documents')

# PDF URLs
PDF_URLS = [
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/900368.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/900451.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/900456.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/910513.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/921098.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/952035.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/952090.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/955332.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/955424.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/961105.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/961125.pdf",
]

if not all([CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, OPENROUTER_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required environment variables")

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

class PDFProcessor:
    def __init__(self):
        self.cached_text: Optional[str] = None
        self.index = self._connect_pinecone(pc)
        self._populate_index()

    def _connect_pinecone(self, pc_instance):
        if PINECONE_INDEX_NAME not in pc_instance.list_indexes().names():
            pc_instance.create_index(PINECONE_INDEX_NAME, dimension=1536, metric="cosine")
            time.sleep(1)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
        else:
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")
        return pc_instance.Index(PINECONE_INDEX_NAME)

    def download_pdf(self, url: str) -> Optional[BytesIO]:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            return BytesIO(res.content)
        except Exception as e:
            logger.error(f"Download error [{url}]: {e}")
            return None

    def extract_text(self, stream: BytesIO) -> str:
        try:
            reader = PyPDF2.PdfReader(stream)
            return "\n".join(filter(None, (page.extract_text() for page in reader.pages)))
        except Exception as e:
            logger.error(f"Extract error: {e}")
            return ""

    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {"model": "text-embedding-ada-002", "input": text}
            res = requests.post("https://openrouter.ai/api/v1/embeddings", headers=headers, json=data, timeout=15)
            res.raise_for_status()
            return res.json()['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def _populate_index(self):
        all_text = []
        for pdf_url in PDF_URLS:
            try:
                pdf_stream = self.download_pdf(pdf_url)
                if not pdf_stream:
                    continue
                text = self.extract_text(pdf_stream)
                if not text:
                    continue
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                for i, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    if embedding:
                        self.index.upsert(
                            vectors=[{"id": f"{pdf_url.split('/')[-1]}-{i}", "values": embedding, "metadata": {"text": chunk, "source": pdf_url}}],
                            namespace="pdf_documents"
                        )
                logger.info(f"Processed and indexed: {pdf_url}")
                all_text.append(text)
            except Exception as e:
                logger.error(f"Error processing {pdf_url}: {e}")
        self.cached_text = "\n".join(all_text)
        logger.info("PDF documents processed and indexed.")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        emb = self.get_embedding(query)
        if not emb:
            return []
        results = self.index.query(vector=emb, top_k=top_k, include_metadata=True)
        return [m['metadata']['text'] for m in results.get('matches', []) if 'metadata' in m and 'text' in m['metadata']]

pdf_processor = PDFProcessor()

def query_openrouter(question: str, context: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-render-app-url.onrender.com",
        "X-Title": "PDF Chatbot"
    }
    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "คุณเป็นผู้ช่วยที่ตอบคำถามจากข้อมูลที่ให้มาเท่านั้น ตอบให้สั้น กระชับ เข้าใจง่าย และเป็นภาษาไทยเท่านั้น"},
            {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}\n\nคำตอบ:"}
        ],
        "temperature": 0.3
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=15)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature")
        abort(400)
    except Exception as e:
        logger.error(f"Callback error: {e}")
        abort(500)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    try:
        user_msg = event.message.text
        user_id = event.source.user_id
        logger.info(f"Message from {user_id}: {user_msg}")

        positive = ["ขอบคุณ", "เก่งมาก", "ดีมาก", "เยี่ยมเลย"]
        negative = ["แย่มาก", "ไม่ดีเลย", "ผิดหวัง"]
        rude = ["ไอ้", "อี", "เหี้ย", "สัส"]
        greetings = ["หวัดดี", "สวัสดี", "Hi", "Hello", "ไง"]

        if user_msg.lower() in map(str.lower, positive):
            reply = "ขอบคุณมากครับ/ค่ะ ยินดีที่ให้บริการครับ/ค่ะ"
        elif any(w in user_msg.lower() for w in negative):
            reply = "ขออภัยเป็นอย่างสูงสำหรับประสบการณ์ที่ไม่ดีครับ/ค่ะ"
        elif any(w in user_msg.lower() for w in rude):
            reply = "เช่นกันครับ"
        elif user_msg.lower() in map(str.lower, greetings):
            reply = "สวัสดีครับ/ค่ะ มีอะไรให้ผม/ดิฉันช่วยค้นหาจากข้อมูลในเอกสารได้บ้างครับ?"
        else:
            context = "\n\n".join(pdf_processor.search(user_msg))
            reply = query_openrouter(user_msg, context) if context else "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณ"

        messaging_api.reply_message_with_http_info(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text=reply)])
        )
    except Exception as e:
        logger.error(f"Handle error: {e}")
        messaging_api.reply_message_with_http_info(
            ReplyMessageRequest(reply_token=event.reply_token, messages=[TextMessage(text="ขออภัย เกิดข้อผิดพลาด")])
        )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
