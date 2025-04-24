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
from pinecone import Pinecone, IndexSpec  # Import IndexSpec
import time

# โหลด environment variables
load_dotenv()

# ตั้งค่าการบันทึกเหตุการณ์
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ข้อมูลการกำหนดค่า
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pdf-documents')

# URL ของ PDF ใน GitHub
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

# ตรวจสอบค่าที่จำเป็น
if not all([CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, OPENROUTER_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required environment variables")

# เริ่มต้นแอป Flask
app = Flask(__name__)

# ตั้งค่า LINE
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

# ตั้งค่า Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

class PDFProcessor:
    def __init__(self):
        self.cached_text: Optional[str] = None
        self.namespace = "pdf_documents"

        # ตรวจสอบและสร้าง index หากจำเป็น
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            index_spec = IndexSpec(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # ขนาดของ embedding vector ที่ได้จาก text-embedding-ada-002
                metric="cosine"
            )
            pc.create_index(index_spec=index_spec)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' created.")
            # รอให้ index พร้อมใช้งาน
            time.sleep(1)
        else:
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

        self.vector_index = pc.Index(PINECONE_INDEX_NAME)
        self.process_pdfs()  # โหลดและประมวลผล PDFs เมื่อเริ่มต้น

    def download_pdf(self, url: str) -> Optional[BytesIO]:
        """ดาวน์โหลด PDF จาก URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_stream: BytesIO) -> str:
        """ดึงข้อความจาก PDF"""
        try:
            reader = PyPDF2.PdfReader(pdf_stream)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def get_embedding(self, text: str) -> List[float]:
        """รับ embedding vector สำหรับข้อความโดยใช้ OpenRouter"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None

    def process_pdfs(self):
        """ประมวลผลและเก็บ PDF ทั้งหมดใน Pinecone"""
        all_text = []

        for pdf_url in PDF_URLS:
            try:
                pdf_stream = self.download_pdf(pdf_url)
                if not pdf_stream:
                    continue

                text = self.extract_text_from_pdf(pdf_stream)
                if not text:
                    continue

                # แบ่งข้อความเป็น chunks (ประมาณ 1000 ตัวอักษรต่อ chunk)
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

                for i, chunk in enumerate(chunks):
                    embedding = self.get_embedding(chunk)
                    if embedding:
                        # เก็บใน Pinecone
                        self.vector_index.upsert(
                            vectors=[{
                                "id": f"{pdf_url.split('/')[-1]}-{i}",
                                "values": embedding,
                                "metadata": {
                                    "text": chunk,
                                    "source": pdf_url
                                }
                            }],
                            namespace=self.namespace
                        )
                logger.info(f"Successfully processed and indexed: {pdf_url}")

                all_text.append(text)

            except Exception as e:
                logger.error(f"Error processing {pdf_url}: {str(e)}")
                continue

        self.cached_text = "\n".join(all_text)

    def search_relevant_text(self, query: str, top_k: int = 3) -> List[str]:
        """ค้นหาข้อความที่เกี่ยวข้องใน Pinecone"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results = self.vector_index.query(
            namespace=self.namespace,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return [match['metadata']['text'] for match in results['matches']]

# สร้าง instance ของ PDFProcessor
pdf_processor = PDFProcessor()

def query_openrouter(question: str, context: str) -> str:
    """ส่งคำถามไปยัง OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-render-app-url.onrender.com",
        "X-Title": "PDF Chatbot"
    }

    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {
                "role": "system",
                "content": "คุณเป็นผู้ช่วยที่ตอบคำถามจากข้อมูลที่ให้มาเท่านั้น ตอบให้สั้น กระชับ เข้าใจง่าย และเป็นภาษาไทยเท่านั้น"
            },
            {
                "role": "user",
                "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}\n\nคำตอบ:"
            }
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        response.raise_for_status()

        response_data = response.json()
        if not response_data.get('choices'):
            raise ValueError("Invalid response format from OpenRouter")

        return response_data['choices'][0]['message']['content'].strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter request failed: {str(e)}")
        return "ขออภัย เกิดปัญหาการเชื่อมต่อ"
    except Exception as e:
        logger.error(f"OpenRouter processing error: {str(e)}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"

@app.route("/callback", methods=['POST'])
def callback():
    """จัดการ LINE Webhook"""
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature detected")
        abort(400)
    except Exception as e:
        logger.error(f"Webhook handler error: {str(e)}")
        abort(500)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """จัดการข้อความจากผู้ใช้"""
    try:
        user_message = event.message.text
        user_id = event.source.user_id
        logger.info(f"Received message from {user_id}: {user_message}")

        # ตรวจสอบว่าเป็นคำชมหรือไม่
        positive_feedback = ["ขอบคุณ", "เก่งมาก", "ดีมาก", "เยี่ยมเลย"]
        if user_message.lower() in [feedback.lower() for feedback in positive_feedback]:
            reply = "ขอบคุณมากครับ/ค่ะ ยินดีที่ให้บริการครับ/ค่ะ"
        # ตรวจสอบว่าเป็นคำตำหนิหรือไม่
        elif any(word.lower() in user_message.lower() for word in ["แย่มาก", "ไม่ดีเลย", "ผิดหวัง"]):
            reply = "ขออภัยเป็นอย่างสูงสำหรับประสบการณ์ที่ไม่ดีที่ท่านได้รับครับ/ค่ะ"
        # ตรวจสอบว่าเป็นคำหยาบหรือไม่สุภาพ
        elif any(word.lower() in user_message.lower() for word in ["ไอ้", "อี", "เหี้ย", "สัส"]):
            reply = "เช่นกันครับ"
        # ตรวจสอบว่าเป็นคำทักทายหรือไม่
        else:
            greetings = ["หวัดดี", "สวัสดี", "Hi", "Hello", "ไง"]
            if user_message.lower() in [greet.lower() for greet in greetings]:
                reply = f"สวัสดีครับ/ค่ะ ยินดีที่ได้คุยกับคุณ มีอะไรให้ผม/ดิฉันช่วยค้นหาจากข้อมูลในเอกสารได้บ้างครับ?"
            else:
                # ค้นหาข้อความที่เกี่ยวข้องจาก Pinecone
                relevant_texts = pdf_processor.search_relevant_text(user_message)
                context = "\n\n".join(relevant_texts)

                if not context:
                    reply = "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณ"
                else:
                    response_from_ai = query_openrouter(user_message, context)
                    if "ขออภัย" in response_from_ai:
                        reply = f"{response_from_ai}\nหากต้องการให้เจ้าหน้าที่ติดต่อกลับ รบกวนฝาก ชื่อ, เบอร์ติดต่อกลับ, อีเมล, และชื่อบริษัท ด้วยครับ/ค่ะ"
                    else:
                        reply = response_from_ai

        # ส่งคำตอบกลับ
        messaging_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )

    except Exception as e:
        logger.error(f"Message handling error: {str(e)}")
        messaging_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="ขออภัย เกิดข้อผิดพลาดในการประมวลผล")]
            )
        )

if __name__ == "__main__":
    # เริ่มเซิร์ฟเวอร์
    app.run(host='0.0.0.0', port=5000)
