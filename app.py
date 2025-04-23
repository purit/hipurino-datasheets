import os
import logging
import requests
import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init Flask App
app = Flask(__name__)

# Load environment variables
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
PORT = int(os.environ.get('PORT', 5000))

# Validate required environment variables
required_env_vars = {
    'LINE_CHANNEL_ACCESS_TOKEN': CHANNEL_ACCESS_TOKEN,
    'LINE_CHANNEL_SECRET': CHANNEL_SECRET,
    'OPENROUTER_API_KEY': OPENROUTER_API_KEY,
    'SUPABASE_URL': SUPABASE_URL,
    'SUPABASE_KEY': SUPABASE_KEY
}

for name, value in required_env_vars.items():
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        raise ValueError(f"Missing required environment variable: {name}")

# Initialize services
try:
    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # ใช้โมเดลขนาดเล็กกว่า (แก้ตรงนี้)
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",  # โมเดลขนาด ~120MB
        device="cpu"  # บังคับใช้ CPU เพื่อเสถียรภาพ
    )
    logger.info("โหลดโมเดล SentenceTransformer สำเร็จแล้ว")
    
    # Initialize LINE Messaging API
    configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
    api_client = ApiClient(configuration)
    messaging_api = MessagingApi(api_client)
    handler = WebhookHandler(CHANNEL_SECRET)
    
except Exception as e:
    logger.error(f"เกิดข้อผิดพลาดในการตั้งค่าเซอร์วิส: {str(e)}")
    raise

def query_openrouter(question: str, context: str) -> str:
    """ ส่งคำถามไปยัง OpenRouter """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        f"จากข้อมูลสินค้าต่อไปนี้: {context}\n\n"
        f"ตอบคำถามเกี่ยวกับสินค้าของผู้ใช้เป็นภาษาไทยให้สั้นและกระชับที่สุดโดยอิงจากข้อมูลเท่านั้น: {question}"
    )
    
    data = {
        "model": "google/gemini-flash-1.5-8b-exp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15  # เพิ่มเวลา timeout
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        logger.error(f"OpenRouter เกิดข้อผิดพลาด: {str(e)}")
        return "ขออภัย ระบบไม่สามารถประมวลผลคำถามได้ในขณะนี้"

def get_relevant_products(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """ ค้นหาสินค้าที่เกี่ยวข้องจาก Supabase """
    try:
        # สร้าง embedding ด้วยโมเดลขนาดเล็ก
        query_embedding = model.encode(query, convert_to_tensor=False).tolist()
        
        response = supabase.rpc(
            'match_content',
            {
                'query_embedding': query_embedding,
                'match_count': top_k
            }
        ).execute()
        
        return response.data if response.data else []
    
    except Exception as e:
        logger.error(f"ค้นหาสินค้าผิดพลาด: {str(e)}")
        return []

@app.route("/callback", methods=['POST'])
def callback():
    """ รับการเรียกกลับจาก LINE """
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        logger.error(f"จัดการ webhook ผิดพลาด: {str(e)}")
        abort(500)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """ จัดการข้อความจากผู้ใช้ """
    try:
        user_message = event.message.text
        logger.info(f"ได้รับข้อความจากผู้ใช้: {user_message}")
        
        # ค้นหาสินค้า
        relevant_items = get_relevant_products(user_message)
        
        if not relevant_items:
            reply = "ไม่พบสินค้าที่เกี่ยวข้อง"
        else:
            context = "\n---\n".join(item['content'] for item in relevant_items)
            reply = query_openrouter(user_message, context)
        
        # ส่งคำตอบกลับ
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
    
    except Exception as e:
        logger.error(f"จัดการข้อความผิดพลาด: {str(e)}")
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="ขออภัย ระบบขัดข้องชั่วคราว")]
            )
        )

@app.route('/health', methods=['GET'])
def health_check():
    """ ตรวจสอบสถานะเซอร์วิส """
    return {'status': 'พร้อมทำงาน', 'model': 'paraphrase-multilingual-MiniLM-L12-v2'}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT)
