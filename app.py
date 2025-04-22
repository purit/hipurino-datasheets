import os
import requests
import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
from urllib.parse import urlparse
import re
import logging

# Init App
app = Flask(__name__)

# Configure Logging (Optional, but highly recommended for production)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LINE Credentials
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# GitHub URL ของไฟล์ all_products.json
JSON_FILE_URL = "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/data/all_products.json"  # แทนที่ด้วย URL ที่ถูกต้อง

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def read_json_from_url(url):
    """
    ดาวน์โหลดและอ่านข้อมูล JSON จาก URL ที่ระบุ.

    Args:
        url (str): URL ของไฟล์ JSON.

    Returns:
        list: รายการของข้อมูลสินค้า (list of dicts) หากสำเร็จ, มิฉะนั้น None.
    """
    try:
        logging.info(f">>> กำลังดาวน์โหลด JSON จาก: {url}")
        response = requests.get(url)
        response.raise_for_status()  # ตรวจสอบ HTTP status code (raise exception ถ้ามี error)
        data = response.json()
        logging.info(f">>> อ่าน JSON จาก {url} เสร็จสิ้น")
        return data.get('products', [])  # ดึง 'products' ออกมา, คืน list ว่างถ้าไม่มี
    except requests.exceptions.RequestException as e:
        logging.error(f"เกิดข้อผิดพลาดในการดาวน์โหลด JSON จาก {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"เกิดข้อผิดพลาดในการ Parse JSON จาก {url}: {e}")
        return None
    except KeyError:
        logging.error(f"Error: 'products' key not found in JSON data from {url}")
        return None

def query_openrouter(question, context):
    """
    ส่งคำถามไปยัง OpenRouter API และรับคำตอบ.
    (โค้ด Function query_openrouter เหมือนเดิม)
    """
    # ... (Function query_openrouter เหมือนเดิม) ...
    # Placeholder for OpenRouter API call
    logging.info(f">>> Calling OpenRouter with question: {question}, context: {context}")
    return "Placeholder reply from OpenRouter"  # Replace with actual API call
    pass

@app.route("/callback", methods=['POST'])
def callback():
    """
    Endpoint สำหรับรับ Webhook จาก LINE.
    """
    body = request.get_data(as_text=True)
    logging.info(f">>> รับ Webhook จาก LINE: {body}")  # Log the entire webhook body
    try:
        handler.handle(body, request.headers['X-Line-Signature'])
    except InvalidSignatureError:
        logging.warning(">>> Invalid signature. Aborting.")
        abort(400)
    except Exception as e:
        logging.error(f">>> Error processing webhook: {e}")
        abort(500)  # Internal Server Error
    return '', 200

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    จัดการกับ MessageEvent จาก LINE (ข้อความที่ผู้ใช้ส่งมา).
    """
    logging.info(">>> handle_message ถูกเรียกใช้งาน")
    user_message = event.message.text
    user_id = event.source.user_id
    logging.info(f">>> ข้อความที่ผู้ใช้ส่งมา: {user_message}, User ID: {user_id}")

    products_data = read_json_from_url(JSON_FILE_URL)
    best_reply = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง"

    if products_data:
        for product in products_data:
            # ค้นหาใน 'product_id', 'name', และ 'description' (ปรับได้ตามต้องการ)
            if ("product_id" in product and re.search(re.escape(user_message), product["product_id"], re.IGNORECASE)) or \
               ("name" in product and re.search(re.escape(user_message), product["name"], re.IGNORECASE)) or \
               ("description" in product and re.search(re.escape(user_message), product["description"], re.IGNORECASE)):
                context = json.dumps(product, ensure_ascii=False)  # ส่งข้อมูลสินค้าทั้ง Object เป็น Context
                ai_reply = query_openrouter(user_message, context)
                if ai_reply and "ขออภัย" not in ai_reply:
                    best_reply = ai_reply
                    break  # เจอสินค้าแล้วก็หยุด loop
            # คุณสามารถเพิ่ม Logic การค้นหาใน Fields อื่นๆ ได้ เช่น 'part_no'

    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=best_reply)]
            )
        )
        logging.info(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ")
    except Exception as e:
        logging.error(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logging.info(f">>> Starting app on port: {port}")
    app.run(host='0.0.0.0', port=port)
