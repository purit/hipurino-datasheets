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

# Configure Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# LINE Credentials
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # URL ของ OpenRouter API
OPENROUTER_MODEL = "anthropic/claude-2"  # เลือก Model ที่ต้องการใช้

# GitHub URL ของไฟล์ all_products.json
JSON_FILE_URL = "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/data/all_products.json"

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def read_json_from_url(url):
    """
    ดาวน์โหลดและอ่านข้อมูล JSON จาก URL ที่ระบุ.
    """
    try:
        logging.info(f">>> กำลังดาวน์โหลด JSON จาก: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f">>> อ่าน JSON จาก {url} เสร็จสิ้น")
        return data.get('products', [])
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
    """
    logging.info(f">>> Calling OpenRouter with question: {question}, context: {context}")

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"Answer the question '{question}' using the following product information: {context}. Question: {question}"
        payload = {
            "prompt": prompt,
            "model": OPENROUTER_MODEL,
            "max_tokens": 200,  # ปรับตามความเหมาะสม
        }
        logging.info(f">>> Sending to OpenRouter: {payload}")

        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        ai_response = response.json()
        logging.info(f">>> Raw OpenRouter response: {ai_response}")

        # ดึงคำตอบจาก API Response (ปรับตามโครงสร้าง Response ของ OpenRouter)
        ai_text = ai_response["choices"][0]["message"]["content"].strip()
        logging.info(f">>> Parsed OpenRouter response: {ai_text}")
        return ai_text

    except requests.exceptions.RequestException as e:
        logging.error(f">>> Error calling OpenRouter API: {e}")
        return "ขออภัย เกิดข้อผิดพลาดในการดึงข้อมูลจาก AI"
    except (KeyError, json.JSONDecodeError) as e:
        logging.error(f">>> Error parsing OpenRouter response: {e}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำตอบจาก AI"

@app.route("/callback", methods=['POST'])
def callback():
    """
    Endpoint สำหรับรับ Webhook จาก LINE.
    """
    body = request.get_data(as_text=True)
    logging.info(f">>> รับ Webhook จาก LINE: {body}")
    try:
        handler.handle(body, request.headers['X-Line-Signature'])
    except InvalidSignatureError:
        logging.warning(">>> Invalid signature. Aborting.")
        abort(400)
    except Exception as e:
        logging.error(f">>> Error processing webhook: {e}")
        abort(500)
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
    product_found = False

    if products_data:
        for product in products_data:
            # ค้นหาใน 'product_id', 'name', และ 'description' (ปรับได้ตามต้องการ)
            if ("product_id" in product and re.search(re.escape(user_message), product["product_id"], re.IGNORECASE)) or \
               ("name" in product and re.search(re.escape(user_message), product["name"], re.IGNORECASE)) or \
               ("description" in product and re.search(re.escape(user_message), product["description"], re.IGNORECASE)):
                context = json.dumps(product, ensure_ascii=False, indent=2)  # เพิ่ม indent เพื่อให้อ่านง่าย
                ai_reply = query_openrouter(user_message, context)
                if ai_reply and "ขออภัย" not in ai_reply.lower():  # ตรวจสอบ case-insensitive
                    best_reply = ai_reply
                    product_found = True
                    break

    if not product_found:
        logging.info(f">>> ไม่พบสินค้าที่ตรงกับคำค้นหา: {user_message}")
        best_reply = f"ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องกับ '{user_message}' กรุณาลองใหม่อีกครั้ง"

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
