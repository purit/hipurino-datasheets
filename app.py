from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import os
import requests
import json
import re
import PyPDF2
from io import BytesIO

import os

# LINE Credentials
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# PDF URL List on GitHub
PDF_URLS = [
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/900368.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90045.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90046.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90153.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90158.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90203.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90259.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90302.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90404.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90405.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/90413.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/DSNY RED BOOK_2015_e_complete.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/datarecord-ec575-en.pdf",
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/datarecord-flag-2100-en-po.pdf",
    # เพิ่ม URL ของไฟล์ PDF อื่นๆ ที่นี่
]

# Init App
app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def read_pdfs_from_urls(pdf_urls):
    all_text = ""
    for url in pdf_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            with BytesIO(response.content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"
        except requests.exceptions.RequestException as e:
            print(f"เกิดข้อผิดพลาดในการดาวน์โหลด PDF จาก {url}: {e}")
        except PyPDF2.errors.PdfReadError as e:
            print(f"เกิดข้อผิดพลาดในการอ่าน PDF จาก {url}: {e}")
        except Exception as e:
            print(f"ข้อผิดพลาดที่ไม่คาดคิดในการประมวลผล PDF จาก {url}: {e}")
    return all_text.strip()

def query_openrouter(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": f"จากข้อมูลนี้: {context}\n\nตอบคำถาม: {question}"}
        ]
    }
    print(f"OpenRouter Request Headers: {headers}")
    print(f"OpenRouter Request Body: {json.dumps(data, ensure_ascii=False)}")

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data, ensure_ascii=False))
        response.raise_for_status()
        response_json = response.json()
        print(f"OpenRouter Response: {response_json}")

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content']
        else:
            print("OpenRouter Response ไม่ถูกต้อง:", response_json)
            return "ขออภัย ระบบไม่สามารถประมวลผลคำถามได้ในขณะนี้ (OpenRouter response error)"

    except requests.exceptions.RequestException as e:
        print(f"OpenRouter error (Request): {e}")
        return "ขออภัย ระบบไม่สามารถเชื่อมต่อกับ OpenRouter ได้"
    except json.JSONDecodeError as e:
        print(f"OpenRouter error (JSON Decode): {e}")
        return "ขออภัย มีปัญหาในการประมวลผลข้อมูลจาก OpenRouter"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print("Parse error:", e)
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    user_id = event.source.user_id
    pdf_text = read_pdfs_from_urls(PDF_URLS)
    ai_reply = query_openrouter(user_message, pdf_text)

    messaging_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=ai_reply)]
        )
    )

if __name__ == "__main__":
    app.run(port=5000)
