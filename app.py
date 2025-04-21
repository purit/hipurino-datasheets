from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import os
import requests
import json
import re
import PyPDF2

# LINE Credentials
CHANNEL_ACCESS_TOKEN = '6s7HGz0kHduwfV86IZrqIt6qBWXPCV4xF96uAPh6UIfuX+4skuvWiUUHi4YZ57CS34PGqypaJGfkdLPsWt7qN7QmdmhYIDWYXsBwf05jTxsEuXOgot/AR+C+Mvmhl9XKxIcVcnGEsNXdWOLwv77ZiQdB04t89/1O/w1cDnyilFU='
CHANNEL_SECRET = '8137cf114e2719c5a58d46705071c558'

# OpenRouter API
OPENROUTER_API_KEY = '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'

# PDF Folder Path
PDF_FOLDER_PATH = "C:\\Users\\purit\\Desktop\\chat-bot\\pdf"

# Init App
app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def read_pdfs_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(folder_path, filename), 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"
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
    pdf_text = read_pdfs_from_folder(PDF_FOLDER_PATH)
    ai_reply = query_openrouter(user_message, pdf_text)

    messaging_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=ai_reply)]
        )
    )

if __name__ == "__main__":
    app.run(port=5000)