import os
import requests
import json
import PyPDF2
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
from io import BytesIO
from urllib.parse import urlparse
import re

# Init App (ประกาศ app ก่อนใช้งาน)
app = Flask(__name__)

# LINE Credentials
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

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

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def read_pdf_from_url(url):
    all_text = ""
    try:
        print(f">>> กำลังดาวน์โหลด PDF จาก: {url}")
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                all_text += page.extract_text() + "\n"
        print(f">>> อ่าน PDF จาก {url} เสร็จสิ้น")
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
    prompt = f"จากข้อมูลนี้: {context}\n\nตอบคำถามต่อไปนี้เป็นภาษาไทยให้สั้นและกระชับที่สุดโดยอิงจากข้อมูลเท่านั้น: {question}"
    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,  # เพิ่ม max_tokens เล็กน้อยเผื่อข้อมูลมากขึ้น
        "temperature": 0.2   # คงค่า temperature ต่ำ
    }
    print(f"OpenRouter Request Body: {json.dumps(data, ensure_ascii=False)}")

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data, ensure_ascii=False))
        response.raise_for_status()
        response_json = response.json()
        print(f"OpenRouter Response: {response_json}")

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content'].strip()
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
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    print(">>> /callback ถูกเรียกใช้งาน")
    print(f">>> Body ที่ได้รับ: {body}")
    print(f">>> Signature ที่ได้รับ: {signature}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print(">>> InvalidSignatureError เกิดขึ้น!")
        abort(400)
    except Exception as e:
        print(f">>> ข้อผิดพลาดในการ Handle Webhook: {e}")
        abort(400)

    return 'OK'

def find_relevant_text(user_message, pdf_text):
    keywords = user_message.lower().split()
    relevant_sentences = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', pdf_text) # Split text into sentences

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for keyword in keywords:
            if keyword in sentence_lower:
                relevant_sentences.append(sentence)
                break # Move to the next sentence once a keyword is found
    return "\n".join(relevant_sentences)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    print(">>> handle_message ถูกเรียกใช้งาน")
    user_message = event.message.text
    user_id = event.source.user_id
    print(f">>> ข้อความที่ผู้ใช้ส่งมา: {user_message}, User ID: {user_id}")

    relevant_text_from_pdf = None
    best_reply = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
    found_relevant_info = False

    for url in PDF_URLS:
        pdf_text = read_pdf_from_url(url)
        if pdf_text:
            # ค้นหาข้อความที่เกี่ยวข้องจากเนื้อหา PDF
            found_text = find_relevant_text(user_message, pdf_text)
            if found_text:
                relevant_text_from_pdf = found_text
                found_relevant_info = True
                break # Found relevant info in one PDF, no need to check others

    if found_relevant_info and relevant_text_from_pdf:
        ai_reply = query_openrouter(user_message, relevant_text_from_pdf)
        if ai_reply and "ขออภัย" not in ai_reply:
            best_reply = ai_reply
    elif not found_relevant_info:
        # หากไม่พบข้อมูลที่เกี่ยวข้อง ลองตอบแบบทั่วไป หรือแจ้งว่าไม่พบ
        best_reply = "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในเอกสารที่มี"

    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=best_reply)]
            )
        )
        print(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ")
    except Exception as e:
        print(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ: {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f">>> Starting app on port: {port}")
    app.run(host='0.0.0.0', port=port)
