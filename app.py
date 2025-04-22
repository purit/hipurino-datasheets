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
from sentence_transformers import SentenceTransformer
import faiss
from urllib.parse import urlparse
import numpy as np
import re

# Init App
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

# FAISS Indexing
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
INDEX_FILE = "pdf_index.faiss"
TEXTS_FILE = "pdf_texts.json"
K_SEARCH = 3  # จำนวนผลลัพธ์ที่ต้องการจาก FAISS

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)
model = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = None
pdf_texts = []

def get_filename_from_url(url):
    return os.path.basename(urlparse(url).path)

def download_and_extract_text(url):
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

def create_faiss_index(pdf_urls):
    global faiss_index, pdf_texts
    texts = []
    for url in pdf_urls:
        text = download_and_extract_text(url)
        texts.append(text)
    pdf_texts = texts
    with open(TEXTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(pdf_texts, f, ensure_ascii=False)

    embeddings = model.encode(texts)
    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, INDEX_FILE)
    print(">>> สร้าง FAISS Index เสร็จสิ้น")

def load_faiss_index():
    global faiss_index, pdf_texts
    if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, 'r', encoding='utf-8') as f:
            pdf_texts = json.load(f)
        print(">>> โหลด FAISS Index และ Texts จากไฟล์")
        return True
    return False

def search_faiss(query, k=K_SEARCH):
    if faiss_index is None:
        print(">>> FAISS Index ยังไม่ถูกโหลด")
        return []
    query_embedding = model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding), k)
    results = [(I[0][i], pdf_texts[I[0][i]]) for i in range(len(I[0]))]
    return results

def query_openrouter(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"จากข้อมูลนี้: {context}\n\nตอบคำถามต่อไปนี้เป็นภาษาไทยให้สั้นและกระชับที่สุด: {question}"
    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,  # กำหนดจำนวน Tokens สูงสุดของคำตอบ
        "temperature": 0.2  # กำหนดค่า Temperature ให้ต่ำลง เพื่อลดความสร้างสรรค์
    }
    print(f"OpenRouter Request Body: {json.dumps(data, ensure_ascii=False)}")

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data, ensure_ascii=False))
        response.raise_for_status()
        response_json = response.json()
        print(f"OpenRouter Response: {response_json}")

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content'].strip() # เพิ่ม .strip() เพื่อลบ Whitespace หน้าหลัง
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

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    print(">>> handle_message ถูกเรียกใช้งาน")
    user_message = event.message.text
    user_id = event.source.user_id
    print(f">>> ข้อความที่ผู้ใช้ส่งมา: {user_message}, User ID: {user_id}")

    if faiss_index is None:
        print(">>> กำลังสร้าง/โหลด FAISS Index...")
        if not load_faiss_index():
            create_faiss_index(PDF_URLS)

    relevant_pdf_text = None
    found_relevant_pdf_by_name = False

    # ลองหาไฟล์ PDF ที่มีชื่อเกี่ยวข้องกับคำถาม (เช่น หมายเลข Part)
    for url in PDF_URLS:
        filename = get_filename_from_url(url)
        if re.search(re.escape(user_message), filename, re.IGNORECASE):
            relevant_pdf_text = download_and_extract_text(url)
            found_relevant_pdf_by_name = True
            break

    if found_relevant_pdf_by_name and relevant_pdf_text:
        search_results = search_faiss(user_message, k=K_SEARCH)
        if search_results:
            # กรองผลลัพธ์ FAISS ที่มาจาก PDF ที่เราพบจากชื่อไฟล์ (ถ้าต้องการ)
            # ในตัวอย่างนี้เราจะใช้ผลลัพธ์ทั้งหมดที่ FAISS หาเจอ
            context = "\n".join([text for index, text in search_results])
            ai_reply = query_openrouter(user_message, context)
            if ai_reply and "ขออภัย" not in ai_reply:
                try:
                    messaging_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=ai_reply)]
                        )
                    )
                    print(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ (FAISS + ชื่อไฟล์)")
                except Exception as e:
                    print(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ (FAISS + ชื่อไฟล์): {e}")
                return
        else:
            print(">>> ไม่พบข้อมูลที่เกี่ยวข้องจาก FAISS ในไฟล์ที่ชื่อตรงกัน")
            # Fallback: อาจจะใช้เนื้อหาทั้งหมดของไฟล์ที่ชื่อตรงกัน หรือตอบว่าไม่พบ
            ai_reply = query_openrouter(user_message, relevant_pdf_text)
            if ai_reply and "ขออภัย" not in ai_reply:
                try:
                    messaging_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=ai_reply)]
                        )
                    )
                    print(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ (ชื่อไฟล์)")
                except Exception as e:
                    print(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ (ชื่อไฟล์): {e}")
                return

    # หากไม่พบไฟล์ที่ชื่อตรงกัน หรือไม่มีผลลัพธ์จาก FAISS
    search_results = search_faiss(user_message)
    if search_results:
        context = "\n".join([text for index, text in search_results])
        ai_reply = query_openrouter(user_message, context)
        if ai_reply and "ขออภัย" not in ai_reply:
            try:
                messaging_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=ai_reply)]
                    )
                )
                print(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ (FAISS ทั่วไป)")
            except Exception as e:
                print(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ (FAISS ทั่วไป): {e}")
            return

    # Fallback สุดท้าย
    ai_reply = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง"
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )
        print(">>> ส่งข้อความตอบกลับไปยัง LINE สำเร็จ (Fallback)")
    except Exception as e:
        print(f">>> เกิดข้อผิดพลาดในการส่งข้อความตอบกลับ (Fallback): {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f">>> Starting app on port: {port}")
    if not load_faiss_index():
        create_faiss_index(PDF_URLS)
    app.run(host='0.0.0.0', port=port)
