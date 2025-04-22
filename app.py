import os
import requests
import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Init App
app = Flask(__name__)

# LINE Credentials (ควรตั้งค่าใน Environment Variables บน Render)
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API (ควรตั้งค่าใน Environment Variables บน Render)
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# GitHub URL ของไฟล์ all_products.txt
TEXT_FILE_URL = "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/data/all_products.txt"

# FAISS Indexing
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
INDEX_DIR = "product_indices"  # Directory สำหรับเก็บ Index แยก
TEXTS_DIR = "product_texts"  # Directory สำหรับเก็บ texts แยก
K_SEARCH = 3  # จำนวนผลลัพธ์ที่ต้องการจาก FAISS

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)
model = SentenceTransformer(EMBEDDING_MODEL)

# Dictionary สำหรับเก็บ Index ที่โหลดแล้ว (เพื่อไม่ให้โหลดซ้ำ)
loaded_indices = {}
# Dictionary เพื่อติดตามว่า Index ของ Product ใดถูกสร้างแล้ว
indices_created = {}
# เก็บข้อมูล Product ทั้งหมดใน Memory ชั่วคราว
all_products_data = []

def read_organized_text_from_github(url, product_start_marker, product_end_marker):
    products_data = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        products = content.split(product_start_marker)[1:]
        for product_text in products:
            if product_end_marker:
                product_text = product_text.split(product_end_marker)[0].strip()
            product_info = {}
            lines = product_text.strip().split('\n')
            product_name = None
            for line in lines:
                if "PRODUCT:" in line:
                    product_name = line.split("PRODUCT:")[1].strip()
                    product_info['PRODUCT'] = product_name
                elif ":" in line:
                    key, value = line.split(":", 1)
                    product_info[key.strip()] = value.strip()
            if product_name and product_info:
                products_data.append(product_info)
        return products_data
    except requests.exceptions.RequestException as e:
        print(f"เกิดข้อผิดพลาดในการอ่าน Text File จาก {url}: {e}")
        return []

def create_faiss_index_for_product(product_name, product_info):
    text_to_embed = f"{product_info.get('PRODUCT', '')} {product_info.get('Description', '')} {product_info.get('Type', '')} {product_info.get('Part No.', '')} {product_info.get('Nominal voltage (a.c.) (UN)', '')} {product_info.get('Voltage protection level [L-N]/[N-PE] (UP)', '')}"
    embedding = model.encode([text_to_embed])
    d = embedding.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embedding.reshape(1, -1)) # Reshape ให้เป็น Matrix (1 x d)

    os.makedirs(INDEX_DIR, exist_ok=True)
    index_file = os.path.join(INDEX_DIR, f"{product_name.replace(' ', '_')}.faiss")
    faiss.write_index(index, index_file)

    os.makedirs(TEXTS_DIR, exist_ok=True)
    text_file = os.path.join(TEXTS_DIR, f"{product_name.replace(' ', '_')}.json")
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump(product_info, f, ensure_ascii=False)
    print(f">>> สร้าง FAISS Index สำหรับ {product_name}")

def load_faiss_index_for_product(product_name):
    index_file = os.path.join(INDEX_DIR, f"{product_name.replace(' ', '_')}.faiss")
    text_file = os.path.join(TEXTS_DIR, f"{product_name.replace(' ', '_')}.json")
    if os.path.exists(index_file) and os.path.exists(text_file):
        index = faiss.read_index(index_file)
        with open(text_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f">>> โหลด FAISS Index สำหรับ {product_name}")
        return index, metadata
    return None, None

def get_faiss_index_and_metadata(product_name):
    if product_name not in loaded_indices:
        index, metadata = load_faiss_index_for_product(product_name)
        if index:
            loaded_indices[product_name] = (index, metadata)
        return index, metadata
    return loaded_indices[product_name]

def search_faiss(query, product_name):
    index, metadata = get_faiss_index_and_metadata(product_name)
    if index is None:
        print(f">>> ไม่พบ/โหลด FAISS Index สำหรับ {product_name}")
        return []
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), K_SEARCH)
    results = [(I[0][i], metadata) for i in range(len(I[0]))]
    return results

def query_openrouter(question, context):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-medium",
        "messages": [
            {"role": "user", "content": f"จากข้อมูลนี้: {context}\n\nตอบคำถาม: {question}"}
        ],
        "max_tokens": 200,
        "top_p": 0.8
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f">>> OpenRouter API Request Error: {e}")
        return "ขออภัย มีปัญหาในการเชื่อมต่อกับ OpenRouter"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f">>> OpenRouter API Response Error: {e}, Response: {response.text}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำตอบ"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text.strip()
    print(f">>> ผู้ใช้ส่งข้อความ: {user_message}")

    relevant_product = None
    global all_products_data
    if not all_products_data:
        all_products_data = read_organized_text_from_github(TEXT_FILE_URL, "==== PRODUCT:", "==== END_PRODUCT ====")

    for product in all_products_data:
        if product.get('PRODUCT') and re.search(re.escape(user_message), product['PRODUCT'], re.IGNORECASE):
            relevant_product = product['PRODUCT']
            current_product_data = product
            break

    if relevant_product:
        if relevant_product not in indices_created:
            create_faiss_index_for_product(relevant_product, current_product_data)
            indices_created[relevant_product] = True

        search_results = search_faiss(user_message, relevant_product)
        if search_results:
            context = search_results[0][1].get('Description', '')
            ai_reply = query_openrouter(user_message, context)
            if ai_reply and "ขออภัย" not in ai_reply:
                try:
                    messaging_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=ai_reply)]
                        )
                    )
                    print(f">>> ส่งข้อความตอบกลับ (FAISS สำหรับ {relevant_product})")
                except Exception as e:
                    print(f">>> ข้อผิดพลาดในการตอบกลับ (FAISS สำหรับ {relevant_product}): {e}")
                return

    # Fallback
    ai_reply = "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้อง"
    try:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )
        print(">>> ส่งข้อความตอบกลับ (Fallback)")
    except Exception as e:
        print(f">>> ข้อผิดพลาดในการตอบกลับ (Fallback): {e}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f">>> Starting app on port: {port}")
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(TEXTS_DIR, exist_ok=True)
    # โหลดข้อมูล Product ทั้งหมดเมื่อเริ่มต้น แต่ยังไม่สร้าง Index
    all_products_data = read_organized_text_from_github(TEXT_FILE_URL, "==== PRODUCT:", "==== END_PRODUCT ====")
    app.run(host='0.0.0.0', port=port)
