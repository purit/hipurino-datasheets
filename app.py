import os
import requests
import json
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# Init App (ประกาศ app ก่อนใช้งาน)
app = Flask(__name__)

# LINE Credentials
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')

# OpenRouter API
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# Supabase Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sentence Transformer Model
model = SentenceTransformer("all-mpnet-base-v2")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

def query_openrouter(question, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"จากข้อมูลสินค้าต่อไปนี้: {context}\n\nตอบคำถามเกี่ยวกับสินค้าของผู้ใช้เป็นภาษาไทยให้สั้นและกระชับที่สุดโดยอิงจากข้อมูลเท่านั้น: {question}"
    data = {
        "model": "google/gemini-flash-1.5-8b-exp",  # ใช้ Gemini Flash
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }
    print(f"OpenRouter Request Body: {json.dumps(data, ensure_ascii=False)}")
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        print(f"OpenRouter Response: {response_json}")
        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            return response_json['choices'][0]['message']['content'].strip()
        else:
            print("OpenRouter Response ไม่ถูกต้อง:", response_json)
            return "ขออภัย ระบบไม่สามารถประมวลผลคำถามได้ในขณะนี้ (OpenRouter error)"
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

async def get_relevant_products(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    try:
        response = await supabase.rpc(
            'match_content',  # ใช้ Function ใหม่ที่ทำงานกับ pdf_embeddings
            {'query_embedding': query_embedding, 'match_count': top_k}
        ).execute()
        if response.error:
            print(f"Error from Supabase function (match_content): {response.error}")
            return []
        return response.data
    except Exception as e:
        print(f"Error querying Supabase (get_relevant_products): {e}")
        return []

@handler.add(MessageEvent, message=TextMessageContent)
async def handle_message(event):
    print(">>> handle_message ถูกเรียกใช้งาน")
    user_message = event.message.text
    user_id = event.source.user_id
    print(f">>> ข้อความที่ผู้ใช้ส่งมา: {user_message}, User ID: {user_id}")

    best_reply = "ขออภัย ไม่พบข้อมูลสินค้าที่เกี่ยวข้อง"

    relevant_items = await get_relevant_products(user_message)
    if relevant_items:
        context = "\n---\n".join([f"ข้อมูลสินค้า: {item['content']}" for item in relevant_items])
        ai_reply = query_openrouter(user_message, context)
        if ai_reply and "ขออภัย" not in ai_reply:
            best_reply = ai_reply

    try:
        await messaging_api.reply_message(
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
