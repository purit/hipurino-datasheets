import os
import requests
import json
import PyPDF2
from io import BytesIO
from flask import Flask, request, jsonify

# --- การตั้งค่าคีย์สำคัญจาก Environment Variables ---
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# --- รายการ URL ของไฟล์ PDF บน GitHub ---
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
]

app = Flask(__name__)

def read_pdfs_from_urls(pdf_urls):
    all_text = ""
    for url in pdf_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
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
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)

    # --- ส่วนนี้คุณอาจจะต้องใช้ Line SDK ของ Netlify หรือจัดการ HTTP Request เอง ---
    # --- ตัวอย่างเบื้องต้น (อาจต้องปรับแก้ให้เข้ากับ Netlify) ---
    try:
        # handler.handle(body, signature) # ถ้ามี SDK ของ Netlify
        # --- หรือถ้าจัดการเอง ---
        # ตรวจสอบ Signature (สำคัญ!)
        # Parse body
        # ประมวลผล Event
        # สร้าง Response
        return jsonify({"message": "Webhook received!"}), 200
    except Exception as e:
        print("Parse error:", e)
        return jsonify({"error": "Parse error"}), 400

def handler(event, context):
    with app.request_context(event):
        return callback()

if __name__ == "__main__":
    app.run(port=5000)
