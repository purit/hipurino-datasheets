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
    
    # Initialize Sentence Transformer Model (load only once)
    model = SentenceTransformer("all-mpnet-base-v2")
    logger.info("SentenceTransformer model loaded successfully")
    
    # Initialize LINE Messaging API
    configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
    api_client = ApiClient(configuration)
    messaging_api = MessagingApi(api_client)
    handler = WebhookHandler(CHANNEL_SECRET)
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

def query_openrouter(question: str, context: str) -> str:
    """
    Query OpenRouter API with the given question and context
    
    Args:
        question: User's question
        context: Product information context
    
    Returns:
        str: Response from OpenRouter or error message
    """
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
    
    logger.debug(f"OpenRouter Request Body: {json.dumps(data, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10  # Add timeout to prevent hanging
        )
        response.raise_for_status()
        
        response_json = response.json()
        logger.debug(f"OpenRouter Response: {response_json}")
        
        if response_json.get('choices') and response_json['choices'][0].get('message'):
            return response_json['choices'][0]['message']['content'].strip()
        
        logger.error(f"Unexpected OpenRouter response format: {response_json}")
        return "ขออภัย ระบบไม่สามารถประมวลผลคำถามได้ในขณะนี้ (รูปแบบคำตอบไม่ถูกต้อง)"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter request failed: {str(e)}")
        return "ขออภัย ระบบไม่สามารถเชื่อมต่อกับ OpenRouter ได้"
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode OpenRouter response: {str(e)}")
        return "ขออภัย มีปัญหาในการประมวลผลข้อมูลจาก OpenRouter"
    except Exception as e:
        logger.error(f"Unexpected error in query_openrouter: {str(e)}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำถาม"

def get_relevant_products(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Get relevant products from Supabase using vector search
    
    Args:
        query: User's query
        top_k: Number of results to return
    
    Returns:
        List of relevant products or empty list on error
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode(query).tolist()
        
        # Call Supabase function
        response = supabase.rpc(
            'match_content',
            {
                'query_embedding': query_embedding,
                'match_count': top_k
            }
        ).execute()
        
        if response.data:
            return response.data
        return []
    
    except Exception as e:
        logger.error(f"Error in get_relevant_products: {str(e)}")
        return []

@app.route("/callback", methods=['POST'])
def callback():
    """
    Handle LINE webhook callback
    """
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    
    logger.info("Received LINE webhook callback")
    logger.debug(f"Request body: {body}")
    logger.debug(f"Signature: {signature}")
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        abort(400)
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        abort(500)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    Handle text message from LINE
    """
    try:
        user_message = event.message.text
        user_id = event.source.user_id
        
        logger.info(f"Received message from user {user_id}: {user_message}")
        
        # Default reply if no relevant products found
        best_reply = "ขออภัย ไม่พบข้อมูลสินค้าที่เกี่ยวข้อง"
        
        # Get relevant products
        relevant_items = get_relevant_products(user_message)
        if relevant_items:
            context = "\n---\n".join(
                f"ข้อมูลสินค้า: {item['content']}" 
                for item in relevant_items
            )
            
            # Query OpenRouter with context
            ai_reply = query_openrouter(user_message, context)
            
            if ai_reply and "ขออภัย" not in ai_reply:
                best_reply = ai_reply
        
        # Send reply
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=best_reply)]
            )
        )
        
        logger.info("Successfully replied to user")
    
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        # Attempt to send error message to user
        try:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำถาม")]
                )
            )
        except Exception as inner_e:
            logger.error(f"Failed to send error message: {str(inner_e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return {'status': 'healthy'}, 200

if __name__ == "__main__":
    logger.info(f"Starting application on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)
