from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError
import os
import requests
import json
import logging
from typing import Optional, List
import PyPDF2
from io import BytesIO
from dotenv import load_dotenv
from pinecone import Pinecone  # เปลี่ยนการ import

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'pdf-documents')

# PDF URLs
PDF_URLS = [
    "https://raw.githubusercontent.com/purit/hipurino-datasheets/main/pdfs/900368.pdf",
    # ... รายการ URL อื่นๆ ...
]

if not all([CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET, OPENROUTER_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Missing required environment variables")

app = Flask(__name__)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
messaging_api = MessagingApi(api_client)
handler = WebhookHandler(CHANNEL_SECRET)

# เริ่มต้น Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

class PDFProcessor:
    def __init__(self):
        self.cached_text: Optional[str] = None
        
        # ตรวจสอบและสร้าง index หากจำเป็น
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine"
            )
        
        self.vector_index = pc.Index(PINECONE_INDEX_NAME)

    # ... เมธอดอื่นๆคงเดิม ...
