import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # free, local
CHAT_MODEL       = "gemini-2.5-flash"   
TEMPERATURE      = 0.2
MAX_TOKENS       = 1000
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200