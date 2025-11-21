import os
import sys
from dotenv import load_dotenv
from huggingface_hub import login

from src.config import DATABASE_PATH
from src.data_loader import load_pdfs_and_chunk
from src.bot import NyayaSetuBot
from src.metrics import run_multilingual_showcase, plot_success_metrics

def main():
    # 1. Environment Setup
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file.")
        sys.exit(1)
        
    print("🔐 Logging into Hugging Face...")
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    # 2. Load Data
    knowledge_base = load_pdfs_and_chunk(DATABASE_PATH)
    if not knowledge_base:
        print("❌ Please add PDFs to the 'database' folder.")
        sys.exit(1)

    # 3. Initialize Bot
    try:
        bot = NyayaSetuBot(knowledge_base)
    except Exception as e:
        print(f"❌ Model Initialization failed: {e}")
        sys.exit(1)

    # 4. Run Demo & Metrics
    performance_data = run_multilingual_showcase(bot)
    plot_success_metrics(performance_data)

    # 5. Interactive Loop (Optional)
    print("\n⚖️  Bot Ready! Type 'exit' to quit.")
    while True:
        q = input("\nYour Question (en/hi/pa): ")
        if q.lower() == 'exit': break
        
        # Simple detection for demo purposes (defaults to English)
        lang = "en"
        if any(char >= '\u0900' and char <= '\u097F' for char in q): lang = "hi" # Hindi range
        if any(char >= '\u0a00' and char <= '\u0a7f' for char in q): lang = "pa" # Punjabi range
        
        print(f"Thinking ({lang})...")
        ans = bot.run_pipeline(q, lang)
        print(f"NyayaSetu: {ans}")

if __name__ == "__main__":
    main()