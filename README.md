# NyayaSetu: Multilingual Legal R-RAG Bot

NyayaSetu ("Justice Bridge") is an AI-powered legal assistant designed to explain complex Indian laws to citizens in their native language (English, Hindi, Punjabi, etc.).

It utilizes a Retrieval-Augmented Generation (RAG) pipeline with a specialized Retriever-Reranker architecture to ensure high accuracy when citing legal sections (e.g., Motor Vehicles Act, BNS, Consumer Rights).

# Features

Multilingual Support: Answers queries in English, Hindi, Punjabi, Tamil, etc.

R-RAG Pipeline: \* Retriever: Bi-Encoder (paraphrase-multilingual-mpnet) for fast search.

Reranker: Cross-Encoder (ms-marco-MiniLM) for precise relevance scoring.

Quantized LLM: Uses Gemma-2b-it (4-bit) for efficient generation on consumer hardware.

Dynamic Knowledge: Loads knowledge directly from PDF documents.

Performance Metrics: Built-in visualization of latency and retrieval accuracy.

# Project Structure

nyayasetu-rag-bot/
├── database/ # ⚠️ Put your Legal PDFs here
├── src/
│ ├── bot.py # RAG Logic (Retriever+Reranker+LLM)
│ ├── data_loader.py # PDF Text Extraction
│ ├── metrics.py # Success Charts Generation
│ └── config.py # Paths and Constants
├── main.py # Main entry point
└── requirements.txt # Python Dependencies

🛠️ Installation

Clone the repository:

git clone [https://github.com/your-username/nyayasetu.git](https://github.com/your-username/nyayasetu.git)
cd nyayasetu

Install dependencies:

pip install -r requirements.txt

Setup API Key:
Create a file named .env in the root folder and add your Hugging Face token:

HF_TOKEN=hf_your_secret_token_here

Add Documents:
Place your legal PDF documents (e.g., Consumer_Act.pdf, Motor_Vehicles.pdf) inside the database/ folder.

# Usage

Run the main script to initialize the bot, run performance tests, and start the chat loop:

python main.py

# Performance Metrics

When you run main.py, the system automatically generates a performance report image (performance_metrics.png) showing:

Response Latency per Language

Retrieval Recall Rates

Knowledge Base Distribution

# Architecture

Ingestion: PDFs are parsed and chunked into paragraphs.

Indexing: Chunks are embedded into a FAISS Vector Database.

Inference:

User Query -> Vector Search (Top-5)

Reranking (Top-5 -> Top-3) using Cross-Encoder.

Generation: LLM receives context + query to generate a legal explanation.

Disclaimer: NyayaSetu is an AI assistant. Always consult a professional lawyer for official legal counsel.
