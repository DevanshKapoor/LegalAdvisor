import os
import glob
from pypdf import PdfReader

def load_pdfs_and_chunk(database_path):
    """
    Loads all PDFs from a folder, extracts text, and chunks it.
    Returns a list of text segments.
    """
    print(f"\n📚 Scanning '{database_path}' for PDF files...")
    documents = []
    
    # Look for PDFs in the specified folder
    pdf_files = glob.glob(os.path.join(database_path, "*.pdf"))

    if not pdf_files:
        print(f"⚠️ No PDF files found in '{database_path}'.")
        return []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            
            # Chunking: Split by double newlines (paragraphs)
            # Filter out chunks < 50 chars (headers/footers)
            chunks = [para.strip() for para in pdf_text.split('\n\n') if len(para.strip()) > 50]
            documents.extend(chunks)
            print(f"    > Loaded & chunked '{os.path.basename(pdf_path)}' into {len(chunks)} segments.")
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
            
    print(f"✅ Knowledge Base loaded with {len(documents)} chunks.")
    return documents