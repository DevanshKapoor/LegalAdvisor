import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import RETRIEVER_MODEL_ID, RERANKER_MODEL_ID, LLM_MODEL_ID, MAX_NEW_TOKENS

class NyayaSetuBot:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Bot initializing on device: {self.device}")
        
        self._load_models()
        self._build_vector_db()

    def _load_models(self):
        print("    > Loading Retriever, Reranker, and LLM...")
        
        # 1. Retriever
        self.retriever_model = SentenceTransformer(RETRIEVER_MODEL_ID, device=self.device)
        
        # 2. Reranker
        self.reranker_model = CrossEncoder(RERANKER_MODEL_ID, max_length=512, device=self.device)

        # 3. LLM (Quantized)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, 
            quantization_config=quant_config, 
            device_map=self.device
        )

    def _build_vector_db(self):
        if not self.knowledge_base:
            print("❌ Knowledge Base is empty.")
            return
            
        embeddings = self.retriever_model.encode(self.knowledge_base, convert_to_tensor=True, show_progress_bar=False)
        self.vector_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.vector_index.add(embeddings.cpu().numpy())
        print("    ✅ Vector database built successfully")

    def retrieve_and_rerank(self, query, top_k=5, rerank_top_n=3):
        # 1. Fast Retrieval
        query_embedding = self.retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
        _, indices = self.vector_index.search(query_embedding, top_k)
        initial_docs = [self.knowledge_base[i] for i in indices[0]]
        
        # 2. Accurate Reranking
        rerank_pairs = [[query, doc] for doc in initial_docs]
        scores = self.reranker_model.predict(rerank_pairs)
        
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_scores[:rerank_top_n]]

    def generate(self, query, context, language_name):
        prompt = f"""<start_of_turn>user
You are 'NyayaSetu', an expert Indian Legal Advisor. 
Your goal is to explain laws simply to common citizens based STRICTLY on the provided context.

Rules:
1. If the context mentions a specific Section or Act, cite it clearly.
2. If the answer is not in the context, say "I do not have information on this specific law."
3. Provide the answer in the {language_name} language only.

CONTEXT:
{" ".join(context)}

QUESTION:
{query}<end_of_turn>
<start_of_turn>model
"""
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_length = inputs.input_ids.shape[1]
        
        outputs = self.llm_model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            eos_token_id=self.llm_tokenizer.eos_token_id
        )
        generated_token_ids = outputs[0, input_token_length:]
        return self.llm_tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    def run_pipeline(self, query_text, language_code):
        from src.config import LANGUAGE_OPTIONS
        lang_name = LANGUAGE_OPTIONS.get(language_code, "English")
        
        reranked_docs = self.retrieve_and_rerank(query_text)
        if not reranked_docs:
            return "Sorry, I could not find relevant legal documents."
            
        return self.generate(query_text, reranked_docs, lang_name)
    
    def get_model_info(self):
        return {
            "device": self.device,
            "kb_size": len(self.knowledge_base),
            "llm": LLM_MODEL_ID
        }