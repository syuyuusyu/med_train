import faiss
import numpy as np
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

class CustomRetriever:
    def __init__(self, passages, embedder_model="intfloat/multilingual-e5-small"):
        if not passages:
            raise ValueError("Passages list cannot be empty")
        self.passages = passages
        self.embedder = SentenceTransformer(embedder_model)
        try:
            embeddings = self.embedder.encode(
                passages,
                convert_to_tensor=True,
                show_progress_bar=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to encode passages: {str(e)}")
        embeddings_np = embeddings.cpu().numpy()
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)
        self.dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query, k=3):
        if not query:
            raise ValueError("Query cannot be empty")
        query_embedding = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.passages[i] for i in indices[0]]
        return retrieved_chunks

class CustomRAG:
    def __init__(self, passages, embedder_model="intfloat/multilingual-e5-small", 
                generator_model="google/mt5-small"):
        self.retriever = CustomRetriever(passages, embedder_model)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model, use_fast=False, legacy=True)
        self.model = MT5ForConditionalGeneration.from_pretrained(generator_model)

    def generate(self, query, k=3, max_length=200):
        retrieved_chunks = self.retriever.retrieve(query, k)
        context = " ".join(retrieved_chunks)
        input_text = f"qa: question: {query} context: {context}"
        print("Input text:", input_text)
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        generated = self.model.generate(**inputs, max_length=max_length, num_beams=5, min_length=10)
        print("Generated tokens:", generated[0].tolist())
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print("Decoded answer:", answer)
        return answer

# 测试代码
chinese_chunks = [
    "这是一个中文文档片段，包含一些关键信息。",
    "产品规格：屏幕尺寸为 6.5 英寸，分辨率为 2400x1080。"
]
english_chunks = [
    "This is an English document snippet with key information.",
    "The project timeline is set for 2025, with a budget of $1M."
]
chunks = chinese_chunks + english_chunks

rag = CustomRAG(passages=chunks)
query = "文件中有哪些关键信息？"
answer = rag.generate(query, k=2)
print("生成的答案:", answer)