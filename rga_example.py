from sentence_transformers import SentenceTransformer
from transformers import DPRQuestionEncoder, DPRContextEncoder, BartForConditionalGeneration, AutoTokenizer
import faiss
import numpy as np
chunks = [
    "这是一个中文文档片段，包含一些关键信息。",]


# 嵌入模型（用于检索）
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
embeddings = embedder.encode(chunks, convert_to_tensor=True)
embeddings_np = embeddings.cpu().numpy()

# FAISS 索引
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# 检索
query = "文件中有哪些关键信息？"
query_embedding = embedder.encode(query, convert_to_tensor=True).cpu().numpy()
k = 3
distances, indices = index.search(query_embedding, k)
retrieved_chunks = [chunks[i] for i in indices[0]]

# 加载中文 BART 模型
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Randeng-BART-139M-Chinese")
model = BartForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-BART-139M-Chinese")

# 准备输入（问题 + 检索到的片段）
context = " ".join(retrieved_chunks)
input_text = f"问题：{query} 上下文：{context}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# 生成答案
generated = model.generate(**inputs, max_length=100)
answer = tokenizer.decode(generated[0], skip_special_tokens=True)
print("生成的答案:", answer)