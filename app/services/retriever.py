import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

class RetrieverService:
    def __init__(self, passages, embedder_model="intfloat/multilingual-e5-large"):
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

        # 向量检索（L2 距离）
        query_embedding = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        # 过滤离散距离
        retrieved_chunks = [self.passages[i] for i in indices[0]]
        for chunk in retrieved_chunks:
            print('--' * 20)
            print(chunk)
        return retrieved_chunks
    
    def reload(self, passages):
        self.__init__(passages)

    def save(self, index_path="faiss_index.bin", passages_path="passages.pkl"):
        """
        Save the FAISS index and passages to disk.
        
        Args:
            index_path (str): File path to save the FAISS index.
            passages_path (str): File path to save the passages list.
        """
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save the passages using pickle
        with open(passages_path, 'wb') as f:
            pickle.dump(self.passages, f)
        
        print(f"Index saved to {index_path}, passages saved to {passages_path}")

    @classmethod
    def load(cls, index_path="faiss_index.bin", passages_path="passages.pkl", embedder_model="intfloat/multilingual-e5-large"):
        """
        Load the FAISS index and passages from disk and create a new CustomRetriever instance.
        
        Args:
            index_path (str): File path to load the FAISS index from.
            passages_path (str): File path to load the passages list from.
            embedder_model (str): The embedder model to use (default: "intfloat/multilingual-e5-large").
        
        Returns:
            CustomRetriever: A new instance with loaded index and passages.
        """
        if not os.path.exists(index_path) or not os.path.exists(passages_path):
            raise FileNotFoundError("Index or passages file not found")

        # Load the FAISS index
        index = faiss.read_index(index_path)
        
        # Load the passages
        with open(passages_path, 'rb') as f:
            passages = pickle.load(f)
        
        # Create a new instance without re-computing embeddings
        retriever = cls(passages, embedder_model)
        retriever.index = index  # Replace the default index with the loaded one
        
        print(f"Loaded index from {index_path} and passages from {passages_path}")
        return retriever
