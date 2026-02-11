import numpy as np 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
         Initialize the embedding manager
         
         Args:
           model_name: HuggingFace model name for sentence embeddings
        """

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """load the sentence transformer model"""

        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
         texts: Lists of text strings to embed 

        Return:
         numpy array of embedding with shape(len(texts), embedding_dim)
         """

        if not self.model:
            raise ValueError("Model not found")

        print(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


# Initialize the manager 
embedding_manager = EmbeddingManager()
print(embedding_manager)
