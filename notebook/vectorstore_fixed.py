import os
import numpy as np
import chromadb
import uuid
from typing import List, Any

class VectorStore:

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory  # Fixed typo: presists -> persist
        self.client = None 
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        
        try:
            os.makedirs(self.persist_directory, exist_ok=True)  # Fixed: removed extra 'self,' and used correct variable
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(  # Fixed: get_or_create_collection (singular)
                name=self.collection_name,
                metadata={"description": "pdf doc embeddings for rag"}  # Fixed typo: metdata -> metadata
            )
            print(f"Vector store initialized successfully with collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):  # Fixed: document -> documents
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store")
        
        ids = []
        metadata_list = []  # Fixed: metdata -> metadata_list
        document_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"  # Fixed: removed extra underscore
            ids.append(doc_id)
            
            doc_metadata = dict(doc.metadata)  # Fixed: metdata -> metadata
            doc_metadata['doc_index'] = i
            doc_metadata['content_length'] = len(doc.page_content)
            metadata_list.append(doc_metadata)  # Fixed: appending to correct list
            
            document_text.append(doc.page_content)  # Fixed: documents_text -> document_text
            
            embeddings_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadata_list,  # Fixed: metdata -> metadatas (ChromaDB expects 'metadatas')
                documents=document_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise


# Initialize the VectorStore OUTSIDE the class
vectorstore = VectorStore()
print(vectorstore)
