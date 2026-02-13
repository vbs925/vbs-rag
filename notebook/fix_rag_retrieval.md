# Fix RAG Retrieval - Step by Step Guide

## 🔧 Problem Summary
Your RAG retriever is returning 0 documents because:
1. Documents were added twice (44 duplicates instead of 22 unique)
2. The vector store collection may not be properly initialized

## ✅ Solution: Add These Cells to Your Notebook

### Step 1: Clean Up the Vector Store (Add this as a new cell)

```python
# STEP 1: Clean up and reinitialize vector store
import shutil
import os

# Remove the old vector store to start fresh
vector_store_path = "../data/vector_store"
if os.path.exists(vector_store_path):
    shutil.rmtree(vector_store_path)
    print(f"✅ Removed old vector store at {vector_store_path}")
else:
    print(f"ℹ️  No existing vector store found")

# Reinitialize with a clean slate
vectorstore = VectorStore()
print(f"✅ Created fresh vector store")
print(f"📊 Current document count: {vectorstore.collection.count()}")
```

### Step 2: Add Documents to Vector Store (Add this as a new cell)

```python
# STEP 2: Add documents to the clean vector store (ONLY ONCE!)
print(f"Adding {len(chunks)} chunks to vector store...")
vectorstore.add_documents(chunks, chunk_embeddings)
print(f"\n✅ Successfully added documents!")
print(f"📊 Total documents in store: {vectorstore.collection.count()}")
```

### Step 3: Reinitialize the Retriever (Add this as a new cell)

```python
# STEP 3: Recreate retriever with clean vector store
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
print("✅ RAG Retriever initialized and ready!")
```

### Step 4: Test Retrieval with Sample Queries (Add this as a new cell)

```python
# STEP 4: Test with actual queries from your PDFs

# Test Query 1: About RAG (should find content from "Rag Application and its use cases.pdf")
print("=" * 70)
print("🔍 TEST QUERY 1: What is RAG?")
print("=" * 70)
results_1 = rag_retriever.retrieve("What is RAG and how does it work?", top_k=3)
print(f"\n📄 Found {len(results_1)} results\n")

for result in results_1:
    print(f"{'='*70}")
    print(f"📍 Rank {result['rank']} | Similarity Score: {result['similarity_score']:.3f}")
    print(f"📄 Source: {result['metadata'].get('source', 'Unknown')}")
    print(f"📖 Content Preview:\n{result['content'][:300]}...")
    print()

# Test Query 2: About Tech Stack
print("\n" + "=" * 70)
print("🔍 TEST QUERY 2: Tech Stack")
print("=" * 70)
results_2 = rag_retriever.retrieve("What technologies are used in this project?", top_k=3)
print(f"\n📄 Found {len(results_2)} results\n")

for result in results_2:
    print(f"{'='*70}")
    print(f"📍 Rank {result['rank']} | Similarity Score: {result['similarity_score']:.3f}")
    print(f"📄 Source: {result['metadata'].get('source', 'Unknown')}")
    print(f"📖 Content Preview:\n{result['content'][:300]}...")
    print()

# Test Query 3: About Varun (should find resume/cover letter content)
print("\n" + "=" * 70)
print("🔍 TEST QUERY 3: Personal Info")
print("=" * 70)
results_3 = rag_retriever.retrieve("Who is Varun Bharadwaj and what are his skills?", top_k=3)
print(f"\n📄 Found {len(results_3)} results\n")

for result in results_3:
    print(f"{'='*70}")
    print(f"📍 Rank {result['rank']} | Similarity Score: {result['similarity_score']:.3f}")
    print(f"📄 Source: {result['metadata'].get('source', 'Unknown')}")
    print(f"📖 Content Preview:\n{result['content'][:300]}...")
    print()
```

### Step 5: Test with ChromaDB Query (Add this as a new cell)

```python
# STEP 5: Verify ChromaDB is working correctly
print("🔍 Testing ChromaDB directly...")
print(f"📊 Total documents in collection: {vectorstore.collection.count()}")

# Get a sample of documents
sample_results = vectorstore.collection.peek(limit=3)
print(f"\n📄 Sample documents in collection:")
for i, doc in enumerate(sample_results['documents']):
    print(f"\nDocument {i+1}:")
    print(f"Preview: {doc[:200]}...")
    print(f"Source: {sample_results['metadatas'][i].get('source', 'Unknown')}")
```

## 🎯 Expected Results

After running these cells, you should see:
- ✅ Document count: 22 (not 44)
- ✅ Each query returns 3-5 relevant results
- ✅ Similarity scores between 0.3 and 1.0
- ✅ Content matches the query topic
- ✅ Source shows the correct PDF file

## 📝 Important Notes

1. **Run cells in order** - Don't skip steps
2. **Only add documents once** - After Step 2, don't run `vectorstore.add_documents()` again
3. **Check document count** - Should be 22, not 44
4. **Test different queries** - Try asking about different topics from your PDFs

## 🚀 Next Steps

Once this is working:
1. Integrate with your Groq API for answer generation
2. Add more sophisticated retrieval (e.g., reranking)
3. Build a simple UI with Streamlit
