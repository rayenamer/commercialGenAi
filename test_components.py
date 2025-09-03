#!/usr/bin/env python3

# Test script to verify LLM and retriever loading
import os
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_llm():
    try:
        print("Testing LLM loading...")
        llm = Llama(
            model_path="C:/Users/User/AppData/Local/llama.cpp/TheBloke_Mistral-7B-Instruct-v0.2-GGUF_mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_threads=12,
            n_ctx=1024,
            verbose=False
        )
        print("✓ LLM loaded successfully")
        return llm
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def document_loader():
    if os.path.exists("doc.txt"):
        print("✓ doc.txt found")
        loader = TextLoader("doc.txt", encoding="utf-8")
        loaded_document = loader.load()
        print(f"✓ Document loaded with {len(loaded_document)} chunks")
        return loaded_document
    else:
        print("✗ doc.txt file not found!")
        return []

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    print(f"✓ Text split into {len(chunks)} chunks")
    return chunks

class SimpleTFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        print(f"✓ TF-IDF retriever initialized with {len(documents)} documents")
    
    def get_relevant_documents(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

def get_retriever():
    splits = document_loader()
    if not splits:
        print("✗ No documents loaded")
        return None
    chunks = text_splitter(splits)
    retriever = SimpleTFIDFRetriever(chunks)
    print("✓ Retriever created successfully")
    return retriever

def load_prompt():
    if os.path.exists("prompt.txt"):
        with open("prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print(f"✓ Prompt loaded: {prompt[:50]}...")
        return prompt
    else:
        print("✗ prompt.txt not found, using default")
        return "Provide an answer based on the document and conversation context."

if __name__ == "__main__":
    print("=== Testing Components ===")
    
    # Test retriever
    print("\n1. Testing Retriever:")
    retriever = get_retriever()
    
    # Test LLM
    print("\n2. Testing LLM:")
    llm = get_llm()
    
    # Test prompt
    print("\n3. Testing Prompt:")
    prompt = load_prompt()
    
    # Test simple query if both work
    if retriever and llm:
        print("\n4. Testing Simple Query:")
        try:
            test_query = "What is this document about?"
            relevant_docs = retriever.get_relevant_documents(test_query)
            print(f"✓ Retrieved {len(relevant_docs)} relevant documents")
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            full_prompt = f"{prompt}\n\nDocument context:\n{context}\n\nUser: {test_query}\nAI: "
            
            print("✓ Testing LLM response...")
            response = llm(full_prompt, max_tokens=50, echo=False)
            if isinstance(response, dict):
                response_text = response.get('choices', [{}])[0].get('text', 'No response')
            else:
                response_text = str(response)
            print(f"✓ LLM Response: {response_text.strip()}")
            
        except Exception as e:
            print(f"✗ Error in query test: {e}")
    
    print("\n=== Test Complete ===")