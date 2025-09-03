from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import time
import threading
from datetime import datetime
import uuid
import os
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for conversations
conversations = {}
active_users = set()
admin_sessions = set()

## Local LLM (copied exactly from working RagApp.py)
def get_llm():
    try:
        llm = Llama(
            model_path="C:/Users/User/AppData/Local/llama.cpp/TheBloke_Mistral-7B-Instruct-v0.2-GGUF_mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            n_threads=12,
            n_ctx=1024,
            verbose=False
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Document loader for text file
## Document loader for text file
def document_loader():
    if os.path.exists("doc.txt"):
        loader = TextLoader("doc.txt", encoding="utf-8")
        loaded_document = loader.load()
        return loaded_document
    else:
        print("doc.txt file not found!")
        return []

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(data)

# Simple TF-IDF based retriever
class SimpleTFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
    
    def get_relevant_documents(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

## Retriever
def get_retriever():
    splits = document_loader()
    if not splits:
        return None
    chunks = text_splitter(splits)
    retriever = SimpleTFIDFRetriever(chunks)
    return retriever

## Load prompt from file
def load_prompt():
    if os.path.exists("prompt.txt"):
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        return "Provide an answer based on the document and conversation context."

class ConversationManager:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm()
        if self.llm is None:
            print("Warning: LLM failed to load. Responses will be limited.")
        if self.retriever is None:
            print("Warning: Document retriever failed to load. RAG functionality disabled.")
    
    def generate_response_stream(self, user_message, conversation_id, user_id):
        """Generate response token by token using local LLM and emit to both user and admin"""
        try:
            # Check if LLM is available (following RagApp.py pattern)
            if not self.llm:
                error_msg = "Error: Could not load the local LLM model."
                socketio.emit('error', {'message': error_msg}, room=user_id)
                return
            
            # Check if retriever is available
            if not self.retriever:
                error_msg = "Error: Could not load the document. Make sure doc.txt exists."
                socketio.emit('error', {'message': error_msg}, room=user_id)
                return
            
            # Get relevant context from RAG
            relevant_docs = self.retriever.get_relevant_documents(user_message)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Load system prompt from prompt.txt
            system_prompt = load_prompt()
            
            # Create full prompt with context (following RagApp.py pattern)
            full_prompt = f"{system_prompt}\n\nDocument context:\n{context}\n\nUser: {user_message}\nAI: "
            
            # Initialize response
            full_response = ""
            
            # Real token-by-token generation using local LLM streaming (exactly like RagApp.py)
            try:
                for token_data in self.llm(full_prompt, max_tokens=300, echo=False, stream=True):
                    token = token_data["choices"][0]["text"]
                    full_response += token
                    
                    # Emit token to user
                    socketio.emit('token_received', {
                        'token': token,
                        'conversation_id': conversation_id,
                        'is_complete': False
                    }, room=user_id)
                    
                    # Emit to admin for live monitoring
                    socketio.emit('admin_token_update', {
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'token': token,
                        'partial_response': full_response,
                        'is_complete': False
                    }, room='admin')
                    
                    # Small delay to make streaming visible
                    time.sleep(0.02)
                    
            except Exception as llm_error:
                error_msg = f"Error generating response: {llm_error}"
                print(f"LLM Error: {llm_error}")
                socketio.emit('error', {'message': error_msg}, room=user_id)
                return
            
            # Clean up response
            full_response = full_response.strip()
            
            # Mark as complete
            socketio.emit('token_received', {
                'token': '',
                'conversation_id': conversation_id,
                'is_complete': True,
                'full_response': full_response
            }, room=user_id)
            
            socketio.emit('admin_token_update', {
                'user_id': user_id,
                'conversation_id': conversation_id,
                'token': '',
                'partial_response': full_response,
                'is_complete': True
            }, room='admin')
            
            # Store in conversation history
            if conversation_id not in conversations:
                conversations[conversation_id] = []
            
            conversations[conversation_id].append({
                'type': 'user',
                'message': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            conversations[conversation_id].append({
                'type': 'assistant',
                'message': full_response,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"Error in generate_response_stream: {e}")
            socketio.emit('error', {'message': error_msg}, room=user_id)
            socketio.emit('admin_error', {
                'user_id': user_id,
                'conversation_id': conversation_id,
                'error': error_msg
            }, room='admin')

conversation_manager = ConversationManager()

@app.route('/')
def user_interface():
    """Main user chat interface"""
    return render_template('chat.html')

@app.route('/admin')
def admin_interface():
    """Admin monitoring interface"""
    return render_template('admin.html')

@app.route('/api/conversations')
def get_conversations():
    """API endpoint to get all conversations for admin"""
    return jsonify(conversations)

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    active_users.discard(request.sid)
    admin_sessions.discard(request.sid)

@socketio.on('join_user')
def handle_join_user(data):
    """User joins their personal room"""
    user_id = data.get('user_id', request.sid)
    join_room(user_id)
    active_users.add(user_id)
    
    # Notify admin of new user
    socketio.emit('user_joined', {
        'user_id': user_id,
        'timestamp': datetime.now().isoformat()
    }, room='admin')
    
    emit('joined', {'user_id': user_id})

@socketio.on('join_admin')
def handle_join_admin():
    """Admin joins monitoring room"""
    join_room('admin')
    admin_sessions.add(request.sid)
    
    # Send current active users and conversations
    emit('admin_data', {
        'active_users': list(active_users),
        'conversations': conversations
    })

@socketio.on('send_message')
def handle_message(data):
    """Handle user message and generate streaming response"""
    user_message = data.get('message', '')
    user_id = data.get('user_id', request.sid)
    conversation_id = data.get('conversation_id', str(uuid.uuid4()))
    
    if not user_message.strip():
        return
    
    # Notify admin of new message
    socketio.emit('admin_new_message', {
        'user_id': user_id,
        'conversation_id': conversation_id,
        'message': user_message,
        'timestamp': datetime.now().isoformat()
    }, room='admin')
    
    # Start response generation in background thread
    thread = threading.Thread(
        target=conversation_manager.generate_response_stream,
        args=(user_message, conversation_id, user_id)
    )
    thread.daemon = True
    thread.start()

@socketio.on('admin_request_conversation')
def handle_admin_conversation_request(data):
    """Admin requests specific conversation history"""
    conversation_id = data.get('conversation_id')
    if conversation_id in conversations:
        emit('conversation_history', {
            'conversation_id': conversation_id,
            'messages': conversations[conversation_id]
        })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
    