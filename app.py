
import json
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- Configuration --- #
# IMPORTANT: For Cloud Run deployment, ensure these files are available in the deployment environment.
# You would typically upload them with your Docker image or fetch from Google Cloud Storage.
CHUNKS_FILE = 'leemai_all_chunks.json'
EMBEDDINGS_FILE = 'leemai_document_embeddings.npy'

# --- Load Model and Data --- #
# These are loaded once when the application starts
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# Load chunks
print(f"Loading chunks from {CHUNKS_FILE}...")
if not os.path.exists(CHUNKS_FILE):
    print(f"Error: {CHUNKS_FILE} not found. Please ensure it's in the same directory as app.py or provide the correct path.")
    all_leemai_chunks = []
else:
    with open(CHUNKS_FILE, 'r') as f:
        all_leemai_chunks = json.load(f)
    print(f"Loaded {len(all_leemai_chunks)} chunks.")

# Load embeddings
print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
if not os.path.exists(EMBEDDINGS_FILE):
    print(f"Error: {EMBEDDINGS_FILE} not found. Please ensure it's in the same directory as app.py or provide the correct path.")
    document_embeddings = np.array([]) # Initialize as empty array if not found
else:
    document_embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded embeddings with shape: {document_embeddings.shape}")


# --- Helper Functions (Copied from previous steps) ---
def get_relevant_chunks(query, model, document_chunks, document_embeddings, top_k=3):
    """Finds the most semantically similar document chunks to a given query."""
    if not document_chunks or document_embeddings.size == 0:
        return []

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    relevant_chunks = []
    for i in top_k_indices:
        relevant_chunks.append({
            'chunk': document_chunks[i],
            'similarity': similarities[i]
        })
    return relevant_chunks

def generate_response_from_chunks(relevant_chunks):
    """Generates a basic response by concatenating relevant chunks with improved formatting."""
    if not relevant_chunks:
        return "I couldn't find relevant information for your query in my knowledge base."

    response_text = "Based on the information I have, here are some relevant details:\n\n"

    for i, item in enumerate(relevant_chunks):
        response_text += f"**Information Block {i+1}:**\n"
        response_text += f"{item['chunk']}\n\n"

    response_text += "I hope these details provide a comprehensive answer to your query!"
    return response_text


# --- Flask Routes ---
@app.route('/')
def home():
    return "LeemAI is running! Use the /ask endpoint to query."

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    if not all_leemai_chunks or document_embeddings.size == 0:
        return jsonify({'error': 'LeemAI knowledge base not loaded. Check server logs.'}), 500

    relevant_chunks = get_relevant_chunks(
        user_query, embedding_model, all_leemai_chunks, document_embeddings
    )
    leemai_response = generate_response_from_chunks(relevant_chunks)

    return jsonify({'response': leemai_response})


# --- Main Entry Point ---
if __name__ == '__main__':
    # In a production environment like Cloud Run, Flask applications are typically
    # run by a WSGI server (like Gunicorn). This block is for local testing.
    # For Cloud Run, ensure your Dockerfile specifies the entrypoint for Gunicorn
    # or equivalent.
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
