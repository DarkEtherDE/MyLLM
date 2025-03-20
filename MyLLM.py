import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import asyncio
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request

# Flask application setup
app = Flask(__name__)

# Configuration
BOOKS_DIR = "D:\\Books\\AI"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "EleutherAI/gpt-neo-2.7B")  # Configurable model

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBEDDING_MODEL).to(device)

# FAISS Index for efficient similarity search
embedding_dim = 384  # MiniLM-L6-v2 output size
index = faiss.IndexFlatL2(embedding_dim)
book_texts = []  # Stores book passages
metadata = []  # Stores book references

# Load Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text


# Load and Process Books
async def load_books(directory):
    book_count = 0
    for root, _, files in os.walk(directory):  # Walk through subdirectories
        for file in files:
            book_path = os.path.join(root, file)
            text = ""

            if file.endswith(".txt"):
                with open(book_path, "r", encoding="utf-8") as f:
                    text = f.read()

            elif file.endswith(".pdf"):
                text = extract_text_from_pdf(book_path)

            else:
                continue

            # Split text into smaller passages
            passages = splitter.split_text(text)

            # Embed and store passages
            for passage in passages:
                embedding = embedder.encode(passage, convert_to_numpy=True)
                index.add(np.array([embedding]))  # Add to FAISS index
                book_texts.append(passage)
                metadata.append(f"{file} (Path: {book_path})")

            book_count += 1
            print(f"Processed: {file} ({len(passages)} chunks)")

    print(f"\n✅ Processed {book_count} books (TXT & PDF) from '{directory}' and subdirectories.\n")

# Search and Retrieve Passages
def retrieve_passages(query, top_k=3):
    if index.ntotal == 0:  # Ensure FAISS has data
        print("⚠️ No book passages found in FAISS. Have you processed any books?")
        return []

    query_embedding = embedder.encode(query, convert_to_numpy=True)
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(book_texts):
            results.append((book_texts[idx], metadata[idx]))

    return results

# Load LLM (configurable via environment variable)
generator = pipeline("text-generation", model=LLM_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# Generate Response with Citations
def generate_response(query):
    # Retrieve relevant passages
    retrieved_passages = retrieve_passages(query)

    # Construct context from retrieved passages
    context = "\n".join([f"{i+1}. {text} (Source: {source})" for i, (text, source) in enumerate(retrieved_passages)])

    # Generate AI response with context
    prompt = f"Using the following excerpts, answer the user's query creatively:\n{context}\n\nUser Query: {query}\nAnswer:"
    ai_response = generator(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']

    # Append citations
    cited_response = f"{ai_response}\n\nSources:\n" + "\n".join([f"- {source}" for _, source in retrieved_passages])

    return cited_response

# Asynchronous query handling
async def handle_query(query, executor):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_response, query)

# Flask Route for the main UI
@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_input = request.form["query"]
        if user_input.strip():
            # Handle queries concurrently
            executor = ThreadPoolExecutor(max_workers=4)  # Use 4 threads for concurrent queries
            response = asyncio.run(handle_query(user_input, executor))
    return render_template("index.html", response=response)

# Start the Flask app
if __name__ == "__main__":
    asyncio.run(load_books(BOOKS_DIR))  # Load books asynchronously
    app.run(debug=True)

