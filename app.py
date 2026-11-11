import os
from flask import Flask, render_template, request
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

app = Flask(__name__)

API_KEY = os.environ.get("GOOGLE_API_KEY", "AQ.Ab8RN6I0Pxt-xvTjrcxTI1jxvKLdXluV7_EUxLjdSa_murnllA")
client = genai.Client(api_key=API_KEY)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DOC_FILES = ["doc1_basics.txt", "doc2_functions.txt", "doc3_oop.txt"]

def split_text_into_chunks(text, chunk_size=200, overlap=30):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def setup_vector_store():
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection("python_docs")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="python_docs")

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    doc_id = 0
    for file_path in DOC_FILES:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = split_text_into_chunks(content)
            for chunk in chunks:
                documents_to_add.append(chunk)
                metadatas_to_add.append({"source": file_path})
                ids_to_add.append(f"id_{doc_id}")
                doc_id += 1

    if not documents_to_add:
        print("[УВАГА] Не знайдено жодного текстового документа!")
        return collection


    embeddings_list = embedding_model.encode(documents_to_add).tolist()

    collection.add(
        documents=documents_to_add,
        embeddings=embeddings_list,
        metadatas=metadatas_to_add,
        ids=ids_to_add
    )
    print(f"[OK] Створено векторну базу. Завантажено {len(documents_to_add)} chunks.")
    return collection

collection = setup_vector_store()

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    retrieved_docs = []
    rag_response = ""
    no_rag_response = ""

    if request.method == "POST":
        query = request.form.get("query")
    
        query_embedding = embedding_model.encode([query]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        retrieved_texts = results['documents'][0] if results['documents'] else []
        retrieved_sources = results['metadatas'][0] if results['metadatas'] else []
        
        for text, meta in zip(retrieved_texts, retrieved_sources):
            retrieved_docs.append({
                'page_content': text,
                'metadata': {'source': meta['source']}
            })

        context_text = "\n\n".join(retrieved_texts)

        rag_prompt = f"""
Відповідай лише українською мовою!
Використовуючи наведений нижче контекст, дай відповідь на запитання користувача.
Якщо відповіді на запитання немає в контексті, чітко повідом про це.

Контекст:
{context_text}

Запитання:
{query}
"""
        rag_res = client.models.generate_content(
            model="gemini-3.5-flash-lite", 
            contents=rag_prompt
        )
        rag_response = rag_res.text
        
        
        no_rag_prompt = f"Відповідай лише українською мовою на це запитання: {query}"
        no_rag_res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=query
        )
        no_rag_response = no_rag_res.text

    return render_template(
        "index.html",
        query=query,
        retrieved_docs=retrieved_docs,
        rag_response=rag_response,
        no_rag_response=no_rag_response
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)