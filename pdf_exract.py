#%%
import pdfplumber
import faiss
import numpy as np
#%%
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Example
pdf_text = extract_text_from_pdf("Gb.pdf")
print(pdf_text[:100])   # print first 1000 characters
#%%
#convert to chunks
def chunks_creation(text):
    chunks = []
    words = text.split()

    for i in range(0,len(words),300):
        c = " ".join(words[i:i+300])
        chunks.append(c)
    return chunks

chunks = chunks_creation(pdf_text)
#
#%%
print(len(chunks))
print("f:",chunks[1])

#%%
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return embed_model.encode(chunks)

chunk_embeddings = embed_chunks(chunks)
print("Embedding shape:", chunk_embeddings.shape)
#it has 4 rows ie 4 chunks and each chunk has 382 vector


# %% Creating vector database
def build_faiss_index(emb):
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb.astype(np.float32))
    return index

index = build_faiss_index(np.array(chunk_embeddings)) 
print("FAISS created")

# %%
def retrieve_chunks(query, index, chunks, k=3):
    query_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(query_emb).astype(np.float32), k)
    
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

query = "real madrid"
top_chunks = retrieve_chunks(query, index, chunks)
print("Retrieved chunks:")
for i, ch in enumerate(top_chunks):
    print(f"--- Chunk {i+1} ---")
    print(ch)

# %%
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



# %%
context = "\n\n".join(top_chunks)


def generate_answer_with_groq(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not present in the context, reply: "Sorry,I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="groq/compound",   # or llama-3-70b, etc.
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content
    #return response.choices[0].message["content"]

# %%
query = input("Your Question on Bale:")



answer = generate_answer_with_groq(query, top_chunks)
print(answer)

# %%
