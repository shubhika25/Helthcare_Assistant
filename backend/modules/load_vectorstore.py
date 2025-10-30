import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Load environment variables
load_dotenv()

# ✅ Get the API key safely
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalassistantindex"

if not PINECONE_API_KEY:
    raise ValueError("❌ Pinecone API key not found! Check your .env or environment variables.")

# ✅ Initialize Pinecone only once
pc = Pinecone(api_key="PINECONE_API_KEY ")
print(f"✅ Pinecone initialized successfully")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

# ✅ Local embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = 384  # model outputs 384-dim embeddings

# ✅ Create Pinecone index if not exists
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"🧠 Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=embedding_dim,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("⏳ Waiting for index to be ready...")
        time.sleep(2)

index = pc.Index(PINECONE_INDEX_NAME)

# ✅ Utility: Batch upsert for large files
def batch_upsert(index, ids, embeddings, metadatas, batch_size=100):
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_emb = embeddings[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        index.upsert(vectors=zip(batch_ids, batch_emb, batch_meta))

# ✅ Main function to embed and upload
def load_vectorstore(uploaded_files, doc_type="lab_report"):
    file_paths = []
    summary = []

    for file in uploaded_files:
        filename = file["filename"]
        content = file["content"]

        save_path = Path(UPLOAD_DIR) / filename
        with open(save_path, "wb") as f:
            f.write(content)
        file_paths.append(str(save_path))

    for file_path in file_paths:
        print(f"📘 Processing file: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "filename": Path(file_path).name,
                "type": doc_type,
                "source": "user_upload",
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content
            }
            for chunk in chunks
        ]

        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc=f"Upserting {Path(file_path).name}") as progress:
            batch_upsert(index, ids, embeddings, metadatas, batch_size=100)
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
        summary.append({"file": Path(file_path).name, "chunks": len(chunks)})

    return {"status": "success", "uploaded_files": summary}
