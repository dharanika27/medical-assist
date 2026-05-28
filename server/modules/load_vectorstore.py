import os
import uuid
import time

from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex-v3"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- PINECONE ---------------- #

pc = Pinecone(api_key=PINECONE_API_KEY)

spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV
)

existing_indexes = [
    i["name"]
    for i in pc.list_indexes()
]

if PINECONE_INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="dotproduct",
        spec=spec
    )

    while not pc.describe_index(
        PINECONE_INDEX_NAME
    ).status["ready"]:

        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# ---------------- VECTORSTORE ---------------- #

def load_vectorstore(uploaded_files):

    embed_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    file_paths = []

    # SAVE FILES

    for file in uploaded_files:

        save_path = Path(UPLOAD_DIR) / file.filename

        with open(save_path, "wb") as f:
            f.write(file.file.read())

        file_paths.append(str(save_path))

    # PROCESS FILES

    for file_path in file_paths:

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=3000,
            chunk_overlap=500
        )

        chunks = splitter.split_documents(documents)

        # REMOVE EMPTY CHUNKS

        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content.strip()
        ]

        texts = [
            chunk.page_content
            for chunk in valid_chunks
        ]

        metadatas = [
            {
                **chunk.metadata,
                "text": chunk.page_content
            }
            for chunk in valid_chunks
        ]

        ids = [
            f"{Path(file_path).stem}-{i}-{uuid.uuid4()}"
            for i in range(len(valid_chunks))
        ]

        print(f"🔍 Embedding {len(texts)} chunks...")

        embeddings = embed_model.embed_documents(texts)

        # CLEAR OLD DATA
        print("🗑 Clearing old vectors from Pinecone...")

        try:
            stats = index.describe_index_stats()

            total_vectors = stats.get("total_vector_count", 0)

            if total_vectors > 0:

                index.delete(delete_all=True)

                # wait until deletion completes
                while True:

                    stats = index.describe_index_stats()

                    remaining = stats.get("total_vector_count", 0)

                    print(f"Remaining vectors: {remaining}")

                    if remaining == 0:
                        break

                    time.sleep(2)

                print("✅ Old vectors cleared successfully")

            else:
                print("ℹ️ No existing vectors found")

        except Exception as e:

            print(f"⚠️ Skipping delete step: {e}")

        vectors = []

        for id_, embedding, metadata in zip(ids, embeddings, metadatas):

            vectors.append({
                "id": id_,
                "values": embedding,
                "metadata": metadata
            })

        print("📤 Uploading to Pinecone...")

        index.upsert(vectors=vectors)

        print(f"✅ Upload complete for {file_path}")