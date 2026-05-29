import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

SERVER_DIR = Path(__file__).resolve().parents[1]
load_dotenv(SERVER_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
CLEAR_INDEX_ON_UPLOAD = os.getenv("CLEAR_INDEX_ON_UPLOAD", "true").lower() == "true"

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = SERVER_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(exist_ok=True)


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=spec,
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


def load_vectorstore(uploaded_files):
    index = get_pinecone_index()
    embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    file_paths = []

    if CLEAR_INDEX_ON_UPLOAD:
        print("🗑 Clearing old vectors from Pinecone...")
        index.delete(delete_all=True)

        while True:
            stats = index.describe_index_stats()
            remaining = stats.get("total_vector_count", 0)
            print(f"Remaining vectors: {remaining}")
            if remaining == 0:
                break
            time.sleep(2)

        print("✅ Old vectors cleared successfully")

    for file in uploaded_files:
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "text": chunk.page_content,
                "source": Path(file_path).name,
            }
            for chunk in chunks
        ]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
        embeddings = []
        with tqdm(total=len(texts), desc="Embedding chunks") as progress:
            for text in texts:
                embeddings.append(embed_model.embed_documents([text])[0])
                progress.update(1)

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} embeddings for {len(texts)} chunks"
            )

        print(f"📤 Uploading {len(embeddings)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")
