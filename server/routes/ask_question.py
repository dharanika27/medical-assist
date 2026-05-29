import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from pydantic import PrivateAttr

from logger import logger
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

SERVER_DIR = Path(__file__).resolve().parents[1]
load_dotenv(SERVER_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

router = APIRouter()


class SimpleRetriever(BaseRetriever):
    _docs: List[Document] = PrivateAttr(default_factory=list)

    def __init__(self, documents: List[Document]):
        super().__init__()
        self._docs = documents

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._docs


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")
        logger.info(
            f"querying pinecone index={PINECONE_INDEX_NAME}, model={EMBEDDING_MODEL}, top_k={RETRIEVAL_TOP_K}"
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        embed_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

        embedded_query = embed_model.embed_query(question)
        res = index.query(
            vector=embedded_query,
            top_k=RETRIEVAL_TOP_K,
            include_metadata=True,
        )

        docs = []
        sources = []
        matches = res.get("matches", [])
        logger.info(f"🔍 Pinecone returned {len(matches)} matches")

        for i, match in enumerate(matches, start=1):
            metadata = match.get("metadata") or {}
            text = metadata.get("text") or metadata.get("page_content") or ""
            if not text.strip():
                logger.warning(f"⚠️ Match {i} has empty text metadata")
                continue

            source = metadata.get("source") or metadata.get("file_path") or metadata.get("filename")
            page = metadata.get("page")
            logger.info(
                f"📄 Match {i}: score={match.get('score')}, source={source}, page={page}, preview={text[:180]!r}"
            )

            metadata = {
                **metadata,
                "score": match.get("score"),
            }
            docs.append(Document(page_content=text, metadata=metadata))

            if source and source not in sources:
                sources.append(source)

        if not docs:
            return {
                "response": "I'm sorry, but I couldn't find relevant information in the provided documents.",
                "sources": [],
            }

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)
        result["sources"] = sources

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
