from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from modules.llm import get_llm_chain
from modules.query_handlers import query_chain

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pinecone import Pinecone
from pydantic import Field

from typing import List
from logger import logger

import os

router = APIRouter()


# ---------------- RETRIEVER ---------------- #

class SimpleRetriever(BaseRetriever):

    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


# ---------------- ASK ROUTE ---------------- #

@router.post("/ask/")
async def ask_question(question: str = Form(...)):

    try:

        logger.info(f"user query: {question}")

        # ---------------- PINECONE ---------------- #

        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        )

        index = pc.Index(
            os.environ["PINECONE_INDEX_NAME"]
        )

        # ---------------- EMBEDDING MODEL ---------------- #

        embed_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview"
        )

        # ---------------- EMBED QUESTION ---------------- #

        embedded_query = embed_model.embed_query(question)

        # ---------------- SEARCH VECTOR DB ---------------- #

        res = index.query(
            vector=embedded_query,
            top_k=10,
            include_metadata=True,
            include_values=False
        )

        matches = res.get("matches", [])

        print(f"\nTOTAL MATCHES: {len(matches)}")

        # ---------------- SCORE FILTER ---------------- #

        SCORE_THRESHOLD = 0.65

        docs = []

        print("\n========== RETRIEVED DOCS ==========\n")

        for i, match in enumerate(matches):

            score = match.get("score", 0)

            metadata = match.get("metadata", {})

            text = metadata.get("text", "")

            print(f"\n--- DOC {i+1} ---")
            print(f"Score: {score}")
            print(text[:1000])

            # FILTER LOW SCORE DOCS

            if score >= SCORE_THRESHOLD:

                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

        print("\n====================================\n")

        # ---------------- NO RELEVANT DOCS ---------------- #

        if not docs:

            return {
                "response": "No relevant medical information found in uploaded documents.",
                "sources": []
            }

        # ---------------- RETRIEVER ---------------- #

        retriever = SimpleRetriever(
            docs=docs
        )

        # ---------------- LLM CHAIN ---------------- #

        chain = get_llm_chain(retriever)

        # ---------------- QUERY CHAIN ---------------- #

        result = query_chain(
            chain,
            question
        )

        logger.info("query successful")

        return result

    except Exception as e:

        logger.exception("Error processing question")

        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )