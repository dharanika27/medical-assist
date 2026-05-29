import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

SERVER_DIR = Path(__file__).resolve().parents[1]
load_dotenv(SERVER_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are MediBot, an AI-powered assistant trained to help users understand uploaded medical documents.

Answer using only the provided PDF context. Do not use outside medical knowledge, even if you know the answer.
A retrieved chunk being present does not mean the question is answered by the PDF.
If the PDF context does not answer the question, say exactly:
"I'm sorry, but I couldn't find relevant information in the provided documents."

Context:
{context}

User Question:
{question}

Answer:
- Answer naturally and directly when the answer is present.
- Include only facts stated in the context.
- Do not mention source files, page numbers, scores, or citations in the answer.
- Do not add extra explanation, prevention tips, treatment advice, or general medical knowledge.
""",
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
