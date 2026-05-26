from fastapi import FastAPI
from routes.upload_pdfs import router as upload_pdfs_router
from routes.ask_question import router as ask_router

app = FastAPI()

app.include_router(upload_pdfs_router)
app.include_router(ask_router)