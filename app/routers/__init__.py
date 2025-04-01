from fastapi import FastAPI
from .triage import router as triage_router

def register_routers(app: FastAPI):
    app.include_router(triage_router, prefix="/ai")