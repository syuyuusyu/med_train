from fastapi import APIRouter, Depends, Request
from ..services import RetrieverService
from app.dependencies import container

router = APIRouter()


@router.post("/triage")
async def triage(request: Request, retriever_service: RetrieverService = Depends(container.retriever_service)):
    data = await request.json()
    query = data.get('query', '')
    retrieved_chunks = retriever_service.retrieve(query, k=10)
    return {"data": "example response", "success": True}

@router.get("/reloadDocinfo")
def reload_docinfo(retriever_service: RetrieverService = Depends(container.retriever_service)):
    retriever_service.reload()
    return {"success": "success", "data": "医生信息已重新加载"}