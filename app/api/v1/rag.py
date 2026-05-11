from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.rag_models import QueryRequest, QueryResponse, UploadResponse, ChunkResult
from app.services.rag_service import rag_service

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, summary="文档解析入库")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="仅支持 PDF 或 TXT 格式的文件入库")

    try:
        chunks_created, time_cost = await rag_service.process_and_store_document(file)
        return UploadResponse(
            status="success",
            chunks_created=chunks_created,
            time_cost_ms=time_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理异常: {str(e)}")


@router.post("/sandbox/query", response_model=QueryResponse, summary="知识检索沙盒")
async def sandbox_query(request: QueryRequest):
    try:
        results, time_cost = rag_service.query_sandbox(request.query, request.top_k)
        formatted_results = [
            ChunkResult(content=res["content"], score=res["score"])
            for res in results
        ]
        return QueryResponse(
            results=formatted_results,
            time_cost_ms=time_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量检索异常: {str(e)}")