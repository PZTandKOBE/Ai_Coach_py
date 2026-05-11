from fastapi import APIRouter
from app.models.sys_models import GlmUsageResponse
from app.services.sys_service import sys_service

router = APIRouter()

@router.get("/monitor/glm-usage", response_model=GlmUsageResponse, summary="获取 GLM Token 消耗大盘")
async def get_glm_usage():
    stats = sys_service.get_stats()
    return GlmUsageResponse(
        total_requests=stats["total_requests"],
        total_tokens=stats["total_tokens"]
    )