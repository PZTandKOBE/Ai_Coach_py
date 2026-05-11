from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models.coach_models import GenerateRequest, VisionResponse
from app.services.llm_service import llm_service
from app.services.vision_service import vision_service

router = APIRouter()


@router.post("/generate/stream", summary="核心流式生成 (SSE)")
async def generate_coach_plan(request: GenerateRequest):
    try:
        # 返回 StreamingResponse，指定媒体类型为 text/event-stream
        return StreamingResponse(
            llm_service.generate_stream(request),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.post("/vision/calories", response_model=VisionResponse, summary="视觉热量识别")
async def recognize_calories(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        return await vision_service.analyze_calories(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))