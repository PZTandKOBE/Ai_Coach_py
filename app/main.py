import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# 占位：导入后续各个模块的路由
# from app.api.v1 import rag, coach
from app.api.v1 import coach, rag, sys

def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例
    """
    app = FastAPI(
        title="AI 健身饮食智能教练 - AI 微服务",
        description="承接高并发的大语言模型推理请求，处理文档文本切片、本地向量检索（RAG）以及流式数据（SSE）下发。",
        version="1.0.0"
    )

    # CORS 跨域配置
    # 由于该微服务可能被 Spring Boot 端、C 端小程序(前端调试时)、B 端 Vue3 后台同时调用
    # 此处允许所有跨域请求，生产环境请务必将 allow_origins 缩小至实际安全域名
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册子路由 (我们会在后续阶段逐一实现并解开注释)
    app.include_router(coach.router, prefix="/api/ai/v1/coach", tags=["智能生成引擎"])
    app.include_router(rag.router, prefix="/api/ai/v1/rag", tags=["RAG知识库实验舱"])
    app.include_router(sys.router, prefix="/api/ai/v1/sys", tags=["运营与监控"])

    return app

app = create_app()

@app.get("/health", tags=["系统探针"])
async def health_check():
    """
    K8s / Spring Boot 监控使用的健康检查接口
    """
    return {
        "status": "ok",
        "message": "AI 微服务运行正常",
        "environment": os.getenv("ENVIRONMENT", "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    # 本地调试运行入口
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)