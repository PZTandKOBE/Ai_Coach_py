from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., description="管理员输入的测试检索语句", examples=["减脂期晚上能吃碳水吗？"])
    top_k: int = Field(default=3, ge=1, le=10, description="期望返回的最相关片段数量")

class ChunkResult(BaseModel):
    content: str = Field(..., description="向量数据库命中的原文切片内容")
    score: float = Field(..., description="相似度得分 (FAISS 默认返回 L2 距离，越小越相似)")

class QueryResponse(BaseModel):
    results: List[ChunkResult]
    time_cost_ms: int = Field(default=0, description="检索总耗时(毫秒)")

class UploadResponse(BaseModel):
    status: str = Field(..., description="状态 (success/failed)")
    chunks_created: int = Field(default=0, description="生成的切片数量")
    time_cost_ms: int = Field(default=0, description="处理入库总耗时(毫秒)")