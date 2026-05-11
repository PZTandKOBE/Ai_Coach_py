from pydantic import BaseModel, Field

class GlmUsageResponse(BaseModel):
    total_requests: int = Field(..., description="总调用频次")
    total_tokens: int = Field(..., description="消耗总Token数")
    period: str = Field(default="current_session", description="统计周期 (当前服务运行期间)")