from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class UserProfile(BaseModel):
    age: int = Field(..., description="年龄")
    gender: str = Field(..., description="性别 (男/女)")
    height_cm: float = Field(..., description="身高(cm)")
    weight_kg: float = Field(..., description="体重(kg)")
    goal: str = Field(..., description="目标 (如: 减脂、增肌、塑形)")
    preference: str = Field(default="无特殊忌口", description="饮食/训练偏好")

class GenerateRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    task_type: str = Field(..., description="任务类型 (diet: 饮食方案, workout: 训练方案)")
    profile_data: UserProfile

class VisionResponse(BaseModel):
    food_name: str = Field(..., description="识别出的食物名称")
    estimated_weight_g: int = Field(..., description="估算重量(克)")
    calories_kcal: int = Field(..., description="估算热量(千卡)")
    protein_g: float = Field(..., description="蛋白质含量(克)")