import os
import base64
import json
from fastapi import UploadFile
from zhipuai import ZhipuAI
from app.models.coach_models import VisionResponse


class VisionService:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))

    async def analyze_calories(self, file: UploadFile) -> VisionResponse:
        # 1. 读取图片并转为 base64
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # 2. 组装 GLM-4V 视觉 Prompt
        prompt = """
        请分析图片中的食物。
        严格以JSON格式返回，必须包含以下字段：
        {"food_name": "食物名", "estimated_weight_g": 重量数字, "calories_kcal": 热量数字, "protein_g": 蛋白数字}
        不要输出任何其他解释文字或Markdown标记，只要纯JSON！
        """

        response = self.client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ]
        )

        # 3. 解析大模型返回的 JSON 字符串
        try:
            result_text = response.choices[0].message.content.strip()
            # 兼容大模型有时手贱加上的 ```json ``` 标记
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()

            data = json.loads(result_text)
            return VisionResponse(**data)
        except Exception as e:
            raise ValueError(f"视觉解析失败或模型未按格式返回: {str(e)}\n原始返回: {result_text}")


vision_service = VisionService()