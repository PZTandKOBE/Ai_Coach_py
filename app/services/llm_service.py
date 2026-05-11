import os
import json
from zhipuai import ZhipuAI
from langchain.prompts import PromptTemplate
from app.services.rag_service import rag_service
from app.models.coach_models import GenerateRequest
from app.services.sys_service import sys_service


class LLMService:
    def __init__(self):
        # 换回最稳定的同步客户端
        self.client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))

        # 预设 System Prompt 模板
        self.system_template = PromptTemplate(
            input_variables=["context", "task_type"],
            template="""你是一位顶级 AI 健身饮食教练。
你的任务是根据用户的身体数据，输出硬核、专业的{task_type}方案。
请参考以下从专业知识库中检索到的最新理论来指导用户：
【知识库参考上下文开始】
{context}
【知识库参考上下文结束】
回答要求：直接输出方案，不要寒暄，语气要专业、冷峻且充满科技感。格式清晰易读。"""
        )

    # 注意：这里去掉了 async，变成标准的同步生成器，FastAPI 会在底层自动用线程池处理流式下发，不阻塞！
    def generate_stream(self, request: GenerateRequest):
        """流式生成方案 (SSE)"""
        # 1. 提取用户特征词，去 RAG 检索本土理论
        query_keyword = f"{request.profile_data.goal} {request.profile_data.preference}"
        rag_results, _ = rag_service.query_sandbox(query=query_keyword, top_k=2)

        # 将命中片段拼接成上下文
        context_str = "\n".join([res["content"] for res in rag_results]) if rag_results else "暂无额外本地知识。"

        task_name = "饮食计划" if request.task_type == "diet" else "训练计划"
        system_prompt = self.system_template.format(context=context_str, task_type=task_name)

        user_prompt = f"我的数据：身高{request.profile_data.height_cm}cm, 体重{request.profile_data.weight_kg}kg, 目标：{request.profile_data.goal}。请给我方案。"

        # 2. 调用智谱 GLM-4 进行流式推理
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.7
        )

        # 3. 标准生成器，按 SSE (Server-Sent Events) 标准格式 yield 数据 (注意这里去掉了 async)
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                # 按照你 API 文档要求的 JSON 格式包装每一滴数据
                data_obj = {"chunk": delta_content}
                yield f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"
            if chunk.usage:
                # 获取本次请求的总 token (包含 prompt_tokens 和 completion_tokens)
                total_tokens_used = chunk.usage.total_tokens
                # 记录到系统监控中
                sys_service.record_usage(total_tokens_used)
        # 流结束标志
        yield "data: [DONE]\n\n"


# 抛出单例
llm_service = LLMService()