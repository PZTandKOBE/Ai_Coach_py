import os
import time
import fitz  # PyMuPDF
from fastapi import UploadFile
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI


class ZhipuEmbeddings(Embeddings):
    """自定义封装智谱 Embedding 模型以完美适配 LangChain"""

    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量向量化 (为防止单次请求过大，这里采用循环请求，生产环境可结合并发优化)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return response.data[0].embedding


class RAGService:
    def __init__(self):
        self.index_path = os.getenv("FAISS_INDEX_PATH", "./app/data/faiss_index")
        self.embeddings = ZhipuEmbeddings()
        # 初始化切片器：500字一块，保留50字重叠防断句丢失上下文
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        """服务启动时尝试加载本地已有的 FAISS 索引"""
        if os.path.exists(self.index_path) and os.path.exists(os.path.join(self.index_path, "index.faiss")):
            # LangChain 新版处于安全考虑，反序列化需显式开启 allow_dangerous_deserialization
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    async def process_and_store_document(self, file: UploadFile) -> tuple[int, int]:
        """解析文件 -> 切片 -> 向量化 -> 入库"""
        start_time = time.time()

        content = ""
        # 1. 提取文本内容
        if file.filename.lower().endswith(".pdf"):
            pdf_bytes = await file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                content += page.get_text()
            doc.close()
        else:
            content = (await file.read()).decode("utf-8")

        # 2. 进行文本切片
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return 0, int((time.time() - start_time) * 1000)

        # 3. 向量化并入库 FAISS
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)

        # 4. 持久化存储到本地磁盘
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

        time_cost = int((time.time() - start_time) * 1000)
        return len(chunks), time_cost

    def query_sandbox(self, query: str, top_k: int = 3):
        """知识库检索沙盒测试"""
        start_time = time.time()
        if self.vector_store is None:
            return [], int((time.time() - start_time) * 1000)

        # 进行相似度检索
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

        results = [
            {"content": doc.page_content, "score": float(score)}
            for doc, score in docs_and_scores
        ]
        time_cost = int((time.time() - start_time) * 1000)
        return results, time_cost


# 抛出单例供路由调用
rag_service = RAGService()