class SysService:
    def __init__(self):
        # 简单在内存中记录，重启服务会清零
        # 生产环境建议改写为写入 Redis (如 INCR 命令)
        self.total_requests = 0
        self.total_tokens = 0

    def record_usage(self, tokens: int):
        """每次大模型调用结束后触发此方法记录"""
        self.total_requests += 1
        self.total_tokens += tokens

    def get_stats(self):
        """提供给接口读取大盘数据"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens
        }

sys_service = SysService()