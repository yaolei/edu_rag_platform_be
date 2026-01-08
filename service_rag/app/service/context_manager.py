import time
import json
from typing import List, Dict, Optional

class SimpleContextManager:
    """简化的上下文管理器，易于集成到现有代码"""

    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.history: List[Dict] = []
        self.conversation_id = None

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到历史"""
        message = {
            "role": role,  # "user" 或 "assistant"
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.history.append(message)

        # 保持历史记录不超过最大限制
        if len(self.history) > self.max_history * 2:  # 每轮包含用户和AI
            self.history = self.history[-(self.max_history * 2):]

    def get_formatted_history(self) -> str:
        """获取格式化的历史记录字符串，用于提示词"""
        if not self.history:
            return ""

        # 只取最近的历史记录
        recent_history = self.history[-(self.max_history * 2):]

        # 格式化为字符串
        formatted = []
        for msg in recent_history:
            if msg["role"] == "user":
                formatted.append(f"用户: {msg['content']}")
            else:
                formatted.append(f"助手: {msg['content']}")

        return "\n".join(formatted)

    def clear(self):
        """清空历史记录"""
        self.history = []
        self.conversation_id = None

    def to_json(self) -> str:
        """转换为JSON字符串，便于前端存储"""
        return json.dumps({
            "conversation_id": self.conversation_id,
            "history": self.history,
            "max_history": self.max_history
        })

    @classmethod
    def from_json(cls, json_str: str):
        """从JSON字符串恢复"""
        try:
            data = json.loads(json_str)
            manager = cls(max_history=data.get("max_history", 3))
            manager.conversation_id = data.get("conversation_id")
            manager.history = data.get("history", [])
            return manager
        except:
            return cls()