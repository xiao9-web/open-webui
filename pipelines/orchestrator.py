"""
在线 API 指挥本地模型 Pipeline

架构：
  用户 → 在线强模型（SiliconFlow/OpenAI）分析任务
       → 强模型判断是否需要本地模型处理子任务
       → 本地 Ollama 执行子任务并返回结果
       → 强模型汇总后回复用户

环境变量：
  STRONG_API_URL   - 在线 API 地址，默认 SiliconFlow
  STRONG_API_KEY   - 在线 API 密钥
  STRONG_MODEL     - 强模型名称
  OLLAMA_URL       - Ollama 地址
  LOCAL_MODEL      - 本地模型名称
"""

import json
import os
import requests
from typing import Generator, Iterator, List, Union


class Pipeline:
    def __init__(self):
        self.name = "在线 API 指挥本地模型"
        self.id = "api_orchestrator"

        self.strong_api_url = os.getenv(
            "STRONG_API_URL", "https://api.siliconflow.cn/v1"
        )
        self.strong_api_key = os.getenv("STRONG_API_KEY", "")
        self.strong_model = os.getenv(
            "STRONG_MODEL", "deepseek-ai/DeepSeek-V3"
        )
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.local_model = os.getenv("LOCAL_MODEL", "qwen3-coder-next")

    async def on_startup(self):
        print(f"[orchestrator] 启动: 强模型={self.strong_model}, 本地={self.local_model}")

    async def on_shutdown(self):
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_local_model",
                    "description": (
                        "将子任务交给本地模型处理。适用场景：代码生成、文本翻译、"
                        "格式转换、数据提取等不需要联网或复杂推理的任务。"
                        "本地模型速度更快且免费。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "给本地模型的完整指令，要清晰具体",
                            }
                        },
                        "required": ["task"],
                    },
                },
            }
        ]

        # 第一轮：强模型分析，决定是否调用本地模型
        first_response = self._call_strong_model(messages, tools=tools)
        message = first_response["choices"][0]["message"]

        if not message.get("tool_calls"):
            # 强模型直接回答，无需本地模型
            return message["content"]

        # 强模型要调用本地模型
        tool_results = []
        for tool_call in message["tool_calls"]:
            if tool_call["function"]["name"] == "call_local_model":
                args = json.loads(tool_call["function"]["arguments"])
                print(f"[orchestrator] 本地模型任务: {args['task'][:80]}...")
                local_result = self._call_local_model(args["task"])
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": local_result,
                    }
                )

        # 第二轮：强模型拿到本地结果后汇总
        combined_messages = messages + [message] + tool_results
        final_response = self._call_strong_model(combined_messages)
        return final_response["choices"][0]["message"]["content"]

    def _call_strong_model(self, messages: list, tools: list = None) -> dict:
        payload = {"model": self.strong_model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = requests.post(
            f"{self.strong_api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.strong_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def _call_local_model(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.local_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
