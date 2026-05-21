import json
import os
import requests
from typing import Generator, Iterator, List, Union
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        strong_api_url: str = "https://api.siliconflow.cn/v1"
        strong_api_key: str = ""
        strong_model: str = "deepseek-ai/DeepSeek-V3"
        ollama_url: str = "http://ollama:11434"
        local_model: str = "qwen3-coder-next"

    def __init__(self):
        self.name = "在线 API 指挥本地模型"
        self.id = "api_orchestrator"
        self.valves = self.Valves(
            strong_api_url=os.getenv("STRONG_API_URL", "https://api.siliconflow.cn/v1"),
            strong_api_key=os.getenv("STRONG_API_KEY", ""),
            strong_model=os.getenv("STRONG_MODEL", "deepseek-ai/DeepSeek-V3"),
            ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
            local_model=os.getenv("LOCAL_MODEL", "qwen3-coder-next"),
        )

    async def on_startup(self):
        print(f"[orchestrator] 强模型={self.valves.strong_model} | 本地={self.valves.local_model}")

    async def on_shutdown(self):
        pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.strong_api_key:
            return "错误：未配置 STRONG_API_KEY，请在 .env 中添加 STRONG_API_KEY=sk-你的硅基流动密钥"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_local_model",
                    "description": (
                        "将子任务委托给本地模型处理。"
                        "适用场景：代码生成与补全、文本翻译、格式转换、数据提取、"
                        "模板填充等不需要联网或深度推理的任务。"
                        "本地模型响应快且不消耗 API 额度。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "给本地模型的完整指令，需清晰、自包含",
                            }
                        },
                        "required": ["task"],
                    },
                },
            }
        ]

        # 第一轮：强模型分析，决定是否委托本地模型
        first_resp = self._call_strong(messages, tools=tools)
        assistant_msg = first_resp["choices"][0]["message"]

        if not assistant_msg.get("tool_calls"):
            # 强模型直接回答，流式输出
            return self._call_strong_streaming(messages)

        # 执行本地模型子任务
        tool_results = []
        for tc in assistant_msg["tool_calls"]:
            if tc["function"]["name"] == "call_local_model":
                args = json.loads(tc["function"]["arguments"])
                print(f"[orchestrator] → 本地模型: {args['task'][:100]}")
                result = self._call_local(args["task"])
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        # 第二轮：强模型汇总，流式输出给用户
        final_messages = messages + [assistant_msg] + tool_results
        return self._call_strong_streaming(final_messages)

    def _call_strong(self, messages: list, tools: list = None) -> dict:
        payload = {"model": self.valves.strong_model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        resp = requests.post(
            f"{self.valves.strong_api_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.valves.strong_api_key}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def _call_strong_streaming(self, messages: list) -> Generator:
        resp = requests.post(
            f"{self.valves.strong_api_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.valves.strong_api_key}"},
            json={
                "model": self.valves.strong_model,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def _call_local(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.valves.ollama_url}/api/generate",
            json={"model": self.valves.local_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
