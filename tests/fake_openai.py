"""A tiny OpenAI-compatible server used for end-to-end tests.

It mimics Ling-3.0-tiny's quirks: a reasoning_content chunk, streamed deltas,
and a scripted main-agent / judge conversation.

    python tests/fake_openai.py 8140
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake-ling-tiny"


def decide(messages: list[dict]) -> str:
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if "可复用经验" in text:
        return "- 在 Windows 下用 echo verified 做最小验证即可\n- 写文件后必须实际运行一次"
    if "评审子代理" in text:
        if "echo verified" in text:
            return '{"status":1}'
        return '{"status":0,"reason":"缺少运行验证，请实际执行命令"}'
    if "上一轮评审未通过" in text:
        if "OBS[terminalbash]" not in text:
            return '{"tool":"terminalbash","args":{"command":"echo verified"}}'
        return '{"final":"已补充运行验证：echo verified 退出码 0"}'
    if "OBS[write]" in text:
        return '{"final":"已创建 demo.txt"}'
    return '{"tool":"write","args":{"path":"demo.txt","content":"hello"}}'


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        return

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        text = decide(messages)
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

        if not payload.get("stream"):
            self._json(
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": text,
                                "reasoning_content": "先想一下再输出。",
                            },
                        }
                    ],
                    "usage": usage,
                }
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj) -> None:
            chunk = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        send({"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "思考中"}}]})
        for index in range(0, len(text), 12):
            send({"choices": [{"index": 0, "delta": {"content": text[index : index + 12]}}]})
            time.sleep(0.005)
        send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8140
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake openai server on http://127.0.0.1:{port}/v1", flush=True)
    server.serve_forever()
