"""Opt-in experiment metadata only: no prompts, media, responses or credentials."""
import json
import hashlib
import os
import re
import sys
import time
from urllib.parse import urlsplit

path = os.environ.get("M3_USAGE_LOG")
if path:
    from providers.llm.openai_llm import OpenAILLM

    original = OpenAILLM._create_completion
    if not getattr(original, "_m3_usage_audit", False):
        def audited(self, **kwargs):
            started = time.time()
            messages = kwargs.get("messages") or []
            parts = [p for message in messages if isinstance(message.get("content"), list)
                     for p in message["content"]]
            text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
            video = re.search(r"^video_id:\s*(\S+)", text, re.MULTILINE)
            clip = re.search(r"^clip_id:\s*(\S+)", text, re.MULTILINE)
            record = {
                "started_at_unix": started,
                "process": os.path.basename(sys.argv[0]),
                "pid": os.getpid(),
                "qa_video_id": os.environ.get("M3_AUDIT_VIDEO"),
                "qa_question_index": os.environ.get("M3_AUDIT_QUESTION_INDEX"),
                "input_video_id": video[1] if video else None,
                "input_clip_id": clip[1] if clip else None,
                "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "requested_model": kwargs.get("model"),
                "endpoint_host": urlsplit(self._base_url).hostname,
                "max_tokens": kwargs.get("max_tokens", kwargs.get("max_completion_tokens")),
                "request_timeout": self._client.timeout,
                "temperature": kwargs.get("temperature"),
                "reasoning_effort": kwargs.get("reasoning_effort"),
                "thinking": (kwargs.get("extra_body") or {}).get("thinking"),
                "images": sum(p.get("type") == "image_url" for p in parts),
                "audio": sum(p.get("type") == "input_audio" for p in parts),
                "tools": [t.get("function", {}).get("name") for t in (kwargs.get("tools") or [])],
            }
            try:
                response = original(self, **kwargs)
                record["status"] = "returned"
                record["response_id"] = getattr(response, "id", None)
                record["reported_model"] = getattr(response, "model", None)
                usage = getattr(response, "usage", None)
                choices = getattr(response, "choices", None) or []
                record["usage"] = usage.model_dump() if usage else None
                record["finish_reasons"] = [c.finish_reason for c in choices]
                shapes = []
                for choice in choices:
                    for call in choice.message.tool_calls or []:
                        try:
                            shapes.append(type(json.loads(call.function.arguments)).__name__)
                        except (ValueError, TypeError):
                            shapes.append("invalid_json")
                record["tool_argument_shapes"] = shapes
                return response
            except Exception as error:
                record["status"] = "error"
                record["exception_type"] = type(error).__name__
                record["http_status"] = getattr(error, "status_code", None)
                raise
            finally:
                record["seconds"] = time.time() - started
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(descriptor, (json.dumps(record, ensure_ascii=False) + "\n").encode())
                finally:
                    os.close(descriptor)
        audited._m3_usage_audit = True
        OpenAILLM._create_completion = audited
