import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests
from dotenv import load_dotenv

from app.decorators.timeit import timeit
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.services.proxy_config_manager import ProxyConfigManager
from app.transcriber.base import Transcriber
from app.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
DONE_CODE = "20000000"
RUNNING_CODES = {"20000001", "20000002"}
DEFAULT_RESOURCE_ID = "volc.seedasr.auc"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 1800.0
DEFAULT_MAX_AUDIO_SIZE_MB = 200.0
SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "ogg", "flac", "aac", "amr"}


class DoubaoASRTranscriber(Transcriber):
    """火山引擎豆包录音文件识别标准版转写器。"""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.resource_id = os.getenv("VOLCENGINE_ASR_RESOURCE_ID", DEFAULT_RESOURCE_ID)
        self.poll_interval = _get_float_env(
            "VOLCENGINE_ASR_POLL_INTERVAL_SECONDS",
            DEFAULT_POLL_INTERVAL_SECONDS,
        )
        self.timeout_seconds = _get_float_env(
            "VOLCENGINE_ASR_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        )
        self.max_audio_size_mb = _get_float_env(
            "VOLCENGINE_ASR_MAX_AUDIO_SIZE_MB",
            DEFAULT_MAX_AUDIO_SIZE_MB,
        )
        self._apply_proxy_config()

    @timeit
    def transcript(self, file_path: str) -> TranscriptResult:
        api_key = os.getenv("VOLCENGINE_ASR_API_KEY")
        if not api_key:
            raise RuntimeError("未配置 VOLCENGINE_ASR_API_KEY，无法使用豆包 ASR。")

        request_id = str(uuid.uuid4())
        logger.info(
            "开始豆包 ASR 标准版转写: file=%s, resource_id=%s, request_id=%s",
            Path(file_path).name,
            self.resource_id,
            request_id,
        )

        headers = self._headers(api_key, request_id)
        payload = self._submit_payload(file_path)
        self._submit(headers, payload, request_id)
        data, status_code, log_id = self._poll(headers, request_id)
        return self._to_transcript_result(
            data,
            request_id=request_id,
            status_code=status_code,
            log_id=log_id,
        )

    def _headers(self, api_key: str, request_id: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        }

    def _submit_payload(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        audio_format = self._infer_audio_format(path)
        audio_size = path.stat().st_size
        if audio_size <= 0:
            raise ValueError("音频文件为空，无法提交豆包 ASR。")

        audio_size_mb = audio_size / 1024 / 1024
        if audio_size_mb > self.max_audio_size_mb:
            raise ValueError(
                f"音频文件过大，无法使用 base64 提交豆包 ASR: "
                f"size={audio_size_mb:.2f}MB, limit={self.max_audio_size_mb:.2f}MB。"
            )

        logger.info("豆包 ASR 提交音频: format=%s, size=%.2fMB", audio_format, audio_size_mb)
        audio_bytes = path.read_bytes()

        return {
            "user": {"uid": "bilinote"},
            "audio": {
                "data": base64.b64encode(audio_bytes).decode("utf-8"),
                "format": audio_format,
                "language": "zh-CN",
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
            },
        }

    def _submit(self, headers: Dict[str, str], payload: Dict[str, Any], request_id: str) -> None:
        response = self.session.post(SUBMIT_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        status_code = _status_code(response, _response_json(response))
        log_id = response.headers.get("X-Tt-Logid", "")
        logger.info(
            "豆包 ASR submit 返回: request_id=%s, status_code=%s, log_id=%s",
            request_id,
            status_code or "unknown",
            log_id or "-",
        )
        if status_code != DONE_CODE:
            raise RuntimeError(
                f"豆包 ASR 提交失败: status_code={status_code or 'unknown'}, "
                f"message={response.headers.get('X-Api-Message', '')}, log_id={log_id or '-'}"
            )

    def _poll(self, headers: Dict[str, str], request_id: str) -> tuple[Dict[str, Any], str, str]:
        deadline = time.monotonic() + self.timeout_seconds
        attempt = 0

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"豆包 ASR 轮询超时: request_id={request_id}, timeout={self.timeout_seconds}s"
                )

            attempt += 1
            response = self.session.post(QUERY_URL, json={}, headers=headers, timeout=60)
            response.raise_for_status()
            data = _response_json(response)
            status_code = _status_code(response, data)
            log_id = response.headers.get("X-Tt-Logid", "")

            if status_code == DONE_CODE:
                logger.info(
                    "豆包 ASR 转写完成: request_id=%s, attempts=%d, log_id=%s",
                    request_id,
                    attempt,
                    log_id or "-",
                )
                return data, status_code, log_id

            if status_code in RUNNING_CODES:
                if attempt == 1 or attempt % 12 == 0:
                    logger.info(
                        "豆包 ASR 处理中: request_id=%s, status_code=%s, attempts=%d, log_id=%s",
                        request_id,
                        status_code,
                        attempt,
                        log_id or "-",
                    )
                time.sleep(self.poll_interval)
                continue

            raise RuntimeError(
                f"豆包 ASR 查询失败: status_code={status_code or 'unknown'}, "
                f"message={response.headers.get('X-Api-Message', '')}, log_id={log_id or '-'}"
            )

    def _to_transcript_result(
        self,
        data: Dict[str, Any],
        request_id: str,
        status_code: str,
        log_id: str,
    ) -> TranscriptResult:
        result = data.get("result") if isinstance(data.get("result"), dict) else data
        full_text = str(result.get("text") or result.get("full_text") or "").strip()
        utterances = _first_list(
            result.get("utterances"),
            result.get("utterance"),
            result.get("segments"),
            result.get("sentences"),
        )

        segments = []
        for item in utterances:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("transcript") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=_start_seconds(item),
                    end=_end_seconds(item),
                    text=text,
                )
            )

        if not full_text and segments:
            full_text = " ".join(seg.text for seg in segments).strip()

        if full_text and not segments:
            segments.append(TranscriptSegment(start=0, end=0, text=full_text))

        if not full_text:
            raise RuntimeError(f"豆包 ASR 返回结果为空: request_id={request_id}, log_id={log_id or '-'}")

        return TranscriptResult(
            language="zh",
            full_text=full_text,
            segments=segments,
            raw={
                "provider": "volcengine-doubao-asr",
                "request_id": request_id,
                "status_code": status_code,
                "log_id": log_id,
                "response": data,
            },
        )

    @staticmethod
    def _infer_audio_format(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in SUPPORTED_FORMATS:
            return suffix
        raise ValueError(
            f"豆包 ASR 不支持当前音频格式: .{suffix or 'unknown'}。"
            f"请先转为以下格式之一: {', '.join(sorted(SUPPORTED_FORMATS))}。"
        )

    def _apply_proxy_config(self) -> None:
        proxy_url = ProxyConfigManager().get_proxy_url()
        if proxy_url:
            if not hasattr(self.session, "proxies"):
                self.session.proxies = {}
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})
            logger.info("豆包 ASR 已使用全局代理配置。")


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("%s=%s 不是有效数字，使用默认值 %s", name, value, default)
        return default


def _response_json(response: requests.Response) -> Dict[str, Any]:
    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _status_code(response: requests.Response, data: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if data:
        body_code = data.get("header", {}).get("code")
        if body_code is not None:
            return str(body_code)
        result = data.get("result")
        if isinstance(result, dict) and (result.get("text") or result.get("utterances")):
            return DONE_CODE
    header_code = response.headers.get("X-Api-Status-Code")
    return str(header_code) if header_code is not None else None


def _first_list(*values: Any) -> list:
    for value in values:
        if isinstance(value, list):
            return value
    return []


def _start_seconds(item: Dict[str, Any]) -> float:
    return _time_seconds(item, ms_keys=("start_time", "start_ms"), second_keys=("start", "begin"))


def _end_seconds(item: Dict[str, Any]) -> float:
    return _time_seconds(item, ms_keys=("end_time", "end_ms"), second_keys=("end", "finish"))


def _time_seconds(
    item: Dict[str, Any],
    ms_keys: Iterable[str],
    second_keys: Iterable[str],
) -> float:
    for key in ms_keys:
        if key in item and item[key] is not None:
            return _float_or_zero(item[key]) / 1000.0
    for key in second_keys:
        if key in item and item[key] is not None:
            return _float_or_zero(item[key])
    return 0.0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
