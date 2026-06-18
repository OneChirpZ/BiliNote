import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_DOUBAO_ASR_RESOURCE_ID = "volc.seedasr.auc"
DEFAULT_DOUBAO_ASR_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_DOUBAO_ASR_TIMEOUT_SECONDS = 1800.0
DEFAULT_DOUBAO_ASR_MAX_AUDIO_SIZE_MB = 200.0
DOUBAO_ASR_NUMERIC_FIELDS = {
    "poll_interval_seconds": DEFAULT_DOUBAO_ASR_POLL_INTERVAL_SECONDS,
    "timeout_seconds": DEFAULT_DOUBAO_ASR_TIMEOUT_SECONDS,
    "max_audio_size_mb": DEFAULT_DOUBAO_ASR_MAX_AUDIO_SIZE_MB,
}


class TranscriberConfigManager:
    """管理转写器配置，存储在 JSON 文件中，支持前端动态修改。"""

    def __init__(self, filepath: str = "config/transcriber.json"):
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._secure_permissions()

    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._secure_permissions()

    def _secure_permissions(self) -> None:
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get_config(self) -> Dict[str, Any]:
        """获取当前转写器配置，fallback 到环境变量默认值。

        whisper 默认 size 从 'medium' (~1.5GB) 改为 'tiny' (~75MB)：
        新装用户没主动设置时不应该被首次下载卡住。想要更高精度可在「音频转写配置」
        页主动切换。
        """
        data = self._read()
        return {
            "transcriber_type": data.get(
                "transcriber_type",
                os.getenv("TRANSCRIBER_TYPE", "fast-whisper"),
            ),
            "whisper_model_size": data.get(
                "whisper_model_size",
                os.getenv("WHISPER_MODEL_SIZE", "tiny"),
            ),
            "doubao_asr": self.get_doubao_asr_config(include_secret=False, data=data),
        }

    def update_config(
        self,
        transcriber_type: str,
        whisper_model_size: Optional[str] = None,
        doubao_asr: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """更新转写器配置并持久化。"""
        data = self._read()
        data["transcriber_type"] = transcriber_type
        if whisper_model_size is not None:
            data["whisper_model_size"] = whisper_model_size
        if doubao_asr is not None:
            data["doubao_asr"] = self._merge_doubao_asr_config(
                data.get("doubao_asr"),
                doubao_asr,
            )
        self._write(data)
        return self.get_config()

    def get_transcriber_type(self) -> str:
        return self.get_config()["transcriber_type"]

    def get_whisper_model_size(self) -> str:
        return self.get_config()["whisper_model_size"]

    def get_doubao_asr_config(
        self,
        include_secret: bool = False,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = self._read() if data is None else data
        raw = data.get("doubao_asr")
        raw = raw if isinstance(raw, dict) else {}

        configured_api_key = str(raw.get("api_key") or "").strip()
        env_api_key = os.getenv("VOLCENGINE_ASR_API_KEY", "").strip()
        api_key = configured_api_key or env_api_key

        config = {
            "resource_id": str(
                raw.get("resource_id")
                or os.getenv("VOLCENGINE_ASR_RESOURCE_ID")
                or DEFAULT_DOUBAO_ASR_RESOURCE_ID
            ).strip(),
            "poll_interval_seconds": self._doubao_numeric_config(
                "poll_interval_seconds",
                raw.get("poll_interval_seconds"),
                "VOLCENGINE_ASR_POLL_INTERVAL_SECONDS",
            ),
            "timeout_seconds": self._doubao_numeric_config(
                "timeout_seconds",
                raw.get("timeout_seconds"),
                "VOLCENGINE_ASR_TIMEOUT_SECONDS",
            ),
            "max_audio_size_mb": self._doubao_numeric_config(
                "max_audio_size_mb",
                raw.get("max_audio_size_mb"),
                "VOLCENGINE_ASR_MAX_AUDIO_SIZE_MB",
            ),
            "api_key_configured": bool(api_key),
            "api_key_source": "config" if configured_api_key else ("env" if env_api_key else ""),
        }
        if include_secret:
            config["api_key"] = api_key
        return config

    def _merge_doubao_asr_config(
        self,
        existing: Any,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(existing) if isinstance(existing, dict) else {}

        api_key = updates.get("api_key")
        if updates.get("clear_api_key"):
            merged.pop("api_key", None)
        elif api_key is not None:
            api_key = str(api_key).strip()
            if api_key:
                merged["api_key"] = api_key

        resource_id = updates.get("resource_id")
        if resource_id is not None:
            resource_id = str(resource_id).strip()
            if resource_id:
                merged["resource_id"] = resource_id

        for key, default in DOUBAO_ASR_NUMERIC_FIELDS.items():
            if key in updates and updates[key] not in (None, ""):
                merged[key] = self._normalize_doubao_numeric(
                    key,
                    self._float_value(updates[key], default),
                )

        return merged

    @staticmethod
    def _float_value(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _float_config(self, value: Any, env_name: str, default: float) -> float:
        if value not in (None, ""):
            return self._float_value(value, default)
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            return self._float_value(env_value, default)
        return default

    def _doubao_numeric_config(self, key: str, value: Any, env_name: str) -> float:
        default = DOUBAO_ASR_NUMERIC_FIELDS[key]
        return self._normalize_doubao_numeric(
            key,
            self._float_config(value, env_name, default),
        )

    @staticmethod
    def _normalize_doubao_numeric(key: str, value: float) -> float:
        default = DOUBAO_ASR_NUMERIC_FIELDS[key]
        return value if value > 0 else default

    def is_model_ready(self) -> Dict[str, Any]:
        """当前转写器是否就绪可用。

        返回 {ready, transcriber_type, model_size, downloading, reason}：
          - 在线引擎 (groq/bcut/kuaishou)：永远 ready（不需要本地模型）
          - doubao-asr：需要 VOLCENGINE_ASR_API_KEY
          - fast-whisper：检查 whisper-{size}/model.bin 落盘
          - mlx-whisper：检查 {repo_id}/config.json 落盘
        给 /generate_note 入口做「开始视频前先确认模型下载好」的门禁用。
        """
        cfg = self.get_config()
        ttype = cfg["transcriber_type"]
        size = cfg["whisper_model_size"]
        result = {
            "ready": True,
            "transcriber_type": ttype,
            "model_size": size,
            "downloading": False,
            "reason": "",
        }
        if ttype == "doubao-asr":
            if self.get_doubao_asr_config(include_secret=True).get("api_key"):
                return result
            result["ready"] = False
            result["reason"] = "豆包 ASR 未配置火山引擎 API Key，请先在「音频转写配置」页填写"
            return result

        if ttype not in ("fast-whisper", "mlx-whisper"):
            return result  # 在线引擎无需本地模型

        # 延迟 import 避免与 routers.config 的循环依赖；只取纯函数，不触发路由副作用
        try:
            from app.routers.config import (
                _check_whisper_model_exists,
                _check_mlx_whisper_model_exists,
                _downloading,
            )
        except Exception as e:
            # 拿不到检查函数时保守放行，不要把用户卡死
            result["reason"] = f"无法检查模型状态: {e}"
            return result

        if ttype == "fast-whisper":
            downloaded = _check_whisper_model_exists(size, "whisper")
            downloading = _downloading.get(size) == "downloading"
        else:  # mlx-whisper
            downloaded = _check_mlx_whisper_model_exists(size)
            downloading = _downloading.get(f"mlx-{size}") == "downloading"

        result["downloading"] = downloading
        if downloaded:
            return result
        result["ready"] = False
        result["reason"] = (
            f"转写模型 {ttype} / {size} 尚未下载就绪"
            + ("，正在下载中，请稍候" if downloading else "，请先在「设置 → 音频转写配置」页下载")
        )
        return result
