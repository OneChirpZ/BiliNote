import json
import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "app" / "transcriber" / "doubao_asr.py"
CONFIG_MANAGER_PATH = ROOT / "app" / "services" / "transcriber_config_manager.py"


def _app_stubs():
    app_mod = types.ModuleType("app")
    decorators_pkg = types.ModuleType("app.decorators")
    timeit_mod = types.ModuleType("app.decorators.timeit")
    models_pkg = types.ModuleType("app.models")
    transcriber_model_mod = types.ModuleType("app.models.transcriber_model")
    services_pkg = types.ModuleType("app.services")
    proxy_config_mod = types.ModuleType("app.services.proxy_config_manager")
    transcriber_pkg = types.ModuleType("app.transcriber")
    base_mod = types.ModuleType("app.transcriber.base")
    utils_pkg = types.ModuleType("app.utils")
    logger_mod = types.ModuleType("app.utils.logger")

    def timeit(func):
        return func

    @dataclass
    class TranscriptSegment:
        start: float
        end: float
        text: str

    @dataclass
    class TranscriptResult:
        language: str | None
        full_text: str
        segments: list
        raw: dict | None = None

    class ProxyConfigManager:
        @staticmethod
        def get_proxy_url():
            return None

    class Transcriber:
        pass

    class _Logger:
        @staticmethod
        def info(*_args, **_kwargs):
            return None

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

        @staticmethod
        def error(*_args, **_kwargs):
            return None

    timeit_mod.timeit = timeit
    transcriber_model_mod.TranscriptSegment = TranscriptSegment
    transcriber_model_mod.TranscriptResult = TranscriptResult
    proxy_config_mod.ProxyConfigManager = ProxyConfigManager
    base_mod.Transcriber = Transcriber
    logger_mod.get_logger = lambda _name: _Logger()

    return {
        "app": app_mod,
        "app.decorators": decorators_pkg,
        "app.decorators.timeit": timeit_mod,
        "app.models": models_pkg,
        "app.models.transcriber_model": transcriber_model_mod,
        "app.services": services_pkg,
        "app.services.proxy_config_manager": proxy_config_mod,
        "app.transcriber": transcriber_pkg,
        "app.transcriber.base": base_mod,
        "app.utils": utils_pkg,
        "app.utils.logger": logger_mod,
    }


def _load_module(module_name, path, stubs=None):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{module_name} module spec not found")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs or {}):
        spec.loader.exec_module(module)
    return module


doubao_asr = _load_module("doubao_asr_under_test", MODULE_PATH, _app_stubs())
transcriber_config_manager = _load_module("transcriber_config_manager_under_test", CONFIG_MANAGER_PATH)
DONE_CODE = doubao_asr.DONE_CODE
QUERY_URL = doubao_asr.QUERY_URL
RUNNING_CODES = doubao_asr.RUNNING_CODES
SUBMIT_URL = doubao_asr.SUBMIT_URL
DoubaoASRTranscriber = doubao_asr.DoubaoASRTranscriber
TranscriberConfigManager = transcriber_config_manager.TranscriberConfigManager


class FakeResponse:
    def __init__(self, headers=None, body=None, status_code=200):
        self.headers = headers or {}
        self._body = body if body is not None else {}
        self.status_code = status_code
        self.content = json.dumps(self._body).encode("utf-8") if self._body is not None else b""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.proxies = {}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP call")
        return self.responses.pop(0)


def _done_headers():
    return {"X-Api-Status-Code": DONE_CODE, "X-Tt-Logid": "log-ok"}


class DoubaoASRTranscriberTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "VOLCENGINE_ASR_API_KEY": "test-key",
                "VOLCENGINE_ASR_POLL_INTERVAL_SECONDS": "0",
                "VOLCENGINE_ASR_TIMEOUT_SECONDS": "5",
            },
            clear=False,
        )
        self.env.start()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        self.tmp.write(b"fake audio")
        self.tmp.close()

    def tearDown(self):
        self.env.stop()
        pathlib.Path(self.tmp.name).unlink(missing_ok=True)

    def test_transcript_converts_utterances(self):
        session = FakeSession([
            FakeResponse(headers=_done_headers()),
            FakeResponse(
                headers=_done_headers(),
                body={
                    "result": {
                        "text": "你好 世界",
                        "utterances": [
                            {"text": "你好", "start_time": 0, "end_time": 1200},
                            {"text": "世界", "start_time": 1200, "end_time": 2500},
                        ],
                    }
                },
            ),
        ])

        result = DoubaoASRTranscriber(session=session).transcript(self.tmp.name)

        self.assertEqual(result.language, "zh")
        self.assertEqual(result.full_text, "你好 世界")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].start, 0)
        self.assertEqual(result.segments[0].end, 1.2)
        self.assertEqual(result.segments[1].text, "世界")
        self.assertEqual(result.raw["provider"], "volcengine-doubao-asr")

        submit_url, submit_kwargs = session.calls[0]
        query_url, query_kwargs = session.calls[1]
        self.assertEqual(submit_url, SUBMIT_URL)
        self.assertEqual(query_url, QUERY_URL)
        self.assertEqual(submit_kwargs["json"]["audio"]["format"], "mp3")
        self.assertIn("data", submit_kwargs["json"]["audio"])
        self.assertEqual(
            submit_kwargs["headers"]["X-Api-Request-Id"],
            query_kwargs["headers"]["X-Api-Request-Id"],
        )

    def test_poll_waits_for_running_status(self):
        running_code = next(iter(RUNNING_CODES))
        session = FakeSession([
            FakeResponse(headers=_done_headers()),
            FakeResponse(headers={"X-Api-Status-Code": running_code}),
            FakeResponse(
                headers=_done_headers(),
                body={"result": {"text": "处理完成"}},
            ),
        ])

        result = DoubaoASRTranscriber(session=session).transcript(self.tmp.name)

        self.assertEqual(result.full_text, "处理完成")
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(result.segments[0].text, "处理完成")

    def test_accepts_body_status_code_when_header_is_missing(self):
        session = FakeSession([
            FakeResponse(body={"header": {"code": int(DONE_CODE)}}),
            FakeResponse(body={"result": {"text": "仅 body 返回"}}),
        ])

        result = DoubaoASRTranscriber(session=session).transcript(self.tmp.name)

        self.assertEqual(result.full_text, "仅 body 返回")

    def test_query_failure_raises_without_secret(self):
        session = FakeSession([
            FakeResponse(headers=_done_headers()),
            FakeResponse(headers={"X-Api-Status-Code": "55000000", "X-Tt-Logid": "log-fail"}),
        ])

        with self.assertRaises(RuntimeError) as ctx:
            DoubaoASRTranscriber(session=session).transcript(self.tmp.name)

        message = str(ctx.exception)
        self.assertIn("55000000", message)
        self.assertIn("log-fail", message)
        self.assertNotIn("test-key", message)

    def test_missing_api_key_raises_clear_error(self):
        with patch.dict(os.environ, {"VOLCENGINE_ASR_API_KEY": ""}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                DoubaoASRTranscriber(session=FakeSession([])).transcript(self.tmp.name)

        self.assertIn("VOLCENGINE_ASR_API_KEY", str(ctx.exception))
        self.assertNotIn("test-key", str(ctx.exception))

    def test_unsupported_audio_format_raises_before_http_call(self):
        path = pathlib.Path(self.tmp.name).with_suffix(".webm")
        path.write_bytes(b"fake audio")
        try:
            session = FakeSession([])

            with self.assertRaises(ValueError) as ctx:
                DoubaoASRTranscriber(session=session).transcript(str(path))

            self.assertIn("不支持当前音频格式", str(ctx.exception))
            self.assertEqual(session.calls, [])
        finally:
            path.unlink(missing_ok=True)

    def test_large_audio_raises_before_reading_payload(self):
        with patch.dict(os.environ, {"VOLCENGINE_ASR_MAX_AUDIO_SIZE_MB": "0.000001"}, clear=False):
            session = FakeSession([])

            with self.assertRaises(ValueError) as ctx:
                DoubaoASRTranscriber(session=session).transcript(self.tmp.name)

        self.assertIn("音频文件过大", str(ctx.exception))
        self.assertEqual(session.calls, [])

    def test_applies_global_proxy_config_to_session(self):
        class FakeProxyConfigManager:
            @staticmethod
            def get_proxy_url():
                return "http://127.0.0.1:7890"

        session = FakeSession([])
        with patch.object(doubao_asr, "ProxyConfigManager", FakeProxyConfigManager):
            DoubaoASRTranscriber(session=session)

        self.assertEqual(session.proxies["https"], "http://127.0.0.1:7890")

    def test_doubao_config_not_ready_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = pathlib.Path(tmp_dir) / "transcriber.json"
            cfg_path.write_text(json.dumps({"transcriber_type": "doubao-asr"}), encoding="utf-8")

            with patch.dict(os.environ, {"VOLCENGINE_ASR_API_KEY": ""}, clear=False):
                result = TranscriberConfigManager(str(cfg_path)).is_model_ready()

        self.assertFalse(result["ready"])
        self.assertIn("VOLCENGINE_ASR_API_KEY", result["reason"])


if __name__ == "__main__":
    unittest.main()
