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


def _install_stubs():
    app_mod = types.ModuleType("app")
    decorators_pkg = types.ModuleType("app.decorators")
    timeit_mod = types.ModuleType("app.decorators.timeit")
    models_pkg = types.ModuleType("app.models")
    transcriber_model_mod = types.ModuleType("app.models.transcriber_model")
    transcriber_pkg = types.ModuleType("app.transcriber")
    base_mod = types.ModuleType("app.transcriber.base")
    utils_pkg = types.ModuleType("app.utils")
    logger_mod = types.ModuleType("app.utils.logger")
    requests_mod = types.ModuleType("requests")
    dotenv_mod = types.ModuleType("dotenv")

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

    class Transcriber:
        pass

    class _RequestsSession:
        def post(self, *_args, **_kwargs):
            raise AssertionError("requests.Session should be replaced by FakeSession in tests")

    class _RequestsResponse:
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
    base_mod.Transcriber = Transcriber
    logger_mod.get_logger = lambda _name: _Logger()
    requests_mod.Session = _RequestsSession
    requests_mod.Response = _RequestsResponse
    dotenv_mod.load_dotenv = lambda: None

    sys.modules.setdefault("app", app_mod)
    sys.modules.setdefault("app.decorators", decorators_pkg)
    sys.modules["app.decorators.timeit"] = timeit_mod
    sys.modules.setdefault("app.models", models_pkg)
    sys.modules["app.models.transcriber_model"] = transcriber_model_mod
    sys.modules.setdefault("app.transcriber", transcriber_pkg)
    sys.modules["app.transcriber.base"] = base_mod
    sys.modules.setdefault("app.utils", utils_pkg)
    sys.modules["app.utils.logger"] = logger_mod
    sys.modules["requests"] = requests_mod
    sys.modules["dotenv"] = dotenv_mod


def _load_doubao_asr_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("doubao_asr", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("doubao_asr module spec not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


doubao_asr = _load_doubao_asr_module()
DONE_CODE = doubao_asr.DONE_CODE
QUERY_URL = doubao_asr.QUERY_URL
RUNNING_CODES = doubao_asr.RUNNING_CODES
SUBMIT_URL = doubao_asr.SUBMIT_URL
DoubaoASRTranscriber = doubao_asr.DoubaoASRTranscriber


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


if __name__ == "__main__":
    unittest.main()
