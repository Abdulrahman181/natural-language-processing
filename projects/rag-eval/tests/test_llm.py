import pytest

from rageval.llm import ChatClient


class FakeResponse:
    def __init__(self, status_code, content="", headers=None):
        self.status_code = status_code
        self.text = content
        self.headers = headers or {}
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json, headers, timeout):
        self.requests.append((url, json, headers))
        return self.responses.pop(0)


def test_client_builds_request_and_returns_content():
    http = FakeHttp([FakeResponse(200, "answer")])
    client = ChatClient("https://example.test/v1/", "model-x", api_key="secret", session=http)
    assert client("hello") == "answer"
    url, payload, headers = http.requests[0]
    assert url == "https://example.test/v1/chat/completions"
    assert payload["model"] == "model-x"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert headers["Authorization"] == "Bearer secret"


def test_client_omits_authorization_without_key():
    http = FakeHttp([FakeResponse(200, "ok")])
    ChatClient("http://localhost:11434/v1", "m", session=http)("x")
    assert "Authorization" not in http.requests[0][2]


def test_client_retries_rate_limits_and_honors_retry_after():
    delays = []
    http = FakeHttp([FakeResponse(429, "slow", {"Retry-After": "3"}), FakeResponse(503), FakeResponse(200, "done")])
    client = ChatClient("https://example.test/v1", "m", session=http, sleep=delays.append)
    assert client("x") == "done"
    assert delays == [3.0, 2]


def test_client_does_not_retry_client_errors():
    http = FakeHttp([FakeResponse(401, "bad key")])
    client = ChatClient("https://example.test/v1", "m", session=http, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="401"):
        client("x")
    assert len(http.requests) == 1


def test_client_raises_after_max_retries():
    http = FakeHttp([FakeResponse(500)] * 3)
    client = ChatClient("https://example.test/v1", "m", max_retries=3, session=http, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="500"):
        client("x")
    assert len(http.requests) == 3
