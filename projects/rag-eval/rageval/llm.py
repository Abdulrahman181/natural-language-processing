import time

import requests


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.3,
        timeout: int = 90,
        max_retries: int = 5,
        session=None,
        sleep=time.sleep,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.http = session if session is not None else requests.Session()
        self.sleep = sleep
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_detail = ""
        for attempt in range(self.max_retries):
            response = self.http.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            last_detail = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                try:
                    delay = float(retry_after) if retry_after else 2 ** attempt
                except ValueError:
                    delay = 2 ** attempt
                self.sleep(min(delay, 60))
                continue
            break
        raise RuntimeError(last_detail)
