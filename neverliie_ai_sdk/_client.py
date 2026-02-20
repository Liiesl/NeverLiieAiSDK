import json
import requests
from typing import Dict, Any, Optional, Iterator
from ._exceptions import APIError, AuthenticationError, RateLimitError, NotFoundError, InvalidRequestError


class HttpClient:
    def __init__(self, base_url: str, api_key: str, default_headers: Dict[str, str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.default_headers = default_headers or {}

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    def _handle_error(self, response: requests.Response):
        status_code = response.status_code
        try:
            error_data = response.json()
            message = error_data.get("error", {}).get("message", response.text)
        except:
            message = response.text
            error_data = None

        if status_code == 401:
            raise AuthenticationError(message, status_code, error_data)
        elif status_code == 404:
            raise NotFoundError(message, status_code, error_data)
        elif status_code == 429:
            raise RateLimitError(message, status_code, error_data)
        elif status_code >= 400:
            raise InvalidRequestError(message, status_code, error_data)

    def post(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        url = self._build_url(endpoint)
        final_headers = {**self.default_headers, **(headers or {})}
        
        response = self.session.post(url, json=data, headers=final_headers, timeout=120)
        
        if not response.ok:
            self._handle_error(response)
        
        return response.json()

    def post_stream(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
    ) -> Iterator[str]:
        url = self._build_url(endpoint)
        final_headers = {**self.default_headers, **(headers or {})}
        
        response = self.session.post(
            url, json=data, headers=final_headers, timeout=120, stream=True
        )
        
        if not response.ok:
            self._handle_error(response)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break
                    yield data_str

    def close(self):
        self.session.close()
