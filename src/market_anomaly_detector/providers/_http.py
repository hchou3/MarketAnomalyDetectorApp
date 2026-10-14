"""Shared HTTP helper for live data providers.

Provides `http_get`: a URL fetcher with SSL/certifi fix (Windows), retries on
5xx and 429 (honouring Retry-After), and immediate failure on other 4xx.
"""

from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request


def _ssl_context() -> ssl.SSLContext:
    # Windows Python often lacks the CA bundle urllib needs; certifi ships one.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _ssl_context()
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}


def http_get(url: str, *, retries: int = 3, backoff: float = 1.0) -> str:
    """Fetch *url* and return the response body as a UTF-8 string.

    Retry policy:
    - HTTP 429: honour Retry-After header; retry up to *retries* times.
    - HTTP 5xx: exponential backoff starting at *backoff* seconds.
    - URLError (DNS failure, timeout, connection reset): same backoff.
    - Any other HTTP 4xx: raise immediately without retrying.
    """
    delay = backoff
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = float(exc.headers.get("Retry-After", delay))
                time.sleep(retry_after)
                delay = max(retry_after * 2, delay * 2)
                last_exc = exc
            elif 500 <= exc.code < 600:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                last_exc = exc
            else:
                raise  # 4xx (except 429) — do not retry
        except urllib.error.URLError as exc:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            last_exc = exc
    raise last_exc
