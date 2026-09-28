"""Moonshot Open Platform authentication configuration and preflight."""

import logging
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SUPPORTED_BASE_URLS = {
    "https://api.moonshot.ai/v1",
    "https://api.moonshot.cn/v1",
}


class ProviderAuthenticationError(RuntimeError):
    """Raised when Moonshot Open Platform authentication cannot be verified."""


def normalize_base_url(base_url: str) -> str:
    """Validate and normalize an explicitly selected Moonshot API endpoint."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError(
            "kimi_base_url is required; select the global "
            "https://api.moonshot.ai/v1 or CN https://api.moonshot.cn/v1 endpoint"
        )
    if normalized not in SUPPORTED_BASE_URLS:
        raise ValueError(
            "Unsupported kimi_base_url. Select https://api.moonshot.ai/v1 "
            "or https://api.moonshot.cn/v1 explicitly."
        )
    return normalized


def configure_agent_env(api_key: str, base_url: str, model: str) -> None:
    """Configure the environment consumed by kimi-agent-sdk without logging secrets."""
    os.environ["KIMI_API_KEY"] = api_key
    os.environ["KIMI_BASE_URL"] = normalize_base_url(base_url)
    os.environ["KIMI_MODEL_NAME"] = model


def preflight_authentication(api_key: str, base_url: str, timeout: float = 10.0) -> None:
    """Verify a Moonshot Open Platform bearer key against its selected region."""
    normalized = normalize_base_url(base_url)
    host = urlparse(normalized).hostname
    logger.info("Checking Moonshot Open Platform authentication at host %s", host)

    request = Request(
        f"{normalized}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise ProviderAuthenticationError(
                    f"Moonshot authentication preflight failed at {host} "
                    f"with HTTP {response.status}"
                )
    except HTTPError as exc:
        if exc.code == 401:
            raise ProviderAuthenticationError(
                f"Moonshot authentication failed at {host} (HTTP 401). "
                "The configured endpoint region may not match the key, the credential "
                "may be a Kimi Code/Kimi-for-Coding credential rather than a Moonshot "
                "Open Platform API key, or the API key may be invalid."
            ) from None
        raise ProviderAuthenticationError(
            f"Moonshot authentication preflight failed at {host} with HTTP {exc.code}"
        ) from None
    except URLError as exc:
        raise ProviderAuthenticationError(
            f"Could not reach the configured Moonshot host {host} for authentication preflight"
        ) from exc

    logger.info("Moonshot Open Platform authentication succeeded at host %s", host)
