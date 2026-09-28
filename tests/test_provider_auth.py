"""Tests for explicit Moonshot provider configuration and authentication."""

import logging
import os
import sys
from urllib.error import HTTPError
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from action_config import ActionConfig
from provider_auth import (
    ProviderAuthenticationError,
    configure_agent_env,
    normalize_base_url,
    preflight_authentication,
)


@pytest.mark.parametrize(
    "base_url",
    ["https://api.moonshot.ai/v1", "https://api.moonshot.cn/v1"],
)
def test_explicit_regional_base_urls(base_url):
    assert normalize_base_url(base_url) == base_url


def test_base_url_must_be_explicit():
    with pytest.raises(ValueError, match="kimi_base_url is required"):
        normalize_base_url("")


def test_action_config_loads_base_url():
    with patch.dict(
        os.environ,
        {
            "INPUT_KIMI_API_KEY": "provider-key",
            "INPUT_KIMI_BASE_URL": "https://api.moonshot.cn/v1",
        },
        clear=True,
    ):
        config = ActionConfig.from_env()

    assert config.kimi_api_key == "provider-key"
    assert config.kimi_base_url == "https://api.moonshot.cn/v1"


def test_api_key_and_endpoint_are_propagated():
    with patch.dict(os.environ, {}, clear=True):
        configure_agent_env(
            "provider-key", "https://api.moonshot.ai/v1", "kimi-k2-thinking"
        )

        assert os.environ["KIMI_API_KEY"] == "provider-key"
        assert os.environ["KIMI_BASE_URL"] == "https://api.moonshot.ai/v1"
        assert os.environ["KIMI_MODEL_NAME"] == "kimi-k2-thinking"


def test_preflight_uses_selected_cn_endpoint():
    response = Mock(status=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    with patch("provider_auth.urlopen", return_value=response) as urlopen:
        preflight_authentication("provider-key", "https://api.moonshot.cn/v1")

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://api.moonshot.cn/v1/models"
    assert request.get_header("Authorization") == "Bearer provider-key"


def test_401_explains_likely_causes_without_disclosing_key(caplog):
    secret = "never-log-this-provider-key"
    error = HTTPError(
        "https://api.moonshot.ai/v1/models", 401, "Unauthorized", {}, None
    )

    with caplog.at_level(logging.INFO), patch(
        "provider_auth.urlopen", side_effect=error
    ), pytest.raises(ProviderAuthenticationError) as raised:
        preflight_authentication(secret, "https://api.moonshot.ai/v1")

    message = str(raised.value)
    assert "HTTP 401" in message
    assert "region" in message
    assert "Kimi Code/Kimi-for-Coding" in message
    assert secret not in message
    assert secret not in caplog.text
    assert "api.moonshot.ai" in caplog.text


def test_main_exits_before_event_processing_when_preflight_fails(caplog):
    import main as main_module

    config = ActionConfig(
        kimi_api_key="never-log-this-provider-key",
        kimi_base_url="https://api.moonshot.ai/v1",
        github_token="github-token",
    )
    failure = ProviderAuthenticationError(
        "Moonshot authentication failed at api.moonshot.ai (HTTP 401)"
    )

    with caplog.at_level(logging.ERROR), patch.object(
        main_module.ActionConfig, "from_env", return_value=config
    ), patch.object(
        main_module, "preflight_authentication", side_effect=failure
    ), patch.object(
        main_module, "GitHubClient"
    ) as github_client, pytest.raises(SystemExit) as raised:
        main_module.main()

    assert raised.value.code == 1
    github_client.assert_not_called()
    assert config.kimi_api_key not in caplog.text
