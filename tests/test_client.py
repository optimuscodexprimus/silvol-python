"""
Smoke tests for the Silvol SDK.

These tests run without a real API key — they verify that the client
initialises correctly and points at the right base URL.
"""

from unittest.mock import MagicMock, patch


def test_silvol_default_base_url():
    """Silvol() should use the Silvol gateway base URL by default."""
    from silvol import Silvol

    with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
        Silvol(api_key="sk-svl-test")
        _, kwargs = mock_init.call_args
        assert kwargs.get("base_url") == "https://api.silvol.ai/v1"


def test_silvol_custom_base_url():
    """Silvol() should respect a caller-supplied base_url."""
    from silvol import Silvol

    custom = "https://my-proxy.example.com/v1"
    with patch("openai.OpenAI.__init__", return_value=None) as mock_init:
        Silvol(api_key="sk-svl-test", base_url=custom)
        _, kwargs = mock_init.call_args
        assert kwargs.get("base_url") == custom


def test_async_silvol_default_base_url():
    """AsyncSilvol() should use the Silvol gateway base URL by default."""
    from silvol import AsyncSilvol

    with patch("openai.AsyncOpenAI.__init__", return_value=None) as mock_init:
        AsyncSilvol(api_key="sk-svl-test")
        _, kwargs = mock_init.call_args
        assert kwargs.get("base_url") == "https://api.silvol.ai/v1"


def test_version_exported():
    """__version__ should be importable from the top-level package."""
    import silvol

    assert isinstance(silvol.__version__, str)
    assert silvol.__version__ != ""


def test_exports():
    """Top-level package should export Silvol, AsyncSilvol, __version__."""
    import silvol

    assert hasattr(silvol, "Silvol")
    assert hasattr(silvol, "AsyncSilvol")
    assert hasattr(silvol, "__version__")
