"""
LangChain integration for Silvol.

Requires: pip install silvol[langchain]
"""

from __future__ import annotations

__all__ = ["SilvolChat"]

_BASE_URL = "https://api.silvol.ai/v1"
_DEFAULT_MODEL = "DeepSeek-R1-Distill-Qwen-7B"


def SilvolChat(
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    **kwargs,
):
    """
    Return a LangChain ``ChatOpenAI`` instance wired to the Silvol gateway.

    Usage::

        from silvol.integrations.langchain import SilvolChat

        llm = SilvolChat(api_key="sk-svl-...")
        print(llm.invoke("Hello"))

    Parameters
    ----------
    api_key:
        Your Silvol API key (``sk-svl-...``).  Falls back to the
        ``OPENAI_API_KEY`` environment variable if omitted.
    model:
        Model ID to use.  Defaults to ``DeepSeek-R1-Distill-Qwen-7B``.
    **kwargs:
        Forwarded verbatim to ``ChatOpenAI``.
    """
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=_BASE_URL,
        model=model,
        **kwargs,
    )
