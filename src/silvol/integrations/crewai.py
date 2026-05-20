"""
CrewAI integration for Silvol.

Requires: pip install silvol[crewai]
"""

from __future__ import annotations

__all__ = ["SilvolLLM"]

_BASE_URL = "https://api.silvol.ai/v1"
_DEFAULT_MODEL = "DeepSeek-R1-Distill-Qwen-7B"


def SilvolLLM(
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    **kwargs,
):
    """
    Return a CrewAI ``LLM`` instance wired to the Silvol gateway.

    Usage::

        from silvol.integrations.crewai import SilvolLLM
        from crewai import Agent

        llm = SilvolLLM(api_key="sk-svl-...")
        agent = Agent(role="researcher", goal="...", llm=llm)

    Parameters
    ----------
    api_key:
        Your Silvol API key (``sk-svl-...``).
    model:
        Model ID to use.  Defaults to ``DeepSeek-R1-Distill-Qwen-7B``.
    **kwargs:
        Forwarded verbatim to ``LLM``.
    """
    from crewai import LLM  # noqa: PLC0415

    return LLM(
        model=f"openai/{model}",
        api_key=api_key,
        base_url=_BASE_URL,
        **kwargs,
    )
