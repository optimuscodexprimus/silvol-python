"""
Silvol Python SDK — sync and async clients.

Drop-in replacements for openai.OpenAI / openai.AsyncOpenAI that point at
the Silvol inference gateway by default.
"""

from openai import OpenAI, AsyncOpenAI

from ._version import __version__

__all__ = ["Silvol", "AsyncSilvol"]

_DEFAULT_BASE_URL = "https://api.silvol.ai/v1"


class Silvol(OpenAI):
    """
    Synchronous Silvol client.

    Usage::

        from silvol import Silvol

        client = Silvol(api_key="sk-svl-...")
        resp = client.chat.completions.create(
            model="DeepSeek-R1-Distill-Qwen-7B",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(resp.choices[0].message.content)
    """

    DEFAULT_BASE_URL: str = _DEFAULT_BASE_URL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )


class AsyncSilvol(AsyncOpenAI):
    """
    Asynchronous Silvol client.

    Usage::

        import asyncio
        from silvol import AsyncSilvol

        async def main():
            client = AsyncSilvol(api_key="sk-svl-...")
            resp = await client.chat.completions.create(
                model="DeepSeek-R1-Distill-Qwen-7B",
                messages=[{"role": "user", "content": "Hello"}],
            )
            print(resp.choices[0].message.content)

        asyncio.run(main())
    """

    DEFAULT_BASE_URL: str = _DEFAULT_BASE_URL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )
