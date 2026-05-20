"""
Silvol — Python SDK for the Silvol inference API.

Drop-in replacement for the OpenAI SDK pointing at Silvol's
decentralised GPU gateway.

Quickstart::

    from silvol import Silvol

    client = Silvol(api_key="sk-svl-...")
    resp = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)
"""

from ._version import __version__
from .client import AsyncSilvol, Silvol

__all__ = ["Silvol", "AsyncSilvol", "__version__"]
