"""
Smoke tests for the Silvol fine-tuning SDK.

These tests verify the client attaches the finetune attribute, that the
dataset encoder accepts multiple input shapes, and that the FinetuneError
class is exported.
"""

import base64
import json
from unittest.mock import patch


def test_finetune_attribute_attached_sync():
    """Silvol() should expose .finetune as a Finetune instance."""
    from silvol import Silvol, Finetune

    with patch("openai.OpenAI.__init__", return_value=None):
        c = Silvol(api_key="sk-svl-test")
    assert isinstance(c.finetune, Finetune)


def test_finetune_attribute_attached_async():
    """AsyncSilvol() should expose .finetune as an AsyncFinetune instance."""
    from silvol import AsyncSilvol, AsyncFinetune

    with patch("openai.AsyncOpenAI.__init__", return_value=None):
        c = AsyncSilvol(api_key="sk-svl-test")
    assert isinstance(c.finetune, AsyncFinetune)


def test_encode_dataset_bytes():
    """_encode_dataset accepts raw bytes."""
    from silvol.finetune import _encode_dataset

    raw = b'{"messages":[{"role":"user","content":"hi"}]}\n'
    encoded = _encode_dataset(raw)
    assert base64.b64decode(encoded) == raw


def test_encode_dataset_string():
    """_encode_dataset accepts a JSONL string."""
    from silvol.finetune import _encode_dataset

    raw = '{"messages":[{"role":"user","content":"hi"}]}'
    encoded = _encode_dataset(raw)
    assert base64.b64decode(encoded).decode("utf-8") == raw


def test_encode_dataset_iterable_of_dicts():
    """_encode_dataset converts an iterable of dicts to JSONL."""
    from silvol.finetune import _encode_dataset

    examples = [
        {"messages": [{"role": "user", "content": "one"}]},
        {"messages": [{"role": "user", "content": "two"}]},
    ]
    encoded = _encode_dataset(examples)
    decoded = base64.b64decode(encoded).decode("utf-8")
    lines = decoded.split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["messages"][0]["content"] == "one"
    assert json.loads(lines[1])["messages"][0]["content"] == "two"


def test_finetune_error_exported():
    """FinetuneError should be importable from the top-level package."""
    import silvol

    assert hasattr(silvol, "FinetuneError")
    assert issubclass(silvol.FinetuneError, Exception)


def test_version_bumped_to_0_2():
    """SDK version should be bumped to 0.2.x for the fine-tuning release."""
    import silvol

    parts = silvol.__version__.split(".")
    assert parts[0] == "0"
    assert int(parts[1]) >= 2, f"expected version >= 0.2.0 for finetune release, got {silvol.__version__}"
