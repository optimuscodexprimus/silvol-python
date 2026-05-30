"""
Silvol — Python SDK for the Silvol inference + fine-tuning API.

Drop-in replacement for the OpenAI SDK pointing at Silvol's
decentralised GPU gateway, plus a fine-tuning client at ``client.finetune``.

Quickstart::

    from silvol import Silvol

    client = Silvol(api_key="sk-svl-...")

    # Inference
    resp = client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)

    # Fine-tuning ($15 flat per run)
    job = client.finetune.submit_job(
        job_name="legal-assistant-v1",
        base_model="llama-3.1-8b",
        dataset_path="training.jsonl",
    )
    client.finetune.upload_dataset(job["dataset_upload_url"], "training.jsonl")
"""

from ._version import __version__
from .client import AsyncSilvol, Silvol
from .finetune import AsyncFinetune, Finetune, FinetuneError

__all__ = [
    "Silvol",
    "AsyncSilvol",
    "Finetune",
    "AsyncFinetune",
    "FinetuneError",
    "__version__",
]
