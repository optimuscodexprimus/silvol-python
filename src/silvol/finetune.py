"""
Silvol fine-tuning API client.

Wraps the gateway endpoints under ``/v1/finetune/*`` with a small typed surface.
Attached to ``Silvol`` and ``AsyncSilvol`` clients as ``client.finetune``.

Quickstart::

    from silvol import Silvol

    client = Silvol(api_key="sk-svl-...")

    # Submit a training job
    with open("dataset.jsonl", "rb") as f:
        job = client.finetune.submit_job(
            job_name="legal-assistant-v1",
            base_model="llama-3.1-8b",
            dataset_file=f,
        )
    print(job["id"], job["price_cents"])

    # Poll until ready
    while True:
        job = client.finetune.get_job(job["id"])
        if job["status"] in ("ready", "failed", "cancelled"):
            break

    # Deploy and use
    if job["status"] == "ready":
        client.finetune.deploy_job(job["id"])
        resp = client.chat.completions.create(
            model=f"silvol/{job['job_name']}",
            messages=[{"role": "user", "content": "What is force majeure?"}],
        )
"""

from __future__ import annotations

import base64
from typing import Any, BinaryIO, Iterable, Literal


BaseModel = Literal["llama-3.1-8b", "mistral-7b"]
JobStatus = Literal[
    "pending", "training", "uploading", "ready",
    "deploying", "deployed", "failed", "cancelled",
]


def _encode_dataset(dataset: bytes | BinaryIO | str | Iterable[dict]) -> str:
    """Accept multiple input shapes and return base64-encoded JSONL."""
    if isinstance(dataset, str):
        # Already JSONL text
        data = dataset.encode("utf-8")
    elif isinstance(dataset, bytes):
        data = dataset
    elif hasattr(dataset, "read"):
        chunk = dataset.read()
        data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
    else:
        # Iterable of message dicts → JSONL
        import json
        data = "\n".join(json.dumps(ex) for ex in dataset).encode("utf-8")
    return base64.b64encode(data).decode("ascii")


def _build_path(*parts: str) -> str:
    return "/" + "/".join(p.strip("/") for p in parts)


# ── Sync ─────────────────────────────────────────────────────────────────────


class Finetune:
    """Synchronous fine-tuning API. Accessed via ``client.finetune``."""

    def __init__(self, client: Any) -> None:
        # `client` is a Silvol (subclass of OpenAI) — reuse its underlying
        # httpx client so we share auth, base URL, timeouts, retries, etc.
        self._client = client

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        # openai SDK exposes `_client._request` on the OpenAI instance; we use
        # `_client._client.request` via the underlying httpx client for raw access.
        url = path  # base_url is already set on the openai client
        # The openai sync client carries a base URL in self.base_url and an
        # httpx client at self._client. We hit that directly for non-OpenAI routes.
        http = self._client._client  # httpx.Client
        full_url = f"{str(self._client.base_url).rstrip('/')}{url}"
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        resp = http.request(method, full_url, headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error") or resp.text
            except Exception:
                detail = resp.text
            raise FinetuneError(f"{method} {path} failed ({resp.status_code}): {detail}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ── Jobs ────────────────────────────────────────────────────────────────

    def submit_job(
        self,
        *,
        job_name: str,
        base_model: BaseModel,
        dataset_file: bytes | BinaryIO | str | Iterable[dict] | None = None,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a training job. Charges $15 via Stripe immediately.

        Provide either ``dataset_file`` (bytes / file object / JSONL string /
        iterable of message dicts) or ``dataset_path`` (path to a .jsonl file).

        Returns the job record including a ``dataset_upload_url`` — the SDK
        does NOT upload your dataset for you in this version. Use the returned
        signed URL with your own HTTP client (or call ``upload_dataset()``).
        """
        if dataset_path and not dataset_file:
            with open(dataset_path, "rb") as f:
                dataset_file = f.read()
        if dataset_file is None:
            raise ValueError("Provide dataset_file or dataset_path")

        return self._request(
            "POST",
            "/v1/finetune/jobs",
            json={
                "jobName":     job_name,
                "baseModel":   base_model,
                "jsonlBase64": _encode_dataset(dataset_file),
            },
        )

    def upload_dataset(self, signed_upload_url: str, dataset_path: str) -> None:
        """
        Upload a JSONL file to the signed URL returned by ``submit_job``.
        Training starts automatically on the next poll cycle (~5 min).
        """
        with open(dataset_path, "rb") as f:
            data = f.read()
        # The signed URL hits Supabase Storage directly, not the gateway —
        # use a fresh httpx call without our Bearer token.
        import httpx
        resp = httpx.put(
            signed_upload_url,
            content=data,
            headers={"Content-Type": "application/jsonl"},
            timeout=60.0,
        )
        if resp.status_code >= 400:
            raise FinetuneError(f"Dataset upload failed ({resp.status_code}): {resp.text}")

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/finetune/jobs")["jobs"]

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", _build_path("v1", "finetune", "jobs", job_id))

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a pending job. Refund issued automatically."""
        return self._request("DELETE", _build_path("v1", "finetune", "jobs", job_id))

    def deploy_job(self, job_id: str) -> dict[str, Any]:
        """Deploy a ready model. Provisions a dedicated GPU node at $0.80/hr."""
        return self._request(
            "POST", _build_path("v1", "finetune", "jobs", job_id, "deploy")
        )

    # ── Models ──────────────────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/finetune/models")["models"]

    def delete_model(self, model_id: str) -> dict[str, Any]:
        """Stop a deployed model or delete a stored one."""
        return self._request(
            "DELETE", _build_path("v1", "finetune", "models", model_id)
        )


# ── Async ────────────────────────────────────────────────────────────────────


class AsyncFinetune:
    """Asynchronous fine-tuning API. Accessed via ``async_client.finetune``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        http = self._client._client  # httpx.AsyncClient
        full_url = f"{str(self._client.base_url).rstrip('/')}{path}"
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        resp = await http.request(method, full_url, headers=headers, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error") or resp.text
            except Exception:
                detail = resp.text
            raise FinetuneError(f"{method} {path} failed ({resp.status_code}): {detail}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    async def submit_job(
        self,
        *,
        job_name: str,
        base_model: BaseModel,
        dataset_file: bytes | BinaryIO | str | Iterable[dict] | None = None,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        if dataset_path and not dataset_file:
            with open(dataset_path, "rb") as f:
                dataset_file = f.read()
        if dataset_file is None:
            raise ValueError("Provide dataset_file or dataset_path")

        return await self._request(
            "POST",
            "/v1/finetune/jobs",
            json={
                "jobName":     job_name,
                "baseModel":   base_model,
                "jsonlBase64": _encode_dataset(dataset_file),
            },
        )

    async def upload_dataset(self, signed_upload_url: str, dataset_path: str) -> None:
        with open(dataset_path, "rb") as f:
            data = f.read()
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.put(
                signed_upload_url,
                content=data,
                headers={"Content-Type": "application/jsonl"},
            )
            if resp.status_code >= 400:
                raise FinetuneError(f"Dataset upload failed ({resp.status_code}): {resp.text}")

    async def list_jobs(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/finetune/jobs"))["jobs"]

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", _build_path("v1", "finetune", "jobs", job_id))

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("DELETE", _build_path("v1", "finetune", "jobs", job_id))

    async def deploy_job(self, job_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", _build_path("v1", "finetune", "jobs", job_id, "deploy")
        )

    async def list_models(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/finetune/models"))["models"]

    async def delete_model(self, model_id: str) -> dict[str, Any]:
        return await self._request(
            "DELETE", _build_path("v1", "finetune", "models", model_id)
        )


# ── Errors ───────────────────────────────────────────────────────────────────


class FinetuneError(Exception):
    """Raised on non-2xx responses from the Silvol fine-tuning API."""

    pass
