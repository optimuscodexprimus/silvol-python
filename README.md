# silvol-python

Python SDK for [Silvol](https://silvol.ai) — an OpenAI-compatible inference API running on
Nosana's decentralised GPU grid.

Drop-in replacement for the OpenAI SDK. Change the base URL and your key; keep the rest
of your code.

---

## Install

```bash
pip install silvol
```

With optional framework integrations:

```bash
pip install silvol[langchain]   # LangChain
pip install silvol[crewai]      # CrewAI
pip install silvol[all]         # both
```

---

## Quickstart

```python
from silvol import Silvol

client = Silvol(api_key="sk-svl-...")          # or set the SILVOL_API_KEY env var

resp = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-7B",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

Async:

```python
import asyncio
from silvol import AsyncSilvol

async def main():
    client = AsyncSilvol(api_key="sk-svl-...")
    resp = await client.chat.completions.create(
        model="DeepSeek-R1-Distill-Qwen-7B",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    async for chunk in resp:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

asyncio.run(main())
```

---

## LangChain

```python
from silvol.integrations.langchain import SilvolChat

llm = SilvolChat(api_key="sk-svl-...")
result = llm.invoke("Summarise the Silvol architecture in one sentence.")
print(result.content)
```

---

## CrewAI

```python
from silvol.integrations.crewai import SilvolLLM
from crewai import Agent, Task, Crew

llm = SilvolLLM(api_key="sk-svl-...")

researcher = Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in AI",
    backstory="...",
    llm=llm,
)
```

---

## Models

| Model ID | Context | Notes |
|---|---|---|
| `DeepSeek-R1-Distill-Qwen-7B` | 32k | Always-on — call directly |
| `meta-llama/Llama-3.1-70B-Instruct` | 128k | On-demand — deploy first |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 32k | On-demand — deploy first |

`DeepSeek-R1-Distill-Qwen-7B` is the always-on model returned by
`GET https://api.silvol.ai/v1/models` — call it directly. Other models are served
**on demand**: provision a dedicated GPU deployment first (from the
[dashboard](https://silvol.ai/dashboard) or the deployments API), then call it by the
HuggingFace model ID you deployed.

---

## Authentication

Get your API key from the [Silvol Dashboard](https://silvol.ai/dashboard).
Keys are prefixed `sk-svl-`. Pass it as `api_key=`, or set the `SILVOL_API_KEY`
environment variable (the SDK reads it automatically). `OPENAI_API_KEY` also works
as a fallback.

---

## Links

- Docs: [silvol.ai/docs](https://silvol.ai/docs)
- Dashboard: [silvol.ai/dashboard](https://silvol.ai/dashboard)
- PyPI: [pypi.org/project/silvol](https://pypi.org/project/silvol)
- GitHub: [github.com/optimuscodexprimus/silvol-python](https://github.com/optimuscodexprimus/silvol-python)
