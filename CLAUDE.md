# CLAUDE.md — silvol-python (PyPI SDK)

> **For Claude:** Full project context lives in the `silvol` repo:
> `C:\Users\otien\AppData\Local\Temp\silvol\CLAUDE.md` — read that first.

---

## This Repo at a Glance

- **Purpose:** Thin Python SDK wrapping the Silvol OpenAI-compatible API
- **PyPI package name:** `silvol`
- **Version:** 0.1.0
- **GitHub:** `https://github.com/optimuscodexprimus/silvol-python`
- **Local clone:** `C:\Users\otien\AppData\Local\Temp\silvol-python`
- **Default base URL:** `https://api.silvol.ai/v1`

## Package Layout

```
src/silvol/
  __init__.py          Exports: Silvol, AsyncSilvol, __version__
  client.py            Silvol (subclasses openai.OpenAI), AsyncSilvol (subclasses openai.AsyncOpenAI)
  _version.py          __version__ = "0.1.0"
  integrations/
    langchain.py       SilvolChat() → ChatOpenAI wired to Silvol gateway
    crewai.py          SilvolLLM()  → crewai.LLM wired to Silvol gateway
tests/
  test_client.py       5 smoke tests (all passing)
.github/workflows/
  publish.yml          push to main → TestPyPI; tagged release → PyPI (OIDC)
```

## Building & Testing

```bash
# Build wheel + sdist
python -m build

# Run tests (from repo root, with src on PYTHONPATH)
PYTHONPATH=src python -m pytest tests/ -v
```

## PyPI Publish Checklist (manual — see Prompt 8 TODOs)

1. Create account at pypi.org + test.pypi.org
2. `python -m twine upload --repository testpypi dist/*`
3. `pip install --index-url https://test.pypi.org/simple/ silvol` (verify)
4. Tag a release → GitHub Actions auto-publishes to production PyPI
