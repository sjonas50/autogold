# Code Review Report
**Date:** 2026-04-03
**Status:** PASS WITH NOTES

## Critical Issues (must fix)

1. **Pickle deserialization of untrusted data** -- `scripts/regime_analyst.py:139` loads a pickle file (`pickle.load(f)`) which is vulnerable to arbitrary code execution if the `data/hmm_regime_model.pkl` file is tampered with. Use `safetensors`, `joblib`, or validate the file hash before loading.

2. **Webhook secret comparison is not timing-safe** -- `src/gold_trading/webhook/validator.py:35` uses `==` to compare `payload.secret` with the expected secret. This is vulnerable to timing side-channel attacks. Use `hmac.compare_digest()` instead.

3. **Missing `OPENAI_API_KEY` from `.env.example`** -- `src/gold_trading/embeddings/client.py:10` requires `OPENAI_API_KEY` but `.env.example` does not list it. Developers will hit a `RuntimeError` at runtime with no guidance on where to get the key.

4. **Dependencies not pinned to exact versions** -- `pyproject.toml:6-24` uses `>=` for all dependencies. A breaking update to `vectorbt`, `hmmlearn`, or `pandas-ta` could silently break the system. Pin exact versions or use `~=` (compatible release) for production trading software.

## Warnings (should fix)

1. **`fetch_all_fred_data` runs sequentially despite docstring claiming parallel** -- `scripts/macro_analyst.py:68-80` stores coroutines in a list then awaits them one by one. Use `asyncio.gather()` for actual concurrent execution.

2. **Missing `db/queries/__init__.py`** -- `src/gold_trading/db/queries/` has no `__init__.py`. This works due to the src layout but is inconsistent with every other package in the project and may break some tooling.

3. **Conditional f-string bug in `simulate_exit`** -- `src/gold_trading/webhook/simulator.py:130-136` uses a ternary expression inside `logger.info()` that produces incorrect output when `r_mult` is falsy (including `0.0`). The `else` branch concatenates two separate f-strings that are not parenthesized together, so the second f-string is always evaluated as the argument.

4. **Module-level `OPENAI_API_KEY` cached at import time** -- `src/gold_trading/embeddings/client.py:10` reads `OPENAI_API_KEY` at import and stores it in a module global. If the env var is set after import (e.g., via `dotenv`), the cached empty string is used as a fallback inside the functions.

5. **Docker Compose uses default credentials** -- `docker-compose.yml:8-9` uses `gold`/`gold` for Postgres. Fine for local dev, but should be documented as dev-only. The hardcoded fallback in `src/gold_trading/db/client.py:17-19` also embeds these credentials as a default.

6. **`_pool._closed` accesses private asyncpg attribute** -- `src/gold_trading/db/client.py:25,39` uses `_pool._closed`, an undocumented private attribute. This could break on asyncpg upgrades.

7. **No `README.md`** -- Project root has no `README.md`. `CLAUDE.md` fills this role partially but a `README.md` is standard for anyone not using Claude Code.

8. **`quant_researcher.py` synthesizes fake trade P&Ls for Monte Carlo** -- `scripts/quant_researcher.py:246-252` generates synthetic trade P&Ls from summary stats rather than extracting actual trade-by-trade results from the vectorbt backtest. This undermines the Monte Carlo simulation's validity (loses autocorrelation, tail distribution, etc.).

9. **Broad `except Exception` handlers suppress failures** -- `scripts/regime_analyst.py:230`, `scripts/quant_researcher.py:73,216` catch `Exception` and log warnings but continue execution. A misconfigured embedding client or broken HMM model will silently degrade system output.

10. **`get_recent_scores` uses inline `__import__("json")`** -- `src/gold_trading/db/queries/sentiment.py:105` calls `__import__("json")` inside a list comprehension. The `json` import already exists at the top of the file via `insert_sentiment_score`; this is a code smell indicating the import was forgotten at module level.

## Suggestions (nice to have)

1. **Add rate limiting to `/webhook/signal`** -- The endpoint has no rate limiter. A misbehaving TradingView alert or attacker could flood the system.

2. **Use `pydantic-settings` `BaseSettings` for all config** -- Risk rules, account size, webhook secret, and API keys are scattered across `os.environ.get()` calls in 8+ files. Centralizing into a single `Settings` class would eliminate duplication and provide validation at startup.

3. **Add `Dockerfile` multi-stage build** -- `Dockerfile` copies `pyproject.toml` and runs `uv sync` but does not copy `uv.lock`, so builds are not reproducible. Also missing `scripts/` directory in the image.

4. **Corpus ingestion deletes all rows before insert** -- `src/gold_trading/embeddings/corpus.py:177` runs `DELETE FROM pinescript_corpus` then inserts. If the embed step fails mid-way, the corpus is empty. Use a transaction with atomic swap or upsert-only.

5. **`scripts/test_agent.py` uses synchronous `httpx`** -- All other scripts use async patterns. Minor inconsistency.

6. **`test_integration.py` hardcodes machine-specific paths** -- `tests/test_integration.py:42-43` hardcodes `/Users/sjonas/tradingview` and `/Users/sjonas/.local/bin/uv`. These will fail on any other developer's machine or CI.

7. **No test for `run_all_entry_checks` in `risk/rules.py`** -- The composite risk check function at line 113 has no dedicated test.

8. **`backtest/engine.py` short-selling logic looks incorrect** -- Lines 78-89 pass `short_entries=signals.entries` and `short_exits=signals.exits` but also swaps `entries`/`exits` in the positional args. This double-inversion may produce unexpected behavior depending on vectorbt version.

9. **Consider adding `py.typed` marker** -- For downstream type checking, `src/gold_trading/py.typed` would declare the package as typed.

10. **`pyproject.toml` target-version says `py311` but requires-python says `>=3.12`** -- `pyproject.toml:43` sets ruff `target-version = "py311"` while line 3 sets `requires-python = ">=3.12"`. These should match.

## Metrics
- Files reviewed: 43 (31 source, 12 test)
- Test count: 132
- Ruff violations: 0
- Security issues: 3 critical, 7 warning
