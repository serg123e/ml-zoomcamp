# Homework 5 — notes

| Q | Answer | How |
|---|---|---|
| Q1 | uv 0.10.11 | `uvx --from uv==0.10.11 uv --version` (local uv 0.8.14 can't fetch Python 3.11.15) |
| Q2 | 1.7.2 | `scikit-learn==1.7.2` in `pyproject.toml` / `uv.lock` |
| Q3 | 0.533 | `uv run python q3_predict.py` |
| Q4 | 0.770 | `uv run uvicorn predict:app --port 9696` + `python q6_test.py` → 0.769799 |
| Q5 | python:3.11.15-slim-bookworm | `FROM` line in `Dockerfile` |
| Q6 | 0.770 | `docker run -p 9696:9696 zoomcamp-model:2026-hw5` + `python q6_test.py` |

`/health` returns `status: ok` and `model_sha256 = 1646bbdc…29f1`, same as `model_metadata.json`
and `sha256sum pipeline.bin`. Image size locally: 686 MB (not graded).
