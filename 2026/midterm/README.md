# Corovans: will the player die in this wave?

ML Zoomcamp 2026 midterm project. The data is real telemetry from my own browser game.

**Play the game:** https://korovany.123automate.it/

## Problem

*Corovans* is a small arcade game where you play a bandit robbing caravans. Guards come in
waves and every wave is harder than the one before. Between waves you buy upgrade cards in a
shop. A run ends when the player dies.

At the start of each wave the model predicts **the probability that the player dies in that
wave**. The game can use this:

- **adaptive difficulty:** when the risk is high, drop a medkit or make the wave a bit weaker;
- **balance analysis:** find waves, cards and versions where players die more often than the
  model expects.

This is binary classification. Only ~11% of waves end in death, so models are compared by
ROC AUC.

## Data

The game uploads every session (one run, from start to death) as JSON with a list of events:
`wave_start`, `player_damaged`, `guard_killed`, `card_picked`, `player_died` and so on.

`prepare_data.py` turns 1052 sessions from 459 players (May–October 2026) into
[`data/waves.csv`](data/waves.csv):

- **one row = one wave a player started.** That gives 9404 rows;
- **features describe the state *before* the wave starts**, so nothing from the wave itself
  leaks in: wave number and size, HP, upgrade cards owned, gold, damage taken in the previous
  wave, the player's earlier sessions and best wave;
- **target:** `died` = 1 if `player_died` happens before the next `wave_start`;
- **anonymized:** player and session ids are hashed, player names are dropped.

HP is logged only when the player takes a hit, while healing (regeneration, medkits, cards) is
silent. `prepare_data.py` rebuilds HP between hits with the game's own formulas. That cuts the
error at the next logged hit from 6.5 to 2.8 HP on average.

The raw sessions are not in the repo. `data/waves.csv` is the processed dataset and is enough
to rerun everything.

## Approach

The full analysis is in [`notebook.ipynb`](notebook.ipynb).

- **EDA:** death rate by wave, HP, cards and game version. Low HP relative to max HP and heavy
  damage in the previous wave are the strongest single signals.
- **Version drift:** game balance changed between versions 1.1 and 1.6. Because of that, the
  test set is **time-based**: the newest version 1.6 (sessions from 2026-08-16, 635 waves,
  12 players). All earlier sessions are used for training.
- **Validation:** 5-fold cross-validation **grouped by player**, so the same player is never in
  both train and validation.
- **Feature engineering** ([`features.py`](features.py)): expected damage per wave =
  damage per guard in the previous wave × guards in this wave. Its ratio to current HP
  (`expected_damage_to_hp`) is the most important feature of the final model.
- **Outliers:** numerical features are clipped at the 99th percentile of train for the linear
  model. Some players left the game open for hours or reached wave 80.
- **Baseline without ML:** a formula, `risk = damage_per_guard × guards^a / HP^b / (1 + best_wave)^x`,
  with exponents picked by grid search.
- **Models:** logistic regression, decision tree, random forest and XGBoost. Hyperparameters are
  tuned by grouped cross-validation.

## Results

| Model | CV AUC | Test AUC (v1.6), 95% CI |
|---|---|---|
| **XGBoost** | **0.842 ± 0.013** | 0.812 (0.75–0.88) |
| Random forest | 0.835 ± 0.016 | 0.816 (0.74–0.88) |
| Logistic regression | 0.816 ± 0.019 | 0.796 (0.72–0.87) |
| Decision tree | 0.796 ± 0.012 | 0.753 (0.68–0.86) |
| Formula (no ML) | 0.792 ± 0.013 | 0.795 (0.65–0.85) |

- **Final model:** XGBoost with `max_depth=6, min_child_weight=20, n_estimators=300,
  learning_rate=0.03`.
- **The simple formula comes surprisingly close.** Most of the signal is "expected damage vs HP
  left", and ML adds about +0.05 AUC in cross-validation.
- **The test set is small** (74 deaths), so its confidence intervals overlap. Models are chosen
  by cross-validation.

The service flags a wave as `high_risk` when the probability is ≥ 0.3. On the test version that
flags 6.5% of waves. 46% of the flagged waves end in death (the base rate is 12%), and they
contain 26% of all deaths.

### Side study: will a new player come back?

I also tried to predict whether a player plays a second game, using telemetry of the first game
([`prepare_retention.py`](prepare_retention.py), [`retention.ipynb`](retention.ipynb)).
There is no signal. Players who die on wave 3 come back as often as those who reach wave 11
(27% vs 30%). Models score AUC ≈ 0.5, the same as on shuffled labels. So the main task stays
death prediction.

## Project structure

| File | What it does |
|---|---|
| `prepare_data.py` | raw session JSON → `data/waves.csv` (needs the raw telemetry, not included) |
| `features.py` | feature engineering shared by the notebook, `train.py` and `predict.py` |
| `notebook.ipynb` | EDA, version drift, model selection and tuning |
| `train.py` | trains the final XGBoost and saves it to `model.bin` |
| `predict.py` | FastAPI web service |
| `test_predict.py` | sends two example waves to the service |
| `Dockerfile`, `pyproject.toml`, `uv.lock` | environment and container |
| `retention.ipynb`, `prepare_retention.py`, `data/first_games.csv` | side study on returning players |

## How to run

You need [uv](https://docs.astral.sh/uv/) 0.10+ and, for the container, Docker.

```bash
cd 2026/midterm
uv sync                      # creates .venv with the locked dependencies (Python 3.11)
```

**Train** (about 30 s; prints the test AUC and writes `model.bin`):

```bash
uv run python train.py
```

**Serve locally:**

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
uv run python test_predict.py          # in another terminal
```

**Docker:**

```bash
docker build -t corovans-death .
docker run -it --rm -p 9696:9696 corovans-death
uv run python test_predict.py          # or the curl below
```

**Notebooks:** `uv run jupyter notebook`. The full `notebook.ipynb` run takes 20–40 minutes
because of the hyperparameter grids.

## API

`POST /predict` takes the player's state at the start of a wave. Only `wave`, `guards` and
`hp_start` are required; fields left out count as 0. Cards and pickups are given as counts.

```bash
curl -X POST http://localhost:9696/predict -H 'Content-Type: application/json' -d '{
  "game_version": "1.7", "prior_sessions": 0, "prior_best_wave": 0,
  "wave": 8, "guards": 16, "hp_start": 30, "max_hp": 100,
  "prev_guards": 13, "prev_damage_taken": 80, "prev_duration_s": 45,
  "cards": {"damage": 2, "glassCannon": 1}
}'
```

```json
{"death_probability": 0.469, "high_risk": true}
```

The interactive docs are at http://localhost:9696/docs, and `GET /health` is a liveness check.
