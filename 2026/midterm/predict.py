"""Web service: probability that the player dies in the wave that is starting.

Input is the player's state at `wave_start` (same fields as data/waves.csv);
card and pickup counts come as dicts. Fields left out count as 0.

Run: uvicorn predict:app --host 0.0.0.0 --port 9696
"""
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import add_features, to_dicts

MODEL_PATH = Path(os.getenv('MODEL_PATH', Path(__file__).with_name('model.bin')))
with MODEL_PATH.open('rb') as f:
    saved = pickle.load(f)
pipeline, categorical, numerical = saved['pipeline'], saved['categorical'], saved['numerical']

# Above this probability the game could step in: drop a medkit, soften the wave.
HIGH_RISK = 0.3

app = FastAPI(title='Corovans: death in the next wave')


class WaveState(BaseModel):
    game_version: str = '1.7'
    started_at: str | None = Field(None, description='session start, ISO; defaults to now')
    prior_sessions: int = 0
    prior_best_wave: int = 0
    wave: int
    boss: bool = False
    caravans: int = 0
    guards: int
    elapsed_s: float = 0
    hp_start: float
    max_hp: float = 100
    regen_per_s: float = 1.5
    prev_guards: int = 0
    prev_damage_taken: float = 0
    prev_duration_s: float = 0
    prev_caravans_escaped: int = 0
    prev_dashes: int = 0
    guards_killed: int = 0
    caravans_robbed: int = 0
    gold_earned: int = 0
    gold_spent: int = 0
    rerolls: int = 0
    flawless_waves: int = 0
    relics: int = 0
    cards_rare_plus: int = 0
    cards: dict[str, int] = Field(default_factory=dict, description='card id -> copies owned')
    pickups: dict[str, int] = Field(default_factory=dict, description='pickup id -> collected')


def to_row(state):
    row = state.model_dump(exclude={'cards', 'pickups'})
    row['started_at'] = row['started_at'] or datetime.now(timezone.utc).isoformat()
    row['boss'] = int(row['boss'])
    row['hp_frac'] = row['hp_start'] / max(row['max_hp'], 1)
    row['cards_total'] = sum(state.cards.values())
    row.update({f'card_{k}': v for k, v in state.cards.items()})
    row.update({f'pickup_{k}': v for k, v in state.pickups.items()})
    df = pd.DataFrame([row])
    for column in numerical:
        if column not in df:
            df[column] = 0
    return add_features(df)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(state: WaveState):
    df = to_row(state)
    probability = float(pipeline.predict_proba(to_dicts(df, categorical, numerical))[0, 1])
    return {
        'death_probability': round(probability, 4),
        'high_risk': probability >= HIGH_RISK,
    }
