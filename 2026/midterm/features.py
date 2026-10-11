"""Feature engineering shared by the notebook, train.py and predict.py.

Game balance changed between versions (regen rebalance on 2026-06-07, new
cards and guards), so raw card counts mean different things in different
versions. The derived ratios below compare expected damage with the HP
buffer and stay stable across versions.
"""
import numpy as np

REGEN_REBALANCE = '2026-06-07'

ID_COLUMNS = ['session_id', 'player_id', 'started_at']
TARGET = 'died'


def add_features(df):
    df = df.copy()
    hp = df.hp_start.clip(lower=1)
    prev_guards = df.prev_guards.where(df.prev_guards > 0, df.guards).clip(lower=1)

    df['era'] = np.where(df.started_at.str[:10] >= REGEN_REBALANCE, 'new', 'old')
    df['dmg_per_guard'] = df.prev_damage_taken / prev_guards
    df['expected_damage'] = df.dmg_per_guard * df.guards
    df['expected_damage_to_hp'] = df.expected_damage / hp
    df['prev_damage_to_hp'] = df.prev_damage_taken / hp
    df['guard_growth'] = df.guards / prev_guards
    df['log_prior_best_wave'] = np.log1p(df.prior_best_wave)
    return df


def risk_formula(df, a=1.25, b=1.5, x=0.5):
    """Non-ML baseline: expected damage vs HP, discounted by player experience."""
    expected = df.dmg_per_guard * df.guards ** a
    return expected / df.hp_start.clip(lower=1) ** b / (1 + df.prior_best_wave) ** x


def feature_columns(df):
    categorical = ['game_version', 'era']
    numerical = [c for c in df.columns if c not in ID_COLUMNS + categorical + [TARGET]]
    return categorical, numerical


def clip_outliers(df, caps):
    df = df.copy()
    df[caps.index] = df[caps.index].clip(upper=caps, axis=1)
    return df


def fit_caps(df, numerical, q=0.99):
    return df[numerical].quantile(q)


def to_dicts(df, categorical, numerical):
    return df[categorical + numerical].to_dict(orient='records')

