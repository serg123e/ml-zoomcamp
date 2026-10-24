"""Build a per-player dataset from Corovans telemetry sessions.

One row = the first game (session) of a player. Features describe only
that game; target `returned` = the player started at least one more game.

A session is uploaded when the player dies, so every row ends with a death.
Players who closed the tab in the middle of their first game are not in
the data at all.

Usage: python prepare_retention.py <sessions_dir> [output_csv]
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd

# Most players who come back start the next game within minutes (median 5),
# so a first game younger than this is too fresh to be labelled "left".
MIN_AGE = timedelta(days=1)


def anon(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:10]


def parse_time(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00'))


def game_version(meta):
    version = (meta.get('build') or {}).get('version')
    if not version:
        return 'unknown'
    return '.'.join(version.split('.')[:2])


def is_v13_plus(version):
    return version != 'unknown' and tuple(map(int, version.split('.'))) >= (1, 3)


def load_sessions(sessions_dir):
    """Human sessions with a known player, oldest first."""
    sessions = []
    for path in glob.glob(os.path.join(sessions_dir, '*.json')):
        with open(path) as f:
            session = json.load(f)
        meta = session.get('meta') or {}
        if meta.get('ai') or meta.get('sim'):
            continue
        start = next((e for e in session['events'] if e['type'] == 'session_start'), {})
        meta['playerId'] = meta.get('playerId') or start.get('playerId')
        meta['playerName'] = meta.get('playerName') or start.get('playerName')
        # Sessions without a player id (May 2026 builds) can't be linked to a
        # later game, so they would all look like players who left.
        if not meta['playerId']:
            continue
        session['meta'] = meta
        sessions.append(session)
    return sorted(sessions, key=lambda s: s['startedAt'])


def session_row(session):
    meta = session['meta']
    version = game_version(meta)
    totals = Counter()
    cards = Counter()
    wave_start_t = 0
    last_wave_damage = 0
    death_source = 'none'
    died = {}

    for e in session['events']:
        kind = e['type']
        if kind == 'wave_start':
            wave_start_t = e['t']
            last_wave_damage = 0
            totals['boss_wave'] = int(bool(e.get('boss')))
        elif kind == 'wave_end':
            totals['waves_completed'] += 1
            totals['caravans_escaped'] += e.get('caravansEscaped', 0)
            if e.get('damageTaken', 0) == 0:
                totals['no_damage_waves'] += 1
        elif kind == 'player_damaged':
            totals['damage_taken'] += e.get('amount', 0)
            totals['hits_taken'] += 1
            last_wave_damage += e.get('amount', 0)
            death_source = e.get('source', 'other').split(':')[0]
        elif kind == 'guard_killed':
            totals['guards_killed'] += 1
        elif kind == 'caravan_robbed':
            totals['caravans_robbed'] += 1
            totals['gold_earned'] += e.get('value', 0)
        elif kind == 'shop_opened':
            totals['shop_visits'] += 1
        elif kind == 'card_picked':
            cards[e['id']] += 1
            totals['gold_spent'] += e.get('cost', 0)
            if e.get('rarity') in ('rare', 'legendary'):
                totals['cards_rare_plus'] += 1
        elif kind == 'reroll':
            totals['rerolls'] += 1
            totals['gold_spent'] += e.get('cost', 0)
        elif kind == 'dash':
            totals['dashes'] += 1
        elif kind == 'pickup_collected':
            totals['pickups'] += 1
        elif kind == 'flawless':
            totals['flawless_waves'] += 1
        elif kind == 'relic_unlocked':
            totals['relics'] += 1
        elif kind == 'rebirth':
            totals['rebirths'] += 1
        elif kind == 'player_died':
            died = e

    end_t = session['events'][-1]['t'] if session['events'] else 0
    wave_reached = died.get('wave', totals['waves_completed'] + 1)
    duration_s = end_t / 1000
    return {
        'player_id': anon(meta['playerId']),
        'started_at': session['startedAt'],
        'game_version': version,
        'hour_utc': parse_time(session['startedAt']).hour,
        'has_name': int(bool(meta.get('playerName'))),
        'wave_reached': wave_reached,
        'waves_completed': totals['waves_completed'],
        'duration_s': round(duration_s, 1),
        'avg_wave_s': round(duration_s / max(wave_reached, 1), 1),
        'last_wave_s': round((end_t - wave_start_t) / 1000, 1),
        'boss_death': totals['boss_wave'],
        'death_source': death_source,
        'got_mercy_heal': int(is_v13_plus(version) and wave_reached >= 4),
        'final_score': died.get('finalScore', 0),
        'damage_taken': totals['damage_taken'],
        'hits_taken': totals['hits_taken'],
        'last_wave_damage': last_wave_damage,
        'no_damage_waves': totals['no_damage_waves'],
        'flawless_waves': totals['flawless_waves'],
        'guards_killed': totals['guards_killed'],
        'caravans_robbed': totals['caravans_robbed'],
        'caravans_escaped': totals['caravans_escaped'],
        'gold_earned': totals['gold_earned'],
        'gold_spent': totals['gold_spent'],
        'shop_visits': totals['shop_visits'],
        'cards_picked': sum(cards.values()),
        'cards_distinct': len(cards),
        'cards_rare_plus': totals['cards_rare_plus'],
        'rerolls': totals['rerolls'],
        'dashes': totals['dashes'],
        'dashes_per_min': round(totals['dashes'] / max(duration_s / 60, 1 / 60), 2),
        'pickups': totals['pickups'],
        'relics': totals['relics'],
        'rebirths': totals['rebirths'],
    }


def build(sessions_dir):
    by_player = defaultdict(list)
    sessions = load_sessions(sessions_dir)
    for session in sessions:
        by_player[session['meta']['playerId']].append(session)

    cutoff = parse_time(sessions[-1]['startedAt']) - MIN_AGE
    rows = []
    for games in by_player.values():
        first = games[0]
        if parse_time(first['startedAt']) > cutoff:
            continue
        row = session_row(first)
        row['returned'] = int(len(games) > 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('started_at', ignore_index=True)


if __name__ == '__main__':
    sessions_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'data/first_games.csv'
    df = build(sessions_dir)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    df.to_csv(output, index=False)
    print(f'{len(df)} players, returned rate {df.returned.mean():.3f} -> {output}')
