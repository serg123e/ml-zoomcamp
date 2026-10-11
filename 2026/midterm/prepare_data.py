"""Build a per-wave dataset from Corovans telemetry sessions.

One row = one wave the player started. Features describe the state
*before* the wave begins (no leakage); target `died` = the player died
during this wave.

Usage: python prepare_data.py <sessions_dir> [output_csv]
"""
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

CARDS = [
    'maxHp', 'damage', 'attackRange', 'cooldown', 'speed', 'lifesteal',
    'regen', 'thorns', 'glassCannon', 'dashCooldown', 'draftSize',
    'wideArc', 'magnet', 'crit', 'cleave', 'doubleStrike', 'fortify',
    'armorBreak', 'phantom',
]
PICKUPS = ['purse', 'frenzy', 'rage', 'guard', 'medkit', 'spice']

# Healing formulas from the game code (corovans/js: utils.js, ui.js, pickups.js,
# combat.js, game.js). Telemetry logs HP only on damage, so HP between hits is
# simulated with these; every `player_damaged.hpAfter` resyncs it to the truth.
# Lifesteal is approximated per kill: damage dealt to a guard ~ its max HP.
PLAYER_MAX_HP = 100
STACK_SCALE = [1, 0.6, 0.3, 0.15, 0.1]  # diminishing returns for the n-th copy of a card
MAXHP_CARD = 25        # +maxHp and full heal
GLASS_CANNON_HP = 15   # -maxHp (min 10)
FORTIFY_HEAL = 3       # heal per guard kill
LIFESTEAL_PCT = 0.07   # share of dealt damage healed
GUARD_HP = {'basic': 40, 'armored': 90, 'archer': 30, 'lancer': 55, 'bombardier': 34}
MEDKIT_HEAL = 20
MERCY_WAVE = 4         # full heal before wave 4 of a player's first session (v1.3+)
REGEN_REBALANCE = '2026-06-07'
# (base regen HP/s, regen card HP/s per copy) before / after the rebalance
REGEN_ERAS = {'old': (2.0, 1.0), 'new': (1.5, 1.85)}


def anon(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:10]


def game_version(meta):
    version = (meta.get('build') or {}).get('version')
    if not version:
        return 'unknown'
    return '.'.join(version.split('.')[:2])


def stack_scale(owned):
    return STACK_SCALE[min(owned, len(STACK_SCALE) - 1)]


class Hp:
    """Player HP between logged damage events. Regen ticks only while a wave is played."""

    def __init__(self, started_at):
        era = 'new' if started_at >= REGEN_REBALANCE else 'old'
        self.base_regen, self.regen_card = REGEN_ERAS[era]
        self.reset()

    def reset(self):
        self.hp = self.max_hp = PLAYER_MAX_HP
        self.regen = self.base_regen
        self.fortify = 0.0
        self.lifesteal = 0.0
        self.in_wave = self.in_shop = False
        self.t = 0

    def advance(self, t):
        if self.in_wave and not self.in_shop:
            self.hp = min(self.max_hp, self.hp + self.regen * (t - self.t) / 1000)
        self.t = t

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + amount)

    def card(self, card_id, owned):
        scale = stack_scale(owned)
        if card_id == 'maxHp':
            self.max_hp += round(MAXHP_CARD * scale)
            self.hp = self.max_hp
        elif card_id == 'glassCannon':
            self.max_hp = max(10, self.max_hp - round(GLASS_CANNON_HP * scale))
            self.hp = min(self.hp, self.max_hp)
        elif card_id == 'regen':
            self.regen += self.regen_card * scale
        elif card_id == 'fortify':
            self.fortify += FORTIFY_HEAL * scale
        elif card_id == 'lifesteal':
            self.lifesteal += LIFESTEAL_PCT * scale

    def kill(self, guard_type, wave):
        # Guards get +3% HP per wave past wave 5 (caravan.js).
        guard_hp = GUARD_HP.get(guard_type, GUARD_HP['basic']) * (1 + max(0, wave - 5) * 0.03)
        self.heal(self.fortify + guard_hp * self.lifesteal)


def is_v13_plus(version):
    return version != 'unknown' and tuple(map(int, version.split('.'))) >= (1, 3)


def load_sessions(sessions_dir):
    sessions = []
    for path in glob.glob(os.path.join(sessions_dir, '*.json')):
        with open(path) as f:
            session = json.load(f)
        meta = session.get('meta') or {}
        if meta.get('ai') or meta.get('sim'):
            continue
        if not meta.get('playerId'):
            start = next((e for e in session['events'] if e['type'] == 'session_start'), {})
            meta['playerId'] = start.get('playerId') or session['id']
        session['meta'] = meta
        sessions.append(session)
    return sorted(sessions, key=lambda s: s['startedAt'])


def session_rows(session, prior_sessions, prior_best_wave):
    rows = []
    hp = Hp(session['startedAt'])
    version = game_version(session['meta'])
    cards = Counter()
    pickups = Counter()
    totals = Counter()
    prev_wave = {}
    prev_guards = 0
    current = None

    def close(row, died):
        row['died'] = int(died)
        rows.append(row)

    for e in session['events']:
        kind = e['type']
        hp.advance(e['t'])
        if kind == 'wave_start':
            if current is not None:
                close(current, False)
            if e['wave'] == MERCY_WAVE and prior_sessions == 0 and is_v13_plus(version):
                hp.heal(hp.max_hp)
            current = {
                'session_id': anon(session['id']),
                'player_id': anon(session['meta']['playerId']),
                'game_version': version,
                'started_at': session['startedAt'],
                'prior_sessions': prior_sessions,
                'prior_best_wave': prior_best_wave,
                'wave': e['wave'],
                'boss': int(bool(e.get('boss'))),
                'caravans': e.get('caravans', 0),
                'guards': e.get('guards', 0),
                'elapsed_s': round(e['t'] / 1000, 1),
                'hp_start': round(hp.hp, 1),
                'max_hp': hp.max_hp,
                'hp_frac': round(hp.hp / hp.max_hp, 3),
                'regen_per_s': round(hp.regen, 2),
                'prev_guards': prev_guards,
                'prev_damage_taken': prev_wave.get('damageTaken', 0),
                'prev_duration_s': round(prev_wave.get('durationMs', 0) / 1000, 1),
                'prev_caravans_escaped': prev_wave.get('caravansEscaped', 0),
                'prev_dashes': totals['dashes_wave'],
                'guards_killed': totals['guards_killed'],
                'caravans_robbed': totals['caravans_robbed'],
                'gold_earned': totals['gold_earned'],
                'gold_spent': totals['gold_spent'],
                'rerolls': totals['rerolls'],
                'flawless_waves': totals['flawless'],
                'relics': totals['relics'],
                'cards_total': sum(cards.values()),
                'cards_rare_plus': totals['rare_plus'],
                **{f'card_{c}': cards[c] for c in CARDS},
                **{f'pickup_{p}': pickups[p] for p in PICKUPS},
            }
            totals['dashes_wave'] = 0
            prev_guards = current['guards']
            hp.in_wave, hp.in_shop = True, False
        elif kind == 'player_damaged':
            hp.hp = e.get('hpAfter', hp.hp)
        elif kind == 'wave_end':
            prev_wave = e
            hp.in_wave = False
        elif kind == 'shop_opened':
            hp.in_shop = True
        elif kind == 'shop_closed':
            hp.in_shop = False
        elif kind == 'dash':
            totals['dashes_wave'] += 1
        elif kind == 'guard_killed':
            totals['guards_killed'] += 1
            hp.kill(e.get('guardType'), e.get('wave', 0))
        elif kind == 'caravan_robbed':
            totals['caravans_robbed'] += 1
            totals['gold_earned'] += e.get('value', 0)
        elif kind == 'card_picked':
            hp.card(e['id'], cards[e['id']])
            cards[e['id']] += 1
            totals['gold_spent'] += e.get('cost', 0)
            if e.get('rarity') in ('rare', 'legendary'):
                totals['rare_plus'] += 1
        elif kind == 'reroll':
            totals['rerolls'] += 1
            totals['gold_spent'] += e.get('cost', 0)
        elif kind == 'flawless':
            totals['flawless'] += 1
        elif kind == 'relic_unlocked':
            totals['relics'] += 1
        elif kind == 'pickup_collected':
            pickups[e['id']] += 1
            if e['id'] == 'medkit':
                hp.heal(MEDKIT_HEAL)
        elif kind == 'rebirth':
            # Rebirth resets the run: stats start over, history of the player stays.
            if current is not None:
                close(current, False)
                current = None
            hp.reset()
            hp.t = e['t']
            cards, pickups, totals, prev_wave, prev_guards = Counter(), Counter(), Counter(), {}, 0
        elif kind == 'player_died' and current is not None:
            close(current, True)
            current = None

    if current is not None:
        close(current, False)
    return rows


def build(sessions_dir):
    history = defaultdict(lambda: {'sessions': 0, 'best_wave': 0})
    rows = []
    for session in load_sessions(sessions_dir):
        player = history[session['meta']['playerId']]
        session_waves = session_rows(session, player['sessions'], player['best_wave'])
        rows.extend(session_waves)
        player['sessions'] += 1
        player['best_wave'] = max([player['best_wave']] + [r['wave'] for r in session_waves])
    return pd.DataFrame(rows)


if __name__ == '__main__':
    sessions_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'data/waves.csv'
    df = build(sessions_dir)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    df.to_csv(output, index=False)
    print(f'{len(df)} waves, {df.session_id.nunique()} sessions, '
          f'{df.player_id.nunique()} players, died rate {df.died.mean():.3f} -> {output}')
