"""Send two example waves to the running service: a calm one and a risky one.

Usage: python test_predict.py [url]
"""
import sys

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696/predict'

calm = {
    'game_version': '1.7', 'prior_sessions': 5, 'prior_best_wave': 12,
    'wave': 3, 'guards': 4, 'hp_start': 100, 'max_hp': 100,
    'prev_guards': 3, 'prev_damage_taken': 5, 'prev_duration_s': 25,
    'cards': {'maxHp': 1, 'damage': 1},
}
risky = {
    'game_version': '1.7', 'prior_sessions': 0, 'prior_best_wave': 0,
    'wave': 8, 'guards': 16, 'hp_start': 30, 'max_hp': 100,
    'prev_guards': 13, 'prev_damage_taken': 80, 'prev_duration_s': 45,
    'cards': {'damage': 2, 'glassCannon': 1},
}

for name, wave in [('calm', calm), ('risky', risky)]:
    response = requests.post(URL, json=wave, timeout=10)
    response.raise_for_status()
    print(name, response.json())
