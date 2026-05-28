import random

from pokemon_pool.pokemon_list import POKEMON_LIST
from pokemon_pool.movesets import POKEMON_MOVES


def generate_random_team():
    selected = random.sample(POKEMON_LIST, 3)
    team = []

    for pokemon in selected:
        if pokemon not in POKEMON_MOVES:
            raise ValueError(f"{pokemon} has no moves defined")

        moves = POKEMON_MOVES[pokemon]["moves"]
        ability = POKEMON_MOVES[pokemon]["ability"]

        team_text = f"""
{pokemon}
Ability: {ability}
Level: 50
- {moves[0]}
- {moves[1]}
- {moves[2]}
- {moves[3]}
"""
        team.append(team_text)

    return "\n".join(team)
