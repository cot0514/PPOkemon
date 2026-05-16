"""State (observation) space definition for the PPOkemon environment.

State vector layout (total 70 dims):
  [0:15]   My active slot   : hp(1) + pokemon_id(9) + stat_boosts(5)
  [15:25]  My bench slot #1 : hp(1) + pokemon_id(9)  — hp=0 implies fainted
  [25:35]  My bench slot #2 : hp(1) + pokemon_id(9)
  [35:50]  Opponent active slot
  [50:60]  Opponent bench slot #1
  [60:70]  Opponent bench slot #2

Pokemon identity is encoded as a 9-dim one-hot using the restricted pool index.
Type info is NOT encoded here; it is carried by the pokemon_pool definitions.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import AbstractBattle, Pokemon

# pokemon_pool.pokemon_list 가 생성되면 아래 import 로 교체할 것:
# from pokemon_pool.pokemon_list import POKEMON_TO_IDX, N_POKEMON_POOL
POKEMON_TO_IDX: dict[str, int] = {}   # 풀 파일 생성 후 채워짐
N_POKEMON_POOL: int = 9

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

STAT_KEYS: list[str] = ["atk", "def", "spa", "spd", "spe"]
N_STATS: int = len(STAT_KEYS)   # 5
MAX_BOOST: float = 6.0

# Per active slot: hp(1) + pokemon_id(9) + boosts(5) = 15
DIM_ACTIVE_SLOT: int = 1 + N_POKEMON_POOL + N_STATS             # 15
# Per bench slot: hp(1) + pokemon_id(9) = 10
DIM_BENCH_SLOT: int = 1 + N_POKEMON_POOL                        # 10
# Per team: active(15) + 2 bench(20) = 35
DIM_TEAM: int = DIM_ACTIVE_SLOT + 2 * DIM_BENCH_SLOT            # 35
# Total: my team + opponent team = 70
OBS_DIM: int = DIM_TEAM * 2                                      # 70

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _pokemon_one_hot(species: str) -> np.ndarray:
    vec = np.zeros(N_POKEMON_POOL, dtype=np.float32)
    idx = POKEMON_TO_IDX.get(species.lower())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _encode_active_slot(pokemon: Pokemon) -> np.ndarray:
    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
        np.array([pokemon.boosts.get(k, 0) / MAX_BOOST for k in STAT_KEYS], dtype=np.float32),
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_ACTIVE_SLOT,), f"Expected {DIM_ACTIVE_SLOT}, got {encoded.shape}"
    return encoded


def _encode_bench_slot(pokemon: Pokemon | None) -> np.ndarray:
    if pokemon is None or pokemon.fainted:
        return np.zeros(DIM_BENCH_SLOT, dtype=np.float32)

    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_BENCH_SLOT,), f"Expected {DIM_BENCH_SLOT}, got {encoded.shape}"
    return encoded


def _get_bench(team: dict, active: Pokemon) -> tuple[Pokemon | None, Pokemon | None]:
    bench = [p for p in team.values() if p is not active]
    return (bench[0] if len(bench) > 0 else None, bench[1] if len(bench) > 1 else None)


# ---------------------------------------------------------------------------
# Main observation builder
# ---------------------------------------------------------------------------


def build_observation(battle: AbstractBattle) -> np.ndarray:
    """Build a flat float32 observation vector from the current battle state.

    Returns:
        np.ndarray of shape (OBS_DIM,) = (82,), values in [-1.0, 1.0].
    """
    my_active = battle.active_pokemon
    my_bench_0, my_bench_1 = _get_bench(battle.team, my_active)

    my_obs = np.concatenate([
        _encode_active_slot(my_active),
        _encode_bench_slot(my_bench_0),
        _encode_bench_slot(my_bench_1),
    ])

    opp_active = battle.opponent_active_pokemon
    if opp_active is not None:
        opp_active_enc = _encode_active_slot(opp_active)
    else:
        opp_active_enc = np.zeros(DIM_ACTIVE_SLOT, dtype=np.float32)

    opp_bench_0, opp_bench_1 = _get_bench(battle.opponent_team, opp_active)

    opp_obs = np.concatenate([
        opp_active_enc,
        _encode_bench_slot(opp_bench_0),
        _encode_bench_slot(opp_bench_1),
    ])

    obs = np.concatenate([my_obs, opp_obs]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"
    return obs
