"""Observation and action space definitions for the PPOkemon environment.

━━━ State vector layout (total 70 dims) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [0 :15]  My active slot   : hp(1) + pokemon_id(9) + stat_boosts(5)
  [15:25]  My bench slot #1 : hp(1) + pokemon_id(9)  — hp=0 implies fainted
  [25:35]  My bench slot #2 : hp(1) + pokemon_id(9)
  [35:50]  Opponent active slot
  [50:60]  Opponent bench slot #1
  [60:70]  Opponent bench slot #2

━━━ Action index layout (total 6 dims) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [0]  move slot 0 — active Pokémon의 첫 번째 기술 사용
  [1]  move slot 1 — active Pokémon의 두 번째 기술 사용
  [2]  move slot 2 — active Pokémon의 세 번째 기술 사용
  [3]  move slot 3 — active Pokémon의 네 번째 기술 사용
  [4]  switch slot 0 — 벤치 포켓몬 #0으로 교체
  [5]  switch slot 1 — 벤치 포켓몬 #1으로 교체

Pokemon identity는 9마리 제한 풀 내 인덱스를 one-hot으로 인코딩한다.
타입 정보는 state에 포함하지 않으며, 에이전트가 보상 신호로 학습한다.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.environment import AbstractBattle, Pokemon
from poke_env.player import BattleOrder

# pokemon_pool.pokemon_list 가 생성되면 아래 import 로 교체할 것:
# from pokemon_pool.pokemon_list import POKEMON_TO_IDX, N_POKEMON_POOL
POKEMON_TO_IDX: dict[str, int] = {}   # 풀 파일 생성 후 채워짐
N_POKEMON_POOL: int = 9

# =============================================================================
# OBSERVATION SPACE
# =============================================================================

# ---------------------------------------------------------------------------
# Observation dimension constants
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
# Observation encoding helpers
# ---------------------------------------------------------------------------


def _pokemon_one_hot(species: str) -> np.ndarray:
    """종 이름을 9차원 one-hot 벡터로 변환한다. 미등록 종은 zero 벡터."""
    vec = np.zeros(N_POKEMON_POOL, dtype=np.float32)
    idx = POKEMON_TO_IDX.get(species.lower())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _encode_active_slot(pokemon: Pokemon) -> np.ndarray:
    """액티브 슬롯 15차원 벡터를 반환한다: [hp | id(9) | boosts(5)]."""
    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
        # 스탯 변화 ÷ MAX_BOOST → [-1, 1] 정규화
        np.array([pokemon.boosts.get(k, 0) / MAX_BOOST for k in STAT_KEYS], dtype=np.float32),
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_ACTIVE_SLOT,), f"Expected {DIM_ACTIVE_SLOT}, got {encoded.shape}"
    return encoded


def _encode_bench_slot(pokemon: Pokemon | None) -> np.ndarray:
    """벤치 슬롯 10차원 벡터를 반환한다: [hp | id(9)]. 기절/부재 시 zero 벡터."""
    if pokemon is None or pokemon.fainted:
        return np.zeros(DIM_BENCH_SLOT, dtype=np.float32)

    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_BENCH_SLOT,), f"Expected {DIM_BENCH_SLOT}, got {encoded.shape}"
    return encoded


def _get_bench(
    team: dict, active: Pokemon
) -> tuple[Pokemon | None, Pokemon | None]:
    """팀 딕셔너리에서 액티브가 아닌 포켓몬 2마리를 순서대로 반환한다."""
    bench = [p for p in team.values() if p is not active]
    return (bench[0] if len(bench) > 0 else None, bench[1] if len(bench) > 1 else None)


# ---------------------------------------------------------------------------
# Main observation builder
# ---------------------------------------------------------------------------


def build_observation(battle: AbstractBattle) -> np.ndarray:
    """현재 배틀 상태로부터 70차원 float32 관측 벡터를 반환한다.

    Returns:
        np.ndarray of shape (OBS_DIM,) = (70,), values in [-1.0, 1.0].
    """
    my_active = battle.active_pokemon
    my_bench_0, my_bench_1 = _get_bench(battle.team, my_active)

    my_obs = np.concatenate([
        _encode_active_slot(my_active),
        _encode_bench_slot(my_bench_0),
        _encode_bench_slot(my_bench_1),
    ])

    opp_active = battle.opponent_active_pokemon
    opp_active_enc = (
        _encode_active_slot(opp_active)
        if opp_active is not None
        else np.zeros(DIM_ACTIVE_SLOT, dtype=np.float32)
    )

    opp_bench_0, opp_bench_1 = _get_bench(battle.opponent_team, opp_active)

    opp_obs = np.concatenate([
        opp_active_enc,
        _encode_bench_slot(opp_bench_0),
        _encode_bench_slot(opp_bench_1),
    ])

    obs = np.concatenate([my_obs, opp_obs]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"
    return obs


# =============================================================================
# ACTION SPACE
# =============================================================================

# ---------------------------------------------------------------------------
# Action dimension constants
# ---------------------------------------------------------------------------

N_MOVES: int = 4
# 포켓몬 한 마리의 최대 기술 슬롯 수.
# 실제 기술이 4개 미만이거나 PP가 0이면 masking으로 차단.

N_SWITCHES: int = 2
# 3v3 배틀에서 벤치 크기: total 3 − active 1 = 2.
# 기절한 벤치 슬롯은 masking으로 차단.

N_ACTIONS: int = N_MOVES + N_SWITCHES   # 6

ACT_SPACE = Discrete(N_ACTIONS)
# Gymnasium Discrete(6): 에이전트가 출력하는 정수 행동 공간.

# ---------------------------------------------------------------------------
# Action mask builder
# ---------------------------------------------------------------------------


def build_action_mask(battle: AbstractBattle) -> np.ndarray:
    """현재 배틀 상태로부터 유효한 행동 마스크를 반환한다.

    shape: (N_ACTIONS,) = (6,), dtype=bool

    True  → 에이전트가 선택 가능한 합법 action.
    False → 불법 action (PPO softmax 직전에 -inf를 더해 확률 0으로 만든다).

    두 가지 턴 유형을 구분한다:
    - 강제 교체 턴 (force_switch=True): active 포켓몬 기절 직후. 교체만 허용.
    - 일반 턴: 기술과 교체 모두 상황에 따라 허용.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    if battle.force_switch:
        # active 포켓몬이 기절했을 때 발생하는 강제 교체 턴.
        # 기술 슬롯(0~3)은 전부 False를 유지한다.
        for i in range(min(len(battle.available_switches), N_SWITCHES)):
            mask[N_MOVES + i] = True
        return mask

    # 일반 턴: poke-env 가 PP=0·봉인 기술 등을 필터링한 뒤 available_moves 를 제공한다.
    for i in range(min(len(battle.available_moves), N_MOVES)):
        mask[i] = True

    # available_switches: 살아있는 벤치 포켓몬만 포함 (active 제외).
    for i in range(min(len(battle.available_switches), N_SWITCHES)):
        mask[N_MOVES + i] = True

    return mask


# ---------------------------------------------------------------------------
# Action → BattleOrder converter
# ---------------------------------------------------------------------------


def action_to_order(action: int, battle: AbstractBattle) -> BattleOrder:
    """이산 action 인덱스를 poke-env의 BattleOrder로 변환한다.

    PPO 에이전트가 결정한 정수 action을 받아 해당 기술 또는 포켓몬 객체를
    BattleOrder에 감싸 반환한다. poke-env는 이 객체를 서버 명령으로 변환한다.

    Args:
        action: [0, N_ACTIONS) 범위의 정수. 0~3 → 기술, 4~5 → 교체.
        battle: 현재 턴의 AbstractBattle 인스턴스.

    Returns:
        poke-env가 실행할 수 있는 BattleOrder 객체.

    Raises:
        ValueError: masking 없이 불법 action이 선택되었을 때.
    """
    if not (0 <= action < N_ACTIONS):
        raise ValueError(f"action={action} is out of range [0, {N_ACTIONS}).")

    if action < N_MOVES:
        # 기술 사용: BattleOrder(move) → 서버에 "/choose move <slot>" 전송.
        move_idx = action
        if move_idx >= len(battle.available_moves):
            raise ValueError(
                f"Move slot {move_idx} is unavailable "
                f"(available_moves has {len(battle.available_moves)} entries)."
            )
        return BattleOrder(battle.available_moves[move_idx])

    # 교체: BattleOrder(pokemon) → 서버에 "/choose switch <slot>" 전송.
    switch_idx = action - N_MOVES
    if switch_idx >= len(battle.available_switches):
        raise ValueError(
            f"Switch slot {switch_idx} is unavailable "
            f"(available_switches has {len(battle.available_switches)} entries)."
        )
    return BattleOrder(battle.available_switches[switch_idx])


# ---------------------------------------------------------------------------
# Fallback safe selector (디버깅·테스트 전용)
# ---------------------------------------------------------------------------


def safe_action(action: int, battle: AbstractBattle) -> int:
    """마스크 위반 시 유효한 action으로 폴백한다.

    정상 학습 루프에서는 masking을 올바르게 적용해야 하므로 이 함수가
    호출된다면 masking 로직에 버그가 있다는 신호다.
    """
    mask = build_action_mask(battle)
    if mask[action]:
        return action
    valid = np.where(mask)[0]
    if valid.size == 0:
        raise RuntimeError("No valid actions available — battle state is broken.")
    return int(valid[0])
