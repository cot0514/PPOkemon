"""Observation and action space definitions for the PPOkemon environment.

━━━ State vector layout (total 168 dims) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Zero-padding 설계: Phase 1부터 최종 크기(168)로 고정.
  아직 사용하지 않는 슬롯은 0.0으로 채워 Phase 간 weight 연속성을 보장한다.

  슬롯 구조:
    [0   : 36]  My active      : hp(1) + id(18) + stats(5) + moves(4×3)
    [36  : 60]  My bench #1    : hp(1) + id(18) + stats(5)
    [60  : 84]  My bench #2    : hp(1) + id(18) + stats(5)
    [84  :120]  Opp active     : 동일 구조 (미공개 기술 슬롯은 0)
    [120 :144]  Opp bench #1   : 동일 구조
    [144 :168]  Opp bench #2   : 동일 구조

  스탯 인코딩 (log-relative):
    내 팀: log(내 스탯 / 상대 액티브 스탯) / log(MAX_STAT) → [-1, 1]
    상대 팀: log(상대 스탯 / 내 액티브 스탯) / log(MAX_STAT) → [-1, 1]

  Phase별 채우는 슬롯:
    Phase 1 (아르세우스 9종):  hp + id[0:9]
    Phase 2 (아르세우스 18종): + stats + moves

━━━ Action index layout (total 6) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [0–3]  기술 슬롯 0–3 사용
  [4–5]  벤치 슬롯 0–1로 교체
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.environment import AbstractBattle, Move, Pokemon
from poke_env.player import BattleOrder

# ---------------------------------------------------------------------------
# 포켓몬 풀 레지스트리 — pokemon_pool.pokemon_list 생성 후 교체
# ---------------------------------------------------------------------------
# from pokemon_pool.pokemon_list import POKEMON_TO_IDX
POKEMON_TO_IDX: dict[str, int] = {}
N_POKEMON_POOL_MAX: int = 18  # Phase 1: 9, Phase 2: 18

# =============================================================================
# 실수치(스탯) 상수
# =============================================================================

BATTLE_STAT_KEYS: list[str] = ["atk", "def", "spa", "spd", "spe"]
N_BATTLE_STATS: int = len(BATTLE_STAT_KEYS)  # 5
MAX_STAT: float = 400.0  # 정규화 상한 (아르세우스·제한 환경 기준)

# =============================================================================
# 기술 인코딩 상수
# =============================================================================

N_MOVES: int = 4
# power_norm(1) + is_special(1) + is_priority(1) = 3
# is_priority: Phase 1(선공기 포함)에서만 1.0, Phase 2/3(선공기 미사용)은 자연스럽게 0.0
N_MOVE_FEATURES: int = 3
MAX_MOVE_POWER: float = 250.0  # 정규화 상한 (폭발 = 250)

# =============================================================================
# 슬롯 차원 (변경 시 OBS_DIM 자동 반영)
# =============================================================================

# hp(1) + id(18) + stats(5) + moves(4×3=12) = 36
DIM_ACTIVE_SLOT: int = (
    1
    + N_POKEMON_POOL_MAX
    + N_BATTLE_STATS
    + N_MOVES * N_MOVE_FEATURES
)

# hp(1) + id(18) + stats(5) = 24
# 벤치 슬롯: 기술 미포함 (교체 후에야 의사결정에 사용됨)
# 벤치 슬롯: 랭크업 미포함 (교체 시 초기화되는 게임 메커니즘)
DIM_BENCH_SLOT: int = (
    1
    + N_POKEMON_POOL_MAX
    + N_BATTLE_STATS
)

DIM_TEAM: int = DIM_ACTIVE_SLOT + 2 * DIM_BENCH_SLOT   # 36 + 48 = 84
OBS_DIM: int = DIM_TEAM * 2                             # 168

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

# =============================================================================
# 행동 공간
# =============================================================================

N_SWITCHES: int = 2   # 3v3: 벤치 2슬롯
N_ACTIONS: int = N_MOVES + N_SWITCHES  # 6

ACT_SPACE = Discrete(N_ACTIONS)

# =============================================================================
# 인코딩 헬퍼
# =============================================================================


def _pokemon_one_hot(species: str) -> np.ndarray:
    """종 이름 → N_POKEMON_POOL_MAX 차원 one-hot. 미등록 종은 zero."""
    vec = np.zeros(N_POKEMON_POOL_MAX, dtype=np.float32)
    idx = POKEMON_TO_IDX.get(species.lower())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _encode_relative_stats(
    pokemon: Pokemon, reference: Pokemon | None
) -> np.ndarray:
    """5-dim log-relative stats: log(pokemon_stat / reference_stat) / log(MAX_STAT).

    Positive → pokemon이 강함; Negative → reference가 강함.
    reference가 None이거나 stats 미확인 시 zero 반환.
    """
    if reference is None:
        return np.zeros(N_BATTLE_STATS, dtype=np.float32)
    stats_a = pokemon.stats
    stats_b = reference.stats
    if not stats_a or not stats_b:
        return np.zeros(N_BATTLE_STATS, dtype=np.float32)

    vals = [
        np.log(max(stats_a.get(k, 1), 1) / max(stats_b.get(k, 1), 1))
        / np.log(MAX_STAT)
        for k in BATTLE_STAT_KEYS
    ]
    return np.clip(np.array(vals, dtype=np.float32), -1.0, 1.0)


def _encode_move(move: Move | None) -> np.ndarray:
    """3-dim 기술 인코딩: power_norm(1) + is_special(1) + is_priority(1).

    미공개·없는 기술 슬롯은 zero 벡터.
    is_priority는 priority > 0인 기술(선공기)에서만 1.0.
    Phase 2/3는 기술 구성에 선공기가 없으므로 is_priority가 자연스럽게 0.0.
    """
    if move is None:
        return np.zeros(N_MOVE_FEATURES, dtype=np.float32)

    power = float(np.clip(move.base_power / MAX_MOVE_POWER, 0.0, 1.0))

    try:
        is_special = 1.0 if "special" in move.category.name.lower() else 0.0
    except AttributeError:
        is_special = 0.0

    is_priority = 1.0 if move.priority > 0 else 0.0

    return np.array([power, is_special, is_priority], dtype=np.float32)


def _encode_active_slot(
    pokemon: Pokemon, reference: Pokemon | None
) -> np.ndarray:
    """Active slot 36-dim: hp + id + relative_stats + moves(4×3).

    relative_stats는 reference(상대 액티브) 대비 log 스케일로 인코딩된다.
    """
    move_list: list[Move | None] = list(pokemon.moves.values())[:N_MOVES]
    while len(move_list) < N_MOVES:
        move_list.append(None)

    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
        _encode_relative_stats(pokemon, reference),
        *[_encode_move(m) for m in move_list],
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_ACTIVE_SLOT,), (
        f"Active slot: expected {DIM_ACTIVE_SLOT}, got {encoded.shape}"
    )
    return encoded


def _encode_bench_slot(
    pokemon: Pokemon | None, reference: Pokemon | None
) -> np.ndarray:
    """Bench slot 24-dim: hp + id + relative_stats. 기절·부재 시 zero.

    relative_stats는 reference(상대 액티브) 대비 log 스케일로 인코딩된다.
    """
    if pokemon is None or pokemon.fainted:
        return np.zeros(DIM_BENCH_SLOT, dtype=np.float32)

    parts: list[np.ndarray] = [
        np.array([pokemon.current_hp_fraction], dtype=np.float32),
        _pokemon_one_hot(pokemon.species),
        _encode_relative_stats(pokemon, reference),
    ]
    encoded = np.concatenate(parts)
    assert encoded.shape == (DIM_BENCH_SLOT,), (
        f"Bench slot: expected {DIM_BENCH_SLOT}, got {encoded.shape}"
    )
    return encoded


def _get_bench(
    team: dict, active: Pokemon
) -> tuple[Pokemon | None, Pokemon | None]:
    """팀에서 액티브가 아닌 포켓몬 2마리를 순서대로 반환한다."""
    bench = [p for p in team.values() if p is not active]
    return (
        bench[0] if len(bench) > 0 else None,
        bench[1] if len(bench) > 1 else None,
    )


# =============================================================================
# 관측 빌더
# =============================================================================


def build_observation(battle: AbstractBattle) -> np.ndarray:
    """현재 배틀 상태로부터 168차원 float32 관측 벡터를 반환한다.

    스탯은 상대 액티브(내 팀) 또는 내 액티브(상대 팀) 대비 log 비율로 인코딩된다.

    Returns:
        np.ndarray of shape (OBS_DIM,) = (168,), values in [-1.0, 1.0].
    """
    my_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    my_bench_0, my_bench_1 = _get_bench(battle.team, my_active)
    opp_bench_0, opp_bench_1 = _get_bench(battle.opponent_team, opp_active)

    # 내 팀: 모두 상대 액티브 대비 log 스케일
    my_obs = np.concatenate([
        _encode_active_slot(my_active, opp_active),
        _encode_bench_slot(my_bench_0, opp_active),
        _encode_bench_slot(my_bench_1, opp_active),
    ])

    # 상대 팀: 모두 내 액티브 대비 log 스케일
    opp_active_enc = (
        _encode_active_slot(opp_active, my_active)
        if opp_active is not None
        else np.zeros(DIM_ACTIVE_SLOT, dtype=np.float32)
    )
    opp_obs = np.concatenate([
        opp_active_enc,
        _encode_bench_slot(opp_bench_0, my_active),
        _encode_bench_slot(opp_bench_1, my_active),
    ])

    obs = np.concatenate([my_obs, opp_obs]).astype(np.float32)
    assert obs.shape == (OBS_DIM,), f"Expected {OBS_DIM}, got {obs.shape}"
    return obs


# =============================================================================
# 행동 마스크 빌더
# =============================================================================


def build_action_mask(battle: AbstractBattle) -> np.ndarray:
    """유효한 행동 마스크를 반환한다.

    shape: (N_ACTIONS,) = (6,), dtype=bool
    True  → 선택 가능한 합법 action.
    False → 불법 action (PPO softmax 직전에 -inf 처리).

    - 강제 교체 턴 (force_switch=True): 교체만 허용.
    - 일반 턴: 기술과 교체 모두 상황에 따라 허용.
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    if battle.force_switch:
        for i in range(min(len(battle.available_switches), N_SWITCHES)):
            mask[N_MOVES + i] = True
        return mask

    for i in range(min(len(battle.available_moves), N_MOVES)):
        mask[i] = True
    for i in range(min(len(battle.available_switches), N_SWITCHES)):
        mask[N_MOVES + i] = True

    return mask


# =============================================================================
# Action → BattleOrder 변환
# =============================================================================


def action_to_order(action: int, battle: AbstractBattle) -> BattleOrder:
    """이산 action 인덱스를 poke-env의 BattleOrder로 변환한다.

    Args:
        action: [0, N_ACTIONS) 범위의 정수. 0–3 → 기술, 4–5 → 교체.
        battle: 현재 턴의 AbstractBattle 인스턴스.

    Raises:
        ValueError: masking 없이 불법 action이 선택되었을 때.
    """
    if not (0 <= action < N_ACTIONS):
        raise ValueError(f"action={action} is out of range [0, {N_ACTIONS}).")

    if action < N_MOVES:
        move_idx = action
        if move_idx >= len(battle.available_moves):
            raise ValueError(
                f"Move slot {move_idx} unavailable "
                f"(available_moves has {len(battle.available_moves)} entries)."
            )
        return BattleOrder(battle.available_moves[move_idx])

    switch_idx = action - N_MOVES
    if switch_idx >= len(battle.available_switches):
        raise ValueError(
            f"Switch slot {switch_idx} unavailable "
            f"(available_switches has {len(battle.available_switches)} entries)."
        )
    return BattleOrder(battle.available_switches[switch_idx])


# =============================================================================
# 폴백 안전 선택기 (디버깅·테스트 전용)
# =============================================================================


def safe_action(action: int, battle: AbstractBattle) -> int:
    """마스크 위반 시 유효한 action으로 폴백한다.

    정상 학습에서 호출된다면 masking 로직에 버그가 있다는 신호다.
    """
    mask = build_action_mask(battle)
    if mask[action]:
        return action
    valid = np.where(mask)[0]
    if valid.size == 0:
        raise RuntimeError("No valid actions available — battle state is broken.")
    return int(valid[0])
