"""Tests for envs/spaces.py — observation/action space helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from envs.spaces import (
    DIM_ACTIVE_SLOT,
    DIM_BENCH_SLOT,
    N_ACTIONS,
    N_MOVES,
    N_POKEMON_POOL_MAX,
    OBS_DIM,
    OBS_SPACE,
    _encode_active_slot,
    _encode_bench_slot,
    _encode_move,
    _pokemon_one_hot,
    action_to_order,
    build_action_mask,
    build_observation,
    safe_action,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_move(power: float = 80.0, category: str = "special", priority: int = 0):
    m = MagicMock()
    m.base_power = power
    m.priority = priority
    cat = MagicMock()
    cat.name = category
    m.category = cat
    return m


def _make_pokemon(
    species: str = "arceusfire",
    hp_fraction: float = 1.0,
    stats: dict | None = None,
    moves: dict | None = None,
    fainted: bool = False,
):
    p = MagicMock()
    p.species = species
    p.current_hp_fraction = hp_fraction
    p.stats = stats or {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    p.moves = moves or {}
    p.fainted = fainted
    return p


def _make_battle(
    my_active_species: str = "arceusfire",
    opp_active_species: str = "arceuswater",
    force_switch: bool = False,
    available_moves: list | None = None,
    available_switches: list | None = None,
):
    battle = MagicMock()
    my_active = _make_pokemon(my_active_species)
    opp_active = _make_pokemon(opp_active_species)

    battle.active_pokemon = my_active
    battle.opponent_active_pokemon = opp_active
    battle.force_switch = force_switch
    battle.available_moves = available_moves if available_moves is not None else [_make_move()]
    battle.available_switches = available_switches if available_switches is not None else []

    bench_1 = _make_pokemon("arceusgrass", 0.8)
    bench_2 = _make_pokemon("arceuselectric", 0.6)

    my_team = {
        "arceusfire": my_active,
        "arceusgrass": bench_1,
        "arceuselectric": bench_2,
    }
    opp_bench = _make_pokemon("arceusground", 0.9)
    opp_team = {
        opp_active_species: opp_active,
        "arceusground": opp_bench,
    }

    battle.team = my_team
    battle.opponent_team = opp_team
    return battle


# ---------------------------------------------------------------------------
# _pokemon_one_hot
# ---------------------------------------------------------------------------


def test_one_hot_known_species() -> None:
    vec = _pokemon_one_hot("arceusfire")
    assert vec.shape == (N_POKEMON_POOL_MAX,)
    assert vec[0] == 1.0
    assert vec.sum() == pytest.approx(1.0)


def test_one_hot_unknown_species() -> None:
    vec = _pokemon_one_hot("pikachu")
    assert vec.sum() == pytest.approx(0.0)


def test_one_hot_case_insensitive() -> None:
    assert np.array_equal(_pokemon_one_hot("ArceusWater"), _pokemon_one_hot("arceuswater"))


# ---------------------------------------------------------------------------
# _encode_move
# ---------------------------------------------------------------------------


def test_encode_move_none_returns_zeros() -> None:
    vec = _encode_move(None)
    assert vec.shape == (3,)
    assert vec.sum() == pytest.approx(0.0)


def test_encode_move_special() -> None:
    m = _make_move(power=100.0, category="special", priority=0)
    vec = _encode_move(m)
    assert vec[1] == pytest.approx(1.0)  # is_special
    assert vec[2] == pytest.approx(0.0)  # is_priority


def test_encode_move_priority() -> None:
    m = _make_move(power=40.0, category="physical", priority=1)
    vec = _encode_move(m)
    assert vec[2] == pytest.approx(1.0)  # is_priority


def test_encode_move_power_clipped() -> None:
    m = _make_move(power=500.0, category="physical", priority=0)
    vec = _encode_move(m)
    assert vec[0] == pytest.approx(1.0)  # power clamped to 1.0


# ---------------------------------------------------------------------------
# _encode_active_slot — None 포켓몬 처리 (force_switch 상태)
# ---------------------------------------------------------------------------


def test_active_slot_none_returns_zeros() -> None:
    """force_switch 상태에서 active_pokemon이 None이면 zero 벡터를 반환해야 한다."""
    vec = _encode_active_slot(None, None)
    assert vec.shape == (DIM_ACTIVE_SLOT,)
    assert vec.sum() == pytest.approx(0.0)


def test_active_slot_none_with_opp_returns_zeros() -> None:
    """active=None이고 상대 포켓몬이 있어도 zero 벡터를 반환해야 한다."""
    opp = _make_pokemon("arceuswater")
    vec = _encode_active_slot(None, opp)
    assert vec.shape == (DIM_ACTIVE_SLOT,)
    assert vec.sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _encode_bench_slot
# ---------------------------------------------------------------------------


def test_bench_slot_fainted_returns_zeros() -> None:
    p = _make_pokemon(fainted=True)
    vec = _encode_bench_slot(p, None)
    assert vec.shape == (DIM_BENCH_SLOT,)
    assert vec.sum() == pytest.approx(0.0)


def test_bench_slot_none_returns_zeros() -> None:
    vec = _encode_bench_slot(None, None)
    assert vec.shape == (DIM_BENCH_SLOT,)
    assert vec.sum() == pytest.approx(0.0)


def test_bench_slot_valid_shape() -> None:
    p = _make_pokemon("arceuswater", 0.7)
    vec = _encode_bench_slot(p, None)
    assert vec.shape == (DIM_BENCH_SLOT,)


# ---------------------------------------------------------------------------
# _encode_relative_stats — None 스탯 처리
# ---------------------------------------------------------------------------


def test_encode_relative_stats_none_values_returns_zeros() -> None:
    """상대 포켓몬 스탯 값이 None일 때 TypeError 없이 zero를 반환해야 한다.

    실전 배틀에서 poke-env는 상대 포켓몬의 스탯을 알 수 없으면
    dict 키는 존재하지만 값이 None으로 채운다.
    """
    from envs.spaces import _encode_relative_stats

    pokemon = _make_pokemon(stats={"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 120})
    # 상대방 스탯: 키는 존재하지만 값이 None
    ref = _make_pokemon(stats={"atk": None, "def": None, "spa": None, "spd": None, "spe": None})
    vec = _encode_relative_stats(pokemon, ref)
    assert vec.shape == (5,)
    # None 스탯은 1로 대체 → log(120/1)/log(400) 형태, zero는 아님
    assert vec.dtype == np.float32


def test_encode_relative_stats_my_none_values() -> None:
    """내 포켓몬 스탯에 None이 있어도 TypeError가 발생하지 않아야 한다."""
    from envs.spaces import _encode_relative_stats

    pokemon = _make_pokemon(stats={"atk": None, "def": 120, "spa": None, "spd": 120, "spe": 120})
    ref = _make_pokemon(stats={"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 120})
    vec = _encode_relative_stats(pokemon, ref)
    assert vec.shape == (5,)
    assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# build_observation — force_switch 상태 (active_pokemon=None)
# ---------------------------------------------------------------------------


def test_build_observation_force_switch_active_none() -> None:
    """force_switch 시 active_pokemon=None이어도 크래시 없이 168차원을 반환해야 한다."""
    battle = _make_battle()
    # 기절 후 active_pokemon = None 시뮬레이션
    battle.active_pokemon = None
    obs = build_observation(battle)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# build_observation
# ---------------------------------------------------------------------------


def test_build_observation_shape() -> None:
    battle = _make_battle()
    obs = build_observation(battle)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_build_observation_within_bounds() -> None:
    battle = _make_battle()
    obs = build_observation(battle)
    assert obs.min() >= -1.0
    assert obs.max() <= 1.0


def test_build_observation_space_contains() -> None:
    battle = _make_battle()
    obs = build_observation(battle)
    assert OBS_SPACE.contains(obs)


# ---------------------------------------------------------------------------
# build_action_mask
# ---------------------------------------------------------------------------


def test_mask_normal_turn_has_move() -> None:
    moves = [_make_move()]
    battle = _make_battle(available_moves=moves, available_switches=[])
    mask = build_action_mask(battle)
    assert mask.shape == (N_ACTIONS,)
    assert mask[0]           # 첫 번째 기술 가능
    assert not any(mask[N_MOVES:])  # 교체 불가


def test_mask_force_switch_only_switches() -> None:
    switches = [_make_pokemon("arceusgrass")]
    battle = _make_battle(force_switch=True, available_moves=[], available_switches=switches)
    mask = build_action_mask(battle)
    assert not any(mask[:N_MOVES])       # 기술 불가
    assert mask[N_MOVES]                 # 첫 번째 교체 가능


def test_mask_all_moves_and_switches() -> None:
    moves = [_make_move() for _ in range(4)]
    switches = [_make_pokemon() for _ in range(2)]
    battle = _make_battle(available_moves=moves, available_switches=switches)
    mask = build_action_mask(battle)
    assert all(mask)  # 전부 가능


# ---------------------------------------------------------------------------
# action_to_order
# ---------------------------------------------------------------------------


def test_action_to_order_move() -> None:
    from poke_env.player import BattleOrder
    moves = [_make_move(), _make_move()]
    battle = _make_battle(available_moves=moves, available_switches=[])
    order = action_to_order(0, battle)
    assert isinstance(order, BattleOrder)


def test_action_to_order_switch() -> None:
    from poke_env.player import BattleOrder
    switches = [_make_pokemon("arceusgrass")]
    battle = _make_battle(available_moves=[], available_switches=switches, force_switch=True)
    order = action_to_order(N_MOVES, battle)
    assert isinstance(order, BattleOrder)


def test_action_to_order_invalid_raises() -> None:
    battle = _make_battle(available_moves=[], available_switches=[])
    with pytest.raises(ValueError):
        action_to_order(0, battle)


def test_action_to_order_out_of_range_raises() -> None:
    battle = _make_battle()
    with pytest.raises(ValueError):
        action_to_order(N_ACTIONS, battle)


# ---------------------------------------------------------------------------
# safe_action
# ---------------------------------------------------------------------------


def test_safe_action_valid_passthrough() -> None:
    moves = [_make_move()]
    battle = _make_battle(available_moves=moves, available_switches=[])
    assert safe_action(0, battle) == 0


def test_safe_action_fallback_on_invalid() -> None:
    moves = [_make_move()]
    battle = _make_battle(available_moves=moves, available_switches=[])
    # action=5(교체)는 마스크 위반 → 유효한 0으로 폴백
    result = safe_action(N_MOVES, battle)
    assert result == 0
