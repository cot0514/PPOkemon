"""Tests for agents/opponents.py — RandomPoolOpponent, MaxDamageOpponent."""

from __future__ import annotations

from unittest.mock import MagicMock


from agents.opponents import MaxDamageOpponent, RandomPoolOpponent


def _make_move(power: float = 80.0, name: str = "tackle") -> MagicMock:
    m = MagicMock()
    m.base_power = power
    m.id = name
    return m


def _make_battle(
    available_moves: list | None = None,
    available_switches: list | None = None,
    force_switch: bool = False,
) -> MagicMock:
    b = MagicMock()
    b.available_moves = available_moves if available_moves is not None else [_make_move()]
    b.available_switches = available_switches if available_switches is not None else []
    b.force_switch = force_switch
    return b


# ---------------------------------------------------------------------------
# RandomPoolOpponent — 교체 루프 방지 확인
# ---------------------------------------------------------------------------


def test_random_opponent_uses_move_when_available() -> None:
    """일반 턴에는 교체 없이 기술을 선택해야 한다."""
    opp = RandomPoolOpponent.__new__(RandomPoolOpponent)
    opp.create_order = MagicMock(side_effect=lambda x: x)

    move = _make_move()
    battle = _make_battle(available_moves=[move], available_switches=[MagicMock()])

    result = opp.choose_move(battle)
    assert result is move


def test_random_opponent_force_switch_picks_first_switch() -> None:
    """강제 교체 상황에서는 available_switches[0]을 선택해야 한다."""
    opp = RandomPoolOpponent.__new__(RandomPoolOpponent)
    switch_target = MagicMock()
    opp.create_order = MagicMock(side_effect=lambda x: x)

    battle = _make_battle(available_moves=[], available_switches=[switch_target], force_switch=True)

    result = opp.choose_move(battle)
    opp.create_order.assert_called_once_with(switch_target)
    assert result is switch_target


def test_random_opponent_no_moves_uses_default_order() -> None:
    """사용 가능한 기술이 없으면 DefaultBattleOrder를 반환해야 한다 (교체 방지)."""
    from poke_env.player.battle_order import DefaultBattleOrder

    opp = RandomPoolOpponent.__new__(RandomPoolOpponent)
    battle = _make_battle(available_moves=[], available_switches=[MagicMock()])

    result = opp.choose_move(battle)
    assert isinstance(result, DefaultBattleOrder)


# ---------------------------------------------------------------------------
# MaxDamageOpponent — 최고 위력 기술 선택
# ---------------------------------------------------------------------------


def test_max_damage_picks_highest_power() -> None:
    """사용 가능한 기술 중 base_power가 가장 높은 기술을 선택해야 한다."""
    opp = MaxDamageOpponent.__new__(MaxDamageOpponent)
    opp.create_order = MagicMock(side_effect=lambda x: x)

    weak = _make_move(40.0, "quick-attack")
    strong = _make_move(120.0, "hyper-beam")
    battle = _make_battle(available_moves=[weak, strong])

    result = opp.choose_move(battle)
    assert result is strong


def test_max_damage_force_switch_picks_first_switch() -> None:
    """강제 교체 상황에서는 available_switches[0]을 선택해야 한다."""
    opp = MaxDamageOpponent.__new__(MaxDamageOpponent)
    switch_target = MagicMock()
    opp.create_order = MagicMock(side_effect=lambda x: x)

    battle = _make_battle(available_moves=[], available_switches=[switch_target], force_switch=True)

    result = opp.choose_move(battle)
    opp.create_order.assert_called_once_with(switch_target)
    assert result is switch_target


def test_max_damage_no_moves_uses_default_order() -> None:
    """사용 가능한 기술이 없으면 DefaultBattleOrder를 반환해야 한다."""
    from poke_env.player.battle_order import DefaultBattleOrder

    opp = MaxDamageOpponent.__new__(MaxDamageOpponent)
    battle = _make_battle(available_moves=[])

    result = opp.choose_move(battle)
    assert isinstance(result, DefaultBattleOrder)
