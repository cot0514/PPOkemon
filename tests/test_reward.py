"""Tests for envs/reward.py — Phase 1 reward shaping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from envs.reward import DEFAULT_CONFIG, RewardCalculator, RewardConfig


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _make_battle(
    my_hp: dict[str, float],
    opp_hp: dict[str, float],
    won: bool = False,
    lost: bool = False,
) -> MagicMock:
    """AbstractBattle 모의 객체를 생성한다."""
    battle = MagicMock()

    my_team: dict[str, MagicMock] = {}
    for name, hp in my_hp.items():
        p = MagicMock()
        p.current_hp_fraction = hp
        my_team[name] = p

    opp_team: dict[str, MagicMock] = {}
    for name, hp in opp_hp.items():
        p = MagicMock()
        p.current_hp_fraction = hp
        opp_team[name] = p

    battle.team = my_team
    battle.opponent_team = opp_team
    battle.won = won
    battle.lost = lost
    return battle


MY_FULL: dict[str, float] = {"a": 1.0, "b": 1.0, "c": 1.0}
OPP_FULL: dict[str, float] = {"x": 1.0, "y": 1.0, "z": 1.0}


# ---------------------------------------------------------------------------
# reset / 첫 호출
# ---------------------------------------------------------------------------


def test_first_compute_returns_zero() -> None:
    """첫 compute() 호출은 기준 스냅샷을 저장하고 0.0을 반환해야 한다."""
    calc = RewardCalculator()
    b = _make_battle(MY_FULL, OPP_FULL)
    assert calc.compute(b) == pytest.approx(0.0)


def test_reset_clears_snapshot() -> None:
    """reset() 이후 첫 compute()는 다시 0.0을 반환해야 한다."""
    calc = RewardCalculator()
    b1 = _make_battle(MY_FULL, OPP_FULL)
    calc.compute(b1)

    b2 = _make_battle({"a": 0.8, "b": 1.0, "c": 1.0}, {"x": 0.9, "y": 1.0, "z": 1.0})
    calc.compute(b2)

    calc.reset()
    b3 = _make_battle(MY_FULL, OPP_FULL)
    assert calc.compute(b3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 딜 보상
# ---------------------------------------------------------------------------


def test_damage_dealt_gives_positive_reward() -> None:
    """상대 HP가 감소하면 hp_delta_scale × Δ만큼 양의 보상이 발생해야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    r = calc.compute(_make_battle({"a": 1.0}, {"x": 0.5}))
    assert r == pytest.approx(cfg.hp_delta_scale * 0.5)


def test_damage_taken_gives_negative_reward() -> None:
    """내 HP가 감소하면 -hp_delta_scale × Δ만큼 음의 보상이 발생해야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    r = calc.compute(_make_battle({"a": 0.6}, {"x": 1.0}))
    assert r == pytest.approx(-cfg.hp_delta_scale * 0.4)


def test_mutual_damage_net_reward() -> None:
    """쌍방 딜 시 보상은 (opp_delta - my_delta) × scale 이어야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    r = calc.compute(_make_battle({"a": 0.7}, {"x": 0.6}))
    expected = cfg.hp_delta_scale * (0.4 - 0.3)
    assert r == pytest.approx(expected)


def test_reward_accumulates_across_turns() -> None:
    """여러 턴에 걸쳐 누적 HP 감소가 올바르게 보상에 반영되어야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    r1 = calc.compute(_make_battle({"a": 1.0}, {"x": 0.7}))
    r2 = calc.compute(_make_battle({"a": 1.0}, {"x": 0.4}))

    assert r1 == pytest.approx(cfg.hp_delta_scale * 0.3)
    assert r2 == pytest.approx(cfg.hp_delta_scale * 0.3)


# ---------------------------------------------------------------------------
# KO 보상 / 기절 패널티
# ---------------------------------------------------------------------------


def test_ko_gives_bonus() -> None:
    """상대 포켓몬이 기절하면 HP 보상에 KO 보너스가 추가되어야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 0.1}))

    r = calc.compute(_make_battle({"a": 1.0}, {"x": 0.0}))
    expected = cfg.hp_delta_scale * 0.1 + cfg.ko_reward * 1
    assert r == pytest.approx(expected)


def test_faint_gives_penalty() -> None:
    """내 포켓몬이 기절하면 HP 패널티에 기절 패널티가 추가되어야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 0.1}, {"x": 1.0}))

    r = calc.compute(_make_battle({"a": 0.0}, {"x": 1.0}))
    expected = -cfg.hp_delta_scale * 0.1 - cfg.faint_penalty * 1
    assert r == pytest.approx(expected)


def test_multiple_ko_and_faint_same_turn() -> None:
    """동시에 상대 KO + 내 기절이 발생할 때 각 항목이 독립적으로 합산되어야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 0.1, "b": 1.0}, {"x": 0.1, "y": 1.0}))

    r = calc.compute(_make_battle({"a": 0.0, "b": 1.0}, {"x": 0.0, "y": 1.0}))
    expected = (
        cfg.hp_delta_scale * 0.1    # opp HP 감소
        - cfg.hp_delta_scale * 0.1  # my HP 감소
        + cfg.ko_reward             # 상대 기절
        - cfg.faint_penalty         # 내 기절
    )
    assert r == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 교체 — switch_cost 없음: HP 변화만 반영
# ---------------------------------------------------------------------------


def test_switch_turn_no_attack_no_reward() -> None:
    """교체 턴에 HP 변화가 없으면 보상도 0이어야 한다 (switch_cost 없음)."""
    calc = RewardCalculator()

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    r = calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))
    assert r == pytest.approx(0.0)


def test_switch_turn_entry_damage_penalized() -> None:
    """교체 후 상대 공격을 맞으면 my_delta 패널티만 발생해야 한다."""
    calc = RewardCalculator()
    cfg = DEFAULT_CONFIG

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    # 교체 턴: 공격 없음(opp HP 그대로) + 입장 피해(내 HP 감소)
    r = calc.compute(_make_battle({"a": 0.7}, {"x": 1.0}))
    assert r == pytest.approx(-cfg.hp_delta_scale * 0.3)


# ---------------------------------------------------------------------------
# 에피소드 종료 보상
# ---------------------------------------------------------------------------


def test_terminal_win() -> None:
    calc = RewardCalculator()
    b = _make_battle({}, {}, won=True)
    assert calc.terminal(b) == pytest.approx(DEFAULT_CONFIG.win_reward)


def test_terminal_loss() -> None:
    calc = RewardCalculator()
    b = _make_battle({}, {}, lost=True)
    assert calc.terminal(b) == pytest.approx(-DEFAULT_CONFIG.lose_reward)


def test_terminal_draw() -> None:
    calc = RewardCalculator()
    b = _make_battle({}, {}, won=False, lost=False)
    assert calc.terminal(b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 커스텀 설정
# ---------------------------------------------------------------------------


def test_custom_config_hp_scale() -> None:
    """커스텀 hp_delta_scale이 보상 계산에 반영되어야 한다."""
    cfg = RewardConfig(hp_delta_scale=1.0)
    calc = RewardCalculator(config=cfg)

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))
    r = calc.compute(_make_battle({"a": 1.0}, {"x": 0.5}))

    assert r == pytest.approx(1.0 * 0.5)


def test_custom_config_win_reward() -> None:
    """커스텀 win_reward가 terminal()에 반영되어야 한다."""
    cfg = RewardConfig(win_reward=2.0)
    calc = RewardCalculator(config=cfg)
    b = _make_battle({}, {}, won=True)
    assert calc.terminal(b) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 신규 포켓몬 등장 (상대 팀 점진적 공개)
# ---------------------------------------------------------------------------


def test_newly_revealed_opponent_no_delta() -> None:
    """이전 스냅샷에 없던 상대 포켓몬이 등장해도 HP delta가 0이어야 한다."""
    calc = RewardCalculator()

    calc.compute(_make_battle({"a": 1.0}, {"x": 1.0}))

    # y가 새로 공개됨 (x HP 변화 없음)
    r = calc.compute(_make_battle({"a": 1.0}, {"x": 1.0, "y": 0.8}))

    # y는 이전 스냅샷 없음 → delta = 0. x 변화 없음 → 전체 reward = 0
    assert r == pytest.approx(0.0)
