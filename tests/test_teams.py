"""Tests for data/teams.py — team specs and random pool teambuilder."""

from __future__ import annotations

import pytest

from data.pokemon_pool import PHASE1_POKEMON
from data.teams import N_POKEMON_TEAM, POKEMON_SPECS, RandomPoolTeambuilder


# ---------------------------------------------------------------------------
# POKEMON_SPECS
# ---------------------------------------------------------------------------


def test_all_phase1_pokemon_have_specs() -> None:
    """PHASE1_POKEMON의 모든 포켓몬이 POKEMON_SPECS에 존재해야 한다."""
    for name in PHASE1_POKEMON:
        assert name in POKEMON_SPECS, f"{name} not in POKEMON_SPECS"


def test_specs_contain_required_fields() -> None:
    """각 스펙 문자열에 필수 필드(Ability, EVs, Nature)가 포함되어야 한다."""
    for name, spec in POKEMON_SPECS.items():
        assert "Ability:" in spec, f"{name}: Ability 누락"
        assert "EVs:" in spec, f"{name}: EVs 누락"
        assert "Nature" in spec, f"{name}: Nature 누락"


def test_specs_have_four_moves() -> None:
    """각 스펙에 기술이 정확히 4개 있어야 한다 (- 로 시작하는 줄)."""
    for name, spec in POKEMON_SPECS.items():
        move_lines = [line.strip() for line in spec.splitlines() if line.strip().startswith("-")]
        assert len(move_lines) == 4, f"{name}: 기술 수가 {len(move_lines)}개"


def test_all_specs_use_hardy_nature() -> None:
    """모든 스펙의 성격이 Hardy여야 한다."""
    for name, spec in POKEMON_SPECS.items():
        assert "Hardy Nature" in spec, f"{name}: Hardy Nature 아님"


def test_arceus_forms_use_correct_plates() -> None:
    """각 아르세우스 폼이 올바른 플레이트를 사용하는지 확인한다."""
    plate_map = {
        "arceusfire": "Flame Plate",
        "arceuswater": "Splash Plate",
        "arceusgrass": "Meadow Plate",
        "arceuselectric": "Zap Plate",
        "arceusground": "Earth Plate",
        "arceusflying": "Sky Plate",
        "arceusrock": "Stone Plate",
        "arceusfighting": "Fist Plate",
        "arceusice": "Icicle Plate",
    }
    for name, expected_plate in plate_map.items():
        spec = POKEMON_SPECS.get(name, "")
        assert expected_plate in spec, f"{name}: {expected_plate} 없음"


# ---------------------------------------------------------------------------
# RandomPoolTeambuilder
# ---------------------------------------------------------------------------


def test_teambuilder_raises_if_pool_too_small() -> None:
    with pytest.raises(ValueError):
        RandomPoolTeambuilder(["arceusfire", "arceuswater"])  # 2 < N_POKEMON_TEAM(3)


def test_teambuilder_accepts_minimum_pool() -> None:
    tb = RandomPoolTeambuilder(PHASE1_POKEMON[:N_POKEMON_TEAM])
    assert tb is not None


def test_yield_team_returns_string() -> None:
    tb = RandomPoolTeambuilder(PHASE1_POKEMON)
    result = tb.yield_team()
    assert isinstance(result, str)
    assert len(result) > 0


def test_yield_team_selects_n_pokemon() -> None:
    """yield_team()이 N_POKEMON_TEAM마리 팀을 구성해야 한다."""
    tb = RandomPoolTeambuilder(PHASE1_POKEMON)
    # packed team 형식은 '|'로 구분된 블록이 6개(포켓몬당) × N마리
    # poke-env join_team 결과에 포켓몬 이름 등장 횟수로 검증
    result = tb.yield_team()
    # packed format: 각 포켓몬 블록은 '|'로 시작
    pokemon_blocks = [b for b in result.split("]") if b.strip()]
    assert len(pokemon_blocks) == N_POKEMON_TEAM


def test_yield_team_no_duplicates() -> None:
    """반복 호출해도 한 팀 내에 중복 포켓몬이 없어야 한다."""
    tb = RandomPoolTeambuilder(PHASE1_POKEMON)
    for _ in range(20):
        packed = tb.yield_team()
        # packed format에서 종 이름은 첫 번째 필드
        species = [block.split("|")[0] for block in packed.split("]") if block.strip()]
        assert len(species) == len(set(species)), "중복 포켓몬 존재"


def test_yield_team_randomness() -> None:
    """충분히 반복하면 서로 다른 팀이 나와야 한다 (고정 팀 아님)."""
    tb = RandomPoolTeambuilder(PHASE1_POKEMON)
    teams = {tb.yield_team() for _ in range(30)}
    assert len(teams) > 1, "30번 호출 중 팀이 전혀 달라지지 않음"
