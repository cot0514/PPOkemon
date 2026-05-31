"""Phase 1 타입 데이터 및 상성 유틸리티.

spaces.py, env.py, eval.py에서 공통으로 사용한다.

━━━ 타입 인코딩 설계 원칙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  관측 벡터의 기술 타입 one-hot은 ALL_TYPES(18개) 기준으로 고정한다.
  Phase 1(9종)에서 사용하지 않는 타입 인덱스는 항상 0이 되어
  Phase 2(18종)로 전이학습 시 관측 차원이 변하지 않는다.
"""

from __future__ import annotations

from poke_env.environment import PokemonType

# ── 전체 18개 타입 (관측 벡터 one-hot 기준; 순서 = 인덱스) ──────────────────
ALL_TYPES: list[PokemonType] = [
    PokemonType.NORMAL,
    PokemonType.FIRE,
    PokemonType.WATER,
    PokemonType.GRASS,
    PokemonType.ELECTRIC,
    PokemonType.GROUND,
    PokemonType.FLYING,
    PokemonType.ROCK,
    PokemonType.FIGHTING,
    PokemonType.ICE,
    PokemonType.POISON,
    PokemonType.BUG,
    PokemonType.GHOST,
    PokemonType.DRAGON,
    PokemonType.DARK,
    PokemonType.STEEL,
    PokemonType.PSYCHIC,
    PokemonType.FAIRY,
]

N_ALL_TYPES: int = len(ALL_TYPES)  # 18

# PokemonType → one-hot 인덱스 (18개 기준)
ALL_TYPE_INDEX: dict[PokemonType, int] = {t: i for i, t in enumerate(ALL_TYPES)}

# ── Phase 1 타입 9개 (상성 계산 전용) ────────────────────────────────────────
PHASE1_TYPES: list[PokemonType] = [
    PokemonType.FIRE,
    PokemonType.WATER,
    PokemonType.GRASS,
    PokemonType.ELECTRIC,
    PokemonType.GROUND,
    PokemonType.FLYING,
    PokemonType.ROCK,
    PokemonType.FIGHTING,
    PokemonType.ICE,
]

N_PHASE1_TYPES: int = len(PHASE1_TYPES)  # 9

# 하위 호환용 별칭 (기존 코드에서 TYPE_INDEX를 사용하는 곳은 ALL_TYPE_INDEX로 교체 권장)
TYPE_INDEX: dict[PokemonType, int] = ALL_TYPE_INDEX

# 공격 타입 → 2배 피해를 주는 방어 타입 목록 (Phase 1 내에서만)
SUPER_EFFECTIVE: dict[PokemonType, list[PokemonType]] = {
    PokemonType.FIRE:     [PokemonType.GRASS, PokemonType.ICE],
    PokemonType.WATER:    [PokemonType.FIRE, PokemonType.GROUND, PokemonType.ROCK],
    PokemonType.GRASS:    [PokemonType.WATER, PokemonType.GROUND, PokemonType.ROCK],
    PokemonType.ELECTRIC: [PokemonType.WATER, PokemonType.FLYING],
    PokemonType.GROUND:   [PokemonType.FIRE, PokemonType.ELECTRIC, PokemonType.ROCK],
    PokemonType.FLYING:   [PokemonType.GRASS, PokemonType.FIGHTING],
    PokemonType.ROCK:     [PokemonType.FIRE, PokemonType.FLYING, PokemonType.ICE],
    PokemonType.FIGHTING: [PokemonType.ROCK, PokemonType.ICE],
    PokemonType.ICE:      [PokemonType.GRASS, PokemonType.GROUND, PokemonType.FLYING],
}

# 아르세우스 종 이름 → PokemonType (종 이름 정규화: 소문자, 하이픈·공백 제거)
ARCEUS_SPECIES_TYPE: dict[str, PokemonType] = {
    "arceusfire":     PokemonType.FIRE,
    "arceuswater":    PokemonType.WATER,
    "arceusgrass":    PokemonType.GRASS,
    "arceuselectric": PokemonType.ELECTRIC,
    "arceusground":   PokemonType.GROUND,
    "arceusflying":   PokemonType.FLYING,
    "arceusrock":     PokemonType.ROCK,
    "arceusfighting": PokemonType.FIGHTING,
    "arceusice":      PokemonType.ICE,
}


def species_to_type(species: str) -> PokemonType | None:
    """종 이름(arceus-fire, Arceus-Fire, arceusfire 등)에서 PokemonType을 반환한다."""
    key = species.lower().replace("-", "").replace(" ", "")
    return ARCEUS_SPECIES_TYPE.get(key)


def matchup_value(my_type: PokemonType | None, opp_type: PokemonType | None) -> int:
    """타입 상성을 +1(유리) / 0(중립) / -1(불리)로 반환한다.

    내가 상대 타입에 2배, 상대가 내 타입에 2배면 상쇄하여 0.
    """
    if my_type is None or opp_type is None:
        return 0
    i_hit_hard = opp_type in SUPER_EFFECTIVE.get(my_type, [])
    they_hit_hard = my_type in SUPER_EFFECTIVE.get(opp_type, [])
    if i_hit_hard and not they_hit_hard:
        return 1
    if they_hit_hard and not i_hit_hard:
        return -1
    return 0
