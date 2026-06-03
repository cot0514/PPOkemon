"""타입 데이터 및 상성 유틸리티 (Phase 1 / Phase 2 공용).

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

# ── Phase 2 타입 18개 = ALL_TYPES ─────────────────────────────────────────────
PHASE2_TYPES: list[PokemonType] = ALL_TYPES

N_PHASE2_TYPES: int = len(PHASE2_TYPES)  # 18

# 하위 호환용 별칭 (기존 코드에서 TYPE_INDEX를 사용하는 곳은 ALL_TYPE_INDEX로 교체 권장)
TYPE_INDEX: dict[PokemonType, int] = ALL_TYPE_INDEX

# 공격 타입 → 2배 피해를 주는 방어 타입 목록 (Phase 2 전체 18타입 기준)
# Phase 1 항목도 Phase 2 신규 방어 타입을 포함해 갱신함.
#   GROUND  : POISON, STEEL 추가
#   FLYING  : BUG 추가
#   ROCK    : BUG 추가
#   FIGHTING: NORMAL, DARK, STEEL 추가
#   ICE     : DRAGON 추가
SUPER_EFFECTIVE: dict[PokemonType, list[PokemonType]] = {
    # Phase 1 타입 (방어 타입 범위 확장)
    PokemonType.FIRE:     [
        PokemonType.GRASS, PokemonType.ICE, PokemonType.BUG, PokemonType.STEEL,
    ],
    PokemonType.WATER:    [
        PokemonType.FIRE, PokemonType.GROUND, PokemonType.ROCK,
    ],
    PokemonType.GRASS:    [
        PokemonType.WATER, PokemonType.GROUND, PokemonType.ROCK,
    ],
    PokemonType.ELECTRIC: [
        PokemonType.WATER, PokemonType.FLYING,
    ],
    PokemonType.GROUND:   [
        PokemonType.FIRE, PokemonType.ELECTRIC,
        PokemonType.POISON, PokemonType.ROCK, PokemonType.STEEL,
    ],
    PokemonType.FLYING:   [
        PokemonType.GRASS, PokemonType.FIGHTING, PokemonType.BUG,
    ],
    PokemonType.ROCK:     [
        PokemonType.FIRE, PokemonType.ICE, PokemonType.FLYING, PokemonType.BUG,
    ],
    PokemonType.FIGHTING: [
        PokemonType.NORMAL, PokemonType.ICE,
        PokemonType.ROCK, PokemonType.DARK, PokemonType.STEEL,
    ],
    PokemonType.ICE:      [
        PokemonType.GRASS, PokemonType.GROUND,
        PokemonType.FLYING, PokemonType.DRAGON,
    ],
    # Phase 2 신규 타입
    PokemonType.POISON:   [
        PokemonType.GRASS, PokemonType.FAIRY,
    ],
    PokemonType.BUG:      [
        PokemonType.GRASS, PokemonType.PSYCHIC, PokemonType.DARK,
    ],
    PokemonType.GHOST:    [
        PokemonType.GHOST, PokemonType.PSYCHIC,
    ],
    PokemonType.DRAGON:   [
        PokemonType.DRAGON,
    ],
    PokemonType.DARK:     [
        PokemonType.GHOST, PokemonType.PSYCHIC,
    ],
    PokemonType.STEEL:    [
        PokemonType.ICE, PokemonType.ROCK, PokemonType.FAIRY,
    ],
    PokemonType.PSYCHIC:  [
        PokemonType.FIGHTING, PokemonType.POISON,
    ],
    PokemonType.FAIRY:    [
        PokemonType.FIGHTING, PokemonType.DRAGON, PokemonType.DARK,
    ],
    # NORMAL: 2배 타입 없음 → 키 자체를 생략 (get 시 [] 반환)
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
    # Phase 2 추가분
    "arceus":         PokemonType.NORMAL,
    "arceuspoison":   PokemonType.POISON,
    "arceusbug":      PokemonType.BUG,
    "arceusghost":    PokemonType.GHOST,
    "arceusdragon":   PokemonType.DRAGON,
    "arceusdark":     PokemonType.DARK,
    "arceussteel":    PokemonType.STEEL,
    "arceuspsychic":  PokemonType.PSYCHIC,
    "arceusfairy":    PokemonType.FAIRY,
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
