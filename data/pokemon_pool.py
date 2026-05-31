"""Pokemon pool definitions for each curriculum phase.

Phase 1: 아르세우스 9종 (단일 타입 9가지)
Phase 2: 아르세우스 18종 (단일 타입 18가지)
Phase 3: 실제 포켓몬 18종 (단일 타입)

━━━ 인덱스 설계 원칙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  POKEMON_TO_IDX는 Phase 2 전체 18마리 기준으로 미리 고정한다.
  Phase 1(9마리)은 인덱스 0~8을 사용하고 나머지 슬롯은 0으로 남는다.
  Phase 2로 확장해도 기존 9마리 인덱스가 변하지 않으므로 전이학습이 가능하다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1 — 아르세우스 9종 (인덱스 0~8)
# ---------------------------------------------------------------------------

PHASE1_POKEMON: list[str] = [
    "arceusfire",       # 0
    "arceuswater",      # 1
    "arceusgrass",      # 2
    "arceuselectric",   # 3
    "arceusground",     # 4
    "arceusflying",     # 5
    "arceusrock",       # 6
    "arceusfighting",   # 7
    "arceusice",        # 8
]

# ---------------------------------------------------------------------------
# Phase 2 — 아르세우스 18종 (Phase 1 이후 인덱스 9~17)
# ---------------------------------------------------------------------------

PHASE2_EXTRA_POKEMON: list[str] = [
    "arceusnormal",     # 9
    "arceuspoison",     # 10
    "arceusbug",        # 11
    "arceusghost",      # 12
    "arceusdragon",     # 13
    "arceusdark",       # 14
    "arceussteel",      # 15
    "arceuspsychic",    # 16
    "arceusfairy",      # 17
]

PHASE2_POKEMON: list[str] = PHASE1_POKEMON + PHASE2_EXTRA_POKEMON

# ---------------------------------------------------------------------------
# 풀 인덱스 매핑 — Phase 2 전체 18마리 기준으로 고정
# ---------------------------------------------------------------------------

POKEMON_TO_IDX: dict[str, int] = {
    name: idx for idx, name in enumerate(PHASE2_POKEMON)
}
