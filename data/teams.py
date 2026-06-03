"""Pokemon team specifications and random pool teambuilder.

각 아르세우스의 Showdown 포맷 문자열을 정의하고,
배틀마다 풀에서 3마리를 랜덤 선택하는 Teambuilder를 제공한다.

닉네임 필수:
  PS 배틀에서 모든 아르세우스 폼의 ident가 'p1: Arceus'로 동일하기 때문에,
  poke-env가 팀 딕셔너리에서 3마리를 하나로 합쳐 available_moves=[]가 된다.
  각 폼에 고유 닉네임(예: arceusfire)을 부여하면 ident가 'p1: arceusfire'처럼
  달라져 poke-env가 각 포켓몬을 독립적으로 추적할 수 있다.

기술 구성 원칙 (Phase 2 기준):
  자속기(Judgment) × 1  +  견제기(coverage) × 3
  각 아르세우스마다 불리 매칭 중 최소 1개는 견제기로 커버 불가 → 교체 강제.
"""

from __future__ import annotations

import random

from poke_env.teambuilder import Teambuilder

N_POKEMON_TEAM: int = 3  # 3v3 배틀 팀 크기

# ---------------------------------------------------------------------------
# Phase 1 — 아르세우스 9종 (인덱스 0~8)
# ---------------------------------------------------------------------------
# 불리 타입 → 견제기 커버 → 교체 강제(미커버)
#   arceusfire     : 물/바위 커버,  땅   미커버
#   arceuswater    : 풀    커버,  전기 미커버
#   arceusgrass    : 불꽃/비행/독 커버, 얼음/벌레 미커버
#   arceuselectric : (유틸 견제기),  땅   미커버
#   arceusground   : 물/얼음 커버,  풀   미커버
#   arceusflying   : 얼음/바위 커버, 전기 미커버
#   arceusrock     : 물/풀/격투/강철 커버, 땅 미커버
#   arceusfighting : 비행/에스퍼 커버, 페어리 미커버
#   arceusice      : 불꽃/바위/강철 커버, 격투 미커버

POKEMON_SPECS: dict[str, str] = {
    "arceusfire": """
arceusfire (Arceus-Fire) @ Flame Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Thunderbolt
- Aura Sphere
- Crunch
""",
    "arceuswater": """
arceuswater (Arceus-Water) @ Splash Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Ice Beam
- Aura Sphere
- Shadow Ball
""",
    "arceusgrass": """
arceusgrass (Arceus-Grass) @ Meadow Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Surf
- Thunderbolt
- Psychic
""",
    "arceuselectric": """
arceuselectric (Arceus-Electric) @ Zap Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Aura Sphere
- Shadow Ball
""",
    "arceusground": """
arceusground (Arceus-Ground) @ Earth Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Thunderbolt
- Aura Sphere
- Shadow Ball
""",
    "arceusflying": """
arceusflying (Arceus-Flying) @ Sky Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Aura Sphere
- Shadow Ball
""",
    "arceusrock": """
arceusrock (Arceus-Rock) @ Stone Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Thunderbolt
- Flamethrower
- Psychic
""",
    "arceusfighting": """
arceusfighting (Arceus-Fighting) @ Fist Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Thunderbolt
- Shadow Ball
- Earth Power
""",
    "arceusice": """
arceusice (Arceus-Ice) @ Icicle Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Surf
- Earth Power
- Shadow Ball
""",
    # -------------------------------------------------------------------------
    # Phase 2 추가분 (인덱스 9~17)
    # -------------------------------------------------------------------------
    # 노말 아르세우스: 플레이트 없음 → Multitype 유지, Judgment = 노말
    #   불리: 격투  /  견제기 없음 → 격투 미커버 (전체 커버 불가)
    "arceus": """
arceus (Arceus)
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Shadow Ball
- Thunderbolt
""",
    # 불리: 땅/에스퍼  /  얼음→땅 커버,  에스퍼 미커버
    "arceuspoison": """
arceuspoison (Arceus-Poison) @ Toxic Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Ice Beam
- Flamethrower
- Meteor Mash
""",
    # 불리: 불꽃/비행/바위  /  물→불꽃·바위 커버,  비행 미커버
    "arceusbug": """
arceusbug (Arceus-Bug) @ Insect Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Surf
- Shadow Ball
- Psychic
""",
    # 불리: 악  /  견제기 없음 → 악 미커버 (고스트는 악 타입 기술에 면역 없음)
    "arceusghost": """
arceusghost (Arceus-Ghost) @ Spooky Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Ice Beam
- Thunderbolt
- Earth Power
""",
    # 불리: 얼음/페어리  /  불꽃→얼음 커버,  페어리 미커버
    "arceusdragon": """
arceusdragon (Arceus-Dragon) @ Draco Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Thunderbolt
- Shadow Ball
""",
    # 불리: 격투/벌레/페어리  /  불꽃→벌레·에스퍼→격투 커버,  페어리 미커버
    "arceusdark": """
arceusdark (Arceus-Dark) @ Dread Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Psychic
- Thunderbolt
""",
    # 불리: 불꽃/격투/땅  /  물→불꽃·땅 커버,  격투 미커버
    "arceussteel": """
arceussteel (Arceus-Steel) @ Iron Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Surf
- Crunch
- Ice Beam
""",
    # 불리: 벌레/고스트/악  /  불꽃→벌레·격투→악 커버,  고스트 미커버
    "arceuspsychic": """
arceuspsychic (Arceus-Psychic) @ Mind Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Aura Sphere
- Thunderbolt
""",
    # 불리: 독/강철  /  불꽃→강철 커버,  독 미커버
    "arceusfairy": """
arceusfairy (Arceus-Fairy) @ Pixie Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Thunderbolt
- Shadow Ball
""",
}


# ---------------------------------------------------------------------------
# 랜덤 풀 팀빌더
# ---------------------------------------------------------------------------


class RandomPoolTeambuilder(Teambuilder):
    """배틀마다 풀에서 3마리를 랜덤 선택해 팀을 구성한다."""

    def __init__(self, pool: list[str]) -> None:
        """Args:
        pool: 선택 대상 포켓몬 이름 리스트 (pokemon_pool.PHASE1_POKEMON 등).
        """
        if len(pool) < N_POKEMON_TEAM:
            raise ValueError(
                f"pool에 최소 {N_POKEMON_TEAM}마리가 필요합니다. (현재 {len(pool)}마리)"
            )
        self.pool = pool

    def yield_team(self) -> str:
        """poke-env가 배틀 시작 전마다 호출한다. N_POKEMON_TEAM마리 팀 문자열을 반환한다."""
        chosen = random.sample(self.pool, N_POKEMON_TEAM)
        team_str = "\n".join(POKEMON_SPECS[name] for name in chosen)
        return self.join_team(self.parse_showdown_team(team_str))
