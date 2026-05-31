"""Pokemon team specifications and random pool teambuilder.

각 아르세우스의 Showdown 포맷 문자열을 정의하고,
배틀마다 풀에서 3마리를 랜덤 선택하는 Teambuilder를 제공한다.

닉네임 필수:
  PS 배틀에서 모든 아르세우스 폼의 ident가 'p1: Arceus'로 동일하기 때문에,
  poke-env가 팀 딕셔너리에서 3마리를 하나로 합쳐 available_moves=[]가 된다.
  각 폼에 고유 닉네임(예: arceusfire)을 부여하면 ident가 'p1: arceusfire'처럼
  달라져 poke-env가 각 포켓몬을 독립적으로 추적할 수 있다.
"""

from __future__ import annotations

import random

from poke_env.teambuilder import Teambuilder

N_POKEMON_TEAM: int = 3  # 3v3 배틀 팀 크기

# ---------------------------------------------------------------------------
# Phase 1 — 아르세우스 9종 개별 스펙
# ---------------------------------------------------------------------------
# 형식: "닉네임 (종) @ 아이템" — 닉네임이 배틀 ident로 사용된다.
# 기술 구성: 자속기(STAB) × 1, 견제기(coverage) × 2, 선공기(priority) × 1

POKEMON_SPECS: dict[str, str] = {
    "arceusfire": """
arceusfire (Arceus-Fire) @ Flame Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Aura Sphere
- Thunderbolt
- Extreme Speed
""",
    "arceuswater": """
arceuswater (Arceus-Water) @ Splash Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Earthpower
- Aura Sphere
- Extreme Speed
""",
    "arceusgrass": """
arceusgrass (Arceus-Grass) @ Meadow Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Surf
- Earthpower
- Extreme Speed
""",
    "arceuselectric": """
arceuselectric (Arceus-Electric) @ Zap Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Air Slash
- Extreme Speed
""",
    "arceusground": """
arceusground (Arceus-Ground) @ Earth Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Power Gem
- Aura Sphere
- Extreme Speed
""",
    "arceusflying": """
arceusflying (Arceus-Flying) @ Sky Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Flamethrower
- Aura Sphere
- Extreme Speed
""",
    "arceusrock": """
arceusrock (Arceus-Rock) @ Stone Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Earthpower
- Ice Beam
- Extreme Speed
""",
    "arceusfighting": """
arceusfighting (Arceus-Fighting) @ Fist Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Earthpower
- Flamethrower
- Extreme Speed
""",
    "arceusice": """
arceusice (Arceus-Ice) @ Icicle Plate
Ability: Multitype
EVs: 252 Atk / 252 SpA / 4 Spe
Hardy Nature
- Judgment
- Thunderbolt
- Energy Ball
- Extreme Speed
""",
    # Phase 2 추가분 (인덱스 9~17)
    # "arceus": """TODO""",
    # "arceuspoison": """TODO""",
    # "arceuspsychic": """TODO""",
    # "arceusbug": """TODO""",
    # "arceusghost": """TODO""",
    # "arceusdragon": """TODO""",
    # "arceusdark": """TODO""",
    # "arceussteel": """TODO""",
    # "arceusfairy": """TODO""",
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
