"""학습 상대 봇 정의.

커리큘럼 단계별로 난이도가 다른 상대를 제공한다.

  RandomPoolOpponent  : 제한 풀에서 랜덤 팀 + 랜덤 기술 선택 (교체 없음)
  MaxDamageOpponent   : 제한 풀에서 랜덤 팀 + 매 턴 최고 위력 기술 선택

교체 루프 방지 설계:
  두 상대 모두 force_switch(기절 후 강제 교체) 상황이 아니면 절대 교체를 선택하지 않는다.

레이스 컨디션 복구:
  PS 서버는 |request|(private)와 |turn|(public)을 별도 WebSocket 메시지로 보낸다.
  |turn|이 먼저 처리되면 available_moves=[]인 상태에서 choose_move가 호출된다.
  이때 DefaultBattleOrder("/choose default")를 PS에 보내면
  "[Invalid choice] There's nothing to choose" 에러가 발생하고 배틀이 멈춘다.
  _handle_battle_message를 오버라이드해서 이 에러를 감지하고
  battle.move_on_next_request=True로 설정하면 뒤이어 |request|가 도착했을 때
  poke-env가 자동으로 재시도한다.
"""

from __future__ import annotations

import logging
import random
from typing import List

from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder, RandomPlayer
from poke_env.player.battle_order import DefaultBattleOrder

logger = logging.getLogger(__name__)

_NOTHING_TO_CHOOSE = "[Invalid choice] There's nothing to choose"


class _RaceConditionMixin:
    """배틀 시작 시 |turn|/|request| 레이스 컨디션을 복구하는 믹스인."""

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        await super()._handle_battle_message(split_messages)  # type: ignore[misc]

        if not split_messages or not split_messages[0]:
            return

        # split_messages[0][0] = ">battle-gen9ppokemonphase1-N"
        raw_tag = split_messages[0][0]
        battle_tag = raw_tag[1:] if raw_tag.startswith(">") else raw_tag
        battle = self._battles.get(battle_tag)  # type: ignore[attr-defined]
        if battle is None:
            return

        for split_message in split_messages[1:]:
            if (
                len(split_message) >= 3
                and split_message[1] == "error"
                and _NOTHING_TO_CHOOSE in split_message[2]
            ):
                # |turn|이 |request|보다 먼저 처리된 레이스 컨디션.
                # 다음 |request| 도착 시 poke-env가 자동으로 재응답하도록 플래그 설정.
                battle.move_on_next_request = True
                logger.info(
                    "%s: race condition recovered — will retry on next |request| (turn=%d)",
                    self.__class__.__name__,
                    battle.turn,
                )


class RandomPoolOpponent(_RaceConditionMixin, RandomPlayer):
    """제한 풀 팀빌더를 주입받아 랜덤 기술을 선택하는 상대.

    force_switch가 아닌 이상 항상 기술을 선택한다.
    """

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.force_switch:
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])
            return DefaultBattleOrder()

        if battle.available_moves:
            chosen = random.choice(battle.available_moves)
            logger.debug(
                "RandomPoolOpponent: turn=%d move=%s (pool=%d moves)",
                battle.turn,
                chosen.id,
                len(battle.available_moves),
            )
            return self.create_order(chosen)

        logger.debug(
            "RandomPoolOpponent: turn=%d available_moves=[] force_switch=%s → DefaultBattleOrder (race condition)",
            battle.turn,
            battle.force_switch,
        )
        return DefaultBattleOrder()


class MaxDamageOpponent(_RaceConditionMixin, RandomPlayer):
    """매 턴 사용 가능한 기술 중 base_power가 가장 높은 기술을 선택하는 상대."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.force_switch:
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])
            return DefaultBattleOrder()

        if battle.available_moves:
            best = max(battle.available_moves, key=lambda m: m.base_power)
            logger.debug(
                "MaxDamageOpponent: turn=%d move=%s power=%d",
                battle.turn,
                best.id,
                best.base_power,
            )
            return self.create_order(best)

        logger.debug(
            "MaxDamageOpponent: turn=%d available_moves=[] → DefaultBattleOrder (race condition)",
            battle.turn,
        )
        return DefaultBattleOrder()
