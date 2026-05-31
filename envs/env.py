"""PPOkemon Gymnasium environment.

poke-env의 Gen9EnvSinglePlayer를 상속하여 MaskablePPO 학습에 적합한
Gymnasium 환경을 제공한다.

━━━ 에피소드 흐름 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  reset()
    └─ RewardCalculator.reset()           HP 스냅샷 초기화
    └─ teambuilder.yield_team()           랜덤 3마리 팀 구성 (poke-env 내부)
    └─ embed_battle()                     초기 관측 반환

  step(action)
    └─ action_masks() 확인 (MaskablePPO)
    └─ action_to_move()                   이산 정수 → BattleOrder
    └─ 서버 응답 대기 (poke-env 내부, 타임아웃 감시 중)
    └─ calc_reward()                      턴 보상 + 종료 보상
    └─ embed_battle()                     다음 관측 반환

━━━ 행동 공간 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [0–3] 기술 슬롯 0–3
  [4–5] 벤치 슬롯 0–1로 교체
"""

from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np
from gymnasium.spaces import Space
from poke_env.concurrency import POKE_LOOP
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder, Gen9EnvSinglePlayer
from poke_env.teambuilder import Teambuilder

from data.type_chart import matchup_value, species_to_type
from envs.reward import DEFAULT_CONFIG, RewardCalculator, RewardConfig
from envs.spaces import (
    N_ACTIONS,
    N_MOVES,
    OBS_DIM,
    OBS_SPACE,
    action_to_order,
    build_action_mask,
    build_observation,
    safe_action,
)

_logger = logging.getLogger(__name__)

# poke-env가 PS 서버 응답을 기다리다 무한 대기에 빠질 수 있다.
# 이 시간(초) 안에 step이 완료되지 않으면 강제 종료 처리한다.
_STEP_TIMEOUT = 30
# reset()은 서버 재시작 대기가 필요할 수 있으므로 더 긴 타임아웃을 사용한다.
_RESET_TIMEOUT = 300


class PPOkemonEnv(Gen9EnvSinglePlayer):
    """MaskablePPO 학습용 Pokemon Showdown 환경.

    Phase 1: 아르세우스 9종, 3v3
    행동 공간: 6 (기술 4 + 교체 2)
    관측 공간: 312차원 float32 벡터
    """

    def __init__(
        self,
        teambuilder: Teambuilder,
        reward_config: RewardConfig = DEFAULT_CONFIG,
        **kwargs,
    ) -> None:
        """Args:
            teambuilder: 배틀마다 랜덤 팀을 생성하는 RandomPoolTeambuilder.
            reward_config: 보상 함수 하이퍼파라미터. 기본값 사용 권장.
            **kwargs: Gen9EnvSinglePlayer에 전달할 추가 인자
                      (예: battle_format, server_configuration).
        """
        # opponent와 start_challenging은 **kwargs로 호출자(train.py)가 전달한다.
        # Gen9EnvSinglePlayer의 기본값: opponent=None, start_challenging=False.
        super().__init__(team=teambuilder, **kwargs)
        self._reward_calc = RewardCalculator(reward_config)
        # 타임아웃된 step() 스레드 참조 — reset()이 큐 정리 후 이 스레드를 기다린다.
        self._step_thread: threading.Thread | None = None
        # 이전 reset()이 타임아웃으로 zero obs를 반환했음을 표시.
        # 다음 reset()에서 poke-env 큐를 강제 정리할 때 사용한다.
        self._reset_timed_out: bool = False

    # ── 관측 ────────────────────────────────────────────────────────────────

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """배틀 상태 → 312차원 관측 벡터."""
        return build_observation(battle)

    def describe_embedding(self) -> Space:
        """observation_space를 반환한다. Gymnasium 규격."""
        return OBS_SPACE

    # ── 보상 ────────────────────────────────────────────────────────────────

    def calc_reward(
        self,
        last_battle: AbstractBattle | None,
        current_battle: AbstractBattle,
    ) -> float:
        """매 턴 보상 + 에피소드 종료 시 승/패 보상.

        poke-env가 서버 응답 후 이 메서드를 호출한다.
        last_battle은 참조하지 않고 RewardCalculator의 내부 스냅샷을 사용한다.
        """
        reward = self._reward_calc.compute(current_battle)
        if current_battle.finished:
            reward += self._reward_calc.terminal(current_battle)
        return reward

    # ── 행동 ────────────────────────────────────────────────────────────────

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """이산 action 인덱스를 poke-env BattleOrder로 변환한다.

        MaskablePPO가 마스킹을 올바르게 적용했다면 safe_action은 무해한 통과.
        배틀 상태 이상으로 유효한 행동이 없으면 PS 기본 행동(DefaultBattleOrder)으로 폴백한다.
        """
        try:
            action = safe_action(action, battle)
            return action_to_order(action, battle)
        except RuntimeError:
            return self.choose_default_move()

    def action_space_size(self) -> int:
        """행동 공간 크기. poke-env 내부에서 Discrete(N) 생성에 사용한다."""
        return N_ACTIONS

    # ── MaskablePPO ─────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO가 매 step 직전에 호출하는 마스크 반환.

        shape: (N_ACTIONS,) = (6,), dtype=bool
        """
        return build_action_mask(self.current_battle)

    # ── 행동 실행 ────────────────────────────────────────────────────────────

    def _terminal_obs(self) -> np.ndarray:
        """현재 배틀 상태에서 관측 벡터를 반환한다. 배틀이 None이면 zero 벡터."""
        b = self.current_battle
        return self.embed_battle(b) if b is not None else np.zeros(OBS_DIM, dtype=np.float32)

    def step(self, action):
        """교체 시 타입 상성 변화에 따라 보상을 조정하는 step 오버라이드.

        force_switch(기절 후 강제 교체)는 적용하지 않는다.
        delta = after_matchup_value - before_matchup_value (+2 ~ -2)
        reward += delta * switch_matchup_scale

        ━━━ 비정상 종료 처리 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        [케이스 1] "Battle is already finished" RuntimeError
          SB3 auto-reset 타이밍 레이스: 이미 끝난 배틀에 step이 들어온 경우.
          terminated=True를 반환해 SB3이 reset을 재시도하도록 한다.

        [케이스 2] _STEP_TIMEOUT 초과 (무한 대기 deadlock)
          poke-env가 PS 서버의 |request| 메시지를 기다리다 멈추는 경우.
          daemon 스레드가 timeout 되면 terminated=True로 강제 반환한다.
          daemon 스레드는 프로세스 종료 시 자동 정리된다.
        """
        matchup_delta = 0.0
        battle = self.current_battle
        if (
            action >= N_MOVES
            and battle is not None
            and not battle.force_switch
        ):
            my_active = battle.active_pokemon
            opp_active = battle.opponent_active_pokemon
            switch_idx = action - N_MOVES
            switches = battle.available_switches

            if (
                my_active is not None
                and opp_active is not None
                and switch_idx < len(switches)
            ):
                my_type = species_to_type(my_active.species)
                opp_type = species_to_type(opp_active.species)
                new_type = species_to_type(switches[switch_idx].species)

                before = matchup_value(my_type, opp_type)
                after = matchup_value(new_type, opp_type)
                matchup_delta = (
                    (after - before) * self._reward_calc.cfg.switch_matchup_scale
                )

        # daemon 스레드로 super().step() 실행 — timeout 초과 시 deadlock 탈출
        result: list = []
        exc_holder: list = []

        def _call_super() -> None:
            try:
                result.append(super(PPOkemonEnv, self).step(action))
            except Exception as e:  # noqa: BLE001 — thread 내 예외를 메인으로 전달
                exc_holder.append(e)

        t = threading.Thread(target=_call_super, daemon=True)
        t.start()
        t.join(timeout=_STEP_TIMEOUT)

        if t.is_alive():
            # 스레드 참조를 보존: reset()이 이 스레드를 기다린 후 진행해야
            # poke-env 내부 상태 충돌(경쟁)을 막을 수 있다.
            self._step_thread = t
            _logger.warning(
                "step() timed out after %ds (turn=%s) — forcing terminal",
                _STEP_TIMEOUT,
                getattr(battle, "turn", "?"),
            )
            return self._terminal_obs(), 0.0, True, False, {}

        self._step_thread = None

        if exc_holder:
            # 모든 예외를 캐치해 종료 처리 — 배틀 상태 이상으로 인한 크래시 방지
            _logger.warning("step() exception: %s", exc_holder[0])
            return self._terminal_obs(), 0.0, True, False, {}

        obs, reward, terminated, truncated, info = result[0]
        reward += matchup_delta
        return obs, reward, terminated, truncated, info

    # ── 에피소드 수명주기 ────────────────────────────────────────────────────

    def _send_forfeit_to_ps(self) -> bool:
        """PS 서버에 /forfeit 명령을 직접 전송해 stuck 배틀을 종료한다.

        성공하면 PS가 배틀을 정상 종료하고 |win| 메시지를 보낸다.
        _battle_finished_callback이 observations 큐에 최종 obs를 투입하여
        잔류 step() 스레드가 자연스럽게 종료된다.
        그 후 _challenge_loop의 battle_against()가 반환되어 새 배틀이 시작된다.

        Returns:
            True if forfeit was sent successfully, False otherwise.
        """
        battle = self.current_battle or self.agent.current_battle
        if battle is None or battle.finished:
            return False
        battle_tag: str | None = getattr(battle, "battle_tag", None)
        if not battle_tag:
            return False

        async def _do_forfeit() -> None:
            await self.agent.ps_client.send_message("/forfeit", battle_tag)

        try:
            asyncio.run_coroutine_threadsafe(_do_forfeit(), POKE_LOOP).result(timeout=5.0)
            _logger.warning("Sent /forfeit for battle %s", battle_tag)
            return True
        except Exception as e:
            _logger.warning("send_forfeit failed: %s", e)
            return False

    def _drain_poke_env_queues(self) -> None:
        """poke-env 내부 actions/observations 큐의 잔류 항목을 정리한다.

        /forfeit 전송이 불가할 때(WebSocket 불가 등) 사용하는 fallback.
        fake obs를 주입해 잔류 step() 스레드가 즉시 종료될 수 있도록 한다.
        단, _challenge_loop의 battle_against() 블록은 해소하지 못하므로
        이후 reset()이 line 321에서 타임아웃될 수 있다.

        POKE_LOOP에서 실행해야 asyncio.Queue 메서드를 스레드 안전하게 호출할 수 있다.
        """
        async def _do_drain() -> None:
            while True:
                try:
                    self._actions.queue.get_nowait()
                except Exception:
                    break
            try:
                self._observations.queue.put_nowait(
                    np.zeros(OBS_DIM, dtype=np.float32)
                )
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_do_drain(), POKE_LOOP).result(timeout=2.0)
        except Exception as e:
            _logger.warning("Queue drain error: %s", e)

    def reset(self, **kwargs):
        """에피소드 초기화. HP 스냅샷을 먼저 리셋한 뒤 poke-env reset을 호출한다.

        ━━━ Stuck 배틀 복구 전략 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        step()이 타임아웃되면 _challenge_loop가 battle_against()에서 영구 블록된다.
        PS가 배틀 종료 메시지를 보내지 않으면 새 배틀이 시작되지 않아 cascade 발생.

        1차 복구: PS에 /forfeit 직접 전송
          - PS가 배틀을 정상 종료 → _battle_finished_callback → observations 큐에 obs 투입
          - 잔류 step() 스레드가 해당 obs를 받아 자연 종료
          - battle_against()가 반환 → _challenge_loop가 새 배틀 시작
          - super().reset()이 새 배틀의 초기 obs를 반환하며 정상 완료

        2차 복구 (fallback): WebSocket 불가 시 큐 drain + fake obs 주입
          - 잔류 step() 스레드는 종료되지만 battle_against()는 여전히 블록
          - super().reset()이 line 321에서 타임아웃될 수 있음 (cascade 지속 가능)
        """
        bad_state = (
            (self._step_thread is not None and self._step_thread.is_alive())
            or self._reset_timed_out
        )
        if bad_state:
            _logger.warning("reset(): bad state detected — forfeiting current battle")
            sent = self._send_forfeit_to_ps()
            if not sent:
                _logger.warning(
                    "reset(): forfeit unavailable — draining poke-env queues"
                )
                self._drain_poke_env_queues()

        # /forfeit 전송 시 _battle_finished_callback 후 step() 스레드가 종료되므로
        # 수 초 내로 join이 성공한다. drain fallback은 즉시 종료된다.
        join_timeout = 15.0 if bad_state else 5.0
        if self._step_thread is not None and self._step_thread.is_alive():
            self._step_thread.join(timeout=join_timeout)
        self._step_thread = None
        self._reset_timed_out = False

        self._reward_calc.reset()

        result: list = []
        exc_holder: list = []

        def _call_reset() -> None:
            try:
                result.append(super(PPOkemonEnv, self).reset(**kwargs))
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        t = threading.Thread(target=_call_reset, daemon=True)
        t.start()
        t.join(timeout=_RESET_TIMEOUT)

        if result:
            return result[0]

        if exc_holder:
            _logger.warning("reset() error — returning zero obs: %s", exc_holder[0])
        else:
            _logger.warning(
                "reset() timed out after %ds — returning zero obs", _RESET_TIMEOUT
            )
        self._reset_timed_out = True
        return np.zeros(OBS_DIM, dtype=np.float32), {}
