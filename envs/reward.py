"""Reward shaping for PPOkemon Phase 1.

Phase 1 목표: 아르세우스 9종으로 타입 상성과 교체 타이밍을 경험 기반 학습.

━━━ 설계 근거 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1에서 아르세우스 9종의 스탯은 완전히 동일하므로
배틀의 유일한 전략 변수는 타입 상성(type matchup)이다.

  타입 상성 학습:
    자속(STAB) + 효과적(SE) 기술은 더 많은 Δ상대HP를 만들어
    → hp_delta 보상이 자연스럽게 높아진다.
    에이전트는 어떤 기술/타입이 많은 딜을 내는지 경험으로 학습한다.

  교체 타이밍 학습:
    불리한 매치업(내 타입이 상대에게 약점)에서는 매 턴 큰 my_delta 패널티.
    교체 후 유리한 매치업으로 전환하면 패널티가 줄고 opp_delta 보상이 늘어
    → switch_cost를 감수할 만한 기대 이득이 발생한다.
    에이전트는 교체 타이밍의 손익을 스스로 학습한다.

━━━ 보상 항목 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  매 턴 (dense reward):
    +hp_delta_scale × Δ상대HP      상대 HP 감소분 → 딜 보상
    -hp_delta_scale × Δ내HP        내 HP 감소분 → 피딜 패널티
    +ko_reward × 상대기절수         KO 보너스
    -faint_penalty × 내기절수       내 기절 패널티
    -turn_penalty (매 턴)           배틀 장기화 억제 (교체 루프 방지)

  에피소드 종료 (sparse reward):
    +win_reward  (승리)
    -lose_reward (패배)
    0.0          (무승부)

━━━ 기술 구성 (Phase 1) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  아르세우스 9종 각각은 다음 4슬롯으로 구성된다:
    자속기(STAB)     × 1 : 자신의 타입 → 위력 × 1.5 보정
    견제기(coverage) × 2 : 다른 타입 공격기 → 위력 × 1.0, 상대 약점 노림
    선공기(priority) × 1 : 위력 낮지만 선제, 역시 공격기

  따라서 에이전트가 학습해야 하는 전략 결정은 두 축이다:
    ① 기술 선택: 견제기로 상대 약점을 찌를 수 있으면 교체 없이도 이득
    ② 교체 선택: 견제기로도 약점을 찌를 수 없을 때 더 유리한 포켓몬으로 교체

━━━ 스케일 근거 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  아르세우스 수치 환경(제한) 기준 턴당 예상 딜량 (기술 종류별):
    자속기  + 효과적(×2)   : STAB×SE    = ×3.0  → ~50% HP → reward ≈ +0.250
    자속기  + 보통 (×1)    : STAB       = ×1.5  → ~25% HP → reward ≈ +0.125
    자속기  + 비효과(×0.5) : STAB×NVE   = ×0.75 → ~12% HP → reward ≈ +0.063
    견제기  + 효과적(×2)   : 비STAB×SE  = ×2.0  → ~33% HP → reward ≈ +0.167
    견제기  + 보통 (×1)    : 비STAB     = ×1.0  → ~17% HP → reward ≈ +0.083
    견제기  + 비효과(×0.5) : 비STAB×NVE = ×0.5  → ~8%  HP → reward ≈ +0.042
    선공기  (저위력·선제)   :            ~15% HP  (상대 선공 차단 시 my_delta=0 추가 이득)

  최대 신호 차이 (자속SE ↔ 견제NVE): 0.250 - 0.042 ≈ 0.208  (충분히 유의미)

  turn_penalty = 0.005:
    매 턴 소액 패널티로 배틀 장기화를 억제한다.
    상대(RandomPoolOpponent)가 항상 공격하므로 에이전트가 교체만 반복하면
    my_delta 패널티 + turn_penalty 누적으로 자연스럽게 공격을 선호하게 된다.
    값이 너무 크면 KO/승리 보상을 압도하여 학습이 불안정해지므로
    KO 보상(0.25)의 2% 수준인 0.005로 설정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.environment import AbstractBattle


# ---------------------------------------------------------------------------
# 보상 가중치 설정
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """보상 함수 하이퍼파라미터. frozen=True로 불변성을 보장한다."""

    hp_delta_scale: float = 0.5
    """HP 변화량에 곱하는 스케일 팩터 (범위: [0, 1] HP fraction 기준)."""

    ko_reward: float = 0.25
    """상대 포켓몬 1마리 기절당 추가 보상."""

    faint_penalty: float = 0.25
    """내 포켓몬 1마리 기절당 추가 패널티."""

    win_reward: float = 1.0
    """에피소드 승리 시 단발 보상."""

    lose_reward: float = 1.0
    """에피소드 패배 시 단발 패널티 (부호는 terminal()에서 음수 처리)."""

    turn_penalty: float = 0.005
    """매 턴 부과하는 소액 패널티. 배틀 장기화 및 교체 루프를 억제한다."""

    switch_matchup_scale: float = 0.3
    """교체로 인한 타입 상성 변화에 곱하는 스케일.

    delta = after_matchup_value - before_matchup_value  (각각 +1/0/-1)
    reward += delta * switch_matchup_scale

    불리→유리(delta=+2): +0.6  불리→중립(delta=+1): +0.3
    중립→불리(delta=-1): -0.3  유리→불리(delta=-2): -0.6
    같은 상성 유지(delta=0):    0.0 (패널티·보너스 없음)
    강제 교체(기절 후)에는 적용하지 않는다."""


DEFAULT_CONFIG = RewardConfig()


# ---------------------------------------------------------------------------
# 보상 계산기
# ---------------------------------------------------------------------------


class RewardCalculator:
    """배틀 HP 변화를 추적하여 매 턴 보상을 계산한다.

    사용 패턴:
        calc = RewardCalculator()
        calc.reset()  # 에피소드 시작

        # 행동 선택 후 서버 응답이 오면:
        r = calc.compute(new_battle)

        # 에피소드 종료:
        r_terminal = calc.terminal(battle)
        calc.reset()
    """

    def __init__(self, config: RewardConfig = DEFAULT_CONFIG) -> None:
        self.cfg = config
        self._prev_my_hp: dict[str, float] = {}
        self._prev_opp_hp: dict[str, float] = {}

    def reset(self) -> None:
        """에피소드 시작 시 내부 HP 스냅샷을 초기화한다."""
        self._prev_my_hp = {}
        self._prev_opp_hp = {}

    @staticmethod
    def _take_snapshot(
        battle: AbstractBattle,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """현재 배틀의 HP 비율 스냅샷(포켓몬명 → HP fraction)을 반환한다."""
        my_hp = {
            name: p.current_hp_fraction for name, p in battle.team.items()
        }
        opp_hp = {
            name: p.current_hp_fraction for name, p in battle.opponent_team.items()
        }
        return my_hp, opp_hp

    def compute(self, battle: AbstractBattle) -> float:
        """서버 응답 후 갱신된 배틀 상태로 턴 보상을 계산한다.

        처음 호출 시(이전 스냅샷 없음)에는 기준 스냅샷만 저장하고 0.0을 반환한다.
        이후 호출마다 이전 스냅샷과 현재 상태의 차이로 보상을 계산한다.

        Args:
            battle: 서버 응답 이후 갱신된 AbstractBattle 인스턴스.

        Returns:
            이번 턴의 float 보상. 첫 호출은 항상 0.0.
        """
        curr_my_hp, curr_opp_hp = self._take_snapshot(battle)

        if not self._prev_my_hp:
            self._prev_my_hp = curr_my_hp
            self._prev_opp_hp = curr_opp_hp
            return 0.0

        # HP 감소량 계산 (감소 → 양수, 증가 → 음수)
        my_delta = sum(
            self._prev_my_hp.get(name, curr_hp) - curr_hp
            for name, curr_hp in curr_my_hp.items()
        )
        opp_delta = sum(
            self._prev_opp_hp.get(name, curr_hp) - curr_hp
            for name, curr_hp in curr_opp_hp.items()
        )

        # 기절 횟수 (HP > 0 → 살아있음)
        prev_my_alive = sum(1 for v in self._prev_my_hp.values() if v > 0.0)
        curr_my_alive = sum(1 for v in curr_my_hp.values() if v > 0.0)
        prev_opp_alive = sum(1 for v in self._prev_opp_hp.values() if v > 0.0)
        curr_opp_alive = sum(1 for v in curr_opp_hp.values() if v > 0.0)

        my_fainted = max(prev_my_alive - curr_my_alive, 0)
        opp_fainted = max(prev_opp_alive - curr_opp_alive, 0)

        reward = (
            self.cfg.hp_delta_scale * opp_delta          # 딜 보상
            - self.cfg.hp_delta_scale * my_delta          # 피딜 패널티
            + self.cfg.ko_reward * opp_fainted            # KO 보너스
            - self.cfg.faint_penalty * my_fainted         # 기절 패널티
            - self.cfg.turn_penalty                       # 배틀 장기화 패널티
        )

        # 다음 턴을 위해 스냅샷 갱신
        self._prev_my_hp = curr_my_hp
        self._prev_opp_hp = curr_opp_hp

        return float(reward)

    def terminal(self, battle: AbstractBattle) -> float:
        """에피소드 종료 시 승/패 보상을 반환한다.

        승리: +win_reward, 패배: -lose_reward, 무승부: 0.0.
        """
        if battle.won:
            return self.cfg.win_reward
        if battle.lost:
            return -self.cfg.lose_reward
        return 0.0
