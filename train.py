"""PPOkemon Phase 1 학습 스크립트.

사용법:
    python train.py                          # 기본값으로 실행
    python train.py --opponent maxdamage     # MaxDamage 상대로 학습
    python train.py --timesteps 1000000      # 스텝 수 변경
    python train.py --n-envs 4               # 병렬 환경 수 변경

Pokemon Showdown 로컬 서버가 실행 중이어야 한다.
    cd <showdown-root> && node pokemon-showdown
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from agents.opponents import MaxDamageOpponent, RandomPoolOpponent
from data.pokemon_pool import PHASE1_POKEMON
from data.teams import RandomPoolTeambuilder
from envs.env import PPOkemonEnv
from envs.reward import RewardConfig

# ---------------------------------------------------------------------------
# 배틀 포맷
# ---------------------------------------------------------------------------
# 아르세우스는 Uber 티어 → gen9ou에서 불법.
# Pokemon Showdown 서버 config/formats.ts에 커스텀 포맷을 추가하거나
# gen9ubers를 사용한다.
BATTLE_FORMAT = "gen9ppokemonphase1"  # custom-formats.ts에 등록된 포맷


# ---------------------------------------------------------------------------
# 인수 파싱
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPOkemon Phase 1 MaskablePPO 학습")
    p.add_argument(
        "--opponent",
        choices=["random", "maxdamage"],
        default="random",
        help="상대 봇 종류 (기본: random)",
    )
    p.add_argument("--n-envs", type=int, default=8, help="병렬 환경 수 (기본: 8)")
    p.add_argument("--timesteps", type=int, default=500_000, help="총 학습 스텝")
    p.add_argument("--lr", type=float, default=3e-4, help="학습률")
    p.add_argument("--n-steps", type=int, default=2048, help="환경당 rollout buffer 크기")
    p.add_argument("--batch-size", type=int, default=64, help="미니배치 크기")
    p.add_argument("--ent-coef", type=float, default=0.01, help="엔트로피 계수 (탐색)")
    p.add_argument("--save-path", type=str, default="models/ppokemon_phase1")
    p.add_argument("--log-dir", type=str, default="logs/")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 로깅 필터
# ---------------------------------------------------------------------------


class _RaceConditionLogFilter(logging.Filter):
    """poke-env가 레이스 컨디션 에러를 CRITICAL로 출력하는 것을 억제한다.

    _RaceConditionMixin이 이미 복구를 처리하므로 이 메시지는 노이즈에 해당한다.
    SubprocVecEnv 서브프로세스별로 적용해야 하므로 _init() 안에서 등록한다.
    """

    _MSG = "[Invalid choice] There's nothing to choose"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._MSG not in record.getMessage()


# ---------------------------------------------------------------------------
# 환경 팩토리
# ---------------------------------------------------------------------------


def make_env_fn(rank: int, opponent_type: str):
    """SubprocVecEnv에 전달할 환경 초기화 함수를 반환한다.

    각 서브프로세스에서 호출되므로 모든 객체를 내부에서 생성해야 한다.
    rank로 username을 구분하여 PS 서버에서 중복 접속을 방지한다.
    """

    def _init() -> PPOkemonEnv:
        # 레이스 컨디션 CRITICAL 로그 억제.
        # poke-env는 username 이름(PPOOpp7 등) 로거를 직접 사용하므로
        # poke_env 로거가 아닌 root 로거에 필터를 적용해야 한다.
        logging.getLogger().addFilter(_RaceConditionLogFilter())
        opp_tb = RandomPoolTeambuilder(PHASE1_POKEMON)
        opp_cfg = AccountConfiguration(f"PPOOpp{rank}", None)
        if opponent_type == "maxdamage":
            opponent: RandomPoolOpponent | MaxDamageOpponent = MaxDamageOpponent(
                account_configuration=opp_cfg,
                server_configuration=LocalhostServerConfiguration,
                battle_format=BATTLE_FORMAT,
                team=opp_tb,
            )
        else:
            opponent = RandomPoolOpponent(
                account_configuration=opp_cfg,
                server_configuration=LocalhostServerConfiguration,
                battle_format=BATTLE_FORMAT,
                team=opp_tb,
            )

        return PPOkemonEnv(
            teambuilder=RandomPoolTeambuilder(PHASE1_POKEMON),
            reward_config=RewardConfig(),
            opponent=opponent,
            account_configuration=AccountConfiguration(f"PPOAgent{rank}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=BATTLE_FORMAT,
            start_challenging=True,
        )

    return _init


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("agents.opponents").setLevel(logging.DEBUG)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"병렬 환경 {args.n_envs}개 초기화 중...")
    vec_env = SubprocVecEnv(
        [make_env_fn(i, args.opponent) for i in range(args.n_envs)],
    )

    # 에이전트·상대 WebSocket 연결(n_envs × 2개) 안정화 대기
    time.sleep(5)
    print(f"환경 {args.n_envs}개 준비 완료. 학습 시작.")

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        tensorboard_log=args.log_dir,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)

    # 전체 환경 승률 합산 (n_won_battles는 속성 → get_attr 사용)
    won_list = vec_env.get_attr("n_won_battles")
    finished_list = vec_env.get_attr("n_finished_battles")
    total_won = sum(won_list)
    total_finished = sum(finished_list)
    print(f"\n모델 저장: {args.save_path}")
    print(
        f"승률: {total_won}/{total_finished} "
        f"({total_won / max(total_finished, 1):.1%})"
    )

    vec_env.close()


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
