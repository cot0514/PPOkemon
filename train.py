"""PPOkemon 학습 스크립트 (Phase 1 / Phase 2 공용).

사용법:
    # Phase 1 신규 학습
    python train.py --phase 1

    # Phase 2 전이학습 (Phase 1 가중치 로드)
    python train.py --phase 2 --load-model models/ppokemon_phase1

    # 기타 옵션
    python train.py --phase 2 --load-model models/ppokemon_phase1 --opponent maxdamage
    python train.py --phase 1 --timesteps 1000000 --n-envs 4

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
from data.pokemon_pool import PHASE1_POKEMON, PHASE2_POKEMON
from data.teams import RandomPoolTeambuilder
from envs.env import PPOkemonEnv
from envs.reward import RewardConfig

# 페이즈별 설정
_PHASE_CONFIG: dict[int, dict] = {
    1: {
        "battle_format": "gen9ppokemonphase1",
        "pokemon_pool": PHASE1_POKEMON,
        "save_path": "models/ppokemon_phase1",
    },
    2: {
        "battle_format": "gen9ppokemonphase2",
        "pokemon_pool": PHASE2_POKEMON,
        "save_path": "models/ppokemon_phase2",
    },
}


# ---------------------------------------------------------------------------
# 인수 파싱
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPOkemon MaskablePPO 학습")
    p.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="학습 페이즈 (1 또는 2)",
    )
    p.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="전이학습용 기존 모델 경로 (Phase 2에서 Phase 1 가중치 로드 시 사용)",
    )
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
    p.add_argument("--save-path", type=str, default=None, help="모델 저장 경로 (기본: 페이즈별 자동)")
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


def make_env_fn(rank: int, opponent_type: str, battle_format: str, pokemon_pool: list[str]):
    """SubprocVecEnv에 전달할 환경 초기화 함수를 반환한다.

    각 서브프로세스에서 호출되므로 모든 객체를 내부에서 생성해야 한다.
    rank로 username을 구분하여 PS 서버에서 중복 접속을 방지한다.
    """

    def _init() -> PPOkemonEnv:
        logging.getLogger().addFilter(_RaceConditionLogFilter())
        opp_tb = RandomPoolTeambuilder(pokemon_pool)
        opp_cfg = AccountConfiguration(f"PPOOpp{rank}", None)
        if opponent_type == "maxdamage":
            opponent: RandomPoolOpponent | MaxDamageOpponent = MaxDamageOpponent(
                account_configuration=opp_cfg,
                server_configuration=LocalhostServerConfiguration,
                battle_format=battle_format,
                team=opp_tb,
            )
        else:
            opponent = RandomPoolOpponent(
                account_configuration=opp_cfg,
                server_configuration=LocalhostServerConfiguration,
                battle_format=battle_format,
                team=opp_tb,
            )

        return PPOkemonEnv(
            teambuilder=RandomPoolTeambuilder(pokemon_pool),
            reward_config=RewardConfig(),
            opponent=opponent,
            account_configuration=AccountConfiguration(f"PPOAgent{rank}", None),
            server_configuration=LocalhostServerConfiguration,
            battle_format=battle_format,
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

    cfg = _PHASE_CONFIG[args.phase]
    battle_format: str = cfg["battle_format"]
    pokemon_pool: list[str] = cfg["pokemon_pool"]
    save_path: str = args.save_path or cfg["save_path"]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"Phase {args.phase} | 포맷: {battle_format} | 풀: {len(pokemon_pool)}마리")
    print(f"병렬 환경 {args.n_envs}개 초기화 중...")

    vec_env = SubprocVecEnv(
        [
            make_env_fn(i, args.opponent, battle_format, pokemon_pool)
            for i in range(args.n_envs)
        ],
    )

    # 에이전트·상대 WebSocket 연결(n_envs × 2개) 안정화 대기
    time.sleep(5)
    print(f"환경 {args.n_envs}개 준비 완료. 학습 시작.")

    if args.load_model is not None:
        # 전이학습: Phase 1 가중치를 그대로 이어받아 Phase 2 환경에 연결
        print(f"전이학습: {args.load_model} 로드")
        model = MaskablePPO.load(
            args.load_model,
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            tensorboard_log=args.log_dir,
        )
    else:
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
    model.save(save_path)

    won_list = vec_env.get_attr("n_won_battles")
    finished_list = vec_env.get_attr("n_finished_battles")
    total_won = sum(won_list)
    total_finished = sum(finished_list)
    print(f"\n모델 저장: {save_path}")
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
