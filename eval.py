"""PPOkemon 학습 결과 평가 스크립트.

학습된 모델을 로드해 실제 배틀을 진행하며
타입 상황별 행동 패턴과 승률을 분석한다.

사용법:
    python eval.py                                        # 기본값 (100 배틀)
    python eval.py --model models/ppokemon_phase1        # 모델 경로 지정
    python eval.py --opponent maxdamage --n-battles 50   # MaxDamage 상대
    python eval.py --verbose                             # 배틀별 상세 출력
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from sb3_contrib import MaskablePPO

from agents.opponents import MaxDamageOpponent, RandomPoolOpponent
from data.pokemon_pool import PHASE1_POKEMON
from data.teams import RandomPoolTeambuilder
from data.type_chart import PHASE1_TYPES, SUPER_EFFECTIVE, matchup_value, species_to_type
from envs.env import PPOkemonEnv
from envs.reward import RewardConfig
from envs.spaces import N_MOVES
from poke_env.environment import PokemonType

BATTLE_FORMAT = "gen9ppokemonphase1"

# PokemonType → 한국어 표시용
_TYPE_KR: dict[PokemonType, str] = {
    PokemonType.FIRE:     "불꽃",
    PokemonType.WATER:    "물",
    PokemonType.GRASS:    "풀",
    PokemonType.ELECTRIC: "전기",
    PokemonType.GROUND:   "땅",
    PokemonType.FLYING:   "비행",
    PokemonType.ROCK:     "바위",
    PokemonType.FIGHTING: "격투",
    PokemonType.ICE:      "얼음",
}

_ = PHASE1_TYPES  # type_chart import 사용 확인용


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------


def get_type_kr(species: str) -> str:
    """종 이름에서 한국어 타입을 반환한다."""
    t = species_to_type(species)
    return _TYPE_KR.get(t, "?") if t is not None else "?"


def matchup_label(species_mine: str, species_opp: str) -> str:
    """두 종 이름의 타입 상성을 '유리' / '불리' / '중립'으로 반환한다."""
    val = matchup_value(species_to_type(species_mine), species_to_type(species_opp))
    return {1: "유리", -1: "불리", 0: "중립"}[val]


# ---------------------------------------------------------------------------
# 환경 생성
# ---------------------------------------------------------------------------


def make_eval_env(opponent_type: str) -> PPOkemonEnv:
    opp_cfg = AccountConfiguration("EvalOpp", None)
    opp_tb = RandomPoolTeambuilder(PHASE1_POKEMON)
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
        account_configuration=AccountConfiguration("EvalAgent", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        start_challenging=True,
    )


# ---------------------------------------------------------------------------
# 평가 루프
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    print(f"모델 로드: {args.model}")
    model = MaskablePPO.load(args.model)

    env = make_eval_env(args.opponent)
    time.sleep(3)
    print(f"환경 준비 완료. {args.n_battles}배틀 시작.\n")

    results: list[bool] = []

    # 타입 상황별 행동 카운터: {상성: {행동종류: 횟수}}
    action_by_matchup: dict[str, Counter[str]] = {
        "유리": Counter(),
        "불리": Counter(),
        "중립": Counter(),
    }
    # 기술 선택 세부 분류: {상성: {기술종류: 횟수}}
    # 기술종류: 자속기 / 견제기(약점) / 견제기(비효과) / 선공기
    move_detail: dict[str, Counter[str]] = {
        "유리": Counter(),
        "불리": Counter(),
        "중립": Counter(),
    }

    for battle_idx in range(args.n_battles):
        obs, _ = env.reset()
        done = False
        prev_won = env.n_won_battles
        battle_log: list[str] = []

        while not done:
            battle = env.current_battle
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)

            # force_switch(기절 후 강제 교체)는 분석 제외
            if battle is not None and not battle.force_switch:
                my_pokemon = battle.active_pokemon
                opp_pokemon = battle.opponent_active_pokemon

                my_species = my_pokemon.species if my_pokemon else "?"
                opp_species = opp_pokemon.species if opp_pokemon else "?"
                my_type = get_type_kr(my_species)
                opp_type = get_type_kr(opp_species)
                situation = matchup_label(my_species, opp_species)

                is_switch = action >= N_MOVES
                if is_switch:
                    idx = action - N_MOVES
                    if battle.available_switches and idx < len(battle.available_switches):
                        sw = battle.available_switches[idx]
                        sw_type = get_type_kr(sw.species)
                        action_label = f"교체→{sw.species}({sw_type})"
                    else:
                        action_label = "교체(없음)"
                    action_by_matchup[situation]["교체"] += 1
                else:
                    if battle.available_moves and action < len(battle.available_moves):
                        mv = battle.available_moves[action]
                        action_label = f"기술:{mv.id}"
                        # 기술 세부 분류
                        # Judgment는 플레이트 타입을 poke-env가 NORMAL로 반환하므로
                        # move ID로 자속기를 판별한다.
                        opp_t = species_to_type(opp_species)
                        if mv.priority > 0:
                            move_kind = "선공기"
                        elif mv.id.lower() == "judgment":
                            move_kind = "자속기"
                        elif opp_t is not None and opp_t in SUPER_EFFECTIVE.get(mv.type, []):
                            move_kind = "견제기(약점)"
                        else:
                            move_kind = "견제기(비효과)"
                        move_detail[situation][move_kind] += 1
                    else:
                        action_label = "기술(없음)"
                    action_by_matchup[situation]["기술"] += 1

                if args.verbose:
                    battle_log.append(
                        f"  턴{battle.turn:2d} | "
                        f"내({my_type}) vs 상대({opp_type}) [{situation}] "
                        f"→ {action_label}"
                    )

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        won = env.n_won_battles > prev_won
        results.append(won)

        if args.verbose:
            print(f"── 배틀 {battle_idx + 1:3d} {'[승]' if won else '[패]'} ──")
            print("\n".join(battle_log))
            print()

        if not args.verbose and (battle_idx + 1) % 10 == 0:
            wr = sum(results) / len(results)
            print(f"  [{battle_idx + 1:3d}/{args.n_battles}] 승률: {wr:.1%}")

    env.close()

    # ── 최종 통계 출력 ──────────────────────────────────────────────────────
    total = len(results)
    wins = sum(results)

    print(f"\n{'='*55}")
    print(f" 총 배틀: {total}   승: {wins}   패: {total - wins}   승률: {wins/total:.1%}")
    print(f"{'='*55}")

    print("\n[타입 상황별 행동 선택]")
    for situation in ("유리", "중립", "불리"):
        cnt = action_by_matchup[situation]
        total_s = cnt["기술"] + cnt["교체"]
        if total_s == 0:
            continue
        switch_r = cnt["교체"] / total_s
        print(
            f"  {situation:2s}: 기술 {cnt['기술']:4d}회 / 교체 {cnt['교체']:4d}회"
            f"  (교체 비율 {switch_r:.1%})"
        )

    print(f"\n{'='*55}")
    print("[기술 선택 세부 분석]")
    print(f"  {'':4s}  {'자속기':>10s}  {'견제기(약점)':>12s}  {'견제기(비효과)':>14s}  {'선공기':>8s}")
    for situation in ("유리", "중립", "불리"):
        d = move_detail[situation]
        total_m = sum(d.values())
        if total_m == 0:
            continue

        def _fmt(k: str) -> str:
            n = d[k]
            return f"{n:4d}회({n / total_m:4.1%})"

        print(
            f"  {situation:2s}:  {_fmt('자속기')}  {_fmt('견제기(약점)')}"
            f"  {_fmt('견제기(비효과)')}  {_fmt('선공기')}"
        )

    print(f"\n{'='*55}")
    print("[해석 기준]")
    print("  불리 상황에서 교체 비율이 높으면 → 타입 교체 전략 학습됨")
    print("  유리 상황에서 교체 비율이 낮으면  → 유리한 위치 유지 학습됨")
    print("  불리 상황 기술 중 견제기(약점) 비율이 높으면 → 커버리지 활용 학습됨")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# 인수 파싱 / 진입점
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPOkemon 모델 평가")
    p.add_argument("--model", default="models/ppokemon_phase1", help="모델 경로")
    p.add_argument(
        "--opponent",
        choices=["random", "maxdamage"],
        default="random",
        help="상대 봇 종류",
    )
    p.add_argument("--n-battles", type=int, default=100, help="평가 배틀 수")
    p.add_argument("--verbose", action="store_true", help="배틀별 턴 상세 출력")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
