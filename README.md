# PPOkemon

Pokemon Showdown 배틀 환경을 기반으로 PPO(Proximal Policy Optimization) 알고리즘을 통해 배틀 전략을 학습하는 강화학습 프로젝트입니다.
Stable-Baselines3의 MaskablePPO를 사용하며, 직접 작성하는 코드는 모두 이 폴더에 위치합니다.

---

## 환경 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.10 |
| PyTorch | CUDA 12.8 이상 |
| Stable-Baselines3 | 최신 |
| sb3-contrib | 최신 (MaskablePPO) |
| poke-env | 최신 |
| Pokemon Showdown 서버 | 로컬 실행 필요 |

```bash
# conda 환경 활성화
conda activate ppokemon

# Pokemon Showdown 서버 시작 (별도 터미널)
cd <showdown-root>
node pokemon-showdown
```

---

## PS 서버 커스텀 포맷 설정 (필수)

아르세우스는 Uber 티어이므로 `gen9ou`에서 불법입니다.
`<showdown-root>/config/formats.ts` 에 아래 포맷을 추가해야 학습이 시작됩니다.

```typescript
{
    name: "Gen 9 PPOkemon Phase 1",
    mod: "gen9",
    ruleset: ["Standard"],
    banlist: [],
    team: "random",  // 팀빌더가 직접 팀을 전달하므로 실질적으로 무시됨
},
```

포맷 ID는 `gen9ppokemonphase1` (소문자, 공백 제거)이어야 합니다.
`train.py`의 `BATTLE_FORMAT = "gen9ppokemonphase1"` 과 일치해야 합니다.

---

## 파일 구조

```
PPOkemon/
├── train.py               # 학습 진입점 (SubprocVecEnv 8개 병렬)
├── eval.py                # 학습 결과 평가 (타입 상황별 행동 분석)
│
├── envs/
│   ├── env.py             # PPOkemonEnv — Gymnasium 환경 (Gen9EnvSinglePlayer 상속)
│   ├── spaces.py          # 관측(312차원) · 행동(6) 공간 정의
│   └── reward.py          # 보상 함수 (RewardConfig, RewardCalculator)
│
├── agents/
│   └── opponents.py       # RandomPoolOpponent, MaxDamageOpponent
│
├── data/
│   ├── pokemon_pool.py    # PHASE1/PHASE2 포켓몬 목록 · POKEMON_TO_IDX
│   ├── teams.py           # 아르세우스 스펙 · RandomPoolTeambuilder
│   └── type_chart.py      # 타입 데이터 · matchup_value() · species_to_type()
│
├── tests/
│   ├── test_spaces.py
│   ├── test_reward.py
│   ├── test_teams.py
│   └── test_opponents.py
│
└── models/                # 학습된 모델 저장 (.zip, gitignore)
```

---

## 제한 환경

전체 포켓몬 풀을 사용하면 학습 시간과 자원이 부족하므로 다음과 같이 제한합니다.

| 항목 | 제한 |
|------|------|
| 배틀 형식 | 3 vs 3 |
| 기술 | 공격기만 허용 (변화기 제외) |
| 아이템 | 사용 안 함 (아르세우스 플레이트 제외) |
| 특성 | 배틀에 영향을 주지 않는 특성만 사용 |
| 랭크업 변화 | State에 포함하지 않음 |

포켓몬 수는 단계별로 확장됩니다 (아래 커리큘럼 학습 참조).

---

## 커리큘럼 학습 (3단계)

관측 벡터를 처음부터 **최종 크기(312차원)** 로 고정하고, 아직 사용하지 않는 슬롯은 `0.0`으로 채웁니다.
이를 통해 단계 전환 시 `MaskablePPO.load()`로 가중치를 그대로 이어받을 수 있습니다.

| 단계 | 포켓몬 풀 | 타입 구성 | 학습 목표 |
|------|-----------|-----------|-----------|
| Phase 1 | 아르세우스 9종 (인덱스 0~8) | 9개 타입 (ALL_TYPES 중 9개 활성) | 기초 타입 상성 + 교체 타이밍 |
| Phase 2 | 아르세우스 18종 (인덱스 0~17) | 18개 타입 전체 활성 | 전체 타입 상성 + 기술 고려 |
| Phase 3 | 실제 포켓몬 18종 | 18개 타입 | 실수치 스탯 기반 전략, 자기대전(self-play) |

### 단계 간 호환성 설계

| 인코딩 | 고정 크기 | Phase 1 | Phase 2 |
|--------|-----------|---------|---------|
| 포켓몬 ID one-hot | 18차원 | 인덱스 0~8 활성 | 인덱스 0~17 전체 활성 |
| 기술 타입 one-hot | 18차원 | FIRE~ICE(인덱스 1~9) 활성 | 전체 18개 활성 |

두 인코딩 모두 처음부터 Phase 2 기준 크기로 고정되어 있어 단계 전환 시 차원이 바뀌지 않습니다.

---

## State (관측 공간) 정의

### 개요

`envs/spaces.py`에 정의된 **312차원 float32 벡터**를 사용합니다. 값의 범위는 `[-1.0, 1.0]`입니다.

### 벡터 레이아웃

```
[0   :156]  내 팀
[156 :312]  상대 팀
```

각 팀은 동일한 구조로 이루어집니다.

```
팀 (156차원)
├── 액티브 슬롯  (108차원)  [0:108]
├── 벤치 슬롯 #1  (24차원)  [108:132]
└── 벤치 슬롯 #2  (24차원)  [132:156]
```

### 액티브 슬롯 (108차원)

| 요소 | 차원 | 범위 | 설명 |
|------|------|------|------|
| HP 비율 | 1 | [0, 1] | `current_hp / max_hp` |
| 포켓몬 ID | 18 | {0, 1} | 18마리 풀 인덱스 one-hot |
| 실수치 스탯 | 5 | [-1, 1] | `log(내 스탯 / 상대 액티브 스탯) / log(400)` |
| 기술 #0~#3 | 21×4 | [0, 1] | 기술당: 위력(1) + 특수여부(1) + 선공기여부(1) + 타입 one-hot(18) |

### 기술 인코딩 (21차원 per move)

```
[0]       power_norm      위력 / 250 (클리핑)
[1]       is_special      특수기 여부 (1.0 / 0.0)
[2]       is_priority     선공기 여부 (priority > 0이면 1.0)
[3~20]    type_one_hot    전체 18개 타입 기준 one-hot
                          (NORMAL=3, FIRE=4, WATER=5, ..., FAIRY=20)
```

타입 one-hot은 Phase 1에서 9개 타입만 활성화되고 나머지 9개는 0.
Phase 2 전환 시 차원 변경 없이 나머지 슬롯이 활성화됩니다.

### 벤치 슬롯 (24차원)

기술 정보와 랭크업 변화는 포함하지 않습니다.
기절·부재 시 슬롯 전체가 `0` 벡터입니다.

| 요소 | 차원 | 범위 | 설명 |
|------|------|------|------|
| HP 비율 | 1 | [0, 1] | 기절 시 0 → 슬롯 전체가 0 벡터 |
| 포켓몬 ID | 18 | {0, 1} | one-hot |
| 실수치 스탯 | 5 | [-1, 1] | `log(내 스탯 / 상대 액티브 스탯) / log(400)` |

### 설계 결정 사항

- **타입 one-hot 18차원 고정**: Phase 2 전이학습을 위해 처음부터 전체 18개 타입 기준으로 인코딩. Phase 1 미사용 타입은 항상 0.
- **상태이상 제외**: 현재 변화기를 사용하지 않으므로 State에서 제외합니다.
- **랭크업 변화 제외**: 현재 로드맵에서 변화기를 사용하지 않으므로 State에서 제외합니다.
- **선공기 여부 포함**: `is_priority` (priority > 0이면 1.0)로 선공기를 명시적으로 인코딩합니다.
- **벤치에 기술 미포함**: 교체 후에야 기술이 의사결정에 사용되므로 불필요합니다.
- **실수치 스탯 log-relative 인코딩**: 포켓몬 대미지 공식이 스탯 비율에 비례하므로 log 스케일이 선형보다 학습 신호를 더 명확하게 표현합니다.

---

## Action (행동 공간) 정의

`envs/spaces.py`에 정의된 **`Discrete(6)`** 이산 행동 공간을 사용합니다.

| 인덱스 | 슬롯 | 설명 |
|--------|------|------|
| 0 | move slot 0 | active 포켓몬의 첫 번째 기술 사용 |
| 1 | move slot 1 | active 포켓몬의 두 번째 기술 사용 |
| 2 | move slot 2 | active 포켓몬의 세 번째 기술 사용 |
| 3 | move slot 3 | active 포켓몬의 네 번째 기술 사용 |
| 4 | switch slot 0 | 벤치 포켓몬 #0으로 교체 |
| 5 | switch slot 1 | 벤치 포켓몬 #1으로 교체 |

### Action Masking

매 턴 실제로 사용 가능한 action은 6개보다 적을 수 있으므로, PPO에게 불법 행동을 차단하는 boolean 마스크를 함께 제공합니다.

| 상황 | 마스킹 결과 |
|------|-------------|
| 기술이 3개뿐인 포켓몬 | `mask[3] = False` |
| 벤치 포켓몬 1마리 기절 | `mask[5] = False` |
| 강제 교체 턴 (active 기절 직후) | `mask[0~3]` 전부 `False` |

---

## Reward (보상 함수) 정의

`envs/reward.py`의 `RewardConfig`로 가중치를 관리합니다.

### 매 턴 보상 (dense)

| 항목 | 공식 | 기본값 |
|------|------|--------|
| 딜 보상 | `+hp_delta_scale × Δ상대HP` | scale=0.5 |
| 피딜 패널티 | `-hp_delta_scale × Δ내HP` | scale=0.5 |
| KO 보너스 | `+ko_reward × 상대기절수` | 0.25 |
| 기절 패널티 | `-faint_penalty × 내기절수` | 0.25 |
| 배틀장기화 패널티 | `-turn_penalty` (매 턴) | 0.005 |

### 교체 보상 (matchup-aware)

강제 교체(기절 후)를 제외한 자발적 교체에만 적용됩니다.

```
delta = matchup_value(새 포켓몬, 상대) - matchup_value(현재 포켓몬, 상대)
reward += delta × switch_matchup_scale
```

| 교체 상황 | delta | 보상 |
|-----------|-------|------|
| 불리 → 유리 | +2 | +0.6 |
| 불리 → 중립 | +1 | +0.3 |
| 중립 유지 | 0 | 0.0 |
| 중립 → 불리 | -1 | -0.3 |
| 유리 → 불리 | -2 | -0.6 |

`matchup_value()`: 내 타입이 상대에게 2배이면 +1, 상대가 내게 2배이면 -1, 상호 또는 중립이면 0.

### 에피소드 종료 보상 (sparse)

| 결과 | 보상 |
|------|------|
| 승리 | +1.0 |
| 패배 | -1.0 |
| 무승부 | 0.0 |

---

## 학습 실행

```bash
# 기본 (랜덤 상대, 8개 병렬 환경, 500K 스텝)
python train.py

# MaxDamage 상대로 학습 (타입 전략 압박 강화)
python train.py --opponent maxdamage

# 스텝 수·병렬 환경 수 지정
python train.py --timesteps 1000000 --n-envs 4

# 저장 경로 지정
python train.py --save-path models/phase1_v2
```

### 병렬 학습 구조

`SubprocVecEnv`로 N개의 독립 환경을 병렬 실행합니다.
각 서브프로세스는 고유한 username(`PPOAgent{rank}` / `PPOOpp{rank}`)을 사용합니다.

```
SubprocVecEnv
├── Subprocess 0: PPOAgent0 vs PPOOpp0
├── Subprocess 1: PPOAgent1 vs PPOOpp1
├── ...
└── Subprocess 7: PPOAgent7 vs PPOOpp7
```

---

## 평가 실행

```bash
# 기본 (100 배틀, random 상대)
python eval.py

# 모델 경로 · 상대 · 배틀 수 지정
python eval.py --model models/ppokemon_phase1 --opponent maxdamage --n-battles 50

# 턴별 상세 출력
python eval.py --verbose
```

### 출력 예시

```
======================================================
 총 배틀: 100   승: 90   패: 10   승률: 90.0%
======================================================

[타입 상황별 행동 선택]
  유리: 기술  350회 / 교체   10회  (교체 비율  2.8%)
  중립: 기술  400회 / 교체   20회  (교체 비율  4.8%)
  불리: 기술  100회 / 교체   80회  (교체 비율 44.4%)

======================================================
[해석 기준]
  불리 상황에서 교체 비율이 높으면 → 타입 교체 전략 학습됨
  유리 상황에서 교체 비율이 낮으면  → 유리한 위치 유지 학습됨
  상황에 관계없이 교체 비율이 동일  → 타입 무관 행동
======================================================
```

---

## 환경 안정성 — Cascade 복구 메커니즘

장시간 학습 중(약 60만 스텝 이상) Pokemon Showdown 서버가 응답하지 않으면
`step()` → `reset()` 연쇄 타임아웃(cascade)이 발생해 학습이 영구 정지할 수 있습니다.
이를 방지하기 위해 `PPOkemonEnv`에 2단계 복구 로직이 구현되어 있습니다.

### Cascade 발생 원인

```
step() → poke-env가 PS의 |request| 대기 → 30s 타임아웃 → terminated 반환
reset() → super().reset() line 321 무한 루프:
    while self.current_battle == self.agent.current_battle:  # 새 배틀 시작 전까지 탈출 불가
        time.sleep(0.01)
→ _challenge_loop의 battle_against()가 PS 응답 없이 블록 → 새 배틀 미시작 → 300s 타임아웃
→ 반복
```

### 복구 전략

**1차 복구 (primary)**: PS 서버에 `/forfeit` 직접 전송

```
reset() bad_state 감지
  → send_message("/forfeit", battle_tag)  # WebSocket을 통해 PS에 직접 전송
  → PS가 배틀 정상 종료 → |win| 전송
  → _battle_finished_callback: observations 큐에 최종 obs 투입
  → 잔류 step() 스레드: obs 수신 후 자연 종료
  → _challenge_loop의 battle_against() 반환 → 새 배틀 시작
  → super().reset() line 321 정상 탈출 → 초기 obs 반환 → 학습 재개
```

**2차 복구 (fallback)**: WebSocket 사용 불가 시 큐 drain + fake obs 주입

> 잔류 스레드는 종료되지만 `battle_against()`가 여전히 블록되므로
> `reset()` 300s 타임아웃이 재발할 수 있습니다.

### 복구 시 출력 로그

```
step() timed out after 30s (turn=3) — forcing terminal
reset(): bad state detected — forfeiting current battle
Sent /forfeit for battle battle-gen9ppokemonphase1-XXXXX
# 이후 정상 학습 재개
```

---

## 테스트 · 린트

```bash
conda activate ppokemon
cd PPOkemon

# 전체 테스트
pytest tests/ -q

# 린트
ruff check .

# 자동 수정
ruff check --fix .
```

---

## Phase 1 기술 구성

각 아르세우스는 기술 4슬롯으로 구성됩니다.

| 슬롯 | 종류 | 설명 |
|------|------|------|
| 0 | 자속기 (STAB) | Judgment — 자신의 타입으로 위력 × 1.5 |
| 1 | 견제기 (coverage) | 다른 타입 공격기 |
| 2 | 견제기 (coverage) | 다른 타입 공격기 |
| 3 | 선공기 (priority) | Extreme Speed — 위력 낮지만 항상 선제 |

에이전트가 학습해야 하는 전략 결정:
1. **기술 선택**: 견제기로 상대 약점을 찌를 수 있으면 교체 없이도 이득
2. **교체 선택**: 견제기로도 약점을 찌를 수 없을 때 더 유리한 포켓몬으로 교체
