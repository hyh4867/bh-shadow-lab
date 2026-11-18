# 처리 흐름 (PIPELINE)

모든 판정 기준과 고정 선택은 [PREREGISTRATION.md](PREREGISTRATION.md)에 있다(자료를 받기 전 커밋 e106140). 이 문서는 코드가 실제로 하는 일과 실행 방법을 적는다.

## 실행

```bash
cd projects/bh-shadow-lab
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # 핵심: numpy scipy astropy matplotlib emcee pytest
uv pip install --python .venv/bin/python -r requirements-optional.txt  # 선택: 교차 확인(ehtim), 4단계(scikit-learn)
# 자료 받기: docs/DATA.md
.venv/bin/python scripts/run_stage1.py   # 약 15초
.venv/bin/python scripts/run_stage2.py   # 약 5분 (fit 40개 + 결합 fit 3개)
.venv/bin/python scripts/run_stage3.py   # 약 45분, 이어 돌리기 가능 (results/stage3_injection.csv)
.venv/bin/python scripts/run_stage4.py   # 선택, ML 비교, 약 4분
.venv/bin/python scripts/run_rml_image.py # 선택, 참고용 RML 시도(ehtim)
# 5단계(추가 분석, PREREGISTRATION 10절). 한 번에 하나씩 실행. fit 요약은 results/stage5_fits/에 저장되어 이어 돌릴 수 있음
.venv/bin/python scripts/run_stage5_systematics.py  # jackknife 13, 확장 모형 4, xs-ring 16, 예산·질량. 약 20분
.venv/bin/python scripts/run_stage5_lmt.py          # LMT 원인 시험, 합성 자료 결합 fit 30개. 약 25분
.venv/bin/python scripts/run_stage5_injection.py    # 현실적 주입 시험 120개. 약 6분
.venv/bin/python scripts/run_stage5_2018.py         # 2018 자료(DATA.md에서 받기). 약 2분
.venv/bin/python scripts/run_stage5_rml_diag.py     # 선택, RML 진단(ehtim). 약 1분
.venv/bin/python scripts/run_stage5_rml_lca_check.py  # 사전 등록 밖 진단: ehtim과 우리 log closure amplitude 비교
.venv/bin/python scripts/run_stage5_summary.py      # 요약 그림(넓은 판, 좁은 판)과 계통 그림
# 6단계(추가 분석, PREREGISTRATION 11절). 한 번에 하나씩, screen + caffeinate -i 권장.
# nested 요약은 results/stage6_fits/, dynesty 체크포인트는 results/stage6_ckpt/(git 제외)에 남아 이어 돌릴 수 있음
.venv/bin/python scripts/run_stage6_models.py      # 주 모형 7개 × nested 3회 + emcee τ. 약 5.5시간(R2G3만 2.2시간)
.venv/bin/python scripts/run_stage6_tau.py         # C3′(emcee 32000단계) 7개, 두 규칙의 중심 모형. 약 22분
.venv/bin/python scripts/run_stage6_systematics.py # C로 jackknife 13, 10초 변형, co-located 제외, 겹침 시험, 예산, 질량. 약 53분
.venv/bin/python scripts/run_stage6_xsring.py      # xs-ring S0–S4 (nested 48회). 약 1시간 45분
.venv/bin/python scripts/run_stage6_2018.py        # 2018 H1–H3, 예산, 비교. 약 33분
.venv/bin/python scripts/run_stage6_rml.py         # 선택, RML 재시도 R0–R3(ehtim). 약 20초
.venv/bin/python scripts/run_stage6_xsring_spread.py  # 사전 등록 밖 진단: X1 실행 간 d̂ 산포(저장된 표본만 읽음)
.venv/bin/python scripts/run_stage6_proc_diag.py   # 사전 등록 밖 진단: 10초 변형에서 M3와 E1 비교. 약 2분
.venv/bin/python scripts/run_stage6_summary.py     # 6단계 요약 그림(넓은 판, 좁은 판)
# scripts/run_stage6_queue.sh 는 models가 끝난 뒤 tau → systematics → xsring → 2018 → rml을 차례로 돌린다
# 7단계(추가 분석, PREREGISTRATION 12절, 큰 모형 수렴). 한 번에 하나씩, screen + caffeinate -i 권장.
# nested 요약은 results/stage7_fits/, 체크포인트는 results/stage7_ckpt/, 표본은 results/chains7/(둘 다 git 제외)
.venv/bin/python scripts/run_stage7_modemap.py     # 차등 진화 최적화 7모형 × 시드 12개(프로세스 4개). 약 45분
.venv/bin/python scripts/run_stage7_models.py      # 래더 A: G1 nlive 50k → 150k(12.2). 약 3시간 20분
.venv/bin/python scripts/run_stage7b.py            # 래더 B: slices 5(3+k), G1 nlive 300 × 5회 → (실패 시) nlive 800 × 5회(12.7)
.venv/bin/python scripts/run_stage7_select.py      # C1–C4 표, 중심 모형, σ_model, 예산, 질량(새 fit 없음)
# scripts/run_stage7_queue.sh 는 modemap → models → select, scripts/run_stage7b_queue.sh 는 stage7b → select
.venv/bin/python -m pytest -v            # 자료가 없으면 자료 기반 회귀 테스트는 skip
```

## 모듈

| 파일 | 하는 일 |
|---|---|
| `src/bhshadow/uvdata.py` | uvfits 읽기(I = RR, σ_I = σ_RR/√2, 켤레), scan 나누기(간격 ≥ 60초), scan 벡터 평균, 1 % 계통 오차, 진폭 편향 보정, SNR ≥ 2와 intra-site 기선 컷 |
| `src/bhshadow/closure.py` | closure phase와 log closure amplitude. scan마다 모든 삼각형·사각형을 σ 순으로 정렬해 rank를 올리는 것만 남김(비중복 집합) |
| `src/bhshadow/models.py` | M1 얇은 ring, M2 흐린 ring, M3 흐린 m=1 ring, D 흐린 disk, G Gaussian, 두 원판 차 crescent(주입용), 픽셀 DFT(시험용) |
| `src/bhshadow/fitting.py` | closure만 쓰는 우도, 차등 진화로 시작점, emcee(walker 32, burn-in 2000, 본 표본 4000), τ 기준 미달이면 두 배 길이로 한 번 재실행 |
| `src/bhshadow/nullfit.py` | 첫 null: A = √(k²(ρ−b₁)² + m²) 가중 최소제곱, scan 단위 bootstrap |
| `src/bhshadow/robust.py` | 강건성 변형: 10초 closure를 scan 안에서 평균 |
| `src/bhshadow/synth.py` | 주입 시험용 합성 자료(실제 10초 uv와 σ, 국 이득, 같은 처리) |
| `src/bhshadow/mass.py` | θ_g = d/α, M = θ_g D c²/G, Monte Carlo 10⁶개 |
| `src/bhshadow/literature.py` | 문헌 값(쪽·표는 PREREGISTRATION 2절) |
| `src/bhshadow/models.py` (5단계 추가) | E1 = M3 + 위치 자유 원형 Gaussian, E2 = m=2 밝기 추가, X0 = xs-ring(Paper VI 부록 B.1.1, 푸리에 변환은 THEORY 8.3에 직접 유도), X1 = xs-ring + 원형 nuisance Gaussian 2개 |
| `src/bhshadow/s5.py` | 5단계 공통: 결합 fit(날마다 β), fit 요약 캐시, walker 수 규칙, 추가 오염(10초 위상 random walk, LMT 10초 이득, 기선별 닫히지 않는 오차) |
| `src/bhshadow/dataset.py` (5단계 추가) | 2018 파일 읽기(국 코드 AX, MM, SW, MG, GL). intra-site 쌍 AA–AX, MM–SW |

| `src/bhshadow/models.py` (6단계 추가) | 타원 Gaussian `egauss`(THEORY 9.1), ring + 타원 Gaussian n개(G1–G3: m=1, R2G1–3: m=2), xs-ring + 타원 Gaussian(XE1–3), 큰 중심 Gaussian 추가(XE3L). 여러 Gaussian은 x 좌표 순서 제약으로 이름 바꿈 대칭을 없앰 |
| `src/bhshadow/nested.py` | dynesty 3.2.0 static nested sampling(bound multi, sample rslice, dlogz 0.1, 프로세스 4개, 120초 체크포인트). 사전분포 변환(x 정렬, β₁+β₂ ≤ 0.5 접기), R̂(주기 매개변수 포함), nested 사후에서 시작하는 emcee τ 확인 |
| `src/bhshadow/s6.py` | 6단계 공통: 결합 fit의 공유 매개변수, 3회 실행 + R̂ + ln Z 범위 + C3/C3′ |
| `src/bhshadow/closure.py` (6단계 추가) | `drop_colocated=True`면 co-located 국 쌍(2017 AA–AP, JC–SM / 2018 AA–AX, MM–SW)을 함께 포함한 사각형을 빼고 비중복 집합을 다시 고름 |
| `src/bhshadow/fitting.py` (7단계 변경) | 같은 group의 데이터셋은 매개변수가 같으므로 (u, v)를 이어 붙여 모형을 한 번만 계산한다. χ²는 데이터셋별 계산과 비트 단위로 같다(tests/test_data_stage7.py). G3 한 번 계산 0.82 → 0.58 ms |
| `src/bhshadow/nested.py` (7단계 추가) | 사전분포 변환 벡터판 `PriorTransform.batch`, `run_nested(slices=…)`, emcee `thin_by`(τ는 단계 단위로 환산) |
| `src/bhshadow/s7.py` | 7단계 공통: nlive = c·k, 다중 실행 + C1/C2 + C3′/C3″, 단위 입방체 위 차등 진화(`de_fit`), 모드 묶기, C4 |

## 단계별 처리

1. **읽기**: 8개 파일. 10초 점 2,367–9,081개. 릴리스 csv와 값 비교(상대 차이 ≤ 1e-4 기준, 실제 최대 약 1.2e-5).
2. **scan 평균**: (scan, 기선)마다 1/σ² 가중 벡터 평균. EHT Paper VI도 scan 평균 자료를 썼다(3쪽 2.2절). 이 평균이 결맞음(phase 안정)을 가정한다는 점은 아래 '한계'에 적었다.
3. **closure 양**: intra-site 기선(AA–AP, JC–SM)을 빼고 만든다. 데이터셋마다 closure phase 48–139개, log closure amplitude 41–114개.
4. **fit**: 데이터셋 8개 × 모형 5개, 그리고 M3 결합 fit(d, w 공유, 날마다 β). 판정용 오차 σ_mine = √((σ_raw·√χ²_ν)² + s₈²).
5. **질량**: α = 11.56 ± 1.32(GC 보정), α = 10.0 ± 0.4(photon ring), D = 16.8 ± 0.8 Mpc.
6. **주입 시험**: 4/06 hi, 4/10 hi coverage에 참값 10종 × 10회를 넣고 M3와 D를 fit. ΔBIC = BIC_D − BIC_M3로 ring과 disk를 분류.

## 독립 확인

- uvfits 읽기 ↔ 릴리스 csv(EHT가 만든 파생 파일): `results/stage1_reader_check.json`.
- 10초 closure phase ↔ eht-imaging 1.3.2 `c_phases` (2017-04-11 hi, 삼각형 4,008개, 최대 차이 6e-6°): `results/stage1_data_summary.json`.
- 모형 visibility ↔ 픽셀 이미지 직접 푸리에 합, ehtim `ftmatrix`(부호 규약): `tests/test_models.py`.
- closure 양의 이득 불변성, 비중복 집합 크기: `tests/test_closure.py`.

## 사전 등록에서 정하지 않아 구현하면서 정한 것

- null fit에서 |V| ≤ σ인 점은 편향 보정 진폭을 0으로 둔다. null fit에는 SNR 컷을 두지 않는다(7.1절에 SNR 컷이 없음).
- SNR 컷은 편향 보정 전의 |V|/σ(1 % 계통 오차 포함)로 계산한다.
- 편향 보정은 4절의 순서대로 계통 오차를 더한 σ로 한다. 계통 오차 1 %의 영향은 SNR이 높을 때 상대 1e-4 수준이다.
- emcee 시작점은 차등 진화 최적점 근처의 작은 공이다(사후분포 자체는 바뀌지 않음). 위치각 매개변수는 원 위의 균일 사전분포로 다룬다.
- 데이터 쪽 켤레(THEORY 2절): 처음 fit에서 4/11 밝은 쪽 위치각이 약 5°로 나와 EHT 값(약 180–207°)과 180° 어긋났다. 원인은 eht-imaging 소스에서 확인했다(+2πi 규약이고 읽을 때 켤레를 취하지 않음). 지름과 판정 기준에는 영향이 없다.

## 5단계 처리 (추가 분석)

1. **jackknife**: M3 결합 fit을 다음 부분 자료에 다시 한다. 망원경 7개를 하나씩 뺀 자료(그 국의 기선을 빼고 closure를 다시 만듦), 날 4개를 하나씩 뺀 자료, 밴드별 자료다. σ = √((n−1)/n Σ(d₍ᵢ₎ − d̄)²)로 계통 항을 만든다.
2. **모형 확장**: E1, E2 결합 fit. 중심값 규칙은 BIC가 M3보다 10 넘게 낮은 모형 중 가장 낮은 것이다.
3. **xs-ring**: X0, X1을 데이터셋마다 fit한다. d̂ = R_out(2 − ψ)는 사슬에서 직접 계산한다.
4. **LMT 원인 시험**: 2단계 최적해(전체, LMT 제외)를 참값으로 실제 8개 coverage에 합성 자료를 만든다. 실현 5회마다 전체와 LMT 제외 결합 fit을 한다.
5. **현실적 주입 시험**: 3단계 오염에 넓은 방출, 10초 위상 random walk, 닫히지 않는 오차를 더한다.
6. **2018**: 같은 처리를 하되 국 코드만 바꾼다. 주 자료는 4/21, 4/25이고 4/22는 부록이다.
7. **요약 그림**: 넓은 판 1440 px, 좁은 판 390 px 폭(2배 해상도 780 px). headless Chrome으로 1440 px 창과 390 px 폭 화면을 찍어 글자가 잘리거나 겹치지 않는지 확인했다.

## 6단계 처리 (추가 분석)

1. **주 모형**: M3, E1, E2, G1–G3, R2G{n*}(n* = G1–G3 중 ln Z가 가장 큰 n, 실제 3)를 2017 8개 결합으로 nested sampling 3회씩 돌린다(시드 다름). 3회 표본을 합쳐 요약하고, R̂, ln Z 범위, emcee τ로 수렴을 본다.
2. **중심 모형**: 수렴한 모형 중 ln Z 최대. ln B < 5 안의 더 단순한 모형이 있으면 그것. 원래 규칙(C3)과 바꾼 규칙(C3′, 11.10)으로 각각 고르고, 후속 분석은 C3′ 쪽(E1)으로 한다.
3. **계통**: 중심 모형으로 jackknife(망원경 7, 날 4, 밴드 2), 10초 closure 변형, co-located 사각형 제외를 각 nested 1회로 다시 fit한다. 겹침 시험은 5단계 M3 jackknife와 E1 jackknife의 차이로 한다.
4. **xs-ring**: 데이터셋마다 S1(X1 nested 3회) → S2(XE3) → S3(XE3, intra-site 기선 포함) → S4(XE3L). S2–S4는 nested 1회다.
5. **2018**: H1–H3은 2단계 emcee 표본 추출기(결과 캐시), 예산은 nested 1회씩이다.
6. **요약 그림**: 넓은 판 1440 px, 좁은 판 390 px 폭(2배 해상도). headless Chrome으로 1440 px 창과 390 px 폭 상자를 찍어 글자가 잘리거나 겹치지 않는지 확인했다(좁은 판의 긴 라벨 두 개를 줄임).

## 7단계 처리 (추가 분석)

1. **모드 지도(12.1)**: 7개 모형마다 scipy `differential_evolution`을 시드 12개로 돌린다. nested와 같은 단위 입방체와 사전분포 변환을 쓰므로 모든 점이 사전분포 안에 있다(popsize 15, maxiter 2000, tol 1e-6, latin hypercube, 끝에 L-BFGS-B polish). χ² 차 10 또는 d 차 0.5 넘게 다르면 다른 모드로 센다.
2. **C4**: 각 실행의 χ²_min이 지금까지 모든 방법(DE, 6·7단계 nested 실행)에서 찾은 최소 χ² + 20 안에 있어야 한다.
3. **래더 A(12.2)**: nlive만 키운다(G1 c = 50 → 150). 둘 다 실패해 멈췄다.
4. **래더 B(12.7)**: rslice 걸음 수 slices를 기본값 3 + k의 5배로 늘린다. G1 nlive 300 × 5회 → 수렴하면 G3, G2, R2G{n*}; 실패하면 G1 nlive 800 × 5회(수렴해도 G2/G3은 미룸).
5. **선택(12.4)**: 6단계 M3, E1, E2 실행에 C4를 더하고, 7단계 표(래더 B가 있으면 그것)로 G 모형 행을 바꾼 뒤 11.3 규칙으로 중심 모형, 11.4 규칙으로 σ_model을 다시 계산한다. 중심 모형이 E1이 아니면 예산은 잠정으로 표시한다.
