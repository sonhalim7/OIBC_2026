# OIBC 2026 – 집합자원 구성 최적화 + 하루전 발전량 예측

## 1. 개요
1. **발전량 예측**: LightGBM(L1 손실, SMP×k 가중)으로 발전소별 시간대 이용률(발전량/설비용량)을 예측한 뒤 설비용량을 곱해 kWh 로 환산
2. **집합자원 구성**: 학습 구간의 주(week) 블록 교차검증 OOF 오차를 이용해, 두 집합의 `Σ SMP·k·|집합 오차 합|` 을 최소화하도록 시뮬레이티드 어닐링으로 253개 발전소를 G1/G2 에 배분 (집합별 설비용량 20,000 kW 이상 제약 유지)

## 2. 사용 데이터 / 출처
- **대회 제공 데이터만 사용**: `sites.csv`, `smp.csv`, `train/*`, `test/forecast_wa|wb`
- 외부 데이터·외부 파일 다운로드 없음. 태양 위치(고도·시간각 등)는 공개된 수식(Spencer 1971 계열 적분 급수식)을 코드(`solar_features`)에서 직접 계산
- `observed_wa/wb`(관측)는 피처로 사용하지 않음 (평가 구간에 존재하지 않는 정보)
- 외부 AI 서비스에 대회 데이터를 입력하지 않음 → **[작성자 확인 필요: 사실과 다르면 수정]**

## 2-1. 예측 편향 보정
- 모델 예측이 시간대별로 한쪽으로 치우친 것(낮 시간대 과대, 아침·저녁 과소)을 곱셈 계수로 보정 (`CFG["bias_mode"]`, 기본 `"peak"`: 피크(k=3)/비피크 2개 계수)
- 계수는 **학습 구간의 교차검증(OOF) 예측과 실적**으로만 구함 (페널티 Σ SMP·k·|집합 합계 오차| 를 최소화하는 가중중앙값). 평가 구간 정보는 사용하지 않음
- `"none"` 으로 끄면 보정 없는 결과와 동일

## 3. 누수 방지 설계
| 점검 항목 | 처리 |
|---|---|
| 대상일 실제 SMP 사용 | 예측 피처에 SMP 없음. SMP×k 는 **학습 행의 sample_weight** 와 구성 최적화(학습 구간 오차 가중)에만 사용. 평가/제출 프레임에는 SMP 를 붙이지 않음(`make_frame(..., with_w=False)`) |
| 평가 구간 통계 사용 | 결측 대체 중앙값은 **학습 구간**에서만 계산해 평가·제출 프레임에 전달(`make_frame(..., med=...)`). 평가 구간 전체 평균/표준화·그룹 분할 없음, 모든 행을 독립 예측 |
| 미래 값 사용 | 예보 결측 보간은 같은 지점의 **과거 값 ffill(최대 6행)** 만 사용 |
| 직전 실적 입력 | 실적 시차(lag) 피처 없음. 실적 기반 피처는 학습 구간의 발전소별 평균 이용률(`site_cf`) 하나뿐 |

## 4. 환경 / 설치
- Python 3.10 이상 권장
- `pip install -r requirements.txt`
- (편의) 코드 내 자동 설치는 **기본 OFF**. 필요할 때만 `OIBC_AUTO_INSTALL=1` 환경변수로 켬
- 제출 시 `out/environment.txt` 에 Python/패키지 버전과 핵심 설정이 기록됨

## 5. 실행 절차
```bash
python oibc2026_tester.py inspect  --data-dir data      # 파일/컬럼/연도별 행 수 확인
python oibc2026_tester.py selftest                      # W 계산 검증
python oibc2026_tester.py test --fit 2022 --eval 2023 --data-dir data   # 연도 검증 (여러 쌍 반복)
python oibc2026_tester.py submit --data-dir data        # out/composition.csv, out/forecast.csv, out/environment.txt 생성 + 형식 점검
python oibc2026_tester.py check                         # 형식 점검만 재실행
```
노트북(Colab)에서는 `CFG["data_dir"]="/content/data"` 설정 후 `inspect() → selftest() → year_grid([...]) → make_submission()` 순서로 실행.

## 6. 재현성
- 시드: 최종 예측 `(0, 1, 2)` 평균, 구성 최적화 시드 동일. LightGBM `deterministic=True`, `force_row_wise=True`, `n_jobs` 고정(`CFG["n_jobs"]`)
- 제출 학습 구간: `CFG["submit_train_from"]` = 2023-06-01 이후
- 출력: `composition.csv`(site_id, group), `forecast.csv`(group, ts, forecast_kwh; 2집합 × 8,760시간 = 17,520행)

## 8. 외부 소스 (코드 패키지)
외부 데이터·사전학습 모델은 사용하지 않았고, 아래 오픈소스 패키지만 사용했다. 인증키는 코드에 없다.
최종 제출 환경(`environment.txt`)의 버전을 고정한다 (Python 3.11.4).

| 패키지 | 버전 | 받는 주소 (공개 시각 근거 = 해당 버전 릴리스 페이지의 업로드 일자) | 라이선스 |
|---|---|---|---|
| numpy | 2.4.6 | https://pypi.org/project/numpy/2.4.6/ | BSD-3-Clause |
| pandas | 3.0.6 | https://pypi.org/project/pandas/3.0.6/ | BSD-3-Clause |
| pyarrow | 26.0.0 | https://pypi.org/project/pyarrow/26.0.0/ | Apache-2.0 |
| lightgbm | 4.6.0 | https://pypi.org/project/lightgbm/4.6.0/ | MIT |
| scikit-learn | 1.9.1 | https://pypi.org/project/scikit-learn/1.9.1/ | BSD-3-Clause |

설치: `python -m pip install numpy==2.4.6 pandas==3.0.6 pyarrow==26.0.0 lightgbm==4.6.0 scikit-learn==1.9.1`
태양 위치(고도·시간각 등)는 제공된 위경도와 공개된 수식으로 `solar_features` 에서 직접 계산한다.

## 9. 최종 제출 재현 절차  (※ 대괄호는 작성자가 기입)
1. 위 환경을 설치하고, 대회 데이터를 `run.py` 와 같은 폴더에 둔다 (`sites.csv`, `smp.csv`, `train/`, `test/`).
2. `run.py` 설정: `SITE_CF_MODE = "[최종 제출에 쓴 값: causal 또는 static]"`, `ENSEMBLE = [None 또는 사용한 비중]`, `BIAS_MODE = "peak"`,
   `SUBMIT_SEEDS = (0, 1, 2)`, `SA_SEEDS = tuple(range(10, 22))`, `N_KEEP = 8`, `RUN_VARIANTS = True` (그 외 스위치 False).
3. `python run.py` → `out_var/v00 … v08` 생성. **최종 제출본 = `out_var/[후보 폴더명]`** (`variants_summary.csv` 의 '출처' 칸: [탐색 시드 기입]).
4. 같은 설정으로 다시 실행했을 때 후보 폴더의 `composition.csv`·`forecast.csv` 가 제출본과 같은지 확인한다 (`check_variants.py` 의 재현 점검 사용).
5. 예측 시점 처리: 대상일 전날 17:00 까지 알 수 있는 정보만 사용한다. 예보는 전날 10시·17시 발행분, SMP 는 학습 구간 표본 가중치로만 사용, `site_cf` 는 `causal` 모드에서 각 대상일의 예측시점(ts 17:00)까지의 실적만 사용한다.
