"""
OIBC 2026 실행용 스크립트 (모델 설정 실험 모드 포함)
- oibc2026_tester.py 와 같은 폴더에 두고:  python run.py
- 데이터는 이 파일이 있는 폴더(sites.csv, smp.csv, train/, test/)를 기준으로 읽습니다.
- 실험 모드는 out 폴더의 제출 파일을 건드리지 않습니다 (RUN_SUBMIT 이 False 인 한 덮어쓰기 없음).
"""
import importlib.util
import os

import pandas as pd

import oibc2026_tester as T

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE
COMPOSITION_MODE = "opt"            # 제출 구성: "opt"(최적화 구성, 기본) / "ref"(원래 구성 current_group 그대로 → 구성 최적화 효과 확인용 A/B 제출)
SA_SEEDS = tuple(range(10, 22))     # 구성 후보 탐색 시드 (후보 만들기용). 이미 쓴 시드(0~4)와 겹치지 않게 둠
N_KEEP = 8                          # 만들 후보 개수 (학습 비용이 낮은 순서, v00 제외)
SUBMIT_SEEDS = (0, 1, 2)            # 제출용 시드 (2차 제출본과 같은 값 → 합계 예측이 2차와 동일해져 깨끗한 비교가 됨)
OUT_DIR = os.path.join(BASE, "out" if COMPOSITION_MODE == "opt" else "out_" + COMPOSITION_MODE)   # 모드별로 폴더를 분리해 기존 out 을 덮어쓰지 않음

START_MONTH = 6                     # 6월~다음해 5월 (평가 구간과 같은 구조)
YEAR_PAIRS = [([2022, 2023], 2024)]         # 연도 검증용 (학습 연도, 평가 연도)
FIT, EVAL = 2023, 2024              # 실험/진단 모드에서 쓰는 한 쌍
BD_FIT, BD_EVAL = [2022, 2023], 2024  # 페널티 분해 진단용: 제출과 같은 2시즌 학습 → 2024 시즌 평가
SITE_CF_MODE = "causal"             # 발전소별 평균 이용률 계산: "causal"(각 대상일 전날 17:00 까지의 실적만 사용, 규정 준수형) / "static"(학습 전체 평균, 기존 제출본과 동일)
ENSEMBLE = None                     # 팀 모델 앙상블: None(기존 단일 모델). 진단 결과가 좋을 때만 예) {"all_l1": 0.5, "A_l1": 0.25, "B_l1": 0.25}
ENS_CHECK_MEMBERS = None            # 앙상블 진단에서 학습할 멤버 (None=전부: all_l1, A_l1, B_l1, all_l2). 시간을 줄이려면 ["all_l1", "A_l1", "B_l1"]
BIAS_MODE = "peak"                  # 예측 편향 보정: "none"(끄기) / "global" / "peak"(권장) / "hour"

# 실행 단계 스위치
RUN_INSPECT = False                 # 파일/컬럼/연도별 행 수 확인
RUN_SELFTEST = False                # W 계산 검증
RUN_TESTER = True                   # 연도 검증 (보정 전 W_opt 와 보정 후 W_opt_bc 를 함께 출력)
RUN_CLOUDBIN = False              # 구름 구간별 보정 진단 (현재 2계수 보정 위에 구름 예보 수준별 계수를 얹으면 나아지는지)
RUN_BREAKDOWN = False              # 페널티 분해 진단 (페널티가 어느 달/시각/날/날씨에 몰려 있는지)
RUN_COND_CHECK = False             # 조건부 편향 진단 (예보 상황별 보정이 현재 2계수 보정보다 나은지, 평가 연도 검증)
RUN_BIAS_CHECK = False             # 편향 진단 (집합 합계 예측이 시간대별로 치우쳤는지 + 곱셈 보정 효과 검증)
RUN_EXPERIMENTS = False             # 모델 설정 실험 (아래 EXPERIMENTS 를 순서대로 실행)
RUN_ENSEMBLE_CHECK = False          # 앙상블 진단 (구조가 다른 모델들의 평균이 단일 팀 모델보다 나은지) — 오래 걸림(멤버 수 × 학습)
RUN_MISSING = False                 # 평가 구간 예보 결손 점검 (결손 규모, 채움 방식, 페널티 가중 노출도)
RUN_VARIANTS = False                # 구성 후보 여러 개 생성 (out_var/v00, v01, ... 폴더) — 리더보드 점수로 최고 구성 고르기용
RUN_SUBMIT = False                  # 제출 파일 생성 (out 폴더를 덮어씀! 먼저 out 을 백업하세요)

# ----------------------------------------------------------------------
# 모델 설정 실험 목록
#  - 각 항목은 '기본 설정'에서 바뀌는 것만 적습니다. (기본값: lr 0.06 / leaves 63 / min_child 100 /
#    subsample 0.8 / colsample 0.7 / 나무 400개)
#  - lgb=LightGBM 설정, n_estimators=나무 수, seeds=시드(생략하면 기본 시드)
#  - 실험 1개 = tester 1회 실행 시간이 걸립니다. 처음에는 앞의 3개만 해 보세요.
# ----------------------------------------------------------------------
EXPERIMENTS = [
    dict(name="기준 (현재 설정)"),
    dict(name="기준·시드1 (흔들림 크기 확인)", seeds=(1,)),
    dict(name="leaves 31 (단순하게)", lgb=dict(num_leaves=31)),
    dict(name="leaves 127 + min_child 200", lgb=dict(num_leaves=127, min_child_samples=200)),
    dict(name="min_child 300 (보수적으로)", lgb=dict(min_child_samples=300)),
    dict(name="lr 0.03 + 나무 800개", lgb=dict(learning_rate=0.03), n_estimators=800),
]


def run_experiments():
    """EXPERIMENTS 를 차례로 tester 로 평가하고 기준 대비 차이표를 출력/저장."""
    if importlib.util.find_spec("lightgbm") is None:
        print("[중단] lightgbm 이 설치되어 있지 않습니다. 설정 실험은 LightGBM 에서만 의미가 있습니다.\n"
              "       python -m pip install lightgbm==4.6.0")
        return None
    base_lgb = dict(T.CFG["lgb_params"])
    base_n = T.CFG["n_estimators"]
    rows = []
    try:
        for i, ex in enumerate(EXPERIMENTS, 1):
            T.CFG["lgb_params"] = {**base_lgb, **ex.get("lgb", {})}
            T.CFG["n_estimators"] = ex.get("n_estimators", base_n)
            print(f"\n===== 실험 {i}/{len(EXPERIMENTS)}: {ex['name']} =====")
            print(f"  lgb={T.CFG['lgb_params']}, 나무 {T.CFG['n_estimators']}개, 시드 {ex.get('seeds') or T.CFG['seeds']}")
            r = T.tester(FIT, EVAL, start_month=START_MONTH, seeds=ex.get("seeds"), verbose=False)
            if T.CFG["backend"] != "lightgbm":
                print("[중단] LightGBM 이 scikit-learn 으로 자동 전환되어 설정이 적용되지 않았습니다. 이후 결과는 무의미합니다.")
                break
            rows.append(dict(실험=ex["name"], W_opt=r["W_opt"], score_model=r["score_model"],
                             score_comp=r["score_comp"], score_total=r["score_total"]))
            print(f"  → W_opt {r['W_opt']:,.0f}  (낮을수록 좋음)")
    finally:
        T.CFG["lgb_params"], T.CFG["n_estimators"] = base_lgb, base_n

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # 비교 지표는 W_opt(절대 페널티, 낮을수록 좋음)만 사용합니다.
    # score_* 는 '기준 모델'도 같은 설정을 쓰기 때문에 설정을 바꾸면 기준선 자체가 같이 변해서 비교에 쓰면 안 됩니다.
    df["ΔW_opt(원)"] = df["W_opt"] - df["W_opt"].iloc[0]
    df["ΔW_opt(%)"] = (df["W_opt"] / df["W_opt"].iloc[0] - 1) * 100
    show = df[["실험", "W_opt", "ΔW_opt(원)", "ΔW_opt(%)"]].copy()
    show["W_opt"] = show["W_opt"].map("{:,.0f}".format)
    show["ΔW_opt(원)"] = show["ΔW_opt(원)"].map("{:+,.0f}".format)
    show["ΔW_opt(%)"] = show["ΔW_opt(%)"].map("{:+.3f}".format)
    print("\n" + "=" * 18 + " 실험 결과 (첫 줄 = 기준, W_opt 가 낮을수록 좋음) " + "=" * 18)
    print(show.to_string(index=False))
    print("※ '기준·시드1' 과 기준의 차이가 이 실험들의 흔들림(노이즈) 크기입니다. 그보다 작은 개선은 믿지 마세요.")
    log = os.path.join(BASE, "experiments_log.csv")
    df.to_csv(log, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 → {log}")
    return df


if __name__ == "__main__":
    T.CFG["data_dir"] = DATA_DIR
    T.CFG["out_dir"] = OUT_DIR
    T.CFG["bias_mode"] = BIAS_MODE
    T.CFG["composition_mode"] = COMPOSITION_MODE
    T.CFG["ensemble"] = ENSEMBLE
    T.CFG["site_cf_mode"] = SITE_CF_MODE
    # T.CFG["backend"] = "sklearn"    # LightGBM 오류가 나면 주석 해제 (설정 실험은 불가)

    if RUN_INSPECT:
        T.inspect()

    if RUN_SELFTEST:
        T.selftest()

    if RUN_TESTER:
        for fit_y, eval_y in YEAR_PAIRS:
            T.tester(fit_y, eval_y, start_month=START_MONTH)   # W_opt(보정 전) / W_opt_bc(보정 후) 비교

    if RUN_CLOUDBIN:
        T.cloud_bin_check(BD_FIT, BD_EVAL, start_month=START_MONTH)

    if RUN_BREAKDOWN:
        T.penalty_breakdown(BD_FIT, BD_EVAL, start_month=START_MONTH)

    if RUN_COND_CHECK:
        T.cond_bias_check(FIT, EVAL, start_month=START_MONTH)

    if RUN_BIAS_CHECK:
        T.bias_check(FIT, EVAL, start_month=START_MONTH)

    if RUN_EXPERIMENTS:
        run_experiments()

    if RUN_ENSEMBLE_CHECK:
        T.ensemble_check(BD_FIT, BD_EVAL, start_month=START_MONTH, members=ENS_CHECK_MEMBERS)

    if RUN_MISSING:
        T.missing_forecast_report()

    if RUN_VARIANTS:
        T.make_submission_variants(pred_seeds=SUBMIT_SEEDS, sa_seeds=SA_SEEDS, n_keep=N_KEEP,
                                   out_root=os.path.join(BASE, "out_var"))

    if RUN_SUBMIT:
        T.make_submission(seeds=SUBMIT_SEEDS)
