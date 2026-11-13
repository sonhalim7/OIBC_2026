"""
OIBC 2026  W 테스터 + 파이프라인 (v5)

■ 목적함수 (온라인 설명회 슬라이드 기준)
    W = Σ_t SMP_t · k_t · |Ŷ(g,t) − A(g,t)|    (G1, G2 각각 '집합 합계'로 계산 후 합산, 집합 간 상쇄 없음)
    k = 3 (ts 라벨 시각 11~15시 = 실제 10~15시), 그 외 1
    정규화 점수 = 100 × (1 − W팀 / W기준선)

■ 테스터: tester(fit_years, eval_years)
    fit_years 데이터로 모델 학습 + 집합자원 구성 최적화 → eval_years 에서 W 를 채점
    (연도만 바꿔서 반복 실행. 예: tester(2022, 2023), tester([2021, 2022], 2024))
    출력 W 3종:  W_base  기준선형(L2·무가중·참조구성)
                 W_ref   팀 모델 + 참조구성
                 W_opt   팀 모델 + 최적화 구성    → 점수 = 100×(1−W/W_base)

■ 제출: make_submission()  →  out/composition.csv, out/forecast.csv  (+ 자동 형식 점검)
        check_submission()  →  무효 조건 6가지 직접 점검 (검증 도구가 없으므로 필수)

■ Colab / Jupyter 사용 예 (이 파일을 셀에 붙여 실행한 뒤)
    CFG["data_dir"] = "/content/data"
    inspect()                          # 파일/컬럼/연도별 행 수 확인 (먼저!)
    selftest()                         # W 계산 검증 (슬라이드 예시)
    r = tester(2022, 2023)             # 연도 2개 비교
    year_grid([(2021, 2022), (2022, 2023), (2023, 2024)])
    make_submission(); check_submission()

■ v5 변경점 (규정 점검 반영)
    - 결측 대체(중앙값)를 '학습 구간 통계'로만 계산 → 평가 구간 전체 통계 사용 제거
    - 예보 결측 보간을 '과거 방향(ffill)'으로만 수행 → 미래 시각 값으로 채우지 않음
    - 평가(예측) 프레임에는 SMP(_w)를 붙이지 않음 (SMP는 학습 가중치/채점에서만 사용)
    - 패키지 자동 설치는 기본 OFF (환경변수 OIBC_AUTO_INSTALL=1 일 때만 동작)
    - LightGBM 재현성 설정(deterministic, n_jobs 고정), 제출 시 out/environment.txt 기록

■ 데이터 폴더 구조 (슬라이드 기준)
    data/sites.csv, data/smp.csv
    data/train/{generation, forecast_wa, forecast_wb, observed_wa, observed_wb}.parquet, data/train/smp.csv
    data/test/{forecast_wa, forecast_wb}.parquet
    (폴더 없이 한 곳에 몰아 올려도 train 파일은 자동 탐색)
"""
import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from importlib import metadata

# ----------------------------------------------------------------------
# [자동 설치] 기본 OFF. 재현 환경을 바꾸지 않기 위해 requirements.txt 로 설치하는 것을 권장.
#   Colab 등에서 편의상 필요하면 실행 전에  OIBC_AUTO_INSTALL=1  환경변수를 설정.
#   (모듈 이름, pip 패키지 이름, 고정 버전 또는 None)
# ----------------------------------------------------------------------
REQUIRED = [
    ("numpy", "numpy", None),
    ("pandas", "pandas", None),
    ("pyarrow", "pyarrow", None),
    ("lightgbm", "lightgbm", "4.6.0"),
    ("sklearn", "scikit-learn", None),      # backend="sklearn" 대체 경로용
]


def _ensure_packages():
    to_install = []
    for mod, pkg, ver in REQUIRED:
        if importlib.util.find_spec(mod) is None:
            to_install.append(f"{pkg}=={ver}" if ver else pkg)
        elif ver:
            try:
                if metadata.version(pkg) != ver:
                    to_install.append(f"{pkg}=={ver}")
            except metadata.PackageNotFoundError:
                pass
    if not to_install:
        return
    print(f"[auto-install] 설치 중: {' '.join(to_install)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *to_install])
    except subprocess.CalledProcessError as e:
        sys.exit(f"[auto-install] 설치 실패 ({e}). 네트워크/권한을 확인하거나 수동으로 설치하세요:\n"
                 f"  {sys.executable} -m pip install {' '.join(to_install)}")
    importlib.invalidate_caches()
    print("[auto-install] 완료")


if os.environ.get("OIBC_AUTO_INSTALL") == "1":
    _ensure_packages()
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# [CONFIG]
# ----------------------------------------------------------------------
CFG = dict(
    data_dir="data",
    out_dir="out",
    files=dict(
        sites="sites.csv",
        smp_eval="smp.csv",                      # 채점 구간(2025-06~2026-05) 시각 목록 + SMP + k
        smp_train="train/smp.csv",               # 학습 구간 SMP
        gen="train/generation.parquet",
        fc_a="train/forecast_wa.parquet",
        fc_b="train/forecast_wb.parquet",
        obs_a="train/observed_wa.parquet",       # inspect 전용 (피처 미사용)
        obs_b="train/observed_wb.parquet",
        test_fc_a="test/forecast_wa.parquet",    # 평가 구간 예보 (제출용)
        test_fc_b="test/forecast_wb.parquet",
    ),
    # 표준 컬럼명 ← 실제 파일의 후보 이름들 (대소문자 무시, 앞에서부터 일치하는 것 사용)
    aliases=dict(
        site=["site_id", "plant_id", "site", "발전소"],
        ts=["ts", "datetime", "timestamp", "time", "date_time", "target_time", "valid_time", "dt"],
        gen=["generation", "generation_kwh", "gen_kwh", "gen", "kwh", "power", "energy", "발전량"],
        cap=["capacity_kw", "capacity", "cap_kw", "cap", "설비용량"],
        lat=["lat", "latitude"], lon=["lon", "lng", "longitude"],
        cur=["current_group", "ref_group"],
        wa=["wa_station"], wb=["wb_station"],
        smp=["smp", "smp_price", "price"],
        k=["k", "k_weight", "weight_k", "weight"],
        rnd=["round", "fc_round", "run", "cycle"],
        issued=["issued_at", "issue_time", "base_time", "init_time"],
        st=["station", "station_id", "stn", "stn_id", "point", "point_id", "obs_station", "지점"],
    ),
    min_cap_kw=20_000.0,        # 집합별 최소 설비용량 (슬라이드: 미달이면 제출 무효)
    start_month=1,              # 연도 구분 시작 월: 1=달력연도, 6=6월~다음해 5월 (평가 구간과 동일 구조)
    k_peak_hours=(11, 12, 13, 14, 15), k_peak=3.0,   # ts 라벨 기준
    # 모델
    models=dict(
        base=dict(objective="regression", weight=False),       # 기준선형
        team=dict(objective="regression_l1", weight=True),     # 팀 모델: L1 + (SMP·k) 가중
    ),
    n_estimators=400,
    # LightGBM 하이퍼파라미터 (기본값 = 기존 제출 코드와 동일). run.py 의 실험 모드에서 바꿔 가며 비교
    lgb_params=dict(learning_rate=0.06, num_leaves=63, subsample=0.8, colsample_bytree=0.7, min_child_samples=100),
    site_cf_mode="static",      # 발전소별 평균 이용률(site_cf): "static"(학습 전체 평균, 기존) / "causal"(각 대상일의 예측시점=전날 17:00 까지의 실적만 사용 — 규정 준수형)
    site_cf_min_rows=200,       # causal 모드에서 이력이 이만큼 쌓이기 전 행은 site_cf 를 결측(NaN)으로 둠
    ensemble=None,              # 팀 모델 앙상블: None(기존 단일 팀 모델) 또는 {"all_l1": 0.5, "A_l1": 0.25, "B_l1": 0.25} 같은 {멤버: 비중}
    composition_mode="opt",     # 제출 구성: "opt"(최적화 구성, 기본) / "ref"(원래 구성 current_group 그대로 — 구성 최적화 효과 확인용 A/B)
    bias_mode="peak",           # 예측 편향 보정: "none"(끄기) / "global"(계수 1개) / "peak"(피크·비피크 2개) / "hour"(시각별)
    n_jobs=4,                   # 스레드 수 고정 (재현성). 환경에 맞게 바꿔도 deterministic=True 라 결과는 같아야 함
    backend="lightgbm",         # lightgbm 이 오류나면 "sklearn" (느리지만 안전)
    cv_folds=3,                 # 주(week) 블록 교차검증 → OOF 오차로 구성 최적화
    opt_iters=200_000,
    seeds=(0,),                 # 최종 예측 시드 (제출 시 (0,1,2) 권장)
    submit_train_from="2023-06-01",   # 제출용 학습 시작일 (기준선과 동일: 최근 2년). None=전체(메모리 주의)
)
C = dict(site="site_id", ts="ts", gen="generation", cap="capacity_kw", lat="lat", lon="lon",
         cur="current_group", wa="wa_station", wb="wb_station", smp="smp", k="k",
         rnd="round", issued="issued_at", st="station")
HINTS = dict(gen=("gen", "kwh"), smp=("smp",), cap=("cap",))


# ----------------------------------------------------------------------
# 입출력 / 정규화
# ----------------------------------------------------------------------
def _find(name, flat_ok=True):
    cands = [os.path.join(CFG["data_dir"], name)]
    if flat_ok:
        cands.append(os.path.join(CFG["data_dir"], os.path.basename(name)))
    for q in cands:
        root, _ = os.path.splitext(q)
        for alt in (q, root + ".parquet", root + ".csv"):
            if os.path.exists(alt):
                return alt
    return None


def read_table(name, flat_ok=True, required=True):
    p = _find(name, flat_ok)
    if p is None:
        if required:
            raise FileNotFoundError(f"'{name}' 을(를) '{CFG['data_dir']}' 아래에서 찾지 못함 → CFG['data_dir'] 확인")
        return None
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def _to_kst(s):
    """시각 → KST naive datetime (우측닫힘 라벨 그대로 유지)"""
    try:
        t = pd.to_datetime(s)
    except (ValueError, TypeError):
        t = pd.to_datetime(s, utc=True)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return t.astype("datetime64[ns]")      # 파일마다 ms/us 단위가 달라 ns 로 통일 (매칭 오류 방지)


def _key(s):
    """관측지점 키를 문자열로 통일 (정수/문자 혼재 방지)"""
    try:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype("Int64").astype(str)
    except (TypeError, ValueError):
        pass
    return s.astype(str).str.strip()


def normalize(df, hints=True):
    df = df.copy()
    if C["ts"] not in [c for c in df.columns] and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": C["ts"]})
    low = {c.lower(): c for c in df.columns}
    for std, cands in CFG["aliases"].items():
        tgt = C[std]
        if tgt in df.columns:
            continue
        hit = next((low[c.lower()] for c in cands if c.lower() in low), None)
        if hit is None and hints and std in HINTS:
            m = [c for c in df.columns if any(h in c.lower() for h in HINTS[std]) and c not in C.values()]
            hit = m[0] if len(m) == 1 else None
        if hit is not None:
            df = df.rename(columns={hit: tgt})
    if C["ts"] in df.columns:
        df[C["ts"]] = _to_kst(df[C["ts"]])
    return df


def inspect(show_rows=False):
    """모든 파일의 스키마·인식 결과·연도별 행 수·결측률을 출력.
    원본 데이터 행은 기본적으로 출력하지 않음 (대회 규정: 원본 행/파일을 외부 AI 서비스에 올리는 것은 금지 → 화면 캡처 공유 시 주의).
    show_rows=True 는 본인 확인용으로만 쓰고 그 출력은 외부에 공유하지 말 것."""
    for key, name in CFG["files"].items():
        flat = not key.startswith("test") and key != "smp_eval"
        p = _find(name, flat_ok=flat if key not in ("smp_train",) else False)
        if p is None:
            print(f"[{key}] {name} 없음\n")
            continue
        raw = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        df = normalize(raw, hints=key in ("gen", "sites", "smp_eval", "smp_train"))
        print(f"[{key}] {p}  ({len(raw):,}행)")
        print(df.dtypes.to_string())
        print("  인식된 표준 컬럼:", {k: v for k, v in C.items() if v in df.columns})
        if C["ts"] in df.columns:
            print("  연도별 행 수:", df[C["ts"]].dt.year.value_counts().sort_index().to_dict())
        na = (df.isna().mean() * 100).round(3)
        na = na[na > 0]
        print("  결측률(%):", na.to_dict() if len(na) else "없음")
        if show_rows:
            print(df.head(2).to_string())
        print()
    g = read_table(CFG["files"]["gen"], required=False)
    if g is not None:
        g = normalize(g)
        if C["site"] in g.columns and C["ts"] in g.columns:
            print("[generation] 연도별 발전소 수:",
                  g.groupby(g[C["ts"]].dt.year)[C["site"]].nunique().to_dict())


# ----------------------------------------------------------------------
# 데이터 로딩 (1회 캐시)
# ----------------------------------------------------------------------
_D = {}


def load_smp():
    parts = []
    for key, flat in (("smp_train", False), ("smp_eval", False)):
        d = read_table(CFG["files"][key], flat_ok=flat, required=False)
        if d is not None:
            parts.append(normalize(d))
    if not parts:
        raise FileNotFoundError("smp.csv / train/smp.csv 를 찾지 못함")
    d = pd.concat(parts, ignore_index=True)
    d = d.drop_duplicates(C["ts"], keep="last").sort_values(C["ts"]).set_index(C["ts"])
    rule = np.where(d.index.hour.isin(CFG["k_peak_hours"]), CFG["k_peak"], 1.0)   # ts 라벨 기준
    if C["k"] in d.columns:
        k = d[C["k"]].astype(float)
        bad = (k.notna() & (k != rule)).mean()
        if bad > 0:
            print(f"[경고] smp 파일의 k 가 규칙(11~15시=3)과 {bad:.1%} 불일치 → 파일 값을 우선 사용")
        d["k"] = k.fillna(pd.Series(rule, index=d.index))
    else:
        d["k"] = rule
    d["smp"] = d[C["smp"]].astype(float)
    full = pd.date_range(d.index.min(), d.index.max(), freq="h")
    d = d[["smp", "k"]].reindex(full)
    d["smp"] = d["smp"].ffill().bfill()
    d["k"] = d["k"].fillna(pd.Series(np.where(full.hour.isin(CFG["k_peak_hours"]), CFG["k_peak"], 1.0), index=full))
    d["w"] = d["smp"] * d["k"]
    return d


def load_fc(name, flat_ok=True):
    """기상 예보 1개 제공사 → (long 테이블[st, ts, _r, vars...], 변수 목록)"""
    fc = normalize(read_table(name, flat_ok), hints=False)
    if C["st"] not in fc.columns:
        cand = [c for c in fc.columns if c not in (C["ts"], C["rnd"], C["issued"])
                and not pd.api.types.is_numeric_dtype(fc[c]) and not pd.api.types.is_datetime64_any_dtype(fc[c])]
        if len(cand) != 1:
            raise KeyError(f"{name}: 관측지점 컬럼을 특정할 수 없음 → CFG['aliases']['st'] 에 추가 ({list(fc.columns)})")
        fc = fc.rename(columns={cand[0]: C["st"]})
    fc[C["st"]] = _key(fc[C["st"]])
    if C["rnd"] in fc.columns:
        fc["_r"] = _key(fc[C["rnd"]])
    elif C["issued"] in fc.columns:
        fc["_r"] = pd.to_datetime(fc[C["issued"]]).dt.hour.astype(str)
    else:
        fc["_r"] = "0"
    skip = {C["st"], C["ts"], "_r", C["rnd"], C["issued"], C["site"], C["lat"], C["lon"], C["cap"], C["gen"]}
    vars_ = [c for c in fc.columns if c not in skip and pd.api.types.is_numeric_dtype(fc[c])
             and not pd.api.types.is_bool_dtype(fc[c])]
    fc[vars_] = fc[vars_].astype("float32")
    return fc[[C["st"], C["ts"], "_r"] + vars_], vars_


def load_data(force=False):
    if _D and not force:
        return _D
    sites = normalize(read_table(CFG["files"]["sites"]))
    for need in ("site", "cap", "lat", "lon", "wa", "wb"):
        if C[need] not in sites.columns:
            raise KeyError(f"sites 에 '{C[need]}' 없음: {list(sites.columns)}")
    sites = sites.sort_values(C["site"]).reset_index(drop=True)
    for c in (C["wa"], C["wb"]):
        sites[c] = _key(sites[c])
    gen = normalize(read_table(CFG["files"]["gen"]))
    for need in ("site", "ts", "gen"):
        if C[need] not in gen.columns:
            raise KeyError(f"generation 에 '{C[need]}' 없음 → CFG['aliases'] 수정: {list(gen.columns)}")
    gen = gen[[C["site"], C["ts"], C["gen"]]].dropna()
    gen = gen[gen[C["site"]].isin(sites[C["site"]])].reset_index(drop=True)

    ref = None
    if C["cur"] in sites.columns:
        ref = sites[C["cur"]].astype(str).str.upper().map({"A": 0, "B": 1, "G1": 0, "G2": 1}).values
        if np.isnan(ref.astype(float)).any():
            ref = None
        else:
            ref = ref.astype(np.int8)
    _D.update(sites=sites, gen=gen, wtab=load_smp(), ref=ref,
              fc={"A": load_fc(CFG["files"]["fc_a"]), "B": load_fc(CFG["files"]["fc_b"])})
    print(f"로딩 완료: 발전소 {len(sites)} | 발전 행 {len(gen):,} | 기간 {gen[C['ts']].min()} ~ {gen[C['ts']].max()}")
    return _D


def season_of(ts, start_month):
    """우측닫힘 라벨 기준 연도 구분: 구간 시작시각(ts-1h)이 속한 연도(start_month 기준)"""
    s = ts - pd.Timedelta(hours=1)
    return s.dt.year - (s.dt.month < start_month).astype(int)


# ----------------------------------------------------------------------
# 피처
# ----------------------------------------------------------------------
def solar_features(ts, lat, lon):
    mid = ts - pd.Timedelta(minutes=30)            # 우측닫힘: 구간 중앙, KST=UTC+9
    doy = mid.dt.dayofyear.values
    hr = (mid.dt.hour + mid.dt.minute / 60).values
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                   - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hr * 60 + eq + 4 * np.asarray(lon) - 60 * 9
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(np.asarray(lat))
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    i0 = 1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    return pd.DataFrame({
        "cosz": cosz.astype("float32"), "etr": (i0 * np.clip(cosz, 0, None)).astype("float32"),
        "ha": ha.astype("float32"), "doy_sin": np.sin(2 * np.pi * doy / 365).astype("float32"),
        "doy_cos": np.cos(2 * np.pi * doy / 365).astype("float32"),
        "hour": mid.dt.hour.values.astype("int8"),
    }, index=ts.index)


def fc_features(fc, vars_, prov, hours, stations, med=None):
    """제공사 1개: 라운드 평균(avg) + 라운드 차이(d, 예보 변동성). 제공사끼리는 섞지 않음(단위계 상이)
    결측 처리는 평가 구간 전체 통계/미래 시각 값을 쓰지 않는다:
      - 라운드 간 대체(같은 시각의 다른 라운드 예보)
      - 같은 지점의 '과거' 값으로 최대 6행 ffill
      - 남은 결측은 학습 구간 중앙값(med). med=None 이면 이 호출 자체가 학습 프레임이므로 여기서 계산해 반환
    반환: (피처 DataFrame, 중앙값 Series)"""
    d = fc[fc[C["st"]].isin(stations) & fc[C["ts"]].isin(hours)]
    g = d.groupby([C["st"], C["ts"], "_r"])[vars_].mean()
    found = set(g.index.get_level_values(0))
    miss = [s for s in stations if s not in found]
    if miss:
        print(f"  [경고] 예보 {prov}: 지점 {len(miss)}개 없음 → 중앙값 대체 (예: {miss[:3]})")
    rounds = sorted(g.index.get_level_values("_r").unique())
    idx = pd.MultiIndex.from_product([stations, hours], names=[C["st"], C["ts"]])
    a = g.xs(rounds[0], level="_r").reindex(idx)
    b = g.xs(rounds[-1], level="_r").reindex(idx) if len(rounds) > 1 else a
    a, b = a.fillna(b), b.fillna(a)                 # 결손: 다른 라운드로 대체 (행 제외 금지)
    out = pd.concat([((a + b) / 2).add_suffix(f"__{prov}_avg"), (b - a).add_suffix(f"__{prov}_d")], axis=1)
    out = out.groupby(level=0).ffill(limit=6)       # 과거 방향으로만 채움 (미래 값 사용 안 함)
    if med is None:
        m = out.median()
    else:
        m = med.reindex(out.columns)
    return out.fillna(m).astype("float32"), m


def make_frame(base, fcs, med=None, with_w=True):
    """base: [site, ts(, generation)] → 낮 시간(cosz>0) 행에 피처 부착. '_row' = base 내 위치
    med    : 결측 대체용 중앙값(Series). None 이면 학습 프레임으로 보고 base 에서 계산 → 결과의 .attrs["med"] 에 저장.
             평가/제출 프레임은 반드시 학습 프레임의 .attrs["med"] 를 넘길 것 (평가 구간 통계 사용 금지).
    with_w : 학습 가중치용 SMP×k(_w) 부착 여부. 평가(예측) 프레임은 False → 예측 경로에 SMP 가 들어가지 않음."""
    D = load_data()
    sites = D["sites"]
    b = base.reset_index(drop=True).merge(
        sites[[C["site"], C["lat"], C["lon"], C["cap"], C["wa"], C["wb"]]], on=C["site"], how="left")
    b["cap_"] = b[C["cap"]].astype("float32")
    b = pd.concat([b, solar_features(b[C["ts"]], b[C["lat"]], b[C["lon"]])], axis=1)
    b["_row"] = np.arange(len(b))
    day = b[b["cosz"] > 0.0].copy()
    hours = pd.DatetimeIndex(np.sort(day[C["ts"]].unique()))
    feats, meds = [], []
    for prov, key in (("A", C["wa"]), ("B", C["wb"])):
        fc, vars_ = fcs[prov]
        w, m = fc_features(fc, vars_, prov, hours, sorted(day[key].unique()), med)
        w.index.names = [key, C["ts"]]
        day = day.merge(w.reset_index(), on=[key, C["ts"]], how="left")
        feats += list(w.columns)
        meds.append(m)
    med_all = pd.concat(meds) if med is None else med
    day[feats] = day[feats].fillna(med_all.reindex(feats))
    if with_w:                                       # 학습 프레임 전용
        wt = D["wtab"]["w"]
        ww = day[C["ts"]].map(wt)
        day["_w"] = ww.fillna(ww.median()).astype("float32")
    base_feats = ["cosz", "etr", "ha", "doy_sin", "doy_cos", "hour", C["lat"], C["lon"], "cap_"]
    out = day.reset_index(drop=True)
    out.attrs["med"] = med_all
    return out, base_feats + feats


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
def make_model(seed, objective):
    try:
        if CFG["backend"] != "lightgbm":
            raise ImportError
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective=objective, n_estimators=CFG["n_estimators"], subsample_freq=1,
                                 random_state=seed, verbose=-1, n_jobs=CFG["n_jobs"], deterministic=True,
                                 force_row_wise=True, **CFG["lgb_params"])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        loss = "absolute_error" if objective in ("regression_l1", "l1") else "squared_error"
        return HistGradientBoostingRegressor(loss=loss, max_iter=250, learning_rate=0.08,
                                             max_leaf_nodes=63, random_state=seed, early_stopping=False)


def _causal_site_cf(query, hist, min_rows=200):
    """query 각 행(대상일 D)의 site_cf = 같은 발전소의 hist 행 중 'D 의 예측시점(전날 17:00)까지 알 수 있는' 행의 평균 이용률.
    ts 는 그 시각에 끝나는 1시간: 전날 ts 17:00 실적까지 사용 가능, ts 18:00 부터는 불가 (그 실적은 D+1 부터 사용).
    자기 자신이나 미래 날짜의 실적은 절대 들어가지 않음."""
    st, ts = C["site"], C["ts"]
    day_of = lambda x: (x - pd.Timedelta(hours=1)).dt.normalize()
    h = hist[[st, ts, "cf"]].copy()
    lab_h = (h[ts] - pd.Timedelta(hours=1)).dt.hour + 1                    # 1..24 (시각 라벨)
    h["avail"] = day_of(h[ts]) + pd.to_timedelta(1 + (lab_h > 17).astype(int), unit="D")
    g = h.groupby([st, "avail"])["cf"].agg(s="sum", n="count").reset_index().sort_values([st, "avail"])
    g[["cs", "cn"]] = g.groupby(st)[["s", "n"]].cumsum()
    q = pd.DataFrame({st: query[st].to_numpy(), "day": day_of(query[ts]).to_numpy(), "_i": np.arange(len(query))}).sort_values("day")
    m = pd.merge_asof(q, g[[st, "avail", "cs", "cn"]].sort_values("avail"), left_on="day", right_on="avail", by=st, direction="backward")
    m = m.sort_values("_i")
    return (m["cs"] / m["cn"]).where(m["cn"] >= min_rows).to_numpy()


def with_site_cf(df, tr):
    """발전소별 평균 이용률. static: 학습 데이터 전체 평균(기존) / causal: 각 행의 예측시점까지의 실적만 사용"""
    if CFG.get("site_cf_mode", "static") == "causal":
        out = df.copy()
        out["site_cf"] = pd.Series(_causal_site_cf(df, tr, CFG["site_cf_min_rows"]), index=out.index).astype("float32")
        return out
    m = tr.groupby(C["site"])["cf"].mean()
    out = df.copy()
    out["site_cf"] = out[C["site"]].map(m).fillna(tr["cf"].mean()).astype("float32")
    return out


def fit_predict(tr, te, feats, spec, seed=0):
    """학습 → te 예측 (kWh)"""
    tr = with_site_cf(tr, tr)
    te = with_site_cf(te, tr)
    f = feats + ["site_cf"]
    # pandas 최신판 ↔ LightGBM 호환 문제(access violation) 방지: 순수 numpy 연속 배열로 변환
    X = np.ascontiguousarray(tr[f].to_numpy(dtype="float32", na_value=np.nan))
    y = np.ascontiguousarray(tr["cf"].to_numpy(dtype="float32", na_value=np.nan))
    sw = np.ascontiguousarray((tr["_w"] / tr["_w"].mean()).to_numpy(dtype="float32")) if spec["weight"] else None
    Xt = np.ascontiguousarray(te[f].to_numpy(dtype="float32", na_value=np.nan))
    m = make_model(seed, spec["objective"])
    try:
        m.fit(X, y, sample_weight=sw)
    except OSError as e:                      # 일부 Windows 환경의 LightGBM 크래시 → sklearn 으로 자동 전환
        print(f"  [경고] LightGBM 오류({e}) → scikit-learn 모델로 자동 전환 (더 느림)")
        CFG["backend"] = "sklearn"
        m = make_model(seed, spec["objective"])
        m.fit(X, y, sample_weight=sw)
    return np.clip(m.predict(Xt), 0, 1.3).astype("float32") * te["cap_"].to_numpy(dtype="float32")


def prep_train(day):
    day = day.copy()
    day["cf"] = (day[C["gen"]] / day["cap_"]).clip(0, 1.3).astype("float32")
    return day[day["cosz"] > 0.02].reset_index(drop=True)


def oof_errors(tr, feats, spec, K, seed=0):
    """주(week) 블록 CV 의 OOF 오차 = 예측 − 실적 (kWh)"""
    assert K >= 2
    week = ((tr[C["ts"]] - tr[C["ts"]].min()).dt.days // 7) % K
    oof = np.zeros(len(tr), dtype="float32")
    for k in range(K):
        m = (week == k).values
        oof[m] = fit_predict(tr[~m], tr[m], feats, spec, seed)
        mae = np.abs(oof[m] - tr.loc[m, C["gen"]].values).mean()
        print(f"    OOF fold {k}: MAE {mae:,.2f} kWh/행")
    return oof - tr[C["gen"]].values


# ----------------------------------------------------------------------
# 팀 모델 앙상블: 구조가 다른 모델(사용 예보·손실함수가 다른)을 평균 → 오차가 덜 겹치는 효과
# ----------------------------------------------------------------------
ENS_MEMBERS = {
    "all_l1": dict(feats="all", spec=None),    # 현재 팀 모델 (모든 예보 피처, L1 손실 + SMP·k 가중)
    "A_l1": dict(feats="A", spec=None),        # A사 예보만 사용
    "B_l1": dict(feats="B", spec=None),        # B사 예보만 사용
    "all_l2": dict(feats="all", spec=dict(objective="regression", weight=True)),   # 모든 예보, L2 손실 + 가중
}


def _member_spec(name):
    sp = ENS_MEMBERS[name]["spec"]
    return sp if sp is not None else CFG["models"]["team"]


def _select_feats(feats, kind):
    """kind: all / A / B  (A: B사 피처 제외, B: A사 피처 제외. 태양위치·발전소 정보 피처는 항상 포함)"""
    if kind == "all":
        return list(feats)
    drop = "__B_" if kind == "A" else "__A_"
    return [f for f in feats if drop not in f]


def _ens_shares(ens):
    unknown = [n for n in ens if n not in ENS_MEMBERS]
    if unknown:
        raise ValueError(f"알 수 없는 앙상블 멤버 {unknown} (가능: {list(ENS_MEMBERS)})")
    tot = float(sum(ens.values()))
    if tot <= 0:
        raise ValueError("앙상블 비중의 합이 0 입니다")
    return {n: v / tot for n, v in ens.items() if v > 0}


def ens_oof_errors(tr, feats, K):
    """팀 모델(또는 앙상블)의 OOF 오차 = 예측 − 실적. 앙상블이면 멤버별 OOF 오차의 비중 가중 평균"""
    ens = CFG.get("ensemble")
    if not ens:
        return oof_errors(tr, feats, CFG["models"]["team"], K)
    err = 0.0
    for name, sh in _ens_shares(ens).items():
        print(f"  [앙상블 멤버 {name}, 비중 {sh:.2f}] OOF")
        err = err + sh * oof_errors(tr, _select_feats(feats, ENS_MEMBERS[name]["feats"]), _member_spec(name), K)
    return np.asarray(err, dtype="float32")


def ens_predict(tr, te, feats, seeds):
    """팀 모델(또는 앙상블)의 te 예측(kWh): 시드 평균 → 멤버 비중 가중 평균"""
    ens = CFG.get("ensemble")
    if not ens:
        return np.mean([fit_predict(tr, te, feats, CFG["models"]["team"], seed=s) for s in seeds], axis=0)
    out = 0.0
    for name, sh in _ens_shares(ens).items():
        f = _select_feats(feats, ENS_MEMBERS[name]["feats"])
        out = out + sh * np.mean([fit_predict(tr, te, f, _member_spec(name), seed=s) for s in seeds], axis=0)
    return np.asarray(out, dtype="float32")


# ----------------------------------------------------------------------
# W 계산 (핵심)
# ----------------------------------------------------------------------
def compute_W(base, pred, assign, wtab):
    """
    base  : DataFrame [site_id, ts, generation(=A, kWh)]   pred: base 와 같은 길이의 예측(kWh)
    assign: Series  site_id → 0(G1)/1(G2)                  wtab: index=ts, 컬럼 smp, k
    W = Σ SMP·k·|Σ_{i∈g} Ŷ_i − Σ_{i∈g} A_i|   (집합·시각별 합계 후 절댓값)
    """
    d = pd.DataFrame({"g": base[C["site"]].map(assign).values, C["ts"]: base[C["ts"]].values,
                      "A": base[C["gen"]].values.astype("float64"), "P": np.asarray(pred, dtype="float64")})
    t = d.groupby(["g", C["ts"]], sort=False)[["A", "P"]].sum().reset_index()
    t = t.merge(wtab[["smp", "k"]], left_on=C["ts"], right_index=True, how="left")
    t["err"] = (t["P"] - t["A"]).abs()
    t["pen"] = t["smp"] * t["k"] * t["err"]
    W = float(t["pen"].sum())
    out = dict(W=W,
               W_G1=float(t.loc[t["g"] == 0, "pen"].sum()), W_G2=float(t.loc[t["g"] == 1, "pen"].sum()),
               peak_share=float(t.loc[t["k"] > 1, "pen"].sum() / W) if W > 0 else np.nan,
               abs_err=float(t["err"].sum()), penalty=0.10 * W, n_nan_smp=int(t["smp"].isna().sum()))
    return out, t


def floor_W(detail):
    """W 의 하한: Σ SMP·k·|전체 오차|.  |E1|+|E2| ≥ |E1+E2| 이므로 어떤 구성으로도 이 값 아래로는 못 내려감
    (E1+E2 = 253개 발전소 전체의 시각별 오차 → 구성과 무관). 구성 최적화로 줄일 수 있는 몫은 W − W_floor 뿐."""
    g = detail.groupby(C["ts"]).agg(A=("A", "sum"), P=("P", "sum"), smp=("smp", "first"), k=("k", "first"))
    return float((g["smp"] * g["k"] * (g["P"] - g["A"]).abs()).sum())


def selftest():
    """슬라이드 예시로 W 계산 검증: k=3, SMP 120, 실적 12,000 / 예측 11,500 → 페널티 18,000원"""
    w = pd.DataFrame({"smp": [120.0, 120.0], "k": [3.0, 1.0], "w": [360.0, 120.0]},
                     index=pd.to_datetime(["2025-07-01 13:00", "2025-07-01 08:00"]))
    mk = lambda sites, ts, a, p: (pd.DataFrame({C["site"]: sites, C["ts"]: pd.to_datetime(ts), C["gen"]: a}), p)
    # (1) 단일 시각 예시
    b, p = mk(["s1"], ["2025-07-01 13:00"], [12000.0], [11500.0])
    r, _ = compute_W(b, p, pd.Series({"s1": 0}), w)
    assert abs(r["W"] - 180_000) < 1e-6 and abs(r["penalty"] - 18_000) < 1e-6, r
    b, p = mk(["s1"], ["2025-07-01 08:00"], [12000.0], [11500.0])
    r, _ = compute_W(b, p, pd.Series({"s1": 0}), w)
    assert abs(r["penalty"] - 6_000) < 1e-6, r
    # (2) 같은 집합 → 반대 오차 상쇄 / 다른 집합 → 상쇄 없음
    b, p = mk(["s1", "s2"], ["2025-07-01 13:00"] * 2, [100.0, 100.0], [110.0, 90.0])
    same, _ = compute_W(b, p, pd.Series({"s1": 0, "s2": 0}), w)
    diff, _ = compute_W(b, p, pd.Series({"s1": 0, "s2": 1}), w)
    assert same["W"] == 0 and abs(diff["W"] - 360 * 20) < 1e-6, (same, diff)
    print("selftest OK  (예시 13시 k=3: 페널티 18,000원 / 08시 k=1: 6,000원 / 집합 간 상쇄 없음 확인)")


# ----------------------------------------------------------------------
# 집합자원 최적화 (SA: swap / move, 설비용량 20,000 kW 제약 유지)
# ----------------------------------------------------------------------
def optimize_groups(Et, caps, w, init, iters, seed, min_cap):
    """Et: (N, T) 오차(예측-실적). cost = Σ_t w_t Σ_g |Σ_{i∈g} Et|"""
    rng = np.random.default_rng(seed)
    N = Et.shape[0]
    keep = np.abs(Et).sum(0) > 0                       # 오차 없는 시각(야간 등) 제외 → 속도
    Et, w = np.ascontiguousarray(Et[:, keep]), w[keep]
    assign = init.copy()
    S = [Et[assign == g].sum(0) for g in (0, 1)]
    cg = [float(caps[assign == g].sum()) for g in (0, 1)]
    cost = float(w @ np.abs(S[0]) + w @ np.abs(S[1]))
    best, best_assign = cost, assign.copy()
    t0 = cost * 0.002 + 1e-9
    for it in range(iters):
        temp = max(t0 * (1 - it / iters), 1e-12)
        if rng.random() < 0.7:                                       # swap
            i = rng.choice(np.flatnonzero(assign == 0))
            j = rng.choice(np.flatnonzero(assign == 1))
            c0 = cg[0] - caps[i] + caps[j]
            c1 = cg[1] + caps[i] - caps[j]
            if c0 < min_cap or c1 < min_cap:
                continue
            S0 = S[0] - Et[i] + Et[j]
            S1 = S[1] + Et[i] - Et[j]
            new = float(w @ np.abs(S0) + w @ np.abs(S1))
            if new < cost or rng.random() < np.exp(-(new - cost) / temp):
                assign[i], assign[j] = 1, 0
                S, cg, cost = [S0, S1], [c0, c1], new
        else:                                                        # move
            i = int(rng.integers(N))
            a, b = int(assign[i]), 1 - int(assign[i])
            ca, cb = cg[a] - caps[i], cg[b] + caps[i]
            if ca < min_cap:
                continue
            Sa, Sb = S[a] - Et[i], S[b] + Et[i]
            new = float(w @ np.abs(Sa) + w @ np.abs(Sb))
            if new < cost or rng.random() < np.exp(-(new - cost) / temp):
                assign[i] = b
                S[a], S[b], cg[a], cg[b], cost = Sa, Sb, ca, cb, new
        if cost < best:
            best, best_assign = cost, assign.copy()
    return best_assign, best


def err_matrix(tr, err, sites, wtab):
    hours = pd.DatetimeIndex(tr[C["ts"]].unique()).sort_values()
    ti = hours.get_indexer(tr[C["ts"]])
    si = pd.Series(np.arange(len(sites)), index=sites[C["site"]].values)[tr[C["site"]]].values
    E = np.zeros((len(sites), len(hours)), dtype="float32")
    E[si, ti] = err
    w = wtab["w"].reindex(hours)
    w = w.fillna(w.median()).values.astype("float64")      # 결측 대체는 학습 시각들의 중앙값
    return E, w


def group_caps(assign, sites):
    c = sites[C["cap"]].values.astype("float64")
    return float(c[assign == 0].sum()), float(c[assign == 1].sum())


def find_composition(Et, sites, w, ref, seeds, iters):
    caps = sites[C["cap"]].values.astype("float64")
    init = ref if ref is not None else (np.arange(len(sites)) % 2).astype(np.int8)
    if min(group_caps(init, sites)) < CFG["min_cap_kw"]:
        print(f"  [경고] 초기 구성이 {CFG['min_cap_kw']:,.0f} kW 제약 미달 → 용량 기준 재분배")
        order = np.argsort(-caps)
        init = np.zeros(len(caps), dtype=np.int8)
        tot = [0.0, 0.0]
        for i in order:                                  # 큰 것부터 작은 쪽에 배정 (그리디)
            g = int(tot[1] < tot[0])
            init[i] = g
            tot[g] += caps[i]
    best = (None, np.inf)
    for s in seeds:
        a, c = optimize_groups(Et, caps, w, init, iters, s, CFG["min_cap_kw"])
        print(f"    seed {s}: OOF cost {c:,.0f}")
        if c < best[1]:
            best = (a, c)
    return best[0], best[1], init


# ----------------------------------------------------------------------
# 테스터
# ----------------------------------------------------------------------
def _as_list(x):
    return [int(v) for v in (x if isinstance(x, (list, tuple, set)) else [x])]


def tester(fit_years, eval_years, start_month=None, seeds=None, verbose=True):
    """
    fit_years 로 학습·구성 최적화 → eval_years 에서 W 채점.
      W_base : 기준선형 (L2·무가중, 참조 구성)       ← 점수 0 기준
      W_ref  : 팀 모델 (L1·SMP·k 가중), 참조 구성     → 예측 개선 효과
      W_opt  : 팀 모델 + 최적화 구성                  → 예측 + 구성 개선 효과
    """
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다 (서로 다른 연도를 지정하세요)")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    for nm, b, ys in (("fit", base_fit, fit_y), ("eval", base_ev, ev_y)):
        if len(b) == 0:
            raise ValueError(f"{nm} 연도 {ys} 에 발전 데이터 없음. 보유 연도: {sorted(sea.unique())} (start_month={sm})")
    n_ev_h = base_ev[C["ts"]].nunique()
    print(f"■ fit={fit_y} → eval={ev_y} (start_month={sm}) | 학습 {len(base_fit):,}행, 평가 {len(base_ev):,}행 / {n_ev_h:,}시각")
    n_fs, n_es = base_fit[C["site"]].nunique(), base_ev[C["site"]].nunique()
    print(f"  데이터가 있는 발전소: fit {n_fs} / eval {n_es} (전체 {len(sites)})")
    if n_fs < len(sites) or n_es < len(sites):
        print("  (참고) 발전소가 연도별로 늘어난 데이터입니다. 실제 평가는 253개소 전체이므로 eval 은 2024년 이후(253개소)를 권장")
    if n_ev_h < 8000:
        print(f"  (참고) 평가 시각이 {n_ev_h:,}개 — 부분 연도라 W 절댓값은 8,760시간 기준보다 작습니다. 점수(%)로 비교하세요")

    # --- 학습·평가 프레임
    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"], med=tr_all.attrs["med"], with_w=False)

    # --- OOF 오차 → 구성 최적화 (fit 연도 안에서만)
    print("  OOF 오차 계산 (팀 모델" + (" 앙상블" if CFG.get("ensemble") else "") + ")...")
    err = ens_oof_errors(tr, feats, CFG["cv_folds"])
    Et, w = err_matrix(tr, err, sites, wtab)
    print("  구성 최적화...")
    opt, opt_cost, init = find_composition(Et, sites, w, ref, seeds, CFG["opt_iters"])
    ref_cost = float(w @ np.abs(Et[ref == 0].sum(0)) + w @ np.abs(Et[ref == 1].sum(0))) if ref is not None else np.nan

    # --- 최종 학습 → 평가 연도 예측 (낮 시간만 예측, 밤은 0)
    print("  최종 학습/예측 (base, team)...")
    preds = {}
    for name in ("base", "team"):
        if name == "team":
            p = ens_predict(tr, ev, feats, seeds)
        else:
            p = np.mean([fit_predict(tr, ev, feats, CFG["models"][name], seed=s) for s in seeds], axis=0)
        full = np.zeros(len(base_ev))
        full[ev["_row"].values] = p
        preds[name] = full

    sid = sites[C["site"]].values
    a_ref = pd.Series(ref if ref is not None else init, index=sid)
    a_opt = pd.Series(opt, index=sid)
    r_base, _ = compute_W(base_ev, preds["base"], a_ref, wtab)
    r_ref, _ = compute_W(base_ev, preds["team"], a_ref, wtab)
    r_opt, detail = compute_W(base_ev, preds["team"], a_opt, wtab)
    scales = fit_bias_scale(tr[[C["site"], C["ts"], C["gen"]]], err + tr[C["gen"]].values, a_opt, wtab, CFG["bias_mode"])
    r_bc, detail_bc = compute_W(base_ev, apply_bias_scale(base_ev, preds["team"], scales, CFG["bias_mode"]), a_opt, wtab)
    sc = lambda r: 100 * (1 - r["W"] / r_base["W"])
    res = dict(fit=fit_y, eval=ev_y, W_base=r_base["W"], W_ref=r_ref["W"], W_opt=r_opt["W"],
               score_model=sc(r_ref), score_total=sc(r_opt), score_comp=100 * (1 - r_opt["W"] / r_ref["W"]),
               peak_share=r_opt["peak_share"], caps=group_caps(opt, sites), n_move=int((opt != a_ref.values).sum()),
               oof_gain=(1 - opt_cost / ref_cost) * 100 if ref_cost == ref_cost else np.nan,
               assign=a_opt, detail=detail, W_opt_bc=r_bc["W"], score_bc=sc(r_bc), bias_scales=scales,
               W_floor=floor_W(detail), W_floor_bc=floor_W(detail_bc))
    if verbose:
        print(f"\n  W_base (기준선형·참조구성) : {r_base['W']:>16,.0f}")
        print(f"  W_ref  (팀 모델·참조구성)  : {r_ref['W']:>16,.0f}   점수 {res['score_model']:+.2f}")
        print(f"  W_opt  (팀 모델·최적구성)  : {r_opt['W']:>16,.0f}   점수 {res['score_total']:+.2f}")
        print(f"  └ 구성 최적화만의 효과(W_ref→W_opt): {res['score_comp']:+.2f}%  | fit 연도 OOF 개선 {res['oof_gain']:+.2f}%  "
              f"(둘의 차이가 크면 과적합 의심)")
        print(f"  피크(k=3) 페널티 비중 {res['peak_share']:.1%} | 집합 용량 G1 {res['caps'][0]:,.1f} / G2 {res['caps'][1]:,.1f} kW"
              f" | 참조 대비 이동 발전소 {res['n_move']}개")
        if scales is not None:
            print(f"  W_opt_bc (+편향보정 {CFG['bias_mode']}): {r_bc['W']:>16,.0f}   점수 {res['score_bc']:+.2f}  | 보정 전 대비 "
                  f"{(r_bc['W'] / r_opt['W'] - 1) * 100:+.3f}% ({r_bc['W'] - r_opt['W']:+,.0f})  계수 {_fmt_scales(scales, CFG['bias_mode'])}")
        wf, wb = (res["W_floor_bc"], r_bc["W"]) if scales is not None else (res["W_floor"], r_opt["W"])
        print(f"  W_floor (합계 오차만의 하한): {wf:>16,.0f}  | 현재 W 는 하한보다 {(wb / wf - 1) * 100:.1f}% 높음 "
              f"→ 구성 최적화로 더 줄일 수 있는 최대치(이론상) {(wb - wf) / wb * 100:.1f}%, 나머지 {wf / wb * 100:.1f}% 는 전체 발전량 예측 정확도 몫")
    return res


def year_grid(pairs, **kw):
    """여러 (fit, eval) 쌍을 연속 실행해 요약표 반환. 예: year_grid([(2021,2022),(2022,2023)])"""
    rows = []
    for f, e in pairs:
        r = tester(f, e, verbose=False, **kw)
        rows.append(dict(fit=r["fit"], eval=r["eval"], W_base=r["W_base"], W_ref=r["W_ref"], W_opt=r["W_opt"],
                         score_model=r["score_model"], score_total=r["score_total"], score_comp=r["score_comp"],
                         oof_gain=r["oof_gain"], peak_share=r["peak_share"], W_opt_bc=r["W_opt_bc"], score_bc=r["score_bc"]))
    df = pd.DataFrame(rows)
    print("\n" + df.round(2).to_string(index=False))
    return df


# ----------------------------------------------------------------------
# 편향 진단: 집합 합계 예측이 시간대별로 한쪽으로 치우쳐 있는가? (곱셈 보정이 평가 연도에서도 통하는지 검증)
# ----------------------------------------------------------------------
def _wmedian(x, w):
    """가중 중앙값"""
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    return float(x[min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(x) - 1)])


def _best_scale(t, bucket_col, lo=0.8, hi=1.2):
    """(집합, 시각)별 합계표 t 에서 버킷마다 Σ w·|c·P − A| 를 최소화하는 c.
    Σ w P |c − A/P| 이므로 해는 'A/P 의 가중중앙값(가중치 w·P)' — 닫힌 형태."""
    c = {}
    for b, d in t.groupby(bucket_col):
        d = d[d["P"] > 0]
        if len(d) < 30:
            c[b] = 1.0
            continue
        c[b] = float(np.clip(_wmedian((d["A"] / d["P"]).to_numpy(), (d["w"] * d["P"]).to_numpy()), lo, hi))
    return pd.Series(c, dtype="float64")


def _bucket_ids(hours, mode):
    h = np.asarray(hours, dtype=int)
    if mode == "global":
        return np.zeros(len(h), dtype=int)
    if mode == "peak":
        return np.isin(h, list(CFG["k_peak_hours"])).astype(int)
    if mode == "hour":
        return h
    raise ValueError(f"bias_mode 는 none/global/peak/hour 중 하나여야 합니다: {mode!r}")


def fit_bias_scale(base, pred, assign, wtab, mode):
    """학습 구간 OOF 예측으로 시간대별 곱셈 보정 계수를 구함 (페널티 Σ SMP·k·|집합 합계 오차| 를 최소화).
    base: [site, ts, generation]  pred: base 와 같은 길이의 OOF 예측(kWh)  assign: site→0/1  → {버킷: 계수} 또는 None"""
    if mode in (None, "none"):
        return None
    _, t = compute_W(base, pred, assign, wtab)
    t = t.copy()
    t["w"] = t["smp"] * t["k"]
    t["hour"] = t[C["ts"]].dt.hour.values
    t["b"] = _bucket_ids(t["hour"].values, mode)
    if mode == "hour":                                       # 발전량이 미미한 일출/일몰 시각은 계수 1.0 으로 둠
        ha = t.groupby("hour")["A"].sum()
        t = t[t["hour"].isin(ha.index[ha >= 0.02 * ha.max()])]
    return _best_scale(t, "b")


def apply_bias_scale(base, pred, scales, mode):
    """base 의 ts 시각에 맞춰 예측에 계수를 곱함 (밤은 예측이 0 이라 영향 없음)"""
    if scales is None:
        return np.asarray(pred, dtype="float64")
    f = pd.Series(_bucket_ids(base[C["ts"]].dt.hour.values, mode)).map(scales).fillna(1.0).to_numpy()
    return np.asarray(pred, dtype="float64") * f


def _fmt_scales(scales, mode):
    if scales is None:
        return "없음"
    if mode == "peak":
        return f"피크 {scales.get(1, 1.0):.3f} / 그 외 {scales.get(0, 1.0):.3f}"
    if mode == "global":
        return f"{scales.iloc[0]:.3f}"
    return f"{scales.min():.3f}~{scales.max():.3f}"


def bias_check(fit_years, eval_years, start_month=None, seeds=None):
    """
    ① fit 연도 OOF 예측으로 시간대별 치우침(ΣP/ΣA)을 측정
    ② 치우침을 없애는 곱셈 계수를 'fit 연도 OOF'에서만 구함 (전체 1개 / 피크·비피크 2개 / 시각별)
    ③ 그 계수를 eval 연도 예측에 적용 → W 가 줄어드는지 확인 (참조 구성 기준, 구성 최적화는 돌리지 않음)
    판단: ③ 의 eval 개선이 흔들림(약 1%)보다 확실히 크면 보정을 파이프라인에 넣을 가치가 있음.
    """
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    if ref is None:
        raise ValueError("current_group(참조 구성) 컬럼이 없어 진단할 수 없습니다")
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    print(f"■ 편향 진단 fit={fit_y} → eval={ev_y} (start_month={sm}) | 학습 {len(base_fit):,}행, 평가 {len(base_ev):,}행")

    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"], med=tr_all.attrs["med"], with_w=False)
    print("  OOF 예측 계산 (팀 모델)...")
    err = oof_errors(tr, feats, CFG["models"]["team"], CFG["cv_folds"])
    print("  평가 연도 예측...")
    p = np.mean([fit_predict(tr, ev, feats, CFG["models"]["team"], seed=s) for s in seeds], axis=0)
    pe = np.zeros(len(base_ev))
    pe[ev["_row"].values] = p

    a_ref = pd.Series(ref, index=sites[C["site"]].values)
    oof_base = tr[[C["site"], C["ts"], C["gen"]]].copy()
    p_oof = (err + tr[C["gen"]].values).astype("float64")

    peak = set(CFG["k_peak_hours"])
    bucket_of = {"global": lambda h: np.zeros(len(h), dtype=int),
                 "peak": lambda h: np.isin(h, list(peak)).astype(int),
                 "hour": lambda h: np.asarray(h, dtype=int)}

    def table(base, pred):
        _, t = compute_W(base, pred, a_ref, wtab)
        t = t.copy()
        t["w"] = t["smp"] * t["k"]
        t["hour"] = t[C["ts"]].dt.hour.values
        for nm, f in bucket_of.items():
            t["b_" + nm] = f(t["hour"].values)
        return t

    t_oof, t_ev = table(oof_base, p_oof), table(base_ev, pe)

    # ① 시간대별 치우침표
    ro = t_oof.groupby("hour")[["A", "P"]].sum()
    re_ = t_ev.groupby("hour")[["A", "P"]].sum()
    keep = ro.index[ro["A"] >= 0.02 * ro["A"].max()]          # 발전량이 미미한 일출/일몰 시각은 제외 (비율이 무의미)
    tab = pd.DataFrame({"학습OOF_ΣP/ΣA": ro["P"] / ro["A"], "평가연도_ΣP/ΣA": re_["P"] / re_["A"]}).loc[keep]
    print("\n[1] 시각(ts 라벨)별 집합 합계 예측 ÷ 실제  (1.00=치우침 없음, >1 과대예측)")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    pk = t_oof["hour"].isin(peak)
    print(f"  피크(k=3) 시간대 전체: OOF {t_oof.loc[pk, 'P'].sum() / t_oof.loc[pk, 'A'].sum():.3f} | "
          f"평가연도 {t_ev.loc[t_ev['hour'].isin(peak), 'P'].sum() / t_ev.loc[t_ev['hour'].isin(peak), 'A'].sum():.3f}")

    # ②③ 보정 → 평가
    W0_oof = float(t_oof["pen"].sum())
    W0_ev = float(t_ev["pen"].sum())
    rows = []
    for nm, label in (("global", "전체 1개 계수"), ("peak", "피크/비피크 2개"), ("hour", "시각별 계수")):
        c = _best_scale(t_oof, "b_" + nm)
        if nm == "hour":
            c = c[c.index.isin(keep)]                         # 제외 시각은 계수 1.0 (map 의 fillna 로 처리)
        h_oof = oof_base[C["ts"]].dt.hour.values
        h_ev = base_ev[C["ts"]].dt.hour.values
        f_oof = pd.Series(bucket_of[nm](h_oof)).map(c).fillna(1.0).to_numpy()
        f_ev = pd.Series(bucket_of[nm](h_ev)).map(c).fillna(1.0).to_numpy()
        W1_oof = compute_W(oof_base, p_oof * f_oof, a_ref, wtab)[0]["W"]
        W1_ev = compute_W(base_ev, pe * f_ev, a_ref, wtab)[0]["W"]
        rows.append({"보정 방식": label,
                     "계수": (f"{c.iloc[0]:.3f}" if nm == "global" else
                             f"피크 {c.get(1, 1.0):.3f} / 그 외 {c.get(0, 1.0):.3f}" if nm == "peak" else
                             f"{c.min():.3f}~{c.max():.3f}"),
                     "학습OOF W 변화(%)": (W1_oof / W0_oof - 1) * 100,
                     "평가연도 W 변화(%)": (W1_ev / W0_ev - 1) * 100,
                     "평가연도 W 변화(원)": W1_ev - W0_ev})
    res = pd.DataFrame(rows)
    out = res.copy()
    for col in ("학습OOF W 변화(%)", "평가연도 W 변화(%)"):
        out[col] = out[col].map("{:+.3f}".format)
    out["평가연도 W 변화(원)"] = out["평가연도 W 변화(원)"].map("{:+,.0f}".format)
    print("\n[2] 곱셈 보정 효과 (계수는 학습 연도 OOF 에서만 구함 / W 변화가 음수면 개선)")
    print(out.to_string(index=False))
    print(f"  (보정 전 W: 학습OOF {W0_oof:,.0f} / 평가연도 {W0_ev:,.0f}, 참조 구성 기준)")
    print("  ※ 핵심은 '평가연도 W 변화'입니다. 학습OOF 에서만 줄고 평가연도에서 안 줄면 과적합입니다.")
    return dict(table=tab, result=res, W0_oof=W0_oof, W0_ev=W0_ev)


# ----------------------------------------------------------------------
# 조건부 편향 진단: 합계 오차의 체계적 치우침이 '예보 상황'(구름 예보 수준, 10시→17시 예보 변화 등)에 따라 달라지는가?
# ----------------------------------------------------------------------
def _small_gbm(seed):
    """보정용 소형 모델: 절댓값 오차(L1), 얕은 나무, 큰 최소 표본 → 과적합 억제"""
    if CFG["backend"] == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(objective="regression_l1", n_estimators=150, learning_rate=0.03, num_leaves=6,
                                     min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=seed, verbose=-1, n_jobs=CFG["n_jobs"], deterministic=True,
                                     force_row_wise=True)
        except ImportError:
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=0.03, max_leaf_nodes=6,
                                         min_samples_leaf=200, random_state=seed)


def _calendar(idx):
    idx = pd.DatetimeIndex(idx)
    doy = idx.dayofyear.values
    return pd.DataFrame({"hour": idx.hour.values, "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                         "doy_cos": np.cos(2 * np.pi * doy / 365.25)}, index=idx)


def cond_fit(t, Xts, seeds=(0, 1, 2)):
    """(집합, 시각)별 합계표 t 로 '실제/예측' 비율을 예보 상황 Xts(시각별 피처)로 설명하는 모델을 학습.
    Σ w|cP − A| = Σ wP|c − A/P| 이므로 목표=A/P, 가중치=SMP·k·P 의 L1 회귀가 페널티 최소화와 같은 문제."""
    d = t[t["P"] > 0]
    if len(d) < 200:
        return None
    X = Xts.reindex(d[C["ts"]].values).to_numpy(dtype="float64")
    y = np.clip((d["A"] / d["P"]).to_numpy(), 0.5, 1.5)
    w = (d["smp"] * d["k"] * d["P"]).to_numpy(dtype="float64")
    return [_small_gbm(sd).fit(X, y, sample_weight=w) for sd in seeds]


def cond_scale(models, Xts, lo=0.9, hi=1.1):
    """시각별 보정 계수 (안전하게 0.9~1.1 로 제한)"""
    c = np.mean([m.predict(Xts.to_numpy(dtype="float64")) for m in models], axis=0)
    return pd.Series(np.clip(c, lo, hi), index=Xts.index)


def cond_bias_check(fit_years, eval_years, start_month=None, seeds=None, corr_seeds=(0, 1, 2)):
    """
    현재 보정(피크/비피크 2계수)보다 '예보 상황별 보정'이 평가 연도에서 W 를 더 줄이는지 검증 (참조 구성 기준).
    계수/모델은 fit 연도 OOF 에서만 학습하고, eval 연도는 채점에만 사용.
    판정: 현재 방식 대비 추가 개선이 -0.2% 보다 크고, 평가 연도 전반·후반 둘 다 현재 방식보다 낮을 때만 '시도 가치'.
    """
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    if ref is None:
        raise ValueError("current_group(참조 구성) 컬럼이 없어 진단할 수 없습니다")
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    print(f"■ 조건부 편향 진단 fit={fit_y} → eval={ev_y} (start_month={sm}) | 학습 {len(base_fit):,}행, 평가 {len(base_ev):,}행")

    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"], med=tr_all.attrs["med"], with_w=False)
    print("  OOF 예측 계산 (팀 모델)...")
    err = oof_errors(tr, feats, CFG["models"]["team"], CFG["cv_folds"])
    print("  평가 연도 예측...")
    p = np.mean([fit_predict(tr, ev, feats, CFG["models"]["team"], seed=sd) for sd in seeds], axis=0)
    pe = np.zeros(len(base_ev))
    pe[ev["_row"].values] = p

    a_ref = pd.Series(ref, index=sites[C["site"]].values)
    oof_base = tr[[C["site"], C["ts"], C["gen"]]].copy()
    p_oof = (err + tr[C["gen"]].values).astype("float64")
    t_oof = compute_W(oof_base, p_oof, a_ref, wtab)[1]
    t_ev0 = compute_W(base_ev, pe, a_ref, wtab)[1]
    mid = t_ev0[C["ts"]].min() + (t_ev0[C["ts"]].max() - t_ev0[C["ts"]].min()) / 2

    def score(po, pe_):
        to = compute_W(oof_base, po, a_ref, wtab)[1]
        te = compute_W(base_ev, pe_, a_ref, wtab)[1]
        m1 = te[C["ts"]] < mid
        return to["pen"].sum(), te["pen"].sum(), te.loc[m1, "pen"].sum(), te.loc[~m1, "pen"].sum()

    base0 = score(p_oof, pe)

    # 현재 방식 (피크/비피크 2계수)
    sc_peak = fit_bias_scale(oof_base, p_oof, a_ref, wtab, "peak")
    cur = score(apply_bias_scale(oof_base, p_oof, sc_peak, "peak"), apply_bias_scale(base_ev, pe, sc_peak, "peak"))

    # 시각별 예보 상황 피처 (전체 발전소 평균)
    cols_all = [f for f in feats if f.endswith("_avg") or f.endswith("_d")]
    Xo_all = tr.groupby(C["ts"])[cols_all].mean()
    Xe_all = ev.groupby(C["ts"])[cols_all].mean()
    cloud = [f for f in cols_all if "cloud" in f.lower()]
    cl_avg = [f for f in cloud if f.endswith("_avg")]
    cl_d = [f for f in cloud if f.endswith("_d")]
    print(f"  인식된 구름 예보 피처: 수준 {cl_avg} / 변화 {cl_d}")
    sets = [("시간·계절만", [])]
    if cl_avg:
        sets.append(("+ 구름 예보 수준", cl_avg))
    if cl_avg and cl_d:
        sets.append(("+ 구름 예보 변화(10→17시)", cl_avg + cl_d))
    if not cloud:
        print("  [경고] 이름에 'cloud' 가 들어간 예보 피처가 없어 구름 기반 조합은 건너뜁니다 (inspect 로 컬럼명 확인).")
    sets.append(("전체 예보 피처(평균)", cols_all))

    rows = [dict(방식="보정 없음", _r=base0), dict(방식="현재 방식 (피크/비피크 2계수)", _r=cur)]
    for name, cols in sets:
        Xo = pd.concat([_calendar(Xo_all.index), Xo_all[cols]], axis=1)
        Xe = pd.concat([_calendar(Xe_all.index), Xe_all[cols]], axis=1)
        ms = cond_fit(t_oof, Xo, corr_seeds)
        if ms is None:
            continue
        c_o, c_e = cond_scale(ms, Xo), cond_scale(ms, Xe)
        fo = oof_base[C["ts"]].map(c_o).fillna(1.0).to_numpy()
        fe = base_ev[C["ts"]].map(c_e).fillna(1.0).to_numpy()
        rows.append(dict(방식=f"조건부: {name}", _r=score(p_oof * fo, pe * fe), _c=(float(c_e.min()), float(c_e.max()))))

    out = []
    for r in rows:
        wo, we, wh1, wh2 = r["_r"]
        d = {"방식": r["방식"],
             "학습 W 변화(%)": (wo / base0[0] - 1) * 100,
             "평가 W 변화(%)": (we / base0[1] - 1) * 100,
             "정산금 환산(원,근사)": -0.1 * (we - base0[1]),
             "현재방식 대비(%)": (we / cur[1] - 1) * 100,
             "전반 대비(%)": (wh1 / cur[2] - 1) * 100,
             "후반 대비(%)": (wh2 / cur[3] - 1) * 100,
             "계수 범위": ("%.3f~%.3f" % r["_c"]) if "_c" in r else ""}
        d["판정"] = ("✓ 시도 가치" if (r["방식"].startswith("조건부") and d["현재방식 대비(%)"] < -0.2 and d["전반 대비(%)"] < 0 and d["후반 대비(%)"] < 0)
                   else ("" if not r["방식"].startswith("조건부") else "–"))
        out.append(d)
    res = pd.DataFrame(out)
    show = res.copy()
    for col in ("학습 W 변화(%)", "평가 W 변화(%)", "현재방식 대비(%)", "전반 대비(%)", "후반 대비(%)"):
        show[col] = show[col].map("{:+.3f}".format)
    show["정산금 환산(원,근사)"] = show["정산금 환산(원,근사)"].map(lambda v: "{:+,.0f}".format(0.0 if abs(v) < 0.5 else v))
    print("\n[조건부 보정 효과] 계수/모델은 학습 연도 OOF 에서만 학습 · W 변화가 음수면 개선 · '대비' 열은 현재 방식(2계수) 기준")
    print(show.to_string(index=False))
    print(f"  (보정 전 W: 학습OOF {base0[0]:,.0f} / 평가연도 {base0[1]:,.0f}, 참조 구성 기준)")
    print("  ※ 핵심은 '현재방식 대비(%)'가 확실히 음수이고 전반·후반 모두 음수인지입니다. 학습 W 만 크게 줄면 과적합입니다.")
    return dict(result=res, W0=base0, cur=cur)


# ----------------------------------------------------------------------
# 페널티 분해 진단: 페널티가 어느 달/시각/날/날씨 유형에 몰려 있는가? (개선 방향을 정하는 용도)
# ----------------------------------------------------------------------
def _prep_oof_eval(fit_years, eval_years, start_month, seeds):
    """fit 연도 OOF 예측 + eval 연도 예측을 한 번에 준비 (참조 구성 기준 진단용 공통 준비 단계)"""
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    if ref is None:
        raise ValueError("current_group(참조 구성) 컬럼이 없어 진단할 수 없습니다")
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    print(f"■ fit={fit_y} → eval={ev_y} (start_month={sm}) | 학습 {len(base_fit):,}행, 평가 {len(base_ev):,}행")
    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"], med=tr_all.attrs["med"], with_w=False)
    print("  OOF 예측 계산 (팀 모델)...")
    err = oof_errors(tr, feats, CFG["models"]["team"], CFG["cv_folds"])
    print("  평가 연도 예측...")
    p = np.mean([fit_predict(tr, ev, feats, CFG["models"]["team"], seed=sd) for sd in seeds], axis=0)
    pe = np.zeros(len(base_ev))
    pe[ev["_row"].values] = p
    return dict(fit_y=fit_y, ev_y=ev_y, wtab=wtab, feats=feats, ev=ev, tr=tr, base_ev=base_ev, pe=pe,
                a_ref=pd.Series(ref, index=sites[C["site"]].values),
                oof_base=tr[[C["site"], C["ts"], C["gen"]]].copy(),
                p_oof=(err + tr[C["gen"]].values).astype("float64"))


def penalty_breakdown(fit_years, eval_years, start_month=None, seeds=None, bias_mode=None, top_days=10):
    """
    eval 연도의 페널티(제출과 같은 편향 보정 적용, 참조 구성 기준)를 분해해서 '어디서 새는지'를 보여 줌.
    · 예측÷실제 > 1 이면 과대예측, < 1 이면 과소예측
    · 편향비중 = |Σ w·오차| / Σ w·|오차| : 오차가 한 방향으로 일관된 정도 (클수록 보정으로 줄일 여지, 작으면 무작위 잡음)
    ※ 검증 연도 기준 분석이라 실제 평가 기간과 패턴이 다를 수 있음 (모델이 어떤 상황에 약한지 보는 용도)
    """
    mode = bias_mode or CFG["bias_mode"]
    Pp = _prep_oof_eval(fit_years, eval_years, start_month, seeds)
    scales = fit_bias_scale(Pp["oof_base"], Pp["p_oof"], Pp["a_ref"], Pp["wtab"], mode)
    pe_c = apply_bias_scale(Pp["base_ev"], Pp["pe"], scales, mode)
    out, t = compute_W(Pp["base_ev"], pe_c, Pp["a_ref"], Pp["wtab"])
    W = float(t["pen"].sum())

    t = t.copy()
    t["E"] = t["P"] - t["A"]
    t["absE"] = t["E"].abs()
    ts_df = t.groupby(C["ts"]).agg(pen=("pen", "sum"), A=("A", "sum"), P=("P", "sum"), absE=("absE", "sum"),
                                   smp=("smp", "first"), k=("k", "first"))
    ts_df["w"] = ts_df["smp"] * ts_df["k"]
    ts_df["E"] = ts_df["P"] - ts_df["A"]
    ts_df["wE"] = ts_df["w"] * ts_df["E"]
    ts_df["wabsE"] = ts_df["w"] * ts_df["E"].abs()
    adj = ts_df.index - pd.Timedelta(hours=1)                 # 시각 라벨은 구간 끝 기준 → 날짜/월은 1시간 앞당겨 배정
    ts_df["date"] = adj.normalize()
    ts_df["month"] = adj.to_period("M")
    ts_df["hour"] = ts_df.index.hour
    day_mask = (ts_df["A"] > 0) | (ts_df["P"] > 0)

    def agg(df, key):
        a = df.groupby(key).agg(pen=("pen", "sum"), A=("A", "sum"), P=("P", "sum"), wE=("wE", "sum"), wabsE=("wabsE", "sum"))
        a["페널티 비중(%)"] = a["pen"] / W * 100
        a["발전량 비중(%)"] = a["A"] / a["A"].sum() * 100
        a["예측÷실제"] = a["P"] / a["A"].replace(0, np.nan)
        a["편향비중(%)"] = a["wE"].abs() / a["wabsE"].replace(0, np.nan) * 100
        return a

    fmt = lambda v: f"{v:.3f}"
    print(f"\n■ 페널티 분해 (보정 {mode}, 참조 구성) | W {W:,.0f} → 실제 페널티 약 {W * 0.1:,.0f}원 (0.1×W 가정)")
    tot_bias = abs(ts_df["wE"].sum()) / ts_df["wabsE"].sum() * 100
    print(f"  전체 편향비중 {tot_bias:.1f}%  (낮을수록 오차가 무작위에 가까워 보정으로 줄이기 어려움)")

    # [1] 월별
    m = agg(ts_df, "month")
    m["배수"] = m["페널티 비중(%)"] / m["발전량 비중(%)"]
    print("\n[1] 월별  (배수 = 페널티 비중 ÷ 발전량 비중, 1.0 보다 크면 그 달 예측이 상대적으로 약함)")
    print(m[["페널티 비중(%)", "발전량 비중(%)", "배수", "예측÷실제", "편향비중(%)"]].to_string(float_format=lambda v: f"{v:.2f}"))

    # [2] 시각별
    h = agg(ts_df[day_mask], "hour")
    h = h[h["페널티 비중(%)"] >= 0.5]
    print("\n[2] 시각(ts 라벨)별  (페널티 비중 0.5% 이상)")
    print(h[["페널티 비중(%)", "예측÷실제", "편향비중(%)"]].to_string(float_format=lambda v: f"{v:.2f}"))

    # [3] SMP×k 가중치 구간별
    d = ts_df[day_mask].copy()
    try:
        d["구간"] = pd.qcut(d["w"], 5, duplicates="drop")
        b = d.groupby("구간", observed=True).agg(시간수=("pen", "size"), pen=("pen", "sum"), A=("A", "sum"), absE=("absE", "sum"))
        b["페널티 비중(%)"] = b["pen"] / W * 100
        b["발전량 비중(%)"] = b["A"] / ts_df["A"].sum() * 100
        b["상대오차율(%)"] = b["absE"] / b["A"] * 100
        b.index = [f"{iv.left:,.0f}~{iv.right:,.0f}" for iv in b.index]
        print("\n[3] 시각별 가중치(SMP×k) 구간  (상대오차율 = Σ|집합 오차| ÷ Σ발전량)")
        print(b[["시간수", "페널티 비중(%)", "발전량 비중(%)", "상대오차율(%)"]].to_string(float_format=lambda v: f"{v:.2f}"))
    except ValueError:
        print("\n[3] 가중치 구간을 나눌 수 없어 생략")

    # [4] 날짜 쏠림
    day = ts_df.groupby("date").agg(pen=("pen", "sum"), A=("A", "sum"), P=("P", "sum"), smp=("smp", "mean"))
    day["비중(%)"] = day["pen"] / W * 100
    day["예측÷실제"] = day["P"] / day["A"].replace(0, np.nan)
    srt = day.sort_values("pen", ascending=False)
    n = len(srt)
    cum = srt["pen"].cumsum() / W * 100
    print(f"\n[4] 날짜별 쏠림 (총 {n}일)  — 균등하다면 상위 5/10/20% 날이 5/10/20%를 차지")
    print("  " + " | ".join(f"상위 {q}%({max(1, int(round(n * q / 100)))}일): {cum.iloc[max(1, int(round(n * q / 100))) - 1]:.1f}%" for q in (5, 10, 20)))

    # 구름 예보 상황 (이름에 cloud 가 들어간 예보 피처)
    ev = Pp["ev"]
    cols = [f for f in Pp["feats"] if "cloud" in f.lower()]
    cl_avg = [f for f in cols if f.endswith("_avg")]
    cl_d = [f for f in cols if f.endswith("_d")]
    if cl_avg:
        ts_df["구름"] = ev.groupby(C["ts"])[cl_avg].mean().mean(axis=1).reindex(ts_df.index)
        day["구름수준"] = ts_df.groupby("date")["구름"].mean()
    if cl_d:
        ts_df["변화"] = ev.groupby(C["ts"])[cl_d].mean().mean(axis=1).reindex(ts_df.index)
        day["구름변화"] = ts_df.groupby("date")["변화"].mean()
        day["변화크기"] = ts_df.assign(변화크기=ts_df["변화"].abs()).groupby("date")["변화크기"].mean()
    top = srt.head(top_days).join(day[[c for c in ("구름수준", "구름변화") if c in day]]).copy()
    top.index = [str(i.date()) for i in top.index]
    show_cols = ["비중(%)", "예측÷실제", "smp"] + [c for c in ("구름수준", "구름변화") if c in top]
    print(f"\n  페널티가 큰 날 상위 {top_days}일  (구름변화 = 17시 예보 − 10시 예보, 양수=더 흐려짐)")
    print(top[show_cols].rename(columns={"smp": "평균SMP"}).to_string(float_format=lambda v: f"{v:.2f}"))

    # [5] 날씨 유형별 (일 단위 3분위)
    def terciles(col, title):
        if col not in day or day[col].notna().sum() < 30:
            return None
        x = day.dropna(subset=[col]).copy()
        x["유형"] = pd.qcut(x[col], 3, labels=["낮음", "중간", "높음"], duplicates="drop")
        r = x.groupby("유형", observed=True).agg(일수=("pen", "size"), pen=("pen", "sum"), A=("A", "sum"), P=("P", "sum"),
                                                  최소=(col, "min"), 최대=(col, "max"))
        r["페널티 비중(%)"] = r["pen"] / W * 100
        r["일평균 페널티 배수"] = (r["pen"] / r["일수"]) / (x["pen"].sum() / len(x))
        r["예측÷실제"] = r["P"] / r["A"]
        r["값 범위"] = [f"{a:.1f}~{b:.1f}" for a, b in zip(r["최소"], r["최대"])]
        print(f"\n[5] {title}별 (일 단위 3분위, 배수 1.0 = 평균적인 날)")
        print(r[["일수", "페널티 비중(%)", "일평균 페널티 배수", "예측÷실제", "값 범위"]].to_string(float_format=lambda v: f"{v:.2f}"))
        return r

    r1 = terciles("구름수준", "구름 예보 수준") if cl_avg else None
    r2 = terciles("변화크기", "구름 예보 변화 크기(|17시−10시|)") if cl_d else None
    if not cl_avg:
        print("\n[5] 이름에 'cloud' 가 들어간 예보 피처가 없어 날씨 유형 분해는 생략 (inspect 로 컬럼명 확인)")
    print("\n※ 읽는 법: [1]~[3] 에서 배수·상대오차율이 유난히 큰 곳, [4] 에서 소수의 날이 큰 비중을 차지하는지, "
          "[5] 에서 어떤 날씨 유형의 배수가 큰지 보세요. 그 부분이 개선 우선순위입니다.")
    return dict(W=W, month=m, hour=h, days=day, top=top, cloud_level=r1, cloud_change=r2)


# ----------------------------------------------------------------------
# 구름 구간별 보정 진단: 구름 예보 수준(3~5구간)별로 곱셈 계수 몇 개만 추가하면 현재 2계수 보정보다 나아지는가?
# ----------------------------------------------------------------------
def _day_cloud(ts_cloud):
    """시각별 구름 예보 수준 → 일 단위 평균 (시각 라벨이 구간 끝 기준이라 1시간 앞당겨 날짜 배정)"""
    d = (ts_cloud.index - pd.Timedelta(hours=1)).normalize()
    return ts_cloud.groupby(d).mean()


def _bin_edges(day_cloud_train, k):
    return np.quantile(day_cloud_train.dropna().to_numpy(), np.linspace(0, 1, k + 1)[1:-1])


def _bins_of(tsvals, day_cloud, edges):
    """각 시각이 속한 날의 구름 구간(0=맑음쪽 .. k-1=흐림쪽), 정보 없으면 -1"""
    d = (pd.DatetimeIndex(tsvals) - pd.Timedelta(hours=1)).normalize()
    v = day_cloud.reindex(d).to_numpy(dtype="float64")
    return np.where(np.isnan(v), -1, np.searchsorted(edges, v, side="right"))


def _bucket(tsvals, bins_row, peak_split):
    if not peak_split:
        return bins_row
    h = pd.DatetimeIndex(tsvals).hour
    return bins_row * 2 + np.isin(h, list(CFG["k_peak_hours"])).astype(int)


def _fit_by_bucket(t, bucket):
    d = pd.DataFrame({"A": t["A"].to_numpy(), "P": t["P"].to_numpy(), "w": (t["smp"] * t["k"]).to_numpy(), "b": bucket})
    return _best_scale(d, "b")


def _apply_bucket(pred, bucket, c):
    return np.asarray(pred, dtype="float64") * pd.Series(bucket).map(c).fillna(1.0).to_numpy()


def cloud_bin_check(fit_years, eval_years, start_month=None, seeds=None, shrink=0.5):
    """
    현재 보정(피크/비피크 2계수) 위에 '구름 예보 수준 구간별 계수'를 얹었을 때 평가 연도 W 가 더 줄어드는지 검증 (참조 구성 기준).
    · 구간 경계(분위수)와 계수는 fit 연도 OOF 에서만 구함 / eval 연도는 채점만
    · 2단계 = 현재 보정을 먼저 적용한 OOF 에서 구간별 '추가' 계수를 구함 (1.0 근처 값) → 절반 반영(shrink) 버전도 평가
    판정: 현재 방식 대비 추가 개선이 -0.2% 보다 크고, 평가 연도 전반·후반 둘 다 현재 방식보다 낮을 때만 '시도 가치'.
    """
    Pp = _prep_oof_eval(fit_years, eval_years, start_month, seeds)
    tr, ev, feats = Pp["tr"], Pp["ev"], Pp["feats"]
    oof_base, p_oof, base_ev, pe, a_ref, wtab = Pp["oof_base"], Pp["p_oof"], Pp["base_ev"], Pp["pe"], Pp["a_ref"], Pp["wtab"]
    cols = [f for f in feats if "cloud" in f.lower() and f.endswith("_avg")]
    print(f"  인식된 구름 예보 피처: {cols}")
    if not cols:
        print("  [중단] 이름에 'cloud' 가 들어간 예보 피처가 없습니다 (inspect 로 컬럼명 확인).")
        return None
    dc_tr = _day_cloud(tr.groupby(C["ts"])[cols].mean().mean(axis=1))
    dc_ev = _day_cloud(ev.groupby(C["ts"])[cols].mean().mean(axis=1))

    t_ev0 = compute_W(base_ev, pe, a_ref, wtab)[1]
    mid = t_ev0[C["ts"]].min() + (t_ev0[C["ts"]].max() - t_ev0[C["ts"]].min()) / 2

    def score(po, pe_):
        to = compute_W(oof_base, po, a_ref, wtab)[1]
        te = compute_W(base_ev, pe_, a_ref, wtab)[1]
        m1 = te[C["ts"]] < mid
        return to["pen"].sum(), te["pen"].sum(), te.loc[m1, "pen"].sum(), te.loc[~m1, "pen"].sum()

    base0 = score(p_oof, pe)
    sc_peak = fit_bias_scale(oof_base, p_oof, a_ref, wtab, "peak")
    po1 = apply_bias_scale(oof_base, p_oof, sc_peak, "peak")
    pe1 = apply_bias_scale(base_ev, pe, sc_peak, "peak")
    cur = score(po1, pe1)
    t0 = compute_W(oof_base, p_oof, a_ref, wtab)[1]
    t1 = compute_W(oof_base, po1, a_ref, wtab)[1]

    rows = [dict(방식="보정 없음", _r=base0, 계수=""),
            dict(방식="현재 방식 (피크/비피크 2계수)", _r=cur, 계수=_fmt_scales(sc_peak, "peak"))]

    def add(label, c, edges, extra_peak, po_in, pe_in, t_fit, coef_txt):
        b_o = _bucket(oof_base[C["ts"]], _bins_of(oof_base[C["ts"]], dc_tr, edges), extra_peak)
        b_e = _bucket(base_ev[C["ts"]], _bins_of(base_ev[C["ts"]], dc_ev, edges), extra_peak)
        rows.append(dict(방식=label, _r=score(_apply_bucket(po_in, b_o, c), _apply_bucket(pe_in, b_e, c)), 계수=coef_txt))

    for k in (3, 5):
        edges = _bin_edges(dc_tr, k)
        bt1 = _bucket(t1[C["ts"]], _bins_of(t1[C["ts"]], dc_tr, edges), False)
        c2 = _fit_by_bucket(t1, bt1)
        c2 = c2[c2.index >= 0]
        txt = " / ".join(f"{c2.get(i, 1.0):.3f}" for i in range(k))
        add(f"2단계: 현재 + 구름 {k}구간 추가계수", c2, edges, False, po1, pe1, t1, f"맑음→흐림 {txt}")
        if k == 3:
            ch = 1 + shrink * (c2 - 1)
            add(f"2단계 절반 반영 (구름 3구간)", ch, edges, False, po1, pe1, t1,
                " / ".join(f"{ch.get(i, 1.0):.3f}" for i in range(3)))
            # 단독 6계수 (구름 3구간 × 피크/비피크) — 현재 보정 없이 한 번에 구함
            bt0 = _bucket(t0[C["ts"]], _bins_of(t0[C["ts"]], dc_tr, edges), True)
            c6 = _fit_by_bucket(t0, bt0)
            c6 = c6[c6.index >= 0]
            add("단독: 구름 3구간 × 피크/비피크 (6계수)", c6, edges, True, p_oof, pe, t0,
                f"{c6.min():.3f}~{c6.max():.3f}")

    out = []
    for r in rows:
        wo, we, w1, w2 = r["_r"]
        d = {"방식": r["방식"], "계수": r["계수"],
             "학습 W 변화(%)": (wo / base0[0] - 1) * 100,
             "평가 W 변화(%)": (we / base0[1] - 1) * 100,
             "정산금 환산(원,근사)": -0.1 * (we - base0[1]),
             "현재방식 대비(%)": (we / cur[1] - 1) * 100,
             "전반 대비(%)": (w1 / cur[2] - 1) * 100,
             "후반 대비(%)": (w2 / cur[3] - 1) * 100}
        isv = r["방식"].startswith(("2단계", "단독"))
        d["판정"] = ("✓ 시도 가치" if (isv and d["현재방식 대비(%)"] < -0.2 and d["전반 대비(%)"] < 0 and d["후반 대비(%)"] < 0)
                   else ("–" if isv else ""))
        out.append(d)
    res = pd.DataFrame(out)
    show = res.copy()
    for col in ("학습 W 변화(%)", "평가 W 변화(%)", "현재방식 대비(%)", "전반 대비(%)", "후반 대비(%)"):
        show[col] = show[col].map("{:+.3f}".format)
    show["정산금 환산(원,근사)"] = show["정산금 환산(원,근사)"].map(lambda v: "{:+,.0f}".format(0.0 if abs(v) < 0.5 else v))
    print("\n[구름 구간별 보정 효과] 경계/계수는 학습 연도 OOF 에서만 구함 · W 변화가 음수면 개선 · '대비' 열은 현재 방식(2계수) 기준")
    print(show.to_string(index=False))
    print(f"  (보정 전 W: 학습OOF {base0[0]:,.0f} / 평가연도 {base0[1]:,.0f}, 참조 구성 기준)")
    print("  ※ 핵심은 '현재방식 대비(%)'가 -0.2% 보다 확실히 음수이고 전반·후반 모두 음수인지입니다. 학습 W 만 줄고 평가가 안 줄면 과적합입니다.")
    return dict(result=res, W0=base0, cur=cur)


# ----------------------------------------------------------------------
# 제출 생성 + 형식 점검
# ----------------------------------------------------------------------
def make_submission(seeds=(0, 1, 2)):
    os.makedirs(CFG["out_dir"], exist_ok=True)
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    g = D["gen"]
    if CFG["submit_train_from"]:
        g = g[g[C["ts"]] >= pd.Timestamp(CFG["submit_train_from"])]
    print(f"학습 행 {len(g):,} ({g[C['ts']].min()} ~ {g[C['ts']].max()})")
    tr_all, feats = make_frame(g.reset_index(drop=True), D["fc"])
    tr = prep_train(tr_all)
    err = ens_oof_errors(tr, feats, CFG["cv_folds"])
    Et, w = err_matrix(tr, err, sites, wtab)
    opt, cost, init = find_composition(Et, sites, w, ref, seeds, CFG["opt_iters"])
    mode = CFG["composition_mode"]
    if mode == "opt":
        final = opt
    elif mode == "ref":
        # 원래 구성으로 제출. 편향 보정 계수는 최적화 구성(opt) 기준으로 그대로 산출하므로
        # G1+G2 합계 예측은 'opt' 제출과 동일하고 그룹 배정만 다름 → 점수 차이 = 구성 최적화의 순수 효과
        if ref is None:
            raise ValueError("composition_mode='ref' 는 sites 파일에 current_group 컬럼이 필요합니다")
        final = np.asarray(ref).astype(np.int8)
        caps_f = group_caps(final, sites)
        if min(caps_f) < CFG["min_cap_kw"]:
            raise ValueError(f"원래 구성이 용량 제약({CFG['min_cap_kw']:,.0f} kW) 미달이라 제출할 수 없습니다: {caps_f}")
        print(f"구성: 원래 구성(current_group) 사용 | 집합 용량 G1 {caps_f[0]:,.1f} / G2 {caps_f[1]:,.1f} kW")
    else:
        raise ValueError(f"composition_mode 는 'opt' 또는 'ref' 여야 합니다: {mode!r}")

    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    eval_hours = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    fcT = {"A": load_fc(CFG["files"]["test_fc_a"], flat_ok=False), "B": load_fc(CFG["files"]["test_fc_b"], flat_ok=False)}
    grid = pd.MultiIndex.from_product([sites[C["site"]].values, eval_hours],
                                      names=[C["site"], C["ts"]]).to_frame(index=False)
    ev, _ = make_frame(grid, fcT, med=tr_all.attrs["med"], with_w=False)
    p = ens_predict(tr, ev, feats, seeds)
    full = np.zeros(len(grid))
    full[ev["_row"].values] = p
    a_opt = pd.Series(opt, index=sites[C["site"]].values)
    scales = fit_bias_scale(tr[[C["site"], C["ts"], C["gen"]]], err + tr[C["gen"]].values, a_opt, wtab, CFG["bias_mode"])
    full = apply_bias_scale(grid, full, scales, CFG["bias_mode"])
    print(f"편향 보정 [{CFG['bias_mode']}]: 계수 {_fmt_scales(scales, CFG['bias_mode'])}")
    grid["P"] = full
    grid["group"] = grid[C["site"]].map(pd.Series(final, index=sites[C["site"]].values))
    agg = grid.groupby(["group", C["ts"]])["P"].sum().reset_index()
    agg["group"] = agg["group"].map({0: "G1", 1: "G2"})
    agg[C["ts"]] = agg[C["ts"]].dt.tz_localize("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    agg = agg.rename(columns={C["ts"]: "ts", "P": "forecast_kwh"})[["group", "ts", "forecast_kwh"]]
    agg["forecast_kwh"] = agg["forecast_kwh"].clip(lower=0.0)
    comp = pd.DataFrame({"site_id": sites[C["site"]].values, "group": np.where(final == 0, "G1", "G2")})
    os.makedirs(CFG["out_dir"], exist_ok=True)       # 실행 중에 폴더가 바뀌거나 지워졌어도 저장 직전에 다시 만든다
    comp.to_csv(os.path.join(CFG["out_dir"], "composition.csv"), index=False)
    agg.to_csv(os.path.join(CFG["out_dir"], "forecast.csv"), index=False)
    print(f"저장 완료 → {CFG['out_dir']}/composition.csv, forecast.csv")
    write_env_report(seeds=seeds)
    return check_submission()


def _save_submission(grid, assign, sites, out_dir):
    """grid[site, ts, P] 와 구성(assign: 0/1 배열)으로 composition.csv / forecast.csv 저장 (make_submission 과 같은 형식)"""
    tmp = grid[[C["site"], C["ts"], "P"]].copy()
    tmp["group"] = tmp[C["site"]].map(pd.Series(assign, index=sites[C["site"]].values))
    agg = tmp.groupby(["group", C["ts"]])["P"].sum().reset_index()
    agg["group"] = agg["group"].map({0: "G1", 1: "G2"})
    agg[C["ts"]] = agg[C["ts"]].dt.tz_localize("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    agg = agg.rename(columns={C["ts"]: "ts", "P": "forecast_kwh"})[["group", "ts", "forecast_kwh"]]
    agg["forecast_kwh"] = agg["forecast_kwh"].clip(lower=0.0)
    comp = pd.DataFrame({"site_id": sites[C["site"]].values, "group": np.where(np.asarray(assign) == 0, "G1", "G2")})
    os.makedirs(out_dir, exist_ok=True)
    comp.to_csv(os.path.join(out_dir, "composition.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "forecast.csv"), index=False)


def _canon(a):
    """G1/G2 이름을 서로 바꾼 구성은 점수가 똑같으므로 같은 구성으로 취급 (0번 발전소가 항상 G1 이 되도록 정규화)"""
    a = np.asarray(a).astype(np.int8)
    return a if a[0] == 0 else (1 - a).astype(np.int8)


def make_submission_variants(pred_seeds=(0, 1, 2), sa_seeds=tuple(range(10, 22)), n_keep=8, out_root=None):
    """
    한 번의 학습으로 '합계 예측은 같고 그룹 배정만 다른' 제출 후보 여러 개를 만든다 (리더보드 점수로 최고 구성을 고르는 용도).
    · v00 = make_submission(seeds=pred_seeds) 와 동일한 제출 (재현 확인용. 이미 제출했다면 다시 제출할 필요 없음)
    · v01.. = 다른 구성 탐색 시드(sa_seeds)로 찾은 구성 중 학습 구간 비용이 낮은 순서
    · 편향 보정 계수는 v00 구성 기준으로 한 번만 구해 모든 후보에 똑같이 적용 → 후보 간 점수 차이는 구성의 차이에서만 나옴
    결과: out_root/v00, v01, ... 폴더(각각 composition.csv, forecast.csv) + variants_summary.csv + environment.txt
    """
    out_root = out_root or os.path.join(CFG["out_dir"], "variants")
    os.makedirs(out_root, exist_ok=True)
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    g = D["gen"]
    if CFG["submit_train_from"]:
        g = g[g[C["ts"]] >= pd.Timestamp(CFG["submit_train_from"])]
    print(f"학습 행 {len(g):,} ({g[C['ts']].min()} ~ {g[C['ts']].max()})")
    tr_all, feats = make_frame(g.reset_index(drop=True), D["fc"])
    tr = prep_train(tr_all)
    err = ens_oof_errors(tr, feats, CFG["cv_folds"])
    Et, w = err_matrix(tr, err, sites, wtab)
    opt0, cost0, init = find_composition(Et, sites, w, ref, pred_seeds, CFG["opt_iters"])

    caps = sites[C["cap"]].values.astype("float64")
    print(f"\n구성 후보 탐색: 구성 탐색 시드 {list(sa_seeds)}")
    cands = []
    seen = {_canon(opt0).tobytes()}
    for sd in sa_seeds:
        a, c = optimize_groups(Et, caps, w, init, CFG["opt_iters"], sd, CFG["min_cap_kw"])
        a = _canon(a)                                   # 이름 교환(G1<->G2) 중복 제거를 위해 정규화
        d_raw = int((a != _canon(opt0)).sum())
        nd = min(d_raw, len(a) - d_raw)                 # 이름 교환을 고려한 실제로 다른 발전소 수
        dup = a.tobytes() in seen
        print(f"    탐색 시드 {sd}: 학습 비용 {c:,.0f} | v00 과 다른 발전소 {nd}개" + (" → 이미 있는 구성이라 제외" if dup else ""))
        if not dup:
            seen.add(a.tobytes())
            cands.append((float(c), sd, a, nd))
    cands.sort(key=lambda x: x[0])
    cands = cands[:n_keep]

    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    eval_hours = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    fcT = {"A": load_fc(CFG["files"]["test_fc_a"], flat_ok=False), "B": load_fc(CFG["files"]["test_fc_b"], flat_ok=False)}
    grid = pd.MultiIndex.from_product([sites[C["site"]].values, eval_hours], names=[C["site"], C["ts"]]).to_frame(index=False)
    ev, _ = make_frame(grid, fcT, med=tr_all.attrs["med"], with_w=False)
    p = ens_predict(tr, ev, feats, pred_seeds)
    full = np.zeros(len(grid))
    full[ev["_row"].values] = p
    a_opt0 = pd.Series(opt0, index=sites[C["site"]].values)
    scales = fit_bias_scale(tr[[C["site"], C["ts"], C["gen"]]], err + tr[C["gen"]].values, a_opt0, wtab, CFG["bias_mode"])
    grid["P"] = apply_bias_scale(grid, full, scales, CFG["bias_mode"])
    print(f"편향 보정 [{CFG['bias_mode']}]: 계수 {_fmt_scales(scales, CFG['bias_mode'])}")

    variants = [("v00", opt0, float(cost0), 0, "v00(기준)")] + [
        (f"v{i:02d}", a, c, nd, f"시드{sd}") for i, (c, sd, a, nd) in enumerate(cands, 1)]
    rows = []
    for name, a, c, nd, tag in variants:
        d = os.path.join(out_root, name)
        _save_submission(grid, a, sites, d)
        print(f"\n[{name}] {tag}")
        ok = check_submission(os.path.join(d, "composition.csv"), os.path.join(d, "forecast.csv"))
        cp = group_caps(np.asarray(a), sites)
        rows.append(dict(후보=name, 출처=tag, 학습비용=c, v00과_다른_발전소=nd, G1용량=round(cp[0], 1), G2용량=round(cp[1], 1),
                         형식점검=bool(ok), 리더보드_정산금=""))
    pd.DataFrame(rows).to_csv(os.path.join(out_root, "variants_summary.csv"), index=False, encoding="utf-8-sig")
    write_env_report(path=os.path.join(out_root, "environment.txt"), seeds=pred_seeds)
    print(f"\n저장 완료 → {out_root}  (후보 {len(variants)}개: v00 은 기존 제출과 동일, v01~ 이 새 후보)")
    print("제출 후 variants_summary.csv 의 '리더보드_정산금' 칸에 점수를 적어 두면 비교하기 편합니다.")
    return rows


def ensemble_check(fit_years, eval_years, start_month=None, seeds=None, members=None):
    """
    구조가 다른 팀 모델 멤버(ENS_MEMBERS)를 각각 학습(OOF + 최종)한 뒤, 여러 평균 조합이 단일 팀 모델(all_l1)보다
    평가 연도 W 를 줄이는지 검증 (제출과 같은 편향 보정 적용, 참조 구성 기준).
    · 보정 계수는 각 조합의 fit 연도 OOF 에서만 구함 / eval 연도는 채점에만 사용
    · 판정: 평가 연도 W 가 기준(all_l1 단독)보다 -0.2% 보다 더 줄고, 평가 연도 전반·후반 모두 줄어든 조합만 '시도 가치'
    ※ 시간: 멤버 수 × (OOF 3회 + 최종 학습 시드 수) 만큼 학습 → 멤버 4개면 단일 모델 진단의 약 4배
    """
    sm = start_month or CFG["start_month"]
    seeds = seeds or CFG["seeds"]
    members = list(members or ENS_MEMBERS)
    fit_y, ev_y = _as_list(fit_years), _as_list(eval_years)
    if set(fit_y) & set(ev_y):
        raise ValueError("fit_years 와 eval_years 가 겹칩니다")
    D = load_data()
    sites, wtab, ref = D["sites"], D["wtab"], D["ref"]
    if ref is None:
        raise ValueError("current_group(참조 구성) 컬럼이 없어 진단할 수 없습니다")
    g = D["gen"]
    sea = season_of(g[C["ts"]], sm)
    base_fit, base_ev = g[sea.isin(fit_y)], g[sea.isin(ev_y)].reset_index(drop=True)
    print(f"■ 앙상블 진단 fit={fit_y} → eval={ev_y} (start_month={sm}) | 멤버 {members} | 시드 {list(seeds)}")
    print("  피처 생성...")
    tr_all, feats = make_frame(base_fit, D["fc"])
    tr = prep_train(tr_all)
    ev, _ = make_frame(base_ev, D["fc"], med=tr_all.attrs["med"], with_w=False)
    a_ref = pd.Series(ref, index=sites[C["site"]].values)
    oof_base = tr[[C["site"], C["ts"], C["gen"]]].copy()
    gen_oof = tr[C["gen"]].to_numpy(dtype="float64")

    Po, Pe = {}, {}
    for m in members:
        f_m = _select_feats(feats, ENS_MEMBERS[m]["feats"])
        print(f"\n[멤버 {m}] 사용 피처 {len(f_m)}개 / 전체 {len(feats)}개 | 손실 {_member_spec(m)['objective']}")
        err_m = oof_errors(tr, f_m, _member_spec(m), CFG["cv_folds"])
        Po[m] = err_m.astype("float64") + gen_oof
        p = np.mean([fit_predict(tr, ev, f_m, _member_spec(m), seed=sd) for sd in seeds], axis=0)
        full = np.zeros(len(base_ev))
        full[ev["_row"].values] = p
        Pe[m] = full

    mode = CFG["bias_mode"]
    t_ref0 = compute_W(base_ev, Pe[members[0]], a_ref, wtab)[1]
    mid = t_ref0[C["ts"]].min() + (t_ref0[C["ts"]].max() - t_ref0[C["ts"]].min()) / 2

    def score(po, pe_):
        sc = fit_bias_scale(oof_base, po, a_ref, wtab, mode)
        po2 = apply_bias_scale(oof_base, po, sc, mode)
        pe2 = apply_bias_scale(base_ev, pe_, sc, mode)
        to = compute_W(oof_base, po2, a_ref, wtab)[1]
        te = compute_W(base_ev, pe2, a_ref, wtab)[1]
        m1 = te[C["ts"]] < mid
        return to["pen"].sum(), te["pen"].sum(), te.loc[m1, "pen"].sum(), te.loc[~m1, "pen"].sum()

    # 멤버 간 평가 연도 합계 오차 상관 (낮을수록 평균의 효과가 큼)
    def tot_err(pe_):
        t = compute_W(base_ev, pe_, a_ref, wtab)[1].groupby(C["ts"]).agg(A=("A", "sum"), P=("P", "sum"))
        t = t[t["A"] > 0]
        return t["P"] - t["A"]
    refm = "all_l1" if "all_l1" in members else members[0]
    e_ref = tot_err(Pe[refm])
    print("\n[멤버 간 평가 연도 합계 오차 상관] (기준 " + refm + " 대비, 1에 가까우면 같은 방향으로 틀림 → 평균해도 이득 작음)")
    for m in members:
        if m != refm:
            print(f"  {m}: {e_ref.corr(tot_err(Pe[m])):.3f}")

    blends = {f"단독: {m}": {m: 1.0} for m in members}
    if len(members) > 1:
        blends["전체 균등"] = {m: 1.0 for m in members}
    if refm in members and len(members) > 2:
        blends[f"{refm} 50% + 나머지 균등 50%"] = {m: (0.5 if m == refm else 0.5 / (len(members) - 1)) for m in members}
    if {"all_l1", "A_l1", "B_l1"} <= set(members):
        blends["all_l1 + A + B 균등"] = {"all_l1": 1, "A_l1": 1, "B_l1": 1}
        blends["all_l1 50% + A 25% + B 25%"] = {"all_l1": 0.5, "A_l1": 0.25, "B_l1": 0.25}
    if {"A_l1", "B_l1"} <= set(members):
        blends["A + B 균등 (all 제외)"] = {"A_l1": 1, "B_l1": 1}
    if {"all_l1", "all_l2"} <= set(members):
        blends["all_l1 + all_l2 균등"] = {"all_l1": 1, "all_l2": 1}

    res = {}
    for name, sh in blends.items():
        tot = float(sum(sh.values()))
        po = sum(v / tot * Po[m] for m, v in sh.items())
        pe_ = sum(v / tot * Pe[m] for m, v in sh.items())
        res[name] = (score(po, pe_), sh)
    base_key = f"단독: {refm}"
    b = res[base_key][0]
    rows = []
    for name, (sc_, sh) in res.items():
        wo, we, w1, w2 = sc_
        d = {"방식": name,
             "학습OOF 대비(%)": (wo / b[0] - 1) * 100, "평가 대비(%)": (we / b[1] - 1) * 100,
             "정산금 환산(원,근사)": -0.1 * (we - b[1]),
             "전반 대비(%)": (w1 / b[2] - 1) * 100, "후반 대비(%)": (w2 / b[3] - 1) * 100}
        if name == base_key:
            d["판정"] = "(기준)"
        else:
            d["판정"] = ("✓ 시도 가치" if (d["평가 대비(%)"] < -0.2 and d["전반 대비(%)"] < 0 and d["후반 대비(%)"] < 0)
                       else ("△ 소폭" if d["평가 대비(%)"] < 0 else "–"))
        rows.append(d)
    out = pd.DataFrame(rows)
    show = out.copy()
    for col in ("학습OOF 대비(%)", "평가 대비(%)", "전반 대비(%)", "후반 대비(%)"):
        show[col] = show[col].map("{:+.3f}".format)
    show["정산금 환산(원,근사)"] = show["정산금 환산(원,근사)"].map(lambda v: "{:+,.0f}".format(0.0 if abs(v) < 0.5 else v))
    print(f"\n[앙상블 효과] 기준 = {base_key} · W 변화가 음수면 개선 · 보정 계수는 각 조합의 학습 OOF 에서 구함")
    print(show.to_string(index=False))
    print(f"  (기준 W: 학습OOF {b[0]:,.0f} / 평가연도 {b[1]:,.0f}, 참조 구성 + 편향 보정 {mode})")
    print("  ※ '평가 대비'가 -0.2% 보다 확실히 음수이고 전반·후반 모두 음수인 조합이 후보입니다. 그 조합의 비중을 run.py 의 ENSEMBLE 에 넣어 검증하세요.")
    return dict(result=out, blends={k: v[1] for k, v in res.items()})


def missing_forecast_report(top=8):
    """
    평가 구간 예보(test)의 결손이 우리 예측에 어떻게 반영되는지 점검 (결손 시각도 채점 대상이므로 모든 시각의 예측이 필요).
    ① 결손 규모(행 자체가 없는 경우 / 값만 비어 있는 경우)  ② 지점별·날짜별 분포·연속 결손 길이
    ③ 낮 시간 '완전 결손'(모든 회차 없음)이 우리 파이프라인에서 어떤 방식으로 채워지는지: 이전 값 이어쓰기(최대 6행) vs 중앙값
    ④ 페널티 가중(SMP×k)과 설비용량 기준 노출도 — 어느 시각에 얼마나 걸리는지
    """
    D = load_data()
    sites, wtab = D["sites"], D["wtab"]
    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    hours = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    n_s, n_h = len(sites), len(hours)
    tt = np.tile(hours.values, n_s)
    sf = solar_features(pd.Series(tt), pd.Series(np.repeat(sites[C["lat"]].to_numpy(), n_h)),
                        pd.Series(np.repeat(sites[C["lon"]].to_numpy(), n_h)))
    day_any = pd.Series(sf["cosz"].to_numpy() > 0).groupby(tt).any()
    day_hours = pd.DatetimeIndex(day_any.index[day_any.to_numpy()])
    pos = pd.Series(np.arange(len(day_hours)), index=day_hours)
    w_ts = wtab["w"].reindex(hours)
    peak = set(CFG["k_peak_hours"])
    cap_tot = float(sites[C["cap"]].sum())
    print(f"■ 평가 시각 {n_h:,}개 (낮 시간 {len(day_hours):,}개) | 발전소 {n_s}개")

    tot_exp = tot_miss = 0
    expo = pd.Series(0.0, index=hours)               # 시각별 '완전 결손' 설비용량 비중 (제공사 A+B 합, 겹침 포함 상한)
    miss_days = []
    for prov, key, fkey in (("A", C["wa"], "test_fc_a"), ("B", C["wb"], "test_fc_b")):
        fc, vars_ = load_fc(CFG["files"][fkey], flat_ok=False)
        stations = sorted(sites[key].unique())
        d = fc[fc[C["st"]].isin(stations) & fc[C["ts"]].isin(hours)]
        rounds = sorted(d["_r"].unique())
        pres = d[[C["st"], C["ts"], "_r"]].drop_duplicates()
        exp = pd.MultiIndex.from_product([stations, hours, rounds])
        miss = exp.difference(pd.MultiIndex.from_frame(pres))
        tot_exp += len(exp)
        tot_miss += len(miss)
        n_any = int(d[vars_].isna().any(axis=1).sum())
        n_all = int(d[vars_].isna().all(axis=1).sum())
        print(f"\n[제공사 {prov}] 지점 {len(stations)}개 · 회차 {rounds} | 기대 행 {len(exp):,} / 존재 행 {len(pres):,} "
              f"→ 행 자체 결손 {len(miss):,} ({len(miss) / len(exp) * 100:.4f}%) | 값 결손 행(일부 변수 NaN) {n_any:,} · (전부 NaN) {n_all:,}")
        if len(miss):
            md = pd.DataFrame(list(miss), columns=["st", "ts", "r"])
            miss_days.append(md.assign(prov=prov))
            print("  지점별 결손 행 상위:", md.groupby("st").size().sort_values(ascending=False).head(top).to_dict())
        # 완전 결손(모든 회차 없음) — 지점×시각
        full = exp_full = pd.MultiIndex.from_product([stations, hours]).difference(
            pd.MultiIndex.from_frame(pres[[C["st"], C["ts"]]].drop_duplicates()))
        fdf = pd.DataFrame(list(full), columns=["st", "ts"])
        fday = fdf[fdf["ts"].isin(day_hours)]
        n_ff = n_med = 0
        runs = []
        for st_, grp in fday.groupby("st"):
            mp = np.sort(pos.reindex(grp["ts"]).to_numpy())
            present = np.ones(len(day_hours), dtype=bool)
            present[mp] = False
            last = pd.Series(np.where(present, np.arange(len(day_hours)), np.nan)).ffill().to_numpy()
            gap = (np.arange(len(day_hours)) - last)[mp]
            ok = ~np.isnan(gap) & (gap <= 6)
            n_ff += int(ok.sum())
            n_med += int((~ok).sum())
            runs += [len(r) for r in np.split(mp, np.where(np.diff(mp) != 1)[0] + 1)]
        print(f"  낮 시간 '완전 결손'(모든 회차 없음) 지점·시각: {len(fday):,}개 → 이전 값 이어쓰기로 채움 {n_ff:,} / 중앙값으로 채움 {n_med:,}")
        if runs:
            r = pd.Series(runs)
            print(f"  연속 결손 길이(시간): 1~2시간 {int((r <= 2).sum())}건 · 3~6시간 {int(((r > 2) & (r <= 6)).sum())}건 · 7시간 이상 {int((r > 6).sum())}건 (최대 {int(r.max())}시간)")
        capshare = sites.groupby(key)[C["cap"]].sum() / cap_tot
        e = fday.assign(c=fday["st"].map(capshare)).groupby("ts")["c"].sum()
        expo = expo.add(e, fill_value=0.0)

    print(f"\n[전체] 행 자체 결손 {tot_miss:,} / {tot_exp:,} = {tot_miss / tot_exp * 100:.4f}%   (공지: 264행, 0.0247% 와 비교)")
    if miss_days:
        md = pd.concat(miss_days)
        dd = (pd.to_datetime(md["ts"]) - pd.Timedelta(hours=1)).dt.normalize()
        print("  결손이 많은 날짜:", {str(k.date()): int(v) for k, v in dd.value_counts().head(top).items()})

    ex = expo[expo > 0]
    if len(ex):
        wsum_day = float(w_ts.reindex(day_hours).sum())
        wexp = float((w_ts.reindex(ex.index) * ex.clip(upper=1.0)).sum() / wsum_day * 100)
        pk = ex[ex.index.hour.isin(peak)]
        print(f"\n[노출도] 낮 시간 중 결손이 있는 시각 {len(ex)}개 (피크 11~15시 라벨 {len(pk)}개)")
        print(f"  SMP×k 가중·설비용량 기준 노출도: 전체 가중 시각의 {wexp:.3f}%  (결손 지점의 발전량 몫이 그만큼 대체값에 의존)")
        big = (w_ts.reindex(ex.index) * ex.clip(upper=1.0)).sort_values(ascending=False).head(top)
        print("  노출이 큰 시각 상위:", {str(k): f"용량 {ex[k] * 100:.1f}%, SMP×k {w_ts[k]:.0f}" for k in big.index})
    else:
        print("\n[노출도] 낮 시간에 '완전 결손'인 지점·시각이 없습니다 → 다른 회차 대체로 충분히 채워짐")
    print("\n※ 참고: 결손 시각도 제출 forecast 에는 포함되어야 하며(우리 제출은 smp 의 모든 시각을 포함), 위 '채움 방식'이 그 시각 예측의 품질을 정합니다.")
    return dict(total_missing=tot_miss, total_expected=tot_exp, exposure=expo)


def write_env_report(path=None, seeds=None):
    """재현용 환경 기록: Python/패키지 버전 + 핵심 설정 → out/environment.txt"""
    path = path or os.path.join(CFG["out_dir"], "environment.txt")
    lines = [f"python {sys.version.split()[0]}"]
    for _, pkg, _v in REQUIRED:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg} (not installed)")
    keep = ("data_dir", "start_month", "n_estimators", "lgb_params", "n_jobs", "backend", "cv_folds", "opt_iters", "bias_mode", "composition_mode", "ensemble", "site_cf_mode",
            "submit_train_from", "min_cap_kw")
    lines += [f"CFG.{k}={CFG[k]}" for k in keep]
    lines.append(f"seeds={tuple(seeds if seeds is not None else CFG['seeds'])}   # 예측 평균·구성 최적화에 실제 사용한 시드")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"환경 기록 → {path}")


def check_submission(comp_path=None, fc_path=None):
    """제출 전 무효 조건 6가지 점검 (형식 오류도 제출 횟수에 포함되므로 반드시 실행)"""
    comp_path = comp_path or os.path.join(CFG["out_dir"], "composition.csv")
    fc_path = fc_path or os.path.join(CFG["out_dir"], "forecast.csv")
    rd = lambda p: pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    comp, fc = rd(comp_path), rd(fc_path)
    D = load_data()
    sites = D["sites"]
    errs = []
    if set(comp["site_id"]) != set(sites[C["site"]]) or len(comp) != len(sites) or comp["site_id"].duplicated().any():
        errs.append(f"① site_id 가 {len(sites)}개 정확히 한 번씩이 아님 (행 {len(comp)})")
    if not set(comp["group"]).issubset({"G1", "G2"}) or comp["group"].nunique() != 2:
        errs.append("② group 값이 G1/G2 이외이거나 한쪽이 비어 있음")
    cap = comp.merge(sites[[C["site"], C["cap"]]], left_on="site_id", right_on=C["site"]).groupby("group")[C["cap"]].sum()
    if (cap < CFG["min_cap_kw"]).any():
        errs.append(f"③ 집합 용량 미달: {cap.round(1).to_dict()}")
    sm = normalize(read_table(CFG["files"]["smp_eval"], flat_ok=False))
    exp_h = pd.DatetimeIndex(sorted(sm[C["ts"]].unique()))
    tsr = fc["ts"].astype(str)
    if not tsr.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$").all():
        errs.append("⑥ ts 에 타임존 표기 없음")
    try:
        k = fc.assign(_t=_to_kst(fc["ts"]))
        got = set(zip(k["group"], k["_t"]))
        want = {(g, t) for g in ("G1", "G2") for t in exp_h}
        if len(fc) != len(want) or got != want:
            errs.append(f"④ (group, ts) 가 {len(want):,}개와 일치하지 않음 (행 {len(fc):,}, 누락 {len(want - got):,}, 초과 {len(got - want):,})")
    except Exception as e:                                           # noqa
        errs.append(f"④ ts 해석 실패: {e}")
    v = pd.to_numeric(fc["forecast_kwh"], errors="coerce").values
    if (~np.isfinite(v)).any() or (v < 0).any():
        errs.append("⑤ 음수/NaN/inf 포함")
    if errs:
        print("✗ 제출 무효 위험:\n  " + "\n  ".join(errs))
    else:
        print(f"✓ 형식 점검 통과 | composition {len(comp)+1}행(헤더 포함) · forecast {len(fc)+1:,}행(헤더 포함) | 집합 용량 {cap.round(1).to_dict()}")
    return not errs


# ----------------------------------------------------------------------
def _in_notebook():
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


if __name__ == "__main__":
    if _in_notebook():
        print("노트북 환경: inspect() / selftest() / tester(2022, 2023) / year_grid([...]) / make_submission() 를 직접 호출하세요.")
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["inspect", "selftest", "test", "submit", "check"])
        ap.add_argument("--fit", type=int, nargs="+")
        ap.add_argument("--eval", type=int, nargs="+")
        ap.add_argument("--start-month", type=int, default=None)
        ap.add_argument("--data-dir", default=None)
        ap.add_argument("--backend", default=None, choices=["lightgbm", "sklearn"])
        a = ap.parse_args()
        if a.data_dir:
            CFG["data_dir"] = a.data_dir
        if a.backend:
            CFG["backend"] = a.backend
        {"inspect": inspect, "selftest": selftest, "submit": make_submission, "check": check_submission,
         "test": lambda: tester(a.fit, a.eval, a.start_month)}[a.cmd]()
