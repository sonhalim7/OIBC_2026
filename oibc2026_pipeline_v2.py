"""
OIBC 2026 파이프라인 v2  (집합자원 2개 구성 + 하루전 발전량 예측)

제공 데이터 명세 반영 사항
 - sites.csv        : 발전소 메타(위경도[노이즈 적용], 설비용량, 배정된 기상관측지점) 253개소 전부 composition에 포함
 - SMP              : 시각별 SMP + 시간대 가중치 k
 - 발전량 실적       : 학습 구간만 존재 → 평가 구간에서는 실적 lag 피처 사용 불가
 - 기상 예보 A/B     : 전일 10시/17시 두 라운드, 학습·평가 모두 제공 (일부 결손 → 보간, 행 제외 금지)
 - 기상 실측         : 학습 구간만 존재 → 피처로 쓰지 않음(예보와 분포 불일치 방지)
 - 시각은 KST, 우측닫힘(ts=01:00 → 00:00~01:00 구간)
 - 위치 원좌표 복원/재식별 시도 금지 → 제공된 (노이즈 포함) 좌표만 그대로 사용

사용법
   python oibc2026_pipeline_v2.py inspect      # 파일 스키마 확인 (먼저 실행)
   python oibc2026_pipeline_v2.py run          # 전체 파이프라인
   python oibc2026_pipeline_v2.py run --sample-sites 30   # 빠른 테스트

!! 실제 파일명/컬럼명은 아래 CFG 를 `inspect` 결과에 맞게 수정하세요.
!! 페널티 산식과 composition/예측 제출 형식은 '대회 규정'·'제출하기' 탭 기준으로 확정하세요.
"""
import argparse
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# [CONFIG]  ← inspect 결과를 보고 수정
# ----------------------------------------------------------------------
CFG = dict(
    data_dir="data",
    out_dir="out",
    files=dict(
        sites="sites.csv",
        smp="smp.csv",
        gen="generation.csv",          # 또는 .parquet (확장자 자동 대체)
        fc="forecast.csv",             # 기상 예보 (long 형식 가정)
    ),
    cols=dict(
        site="site_id", lat="lat", lon="lon", cap="capacity", station="station_id",
        ts="ts", smp="smp", k="k", gen="generation",
        fc_provider="provider",        # 제공사 A/B 구분 컬럼
        fc_round="round",              # 10시/17시 라운드 구분 컬럼 (없으면 None)
        fc_issued="issued_at",         # round 컬럼이 없을 때 발행시각에서 라운드 유도 (없으면 None)
    ),
    cap_scale=1.0,          # 발전량 단위 / 설비용량 단위 보정 (예: gen[kWh], cap[kW] → 1.0, cap[MW] → 1000)
    eval_grid_source="smp",  # 평가 구간 시각 목록을 SMP 파일 기준으로 생성 (결손 시각 누락 방지)
    n_groups=2,
    min_group_size=20,       # 각 집합자원 최소 발전소 수 (규정 확인 후 수정)
    cv_folds=4,              # 주(week) 단위 블록 교차검증 (OOF 오차로 집합자원 최적화)
    tol=0.0,                 # 허용오차율(그룹 설비용량 대비). 규정에 면제구간이 있으면 설정
    opt_iters=300_000,
    seeds=(0, 1, 2),
    composition_cols=("site_id", "group"),   # 제출 컬럼명 (제출하기 탭 확인 후 수정)
    forecast_cols=("ts", "group", "pred"),   # 집합자원별 예측 제출 컬럼명
)
C = CFG["cols"]


# ----------------------------------------------------------------------
# 입출력
# ----------------------------------------------------------------------
def _path(name):
    p = os.path.join(CFG["data_dir"], name)
    if os.path.exists(p):
        return p
    root, ext = os.path.splitext(p)
    alt = root + (".parquet" if ext == ".csv" else ".csv")
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(p)


def read_table(name, **kw):
    p = _path(name)
    if p.endswith(".parquet"):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, **kw)


def inspect():
    for key, name in CFG["files"].items():
        try:
            p = _path(name)
        except FileNotFoundError:
            print(f"[{key}] {name} 없음 → CFG['files'] 수정 필요\n")
            continue
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, nrows=200_000)
        print(f"[{key}] {p}  (표시 {len(df):,}행)")
        print(df.dtypes.to_string())
        print(df.head(3).to_string(), "\n")


# ----------------------------------------------------------------------
# 예보 처리: (provider, round) 별 wide 변환 + 결손 보간 (행 제외 금지)
# ----------------------------------------------------------------------
def load_forecast(sites_key):
    fc = read_table(CFG["files"]["fc"])
    fc[C["ts"]] = pd.to_datetime(fc[C["ts"]])
    key = C["station"] if C["station"] in fc.columns else C["site"]

    rnd = C["fc_round"]
    if not rnd or rnd not in fc.columns:
        if C["fc_issued"] and C["fc_issued"] in fc.columns:
            fc["_round"] = pd.to_datetime(fc[C["fc_issued"]]).dt.hour.astype(str)
        else:
            fc["_round"] = "0"
        rnd = "_round"
    prov = C["fc_provider"] if C["fc_provider"] in fc.columns else None
    if prov is None:
        fc["_prov"] = "X"
        prov = "_prov"

    id_cols = {key, C["ts"], prov, rnd, C["fc_issued"]}
    vars_ = [c for c in fc.columns if c not in id_cols and pd.api.types.is_numeric_dtype(fc[c])]
    fc[vars_] = fc[vars_].astype("float32")

    wide = fc.set_index([key, C["ts"], prov, rnd])[vars_].unstack([prov, rnd])
    wide.columns = [f"{v}__{p}_{r}" for v, p, r in wide.columns]

    # 전체 시간 격자로 reindex → 행 자체가 빠진 결손도 복원
    full = pd.date_range(fc[C["ts"]].min(), fc[C["ts"]].max(), freq="h")
    keys = wide.index.get_level_values(0).unique()
    wide = wide.reindex(pd.MultiIndex.from_product([keys, full], names=[key, C["ts"]]))

    out = {}
    for v in vars_:
        cols = [c for c in wide.columns if c.startswith(v + "__")]
        out[f"{v}__mean"] = wide[cols].mean(axis=1)           # 제공사·라운드 평균
        out[f"{v}__spread"] = wide[cols].std(axis=1)           # 예보 불확실성
        wide[cols] = wide[cols].apply(lambda s: s.fillna(out[f"{v}__mean"]))
    wide = pd.concat([wide, pd.DataFrame(out, index=wide.index)], axis=1)

    # 남은 결손: 관측지점별 시간 보간 → 전역 중앙값
    wide = wide.groupby(level=0, group_keys=False).apply(
        lambda g: g.interpolate(limit=6, limit_direction="both"))
    wide = wide.fillna(wide.median(numeric_only=True))
    return wide.reset_index(), key, [c for c in wide.columns]


# ----------------------------------------------------------------------
# 태양 위치 (우측닫힘: 구간 중앙 = ts - 30분, KST=UTC+9)
# ----------------------------------------------------------------------
def solar_features(ts, lat, lon):
    mid = ts - pd.Timedelta(minutes=30)
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
        "cosz": cosz.astype("float32"),
        "etr": (i0 * np.clip(cosz, 0, None)).astype("float32"),
        "ha": ha.astype("float32"),
        "doy_sin": np.sin(2 * np.pi * doy / 365).astype("float32"),
        "doy_cos": np.cos(2 * np.pi * doy / 365).astype("float32"),
        "hour": mid.dt.hour.values.astype("int8"),
    }, index=ts.index)


def attach(df, sites, fc_wide, fc_key, wvars):
    df = df.merge(sites[[C["site"], C["lat"], C["lon"], C["cap"], fc_key]].drop_duplicates(C["site"]),
                  on=C["site"], how="left")
    df = df.merge(fc_wide, on=[fc_key, C["ts"]], how="left")
    df[wvars] = df[wvars].fillna(fc_wide[wvars].median())     # 최종 안전장치 (행 제외 금지)
    df = pd.concat([df, solar_features(df[C["ts"]], df[C["lat"]], df[C["lon"]])], axis=1)
    df["cap_"] = df[C["cap"]].astype("float32") * CFG["cap_scale"]
    return df


# ----------------------------------------------------------------------
# 모델
# ----------------------------------------------------------------------
def make_model(seed):
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=700, learning_rate=0.05, num_leaves=127,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 min_child_samples=100, random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                             max_leaf_nodes=63, random_state=seed)


def with_site_cf(df, tr):
    """발전소별 평균 이용률 (학습 폴드 기준으로만 계산 → 누수 방지)"""
    m = tr.groupby(C["site"])["cf"].mean()
    out = df.copy()
    out["site_cf"] = out[C["site"]].map(m).fillna(tr["cf"].mean()).astype("float32")
    return out


def fit_predict(tr, te, feats, seed=0):
    tr = with_site_cf(tr, tr)
    te = with_site_cf(te, tr)
    mdl = make_model(seed)
    mdl.fit(tr[feats], tr["cf"])
    return np.clip(mdl.predict(te[feats]), 0, 1.3).astype("float32")


# ----------------------------------------------------------------------
# 집합자원 최적화 (SA 기반 swap/move)
#   cost = Σ_t w_t · max(|Σ_{i∈G} err_it| − tol·cap_G, 0)   (그룹별 합)
#   w_t = k_t · SMP_t  ← 실제 페널티 산식으로 교체 필요
# ----------------------------------------------------------------------
def group_cost(S, capg, w, tol):
    return float((w * np.clip(np.abs(S) - tol * capg, 0, None)).sum())


def optimize_groups(Et, caps, w, iters, seed, tol, min_size):
    """Et: (N, T) 오차(예측-실적) 행렬"""
    rng = np.random.default_rng(seed)
    N, T = Et.shape
    order = rng.permutation(N)
    assign = np.zeros(N, dtype=np.int8)
    assign[order[N // 2:]] = 1
    S = [Et[assign == g].sum(0) for g in (0, 1)]
    cg = [caps[assign == g].sum() for g in (0, 1)]
    cost = sum(group_cost(S[g], cg[g], w, tol) for g in (0, 1))
    best, best_assign = cost, assign.copy()
    t0 = cost * 0.002 + 1e-9
    for it in range(iters):
        temp = t0 * (1 - it / iters)
        if rng.random() < 0.7:                                  # swap
            i = rng.choice(np.flatnonzero(assign == 0))
            j = rng.choice(np.flatnonzero(assign == 1))
            S0 = S[0] - Et[i] + Et[j]; S1 = S[1] + Et[i] - Et[j]
            c0 = cg[0] - caps[i] + caps[j]; c1 = cg[1] + caps[i] - caps[j]
            new = group_cost(S0, c0, w, tol) + group_cost(S1, c1, w, tol)
            if new < cost or rng.random() < np.exp(-(new - cost) / max(temp, 1e-12)):
                assign[i], assign[j] = 1, 0
                S, cg, cost = [S0, S1], [c0, c1], new
        else:                                                   # move
            i = rng.integers(N)
            a, b = assign[i], 1 - assign[i]
            if (assign == a).sum() <= min_size:
                continue
            Sa = S[a] - Et[i]; Sb = S[b] + Et[i]
            ca = cg[a] - caps[i]; cb = cg[b] + caps[i]
            new = group_cost(Sa, ca, w, tol) + group_cost(Sb, cb, w, tol)
            if new < cost or rng.random() < np.exp(-(new - cost) / max(temp, 1e-12)):
                assign[i] = b
                S[a], S[b], cg[a], cg[b], cost = Sa, Sb, ca, cb, new
        if cost < best:
            best, best_assign = cost, assign.copy()
    return best_assign, best


def eval_cost(Et, caps, w, assign, tol):
    return sum(group_cost(Et[assign == g].sum(0), caps[assign == g].sum(), w, tol) for g in (0, 1))


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def run(sample_sites=None):
    os.makedirs(CFG["out_dir"], exist_ok=True)
    sites = read_table(CFG["files"]["sites"])
    if sample_sites:
        sites = sites.sample(sample_sites, random_state=0)
    n_sites_total = len(sites)

    fc_wide, fc_key, wvars = load_forecast(None)
    if fc_key not in sites.columns:
        raise KeyError(f"sites 에 '{fc_key}' (기상 예보 키) 컬럼이 없습니다. CFG['cols'] 확인")

    smp = read_table(CFG["files"]["smp"])
    smp[C["ts"]] = pd.to_datetime(smp[C["ts"]])
    gen = read_table(CFG["files"]["gen"])
    gen[C["ts"]] = pd.to_datetime(gen[C["ts"]])
    gen = gen[gen[C["site"]].isin(sites[C["site"]])]

    # ---- 학습 데이터 (주간만) ----
    tr = attach(gen[[C["ts"], C["site"], C["gen"]]].dropna(), sites, fc_wide, fc_key, wvars)
    tr["cf"] = (tr[C["gen"]] / tr["cap_"]).clip(0, 1.3).astype("float32")
    tr = tr[tr["cosz"] > 0.02].reset_index(drop=True)
    feats = (["cosz", "etr", "ha", "doy_sin", "doy_cos", "hour", C["lat"], C["lon"], "cap_", "site_cf"]
             + wvars)

    # ---- 평가 구간 격자: 학습 마지막 이후의 SMP 시각 전체 (결손 시각 포함) ----
    last = gen[C["ts"]].max()
    eval_ts = pd.Series(sorted(smp.loc[smp[C["ts"]] > last, C["ts"]].unique()))
    ev = pd.MultiIndex.from_product([sites[C["site"]].values, eval_ts],
                                    names=[C["site"], C["ts"]]).to_frame(index=False)
    ev = attach(ev, sites, fc_wide, fc_key, wvars)
    print(f"학습 행 {len(tr):,} | 평가 시각 {len(eval_ts):,} × 발전소 {n_sites_total}")

    # ---- OOF 예측 (주 단위 블록 CV) → 오차 행렬 ----
    K = CFG["cv_folds"]
    fold = ((tr[C["ts"]] - tr[C["ts"]].min()).dt.days // 7) % K
    oof = np.zeros(len(tr), dtype="float32")
    for k in range(K):
        m = (fold == k).values
        oof[m] = fit_predict(tr[~m], tr[m], feats)
        mae = np.abs(oof[m] - tr.loc[m, "cf"]).mean()
        print(f"  fold {k}: CF-MAE {mae:.4f}")
    tr["err"] = (oof - tr["cf"].values) * tr["cap_"].values        # 예측-실적 (에너지 단위)

    all_ts = np.sort(gen[C["ts"]].unique())
    t_idx = pd.Series(np.arange(len(all_ts)), index=all_ts)
    site_ids = sites[C["site"]].values
    s_idx = pd.Series(np.arange(len(site_ids)), index=site_ids)
    Et = np.zeros((len(site_ids), len(all_ts)), dtype="float32")
    Et[s_idx[tr[C["site"]]].values, t_idx[tr[C["ts"]]].values] = tr["err"].values
    caps = sites.set_index(C["site"]).loc[site_ids, C["cap"]].values.astype("float64") * CFG["cap_scale"]

    wdf = smp.set_index(C["ts"])
    w = (wdf[C["k"]] * wdf[C["smp"]]).reindex(all_ts) if C["k"] in wdf.columns else wdf[C["smp"]].reindex(all_ts)
    w = w.fillna(w.mean()).values.astype("float32")

    # ---- 과적합 진단: 앞 70%로 최적화 → 뒤 30%에서 검증 ----
    cut = int(Et.shape[1] * 0.7)
    a_h, _ = optimize_groups(np.ascontiguousarray(Et[:, :cut]), caps, w[:cut],
                             CFG["opt_iters"] // 3, 0, CFG["tol"], CFG["min_group_size"])
    rnd = (np.arange(len(site_ids)) % 2).astype(np.int8)
    c_opt = eval_cost(Et[:, cut:], caps, w[cut:], a_h, CFG["tol"])
    c_rnd = eval_cost(Et[:, cut:], caps, w[cut:], rnd, CFG["tol"])
    print(f"[holdout] 최적화 {c_opt:,.0f} vs 단순배분 {c_rnd:,.0f} ({1 - c_opt / c_rnd:+.1%})")

    # ---- 전체 구간 최적화 (여러 시드 중 최선) ----
    best = (None, np.inf)
    for s in CFG["seeds"]:
        a, c = optimize_groups(Et, caps, w, CFG["opt_iters"], s, CFG["tol"], CFG["min_group_size"])
        print(f"  seed {s}: cost {c:,.0f}")
        if c < best[1]:
            best = (a, c)
    assign = best[0]

    # ---- 최종 학습 → 평가 구간 예측 ----
    ev_pred = np.mean([fit_predict(tr, ev, feats, seed=s) for s in CFG["seeds"]], axis=0)
    ev["pred"] = np.where(ev["cosz"] > 0.0, ev_pred * ev["cap_"].values, 0.0)   # 야간 0
    gmap = dict(zip(site_ids, assign + 1))
    ev["group"] = ev[C["site"]].map(gmap)

    # ---- 제출 파일 ----
    comp = pd.DataFrame({CFG["composition_cols"][0]: site_ids,
                         CFG["composition_cols"][1]: assign + 1})
    agg = ev.groupby([C["ts"], "group"])["pred"].sum().reset_index()
    agg.columns = CFG["forecast_cols"]

    # 검증: 전 발전소 포함, 결손 시각 없음
    assert comp[CFG["composition_cols"][0]].nunique() == n_sites_total, "발전소 누락"
    assert set(comp[CFG["composition_cols"][1]]) == {1, 2}, "집합자원은 정확히 2개여야 함"
    assert agg["pred"].notna().all() and len(agg) == len(eval_ts) * 2, "평가 시각 누락/NaN"

    comp.to_csv(os.path.join(CFG["out_dir"], "composition.csv"), index=False)
    agg.to_csv(os.path.join(CFG["out_dir"], "forecast.csv"), index=False)
    print(f"집합자원 크기: {np.bincount(assign)} → {CFG['out_dir']}/composition.csv, forecast.csv 저장")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "run"])
    ap.add_argument("--sample-sites", type=int, default=None)
    args = ap.parse_args()
    inspect() if args.cmd == "inspect" else run(args.sample_sites)
