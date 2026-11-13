"""
OIBC 2026 파이프라인 v3  (집합자원 2개 구성 + 하루전 발전량 예측)
 - 실제 보유 파일명에 맞춤: forecast_wa / forecast_wb / generation / observed_wa / observed_wb (.parquet)
 - 컬럼명은 별칭(alias) 자동 인식 → `inspect` 로 인식 결과 확인 후 필요 시 CFG["aliases"] 수정
 - sites / smp 파일이 없어도 동작
     · sites 없음 → generation 에 들어있는 메타(위경도·설비용량·관측지점)를 사용,
                    설비용량이 없으면 발전소별 99.5분위 발전량으로 대체(추정)
     · smp 없음   → 페널티 가중치 w_t = 1 (오차 절대량 최소화)
 - 기상 실측(observed_*)은 학습 구간에만 존재 → 피처로 사용하지 않음 (예보와 분포 불일치 방지)
 - 시각은 KST, 우측닫힘(ts=01:00 → 00:00~01:00 구간)
 - 위치 원좌표 복원/재식별 시도 금지 → 제공된 (노이즈 포함) 좌표만 그대로 사용

사용법
   python oibc2026_pipeline_v3.py inspect      # 파일 스키마 + 컬럼 인식 결과 확인 (먼저 실행)
   python oibc2026_pipeline_v3.py run          # 전체 파이프라인
   python oibc2026_pipeline_v3.py run --sample-sites 30   # 빠른 테스트

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
        gen="generation.parquet",
        fc_a="forecast_wa.parquet",     # 기상 예보 제공사 A
        fc_b="forecast_wb.parquet",     # 기상 예보 제공사 B
        obs_a="observed_wa.parquet",    # 기상 실측 A (inspect 전용, 피처 미사용)
        obs_b="observed_wb.parquet",    # 기상 실측 B (inspect 전용, 피처 미사용)
    ),
    optional_files=dict(
        sites="sites.csv",              # 있으면 사용, 없으면 generation 에서 메타 추출
        smp="smp.csv",                  # 있으면 w=k*SMP, 없으면 w=1
    ),
    # 표준 컬럼명 : 실제 파일에서 찾아볼 후보 이름들 (앞에서부터 먼저 일치하는 것 사용)
    aliases=dict(
        site=["site_id", "plant_id", "plant", "site", "id", "발전소", "발전소id"],
        ts=["ts", "datetime", "timestamp", "time", "date_time", "dt", "target_time", "시각", "일시"],
        gen=["generation", "gen", "power", "kwh", "energy", "발전량"],
        cap=["capacity", "cap", "capacity_kw", "capacity_mw", "installed_capacity", "설비용량"],
        lat=["lat", "latitude", "위도"],
        lon=["lon", "lng", "longitude", "경도"],
        station=["station_id", "stn_id", "stn", "station", "obs_id", "관측지점", "지점"],
        smp=["smp", "price", "SMP"],
        k=["k", "weight", "k_weight"],
        fc_round=["round", "fc_round", "run", "cycle"],
        fc_issued=["issued_at", "issue_time", "base_time", "init_time", "announce_time", "발표시각"],
    ),
    cap_scale=1.0,          # 발전량 단위 / 설비용량 단위 보정 (예: gen[kWh], cap[kW] → 1.0, cap[MW] → 1000)
    n_groups=2,
    min_group_size=20,       # 각 집합자원 최소 발전소 수 (규정 확인 후 수정)
    cv_folds=4,              # 주(week) 단위 블록 교차검증 (OOF 오차로 집합자원 최적화)
    tol=0.0,                 # 허용오차율(그룹 설비용량 대비). 규정에 면제구간이 있으면 설정
    opt_iters=300_000,
    seeds=(0, 1, 2),
    composition_cols=("site_id", "group"),   # 제출 컬럼명 (제출하기 탭 확인 후 수정)
    forecast_cols=("ts", "group", "pred"),   # 집합자원별 예측 제출 컬럼명
)
# 표준 컬럼명 (정규화 후 코드 전체에서 사용)
C = dict(site="site_id", lat="lat", lon="lon", cap="capacity", station="station_id",
         ts="ts", smp="smp", k="k", gen="generation", fc_round="round", fc_issued="issued_at")



# ----------------------------------------------------------------------
# 입출력 + 컬럼 정규화
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


def read_optional(name):
    try:
        return read_table(name)
    except FileNotFoundError:
        return None


def normalize(df):
    """별칭 → 표준 컬럼명으로 변경, 시각은 KST naive datetime 으로 통일"""
    df = df.copy()
    low = {c.lower(): c for c in df.columns}
    for std, cands in CFG["aliases"].items():
        target = C[std]
        if target in df.columns:
            continue
        for cand in cands:
            if cand.lower() in low and low[cand.lower()] != target:
                df = df.rename(columns={low[cand.lower()]: target})
                break
    # 시각이 index 에 들어있는 경우
    if C["ts"] not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": C["ts"]})
    if C["ts"] in df.columns:
        t = pd.to_datetime(df[C["ts"]])
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        df[C["ts"]] = t
    return df


def inspect():
    names = {**CFG["files"], **CFG["optional_files"]}
    for key, name in names.items():
        try:
            p = _path(name)
        except FileNotFoundError:
            tag = "(선택 파일)" if key in CFG["optional_files"] else "← CFG['files'] 확인"
            print(f"[{key}] {name} 없음 {tag}\n")
            continue
        raw = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, nrows=200_000)
        df = normalize(raw)
        print(f"[{key}] {p}  ({len(raw):,}행)")
        print(df.dtypes.to_string())
        found = {k: v for k, v in C.items() if v in df.columns}
        print("  인식된 표준 컬럼:", found)
        if C["ts"] in df.columns:
            print(f"  시각 범위: {df[C['ts']].min()} ~ {df[C['ts']].max()}")
        print(df.head(3).to_string(), "\n")


# ----------------------------------------------------------------------
# 예보 처리: (provider, round) 별 wide 변환 + 결손 보간 (행 제외 금지)
# ----------------------------------------------------------------------
def load_forecast():
    """forecast_wa / forecast_wb 를 합쳐 (제공사, 라운드) 별 wide 로 변환 + 결손 보간"""
    parts = []
    for prov, key_ in (("A", "fc_a"), ("B", "fc_b")):
        d = normalize(read_table(CFG["files"][key_]))
        d["_prov"] = prov
        parts.append(d)
    fc = pd.concat(parts, ignore_index=True)

    key = C["station"] if C["station"] in fc.columns else C["site"]
    if key not in fc.columns:
        raise KeyError(f"예보 파일에서 관측지점/발전소 키 컬럼을 못 찾음: {list(fc.columns)}")

    # 라운드(전일 10시/17시) 구분
    if C["fc_round"] in fc.columns:
        fc["_round"] = fc[C["fc_round"]].astype(str)
    elif C["fc_issued"] in fc.columns:
        fc["_round"] = pd.to_datetime(fc[C["fc_issued"]]).dt.hour.astype(str)
    else:
        fc["_round"] = "0"
    prov, rnd = "_prov", "_round"

    skip = {key, C["ts"], prov, rnd, C["fc_round"], C["fc_issued"],
            C["site"], C["station"], C["lat"], C["lon"], C["cap"], C["gen"]}
    vars_ = [c for c in fc.columns
             if c not in skip and pd.api.types.is_numeric_dtype(fc[c])
             and not pd.api.types.is_bool_dtype(fc[c])]
    fc[vars_] = fc[vars_].astype("float32")

    # 중복 행은 평균 후 wide 변환
    wide = fc.groupby([key, C["ts"], prov, rnd])[vars_].mean().unstack([prov, rnd])
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
        wide[cols] = wide[cols].apply(lambda s_: s_.fillna(out[f"{v}__mean"]))
    wide = pd.concat([wide, pd.DataFrame(out, index=wide.index)], axis=1)

    # 남은 결손: 관측지점별 시간 보간 → 전역 중앙값
    wide = wide.groupby(level=0, group_keys=False).apply(
        lambda g: g.interpolate(limit=6, limit_direction="both"))
    wide = wide.fillna(wide.median(numeric_only=True))
    wide = wide.reset_index()
    return wide, key, [c for c in wide.columns if c not in (key, C["ts"])]


def load_sites(gen):
    """sites 파일이 있으면 사용, 없으면 generation 에서 발전소 메타 추출"""
    sites = read_optional(CFG["optional_files"]["sites"])
    if sites is not None:
        sites = normalize(sites)
        print("sites 파일 사용")
    else:
        meta = [C[k] for k in ("lat", "lon", "cap", "station") if C[k] in gen.columns]
        sites = gen.groupby(C["site"])[meta].first().reset_index() if meta else \
            gen[[C["site"]]].drop_duplicates().reset_index(drop=True)
        print(f"sites 파일 없음 → generation 에서 메타 추출 (사용 컬럼: {meta})")

    if C["cap"] not in sites.columns:
        q = gen.groupby(C["site"])[C["gen"]].quantile(0.995)
        sites[C["cap"]] = sites[C["site"]].map(q).values
        print("[경고] 설비용량 컬럼 없음 → 발전소별 99.5분위 발전량으로 대체 (추정치)")
    for k, default in ((C["lat"], 36.0), (C["lon"], 128.0)):
        if k not in sites.columns:
            sites[k] = default
            print(f"[경고] {k} 컬럼 없음 → {default} 로 대체 (태양위치 피처 정확도 저하)")
    return sites


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
    meta_cols = list(dict.fromkeys([C["site"], C["lat"], C["lon"], C["cap"], fc_key]))
    df = df.merge(sites[meta_cols].drop_duplicates(C["site"]), on=C["site"], how="left")
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
    gen = normalize(read_table(CFG["files"]["gen"]))
    for k in ("site", "ts", "gen"):
        if C[k] not in gen.columns:
            raise KeyError(f"generation 에서 '{C[k]}' 컬럼을 못 찾음 → CFG['aliases'] 수정 ({list(gen.columns)})")

    sites = load_sites(gen)
    if sample_sites:
        sites = sites.sample(sample_sites, random_state=0)
    n_sites_total = len(sites)

    fc_wide, fc_key, wvars = load_forecast()
    if fc_key not in sites.columns:
        raise KeyError(f"sites/generation 에 '{fc_key}' (기상 예보 키) 컬럼이 없습니다. "
                       f"발전소↔관측지점 매핑 정보를 확인하세요")

    smp = read_optional(CFG["optional_files"]["smp"])
    if smp is not None:
        smp = normalize(smp)
    else:
        print("smp 파일 없음 → 페널티 가중치 w=1")
    gen = gen[gen[C["site"]].isin(sites[C["site"]])]

    # ---- 학습 데이터 (주간만) ----
    tr = attach(gen[[C["ts"], C["site"], C["gen"]]].dropna(), sites, fc_wide, fc_key, wvars)
    tr["cf"] = (tr[C["gen"]] / tr["cap_"]).clip(0, 1.3).astype("float32")
    tr = tr[tr["cosz"] > 0.02].reset_index(drop=True)
    feats = (["cosz", "etr", "ha", "doy_sin", "doy_cos", "hour", C["lat"], C["lon"], "cap_", "site_cf"]
             + wvars)

    # ---- 평가 구간 격자: 학습 마지막 이후의 SMP 시각 전체 (결손 시각 포함) ----
    last = gen[C["ts"]].max()
    if smp is not None:
        eval_ts = pd.Series(sorted(smp.loc[smp[C["ts"]] > last, C["ts"]].unique()))
    else:                                                     # SMP 없으면 예보 시각 기준
        eval_ts = pd.Series(sorted(fc_wide.loc[fc_wide[C["ts"]] > last, C["ts"]].unique()))
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

    if smp is not None:
        wdf = smp.drop_duplicates(C["ts"]).set_index(C["ts"])
        w = (wdf[C["k"]] * wdf[C["smp"]] if C["k"] in wdf.columns else wdf[C["smp"]]).reindex(all_ts)
        w = w.fillna(w.mean()).values.astype("float32")
    else:
        w = np.ones(len(all_ts), dtype="float32")

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


def _in_notebook():
    import sys
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


if __name__ == "__main__":
    if _in_notebook():
        # Colab/Jupyter: argparse 를 쓰지 않음 → 다음 셀에서 inspect() / run() 을 직접 호출
        print("노트북 환경 감지: 다음 셀에서 inspect() 또는 run() 을 직접 호출하세요.")
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["inspect", "run"])
        ap.add_argument("--sample-sites", type=int, default=None)
        args = ap.parse_args()
        inspect() if args.cmd == "inspect" else run(args.sample_sites)
