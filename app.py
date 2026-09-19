import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

st.set_page_config(page_title="A股尾盘→次日涨停筛选器", page_icon="📈", layout="wide")

st.title("📈 A股尾盘 → 次日涨停机会筛选器")
st.caption("只研究“今天尾盘尚未涨停、明天可能触板/封板”的股票；不会把今天已经涨停的股票当成候选。")

# ============================================================
# 网络
# ============================================================
EAST_SPOT = "https://82.push2.eastmoney.com/api/qt/clist/get"
EAST_HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_HIS = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

SESSION = requests.Session()
retry = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=1.2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                  "AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://quote.eastmoney.com/",
}

def get_json(url, params, timeout=15):
    r = SESSION.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ============================================================
# 股票池
# ============================================================
def board_of(code):
    code = str(code)
    if code.startswith(("60", "00")):
        return "主板"
    if code.startswith(("30", "301")):
        return "创业板"
    if code.startswith(("68", "689")):
        return "科创板"
    if code.startswith(("43", "83", "87", "4", "8")):
        return "北交所"
    return "其他"

def limit_rate(code, date=None):
    """研究用历史涨停幅度。创业板 2020-08-24 起20%；
    科创板自开板即20%；北交所按历史制度简化为10%/30%阶段。
    非ST普通主板按10%。
    """
    code = str(code)
    d = pd.Timestamp(date) if date is not None else pd.Timestamp.today()
    if code.startswith(("300", "301")):
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("43", "83", "87", "8", "4")):
        # 北交所历史规则复杂；当前研究版采用当前常规30%。
        # 严格历史回测应进一步接入制度变更表。
        return 0.30
    return 0.10

@st.cache_data(ttl=180, show_spinner=False)
def get_spot():
    p = {
        "pn": 1, "pz": 6000, "po": 1, "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2, "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:0+t:81,m:1+t:2,m:1+t:23,m:1+t:27",
        "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f20,f21,f23"
    }
    j = get_json(EAST_SPOT, p)
    diff = (j.get("data") or {}).get("diff") or []
    if not diff:
        raise RuntimeError("没有拿到实时股票池。")
    d = pd.DataFrame(diff)
    d["code"] = d["f12"].astype(str)
    d["name"] = d["f14"].astype(str)
    d["price"] = pd.to_numeric(d["f2"], errors="coerce")
    d["pct"] = pd.to_numeric(d["f3"], errors="coerce") / 100.0
    d["amount"] = pd.to_numeric(d["f6"], errors="coerce")  # 元
    d["turnover"] = pd.to_numeric(d["f8"], errors="coerce")
    d["board"] = d["code"].map(board_of)
    return d

# ============================================================
# 历史日线：不复权，专门用于涨停标签
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def east_history(code, beg="20180101", end="20991231"):
    secid = ("1." + code) if str(code).startswith(("6", "68")) else ("0." + code)
    p = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 0,
        "beg": beg,
        "end": end,
    }
    j = get_json(EAST_HIS, p)
    kl = (j.get("data") or {}).get("klines") or []
    if not kl:
        return pd.DataFrame()
    rows = [x.split(",") for x in kl]
    cols = ["date","open","close","high","low","volume","amount",
            "amplitude","pct_chg","turnover","extra"]
    d = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d.dropna(subset=["date","close","high","low"]).sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def tx_history(code, beg="20180101", end="20991231"):
    code = str(code)
    prefix = "sh" if code.startswith(("6", "68")) else "sz"
    symbol = prefix + code
    p = {"param": f"{symbol},day,{beg},{end},5000"}
    try:
        j = get_json(TX_HIS, p)
        data = (j.get("data") or {}).get(symbol) or {}
        arr = data.get("qfqday") or data.get("day") or []
        if not arr:
            return pd.DataFrame()
        # Tencent常见格式：[date,open,close,high,low,volume,...]
        rows = []
        for x in arr:
            if len(x) < 6:
                continue
            rows.append([
                x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]),
                float(x[5]), np.nan, np.nan, np.nan, np.nan
            ])
        d = pd.DataFrame(rows, columns=[
            "date","open","close","high","low","volume",
            "amount","amplitude","pct_chg","turnover"
        ])
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        for c in ["open","close","high","low","volume"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["pct_chg"] = d["close"].pct_change()
        return d.dropna(subset=["date","close","high","low"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def get_history(code):
    try:
        d = east_history(code)
        if len(d) >= 80:
            return d, "东方财富"
    except Exception:
        pass
    try:
        d = tx_history(code)
        if len(d) >= 80:
            return d, "腾讯"
    except Exception:
        pass
    return pd.DataFrame(), "失败"

# ============================================================
# 特征
# ============================================================
FEATURES = [
    "r1","r3","r5","r10","r20",
    "relvol5","relvol20",
    "turnover5","close_pos",
    "dist_high20","dist_high60",
    "limitup20","range5","amp20",
    "vol_chg5","ma_bull","vol_break","volatility20",
]

def build_daily_features(d, code):
    if len(d) < 80:
        return pd.DataFrame()

    x = d.copy().reset_index(drop=True)
    x["r1"] = x.close.pct_change(1)
    x["r3"] = x.close.pct_change(3)
    x["r5"] = x.close.pct_change(5)
    x["r10"] = x.close.pct_change(10)
    x["r20"] = x.close.pct_change(20)

    x["relvol5"] = x.volume / x.volume.rolling(5).mean()
    x["relvol20"] = x.volume / x.volume.rolling(20).mean()
    x["turnover5"] = x.turnover.rolling(5).mean()

    den = (x.high - x.low).replace(0, np.nan)
    x["close_pos"] = (x.close - x.low) / den

    high20 = x.high.rolling(20).max()
    high60 = x.high.rolling(60).max()
    x["dist_high20"] = x.close / high20 - 1
    x["dist_high60"] = x.close / high60 - 1

    rates = np.array([limit_rate(code, dt) for dt in x.date])
    prev_close = x.close.shift(1)
    x["limitup20"] = (
        (x.high / prev_close - 1) >= (rates - 0.002)
    ).rolling(20).sum()

    x["range5"] = ((x.high - x.low) / x.close).rolling(5).mean()
    x["amp20"] = (x.amplitude / 100).rolling(20).mean()
    x["vol_chg5"] = x.volume.pct_change(5).clip(-1, 5)
    x["ma_bull"] = (x.close > x.close.rolling(5).mean()).astype(int)
    x["vol_break"] = (
        (x.volume > x.volume.rolling(20).mean() * 1.8)
        & (x.r1 > 0.03)
    ).astype(int)
    x["volatility20"] = x.r1.rolling(20).std()

    # T+1目标：明天最高价触及涨停
    next_rate = np.array([limit_rate(code, dt) for dt in x.date])
    next_close = x.close.shift(1)
    # 对每一行，下一交易日的涨停幅度使用下一日日期
    future_high = x.high.shift(-1)
    future_close = x.close
    x["touch_target"] = (
        future_high >= future_close * (1 + next_rate) * 0.995
    ).astype(int)

    x["close_limit_target"] = (
        x.close.shift(-1) >= future_close * (1 + next_rate) * 0.995
    ).astype(int)

    # 次日收盘收益
    x["t1_return"] = x.close.shift(-1) / x.close - 1

    return x

# ============================================================
# 历史训练样本
# ============================================================
def load_training_codes(spot, n):
    s = spot.copy()
    s = s[~s["name"].str.contains("ST|退", regex=True, na=False)]
    s = s[s["board"].isin(["主板","创业板","科创板"])]
    s = s[(s["price"] >= 2) & (s["amount"] >= 5e7)]
    return s.sort_values("amount", ascending=False).head(n)["code"].tolist()

def prepare_one(code):
    d, source = get_history(code)
    if len(d) < 80:
        return None
    z = build_daily_features(d, code)
    if len(z) < 80:
        return None
    z["code"] = code
    z["source"] = source
    return z.iloc[:-1].copy()

def train_models(n):
    spot = get_spot()
    codes = load_training_codes(spot, n)

    rows = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(prepare_one, c): c for c in codes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            code = futs[fut]
            try:
                z = fut.result()
                if z is not None and len(z):
                    rows.append(z)
            except Exception:
                pass
            progress.progress(done / max(len(codes),1))
            status.write(f"历史数据：{done}/{len(codes)}")

    progress.empty()
    status.empty()

    if not rows:
        raise RuntimeError("没有拿到足够历史数据。请稍后重试。")

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([np.inf, -np.inf], np.nan)

    need = FEATURES + ["touch_target","close_limit_target","t1_return"]
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["touch_target","close_limit_target"])

    # 严格按时间：最后365天测试
    cutoff = df["date"].max()
    test_start = cutoff - pd.Timedelta(days=365)
    tr = df[df["date"] < test_start].copy()
    te = df[df["date"] >= test_start].copy()

    if len(tr) < 1000 or te["touch_target"].nunique() < 2:
        split = int(len(df) * 0.8)
        tr = df.iloc[:split].copy()
        te = df.iloc[split:].copy()

    med = tr[FEATURES].median().fillna(0)
    Xtr = tr[FEATURES].fillna(med)
    Xte = te[FEATURES].fillna(med)

    def fit_model(ycol):
        pos = tr[ycol].astype(int)
        base = HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            min_samples_leaf=35,
            learning_rate=0.05,
            l2_regularization=2.0,
            random_state=42
        )
        base.fit(Xtr, pos)
        return base

    touch_model = fit_model("touch_target")
    close_model = fit_model("close_limit_target")

    touch_auc = np.nan
    close_auc = np.nan
    touch_brier = np.nan
    close_brier = np.nan

    if te["touch_target"].nunique() == 2:
        p = touch_model.predict_proba(Xte)[:,1]
        touch_auc = roc_auc_score(te["touch_target"], p)
        touch_brier = brier_score_loss(te["touch_target"], p)

    if te["close_limit_target"].nunique() == 2:
        p = close_model.predict_proba(Xte)[:,1]
        close_auc = roc_auc_score(te["close_limit_target"], p)
        close_brier = brier_score_loss(te["close_limit_target"], p)

    # 最近历史数据中的真实T+1收益统计
    t1 = pd.to_numeric(te["t1_return"], errors="coerce").dropna()
    return {
        "touch_model": touch_model,
        "close_model": close_model,
        "median": med,
        "touch_auc": touch_auc,
        "close_auc": close_auc,
        "touch_brier": touch_brier,
        "close_brier": close_brier,
        "samples": len(df),
        "test_t1_mean": float(t1.mean()) if len(t1) else np.nan,
        "test_t1_median": float(t1.median()) if len(t1) else np.nan,
    }

# ============================================================
# 尾盘实时特征
# ============================================================
def make_live_features(row, d, code):
    # d是截至最近交易日的历史日线。
    if len(d) < 80:
        return None

    x = d.copy().reset_index(drop=True)

    # 用今天实时价格替换最后一根K线的收盘价，形成“尾盘状态”
    price = float(row["price"])
    amount = float(row["amount"])
    turnover = float(row["turnover"])

    if not np.isfinite(price) or price <= 0:
        return None

    last = x.iloc[-1].copy()
    prev = x.iloc[-2]

    close = price
    last["close"] = close
    last["amount"] = amount
    last["turnover"] = turnover

    # 当前实时涨幅已由行情快照提供；如果历史最后一天不是今天，则补成当前状态
    x.iloc[-1] = last

    z = build_daily_features(x, code)
    if z.empty:
        return None

    feat = z.iloc[-1].copy()
    feat["code"] = code
    feat["实时涨幅"] = float(row["pct"])
    feat["实时成交额"] = amount
    feat["实时换手"] = turnover
    feat["price"] = close
    return feat

def is_already_limit_up(row):
    code = row["code"]
    pct = row["pct"]
    rate = limit_rate(code)
    # 给四舍五入留一点空间
    return pct >= rate - 0.003

def live_screen(models, boards, n, topk, min_amt):
    spot = get_spot()
    s = spot[spot["board"].isin(boards)].copy()
    s = s[~s["name"].str.contains("ST|退", regex=True, na=False)]
    s = s[(s["price"] >= 2) & (s["amount"] >= min_amt)]
    # 核心：排除已经涨停/接近涨停的票
    s = s[~s.apply(is_already_limit_up, axis=1)]
    # 排除跌停附近
    s = s[s["pct"] > -0.08]
    # 尾盘策略重点：强势但还没有涨停
    s = s[s["pct"] >= 0.02]
    s = s.sort_values(["pct","amount"], ascending=[False,False]).head(n)

    results = []
    progress = st.progress(0)
    status = st.empty()

    def one(row):
        code = row["code"]
        d, source = get_history(code)
        if len(d) < 80:
            return None
        feat = make_live_features(row, d, code)
        if feat is None:
            return None
        X = pd.DataFrame([feat[FEATURES].fillna(models["median"])])
        touch = float(models["touch_model"].predict_proba(X)[:,1][0])
        closep = float(models["close_model"].predict_proba(X)[:,1][0])

        # 一个简单的研究排序：触板概率为主，封板概率为辅，避免已经涨停
        score = 0.60*touch + 0.30*closep + 0.10*min(max(float(row["pct"])/0.10,0),1)

        return {
            "股票代码": code,
            "名称": row["name"],
            "板块": row["board"],
            "现价": row["price"],
            "今日涨幅": row["pct"],
            "触板概率": touch,
            "封板概率": closep,
            "成交额(万元)": row["amount"]/1e4,
            "换手率": row["turnover"],
            "综合研究分": score,
            "数据源": source,
        }

    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(one, row): row["code"] for _, row in s.iterrows()}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception:
                pass
            progress.progress(done/max(len(s),1))
            status.write(f"尾盘候选：{done}/{len(s)}")

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("综合研究分", ascending=False).head(topk)
    return out.reset_index(drop=True)

# ============================================================
# UI
# ============================================================
st.sidebar.header("尾盘参数")
boards = st.sidebar.multiselect(
    "板块",
    ["主板","创业板","科创板"],
    default=["主板","创业板"]
)
train_n = st.sidebar.slider("训练股票数量", 80, 300, 150, 10)
scan_n = st.sidebar.slider("尾盘候选扫描数量", 50, 400, 180, 10)
topk = st.sidebar.slider("最终显示", 5, 30, 10)
min_amt = st.sidebar.number_input("最低成交额（万元）", 1000.0, 100000.0, 5000.0, 500.0)

st.info(
    "使用方式：14:45以后再运行。系统只保留“今日尚未涨停、但已经明显走强”的股票，"
    "目标是预测下一交易日触板/封板，而不是把今天已经涨停的股票再筛一遍。"
)

if st.button("🚀 训练并执行尾盘筛选", type="primary", use_container_width=True):
    now = pd.Timestamp.now()
    if now.hour < 14 or (now.hour == 14 and now.minute < 35):
        st.warning("现在还不到尾盘。建议14:45以后运行，避免把盘中状态误当成尾盘状态。")
    else:
        with st.spinner("正在训练模型并扫描尾盘强势但未涨停股票……"):
            try:
                models = train_models(train_n)
                result = live_screen(models, boards, scan_n, topk, min_amt)

                st.session_state["models_info"] = models
                st.session_state["result"] = result
                st.session_state["run_time"] = str(now)

                if result.empty:
                    st.warning("本次没有得到合格候选。不要为了凑数量强行放宽条件。")
                else:
                    st.success(f"完成：{len(result)}只候选。")
            except Exception as e:
                st.error(f"运行失败：{e}")

if "models_info" in st.session_state:
    m = st.session_state["models_info"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("触板AUC", "—" if pd.isna(m["touch_auc"]) else f'{m["touch_auc"]:.3f}')
    c2.metric("封板AUC", "—" if pd.isna(m["close_auc"]) else f'{m["close_auc"]:.3f}')
    c3.metric("历史样本", f'{m["samples"]:,}')
    c4.metric("测试期T+1中位收益", "—" if pd.isna(m["test_t1_median"]) else f'{m["test_t1_median"]*100:.2f}%')

if "result" in st.session_state and len(st.session_state["result"]):
    r = st.session_state["result"].copy()
    show = r.copy()
    for c in ["今日涨幅","触板概率","封板概率"]:
        show[c] = (show[c]*100).round(2).astype(str) + "%"
    show["现价"] = show["现价"].round(2)
    show["成交额(万元)"] = show["成交额(万元)"].round(0)
    show["换手率"] = show["换手率"].round(2).astype(str) + "%"
    show["综合研究分"] = show["综合研究分"].round(4)

    st.subheader("🎯 尾盘候选：今天没涨停 → 研究明天是否可能涨停")
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button(
        "下载尾盘候选 CSV",
        r.to_csv(index=False).encode("utf-8-sig"),
        "A股尾盘次日候选.csv",
        "text/csv"
    )

st.divider()
st.caption(
    "研究限制：实时尾盘筛选使用当前实时行情；历史训练主要使用日线收盘状态，因此不是完整的历史1分钟级尾盘回放。"
    "若要严格验证14:45/14:50/14:55不同买入时点，需要接入完整历史分钟数据并做逐日无未来函数回测。"
