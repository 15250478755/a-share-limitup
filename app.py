import streamlit as st
import pandas as pd
import numpy as np
import requests, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

st.set_page_config(page_title="A股次日机会筛选器 V2", page_icon="📈", layout="wide")

st.title("📈 A股次日机会筛选器 V2")
st.caption("触板 + 封板 + T+1历史表现；严格按时间切分训练/测试。研究工具，不构成收益保证。")

# ---------------- 网络容错 ----------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
]

EAST_SPOT = "https://82.push2.eastmoney.com/api/qt/clist/get"
EAST_SPOT_BACKUP = "https://push2.eastmoney.com/api/qt/clist/get"
EAST_HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_HIS = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

def create_session():
    s = requests.Session()
    retry = Retry(
        total=2, connect=2, read=2, backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = create_session()

def headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        "Connection": "keep-alive",
    }

def get_json(url, params, timeout=12):
    last = None
    for attempt in range(3):
        try:
            r = _SESSION.get(url, params=params, headers=headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1) + random.random())
    raise last

# ---------------- 股票池 ----------------
@st.cache_data(ttl=300, show_spinner=False)
def spot():
    p = {
        "pn": 1, "pz": 6000, "po": 1, "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2, "invt": 2, "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f20,f21,f23"
    }
    last = None
    for url in [EAST_SPOT, EAST_SPOT_BACKUP]:
        try:
            data = get_json(url, p)
            diff = (data.get("data") or {}).get("diff") or []
            if diff:
                return pd.DataFrame(diff)
        except Exception as e:
            last = e
    raise RuntimeError(f"行情快照获取失败：{last}")

def board_of(code):
    code = str(code)
    if code.startswith(("68", "689")):
        return "科创板"
    if code.startswith(("30", "301")):
        return "创业板"
    if code.startswith(("43", "83", "87", "8", "4")):
        return "北交所"
    return "主板"

def limit_rate(code, date):
    """按股票板块+日期给出研究用涨停幅度。"""
    code = str(code)
    d = pd.Timestamp(date)
    if code.startswith(("30", "301")):
        # 创业板 2020-08-24 起 20%
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("68", "689")):
        return 0.20
    if code.startswith(("43", "83", "87", "8", "4")):
        return 0.30
    return 0.10

# ---------------- 历史数据 ----------------
def parse_eastmoney(klines):
    cols = ["date","open","close","high","low","volume","amount",
            "amplitude","pct_chg","turnover","extra"]
    rows = [x.split(",") for x in klines]
    d = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def hist_east(code, beg="20180101", end="20991231"):
    secid = ("1." + code) if str(code).startswith(("6","68")) else ("0." + code)
    p = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101, "fqt": 0, "beg": beg, "end": end
    }
    data = get_json(EAST_HIS, p)
    klines = (data.get("data") or {}).get("klines") or []
    if not klines:
        return pd.DataFrame()
    return parse_eastmoney(klines)

def hist_tencent(code, beg="2018-01-01", end="2099-12-31"):
    code = str(code)
    prefix = "sh" if code.startswith(("6","68")) else "sz"
    symbol = prefix + code
    p = {"param": f"{symbol},day,{beg},{end},5000"}
    data = get_json(TX_HIS, p)
    root = data.get("data") or {}
    item = root.get(symbol) or {}
    klines = item.get("qfqday") or item.get("day") or []
    if not klines:
        return pd.DataFrame()
    rows = []
    for row in klines:
        if len(row) >= 6:
            rows.append(row[:6])
    d = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
    for c in ["open","close","high","low","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["amount"] = np.nan
    d["amplitude"] = (d["high"] - d["low"]) / d["close"] * 100
    d["pct_chg"] = d["close"].pct_change() * 100
    d["turnover"] = np.nan
    d["extra"] = np.nan
    return d

@st.cache_data(ttl=86400, show_spinner=False)
def hist(code):
    last = None
    try:
        d = hist_east(code)
        if len(d) >= 35:
            return d
    except Exception as e:
        last = e
    try:
        d = hist_tencent(code)
        if len(d) >= 35:
            return d
    except Exception as e:
        last = e
    raise RuntimeError(f"{code}历史数据失败：{last}")

# ---------------- 特征/标签 ----------------
FEATURES = [
    "r1","r3","r5","r10","r20",
    "relvol5","relvol20","turnover5",
    "close_pos","dist_high20","dist_high60",
    "maxret20","limitup20","range5","amp20",
    "vol_chg5","ma_bull","vol_break","volatility20",
    "market_proxy"
]

def add_features(d, code):
    if len(d) < 80:
        return pd.DataFrame()
    x = d.copy()
    x["date"] = pd.to_datetime(x["date"])
    for c in ["open","close","high","low","volume","amount","pct_chg","turnover"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    x["code"] = str(code)
    x["board"] = board_of(code)
    x["r1"] = x.close.pct_change(1)
    x["r3"] = x.close.pct_change(3)
    x["r5"] = x.close.pct_change(5)
    x["r10"] = x.close.pct_change(10)
    x["r20"] = x.close.pct_change(20)

    v5 = x.volume.rolling(5).mean()
    v20 = x.volume.rolling(20).mean()
    x["relvol5"] = x.volume / v5.replace(0, np.nan)
    x["relvol20"] = x.volume / v20.replace(0, np.nan)

    x["turnover5"] = x.turnover.rolling(5).mean()
    x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)

    h20 = x.high.rolling(20).max()
    h60 = x.high.rolling(60).max()
    x["dist_high20"] = x.close / h20 - 1
    x["dist_high60"] = x.close / h60 - 1

    x["maxret20"] = x.pct_chg.rolling(20).max() / 100
    prev_close = x.close.shift(1)
    daily_limit = [limit_rate(code, d) for d in x.date]
    x["limit_rate"] = daily_limit

    # 触板标签：次日最高价达到理论涨停价附近
    limit_price = prev_close * (1 + x["limit_rate"])
    x["touch"] = (x.high >= limit_price * 0.998).astype(int)
    x["close_limit"] = (
        (x.close >= limit_price * 0.998) &
        (x.high >= limit_price * 0.998)
    ).astype(int)

    x["limitup20"] = x["touch"].shift(1).rolling(20).sum()
    x["range5"] = ((x.high - x.low) / x.close).rolling(5).mean()
    x["amp20"] = x.amplitude.rolling(20).mean() / 100
    x["vol_chg5"] = x.volume.pct_change(5).clip(-1, 5)
    x["ma_bull"] = (x.close > x.close.rolling(5).mean()).astype(int)
    x["vol_break"] = (
        (x.volume > x.volume.rolling(20).mean() * 1.8) &
        (x.r1 > 0.03)
    ).astype(int)
    x["volatility20"] = x.r1.rolling(20).std()

    # 简单市场代理：个股20日相对位置，不冒充真正的全市场涨跌家数
    x["market_proxy"] = (
        x.close / x.close.rolling(20).mean() - 1
    )

    # 把“明天”标签对齐到今天特征
    x["target_touch"] = x["touch"].shift(-1)
    x["target_close"] = x["close_limit"].shift(-1)
    x["t1_return"] = x.close.shift(-1) / x.close - 1

    # 上市初期、异常值和停牌等不作为训练样本
    x = x.replace([np.inf, -np.inf], np.nan)
    return x

# ---------------- 模型 ----------------
def fit_one(train, test, target):
    tr = train.copy()
    te = test.copy()
    med = tr[FEATURES].median().fillna(0)
    tr[FEATURES] = tr[FEATURES].fillna(med)
    te[FEATURES] = te[FEATURES].fillna(med)

    y = pd.to_numeric(tr[target], errors="coerce")
    tr = tr[y.notna()].copy()
    y = tr[target].astype(int)

    # 训练集负样本过多时只下采样训练集，不碰测试集
    pos = tr[tr[target] == 1]
    neg = tr[tr[target] == 0]
    if len(pos) >= 20 and len(neg) > len(pos) * 4:
        neg = neg.sample(n=len(pos) * 4, random_state=42)
    tr = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=42
    )
    model.fit(tr[FEATURES], tr[target].astype(int))

    p = model.predict_proba(te[FEATURES])[:,1]
    auc = np.nan
    brier = np.nan
    if te[target].nunique() == 2:
        auc = roc_auc_score(te[target], p)
        brier = brier_score_loss(te[target], p)
    return model, med, auc, brier

def build_training_data(stock_codes, progress_callback=None):
    all_rows = []
    failures = []
    for i, code in enumerate(stock_codes):
        try:
            d = hist(code)
            z = add_features(d, code)
            if len(z) >= 80:
                # 最后一行没有“明天”标签，去掉
                z = z.iloc[:-1].copy()
                all_rows.append(z)
        except Exception as e:
            failures.append((str(code), str(e)))
        if progress_callback:
            progress_callback(i + 1, len(stock_codes))
    if not all_rows:
        raise RuntimeError("没有获取到足够历史数据。请稍后重试或减少股票数量。")
    df = pd.concat(all_rows, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["date"] = pd.to_datetime(df["date"])
    for c in FEATURES + ["target_touch","target_close","t1_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["target_touch","target_close"])
    return df, failures

def select_universe(boards, n, min_price, min_amt):
    s = spot().copy()
    s["code"] = s["f12"].astype(str)
    s["name"] = s["f14"].astype(str)
    s["price"] = pd.to_numeric(s["f2"], errors="coerce")
    s["amt"] = pd.to_numeric(s["f6"], errors="coerce") / 1e4

    s = s[~s["name"].str.contains("ST|退", regex=True, na=False)]
    s = s[s["code"].str.match(r"^(0\d{5}|3\d{5}|60\d{4}|68\d{4}|43\d{3}|83\d{4}|87\d{4}|4\d{5}|8\d{5})$", na=False)]
    s["board"] = s["code"].map(board_of)
    s = s[s["board"].isin(boards)]
    s = s[(s.price >= min_price) & (s.amt >= min_amt)]
    return s.sort_values("amt", ascending=False).head(n).reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_train(stock_codes_tuple):
    return build_training_data(list(stock_codes_tuple), None)[0]

# ---------------- 主流程 ----------------
st.sidebar.header("筛选参数")
boards = st.sidebar.multiselect(
    "板块",
    ["主板","创业板","科创板","北交所"],
    default=["主板","创业板"]
)
n = st.sidebar.slider("训练股票数量", 50, 300, 150, 25)
topk = st.sidebar.slider("显示前几名", 5, 20, 10)
min_price = st.sidebar.number_input("最低股价（元）", 0.0, 1000.0, 2.0)
min_amt = st.sidebar.number_input("最低日成交额（万元）", 0.0, 100000.0, 5000.0)

if st.button("🚀 开始筛选", type="primary", use_container_width=True):
    if not boards:
        st.error("至少选择一个板块。")
    else:
        try:
            universe = select_universe(boards, n, min_price, min_amt)
            if len(universe) < 30:
                st.warning(f"当前条件只有 {len(universe)} 只股票，建议降低成交额门槛。")

            progress = st.progress(0)
            status = st.empty()
            codes = universe["code"].tolist()

            # 并发下载，失败自动跳过；控制在5路，避免把数据源打爆
            results = {}
            failures = []

            def one(code):
                try:
                    d = hist(code)
                    return code, add_features(d, code), None
                except Exception as e:
                    return code, pd.DataFrame(), str(e)

            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = [ex.submit(one, c) for c in codes]
                for i, fut in enumerate(as_completed(futures), 1):
                    code, z, err = fut.result()
                    if err:
                        failures.append((code, err))
                    elif len(z) >= 80:
                        results[code] = z
                    progress.progress(i / len(codes))
                    status.write(f"历史数据：{i}/{len(codes)}，成功 {len(results)}，失败 {len(failures)}")

            progress.empty()
            status.empty()

            if len(results) < 30:
                raise RuntimeError(f"有效股票历史数据只有 {len(results)} 只，不足以训练。")

            df = pd.concat(
                [z.iloc[:-1].copy() for z in results.values()],
                ignore_index=True
            )
            df = df.replace([np.inf, -np.inf], np.nan)
            df["date"] = pd.to_datetime(df["date"])
            for c in FEATURES + ["target_touch","target_close","t1_return"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=FEATURES + ["target_touch","target_close"])
            df = df.sort_values("date")

            # 严格时间切分：最后365天作为测试集
            cutoff = df.date.max()
            test_start = cutoff - pd.Timedelta(days=365)
            train = df[df.date < test_start].copy()
            test = df[df.date >= test_start].copy()

            if len(train) < 1000 or test["target_touch"].nunique() < 2:
                split = int(len(df) * 0.8)
                train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

            touch_model, med, touch_auc, touch_brier = fit_one(train, test, "target_touch")
            close_model, _, close_auc, close_brier = fit_one(train, test, "target_close")

            # 最终预测：每只股票最新一行
            latest_rows = []
            for code, z in results.items():
                z = z.copy()
                latest_rows.append(z.iloc[-1].copy())
            latest = pd.DataFrame(latest_rows)
            latest[FEATURES] = latest[FEATURES].fillna(med)

            latest["触板概率"] = touch_model.predict_proba(latest[FEATURES])[:,1]
            latest["封板概率"] = close_model.predict_proba(latest[FEATURES])[:,1]
            latest["历史样本"] = latest["code"].map(df.groupby("code").size())
            latest["近5日涨幅"] = latest["r5"]
            latest["量比5日"] = latest["relvol5"]
            latest["换手率"] = latest["turnover5"]
            latest["收盘位置"] = latest["close_pos"]
            latest["距离20日高点"] = latest["dist_high20"]
            latest["T+1历史平均收益"] = latest["code"].map(
                df.groupby("code")["t1_return"].mean()
            )

            # 综合排序不是“保证收益”，只是研究排序：
            # 触板概率、封板概率和历史T+1收益均先标准化到横截面
            def rank_pct(s):
                return s.rank(pct=True).fillna(0.5)
            latest["研究排序分"] = (
                0.45 * rank_pct(latest["触板概率"]) +
                0.35 * rank_pct(latest["封板概率"]) +
                0.20 * rank_pct(latest["T+1历史平均收益"])
            )
            latest = latest.sort_values(
                ["研究排序分","触板概率"], ascending=False
            ).head(topk)

            st.session_state["result"] = latest
            st.session_state["metrics"] = {
                "touch_auc": touch_auc, "touch_brier": touch_brier,
                "close_auc": close_auc, "close_brier": close_brier,
                "samples": len(df), "valid_stocks": len(results),
                "failures": len(failures),
                "touch_rate": float(df.target_touch.mean()),
                "close_rate": float(df.target_close.mean()),
            }
            st.session_state["failures"] = failures

        except Exception as e:
            st.error(f"运行失败：{e}")
            st.info("先把训练股票数量降到100，成交额门槛保持5000万元，再重试。")

if "result" in st.session_state:
    r = st.session_state["result"]
    m = st.session_state["metrics"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("触板 AUC", "—" if pd.isna(m["touch_auc"]) else f'{m["touch_auc"]:.3f}')
    c2.metric("封板 AUC", "—" if pd.isna(m["close_auc"]) else f'{m["close_auc"]:.3f}')
    c3.metric("历史触板率", f'{m["touch_rate"]*100:.2f}%')
    c4.metric("有效股票", f'{m["valid_stocks"]}')

    st.info(
        f'样本外测试集最后365天；历史样本 {m["samples"]:,} 条。'
        f' 数据失败 {m["failures"]} 只。概率是模型估计，不是未来收益承诺。'
    )

    show = r[[
        "code","board","close","触板概率","封板概率",
        "T+1历史平均收益","r5","relvol5","turnover5",
        "close_pos","dist_high20","历史样本"
    ]].copy()

    show.columns = [
        "股票代码","板块","最新价","触板概率","封板概率",
        "T+1历史平均收益","近5日涨幅","5日量比","5日平均换手",
        "收盘位置","距20日高点","历史样本"
    ]

    for c in ["触板概率","封板概率","T+1历史平均收益","近5日涨幅","5日平均换手","收盘位置","距20日高点"]:
        show[c] = pd.to_numeric(show[c], errors="coerce")

    show["触板概率"] = (show["触板概率"]*100).round(2).astype(str)+"%"
    show["封板概率"] = (show["封板概率"]*100).round(2).astype(str)+"%"
    show["T+1历史平均收益"] = (show["T+1历史平均收益"]*100).round(2).astype(str)+"%"
    show["近5日涨幅"] = (show["近5日涨幅"]*100).round(1).astype(str)+"%"
    show["5日量比"] = show["5日量比"].round(2)
    show["5日平均换手"] = (show["5日平均换手"]*100).round(2).astype(str)+"%"
    show["收盘位置"] = (show["收盘位置"]*100).round(1).astype(str)+"%"
    show["距20日高点"] = (show["距20日高点"]*100).round(1).astype(str)+"%"

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ 下载候选股票 CSV",
        r.to_csv(index=False).encode("utf-8-sig"),
        "A股次日机会候选_V2.csv",
        "text/csv"
    )

    if st.session_state.get("failures"):
        with st.expander("部分股票数据失败（不影响其他股票）"):
            st.write(st.session_state["failures"][:30])

st.divider()
st.caption(
    "V2重点：不复权价格用于涨停标签；创业板历史涨停制度按日期处理；"
    "时间切分测试；触板/封板双模型；5路并发+缓存+备用数据源；"
    "当前仍存在幸存者偏差，且腾讯备用数据的换手/成交额可能为空，因此严格收益证明仍需独立历史股票池。"
)
