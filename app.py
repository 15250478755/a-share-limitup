import streamlit as st
import pandas as pd
import numpy as np
import requests, time, random, os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

st.set_page_config(page_title="A股次日涨停概率筛选器", page_icon="📈", layout="wide")
st.title("📈 A股次日涨停概率筛选器")
st.caption("T-1 → T：根据历史量价特征估计次日触及涨停的概率。研究工具，不构成收益保证。")


# ==================== 反爬核心：动态请求头 + 会话复用 ====================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]

REFERERS = [
    "https://quote.eastmoney.com/center/gridlist.html",
    "https://www.eastmoney.com/",
    "https://data.eastmoney.com/",
]


def create_session():
    """创建带重试机制的 requests Session，自动重试被拦截的请求"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,                          # 最多重试5次
        backoff_factor=2,                 # 退避因子：等待 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # 设置连接超时和读取超时
    session.timeout = 20
    return session


def get_headers():
    """每次请求随机切换 UA 和 Referer，模拟真人浏览"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": random.choice(REFERERS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
    }


# 全局 Session
_SESSION = create_session()


def safe_request(url, params, retries=3):
    """带自动重试的请求函数，处理 RemoteDisconnected"""
    last_err = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, headers=get_headers(), timeout=20)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                wait = random.uniform(3, 8) * (attempt + 1)   # 递增退避
                time.sleep(wait)
    raise last_err


# ==================== 数据获取 ====================

EAST = "https://82.push2.eastmoney.com/api/qt/clist/get"
EAST_BACKUP = "https://push2.eastmoney.com/api/qt/clist/get"
HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


@st.cache_data(ttl=300, show_spinner=False)
def spot():
    """获取全市场实时快照（带备用域名 + 反爬重试）"""
    p = {
        "pn": 1, "pz": 5000, "po": 1, "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2, "invt": 2, "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": "f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f20,f21,f23",
    }
    # 先试主域名，失败后自动切换备用域名
    for url in [EAST, EAST_BACKUP]:
        try:
            r = safe_request(url, p)
            d = r.json()["data"]["diff"]
            return pd.DataFrame(d)
        except Exception:
            time.sleep(random.uniform(2, 5))
            continue
    raise RuntimeError("行情快照获取失败，请稍后重试或检查网络环境。")


@st.cache_data(ttl=86400, show_spinner=False)
def hist(code, beg="20180101", end="20991231"):
    """获取个股历史K线（带反爬重试 + 动态UA）"""
    p = {
        "secid": ("1." + code if code.startswith(("6", "68")) else "0." + code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101, "fqt": 1, "beg": beg, "end": end,
    }
    r = safe_request(HIS, p)
    kl = r.json().get("data", {}).get("klines", [])
    if not kl:
        return pd.DataFrame()
    rows = [x.split(",") for x in kl]
    cols = ["date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "pct_chg", "turnover", "extra"]
    d = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


# ==================== 特征工程（修复版） ====================

def limit_rate(code):
    """根据板块返回涨停幅度"""
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("430", "440", "830", "8", "4")):
        return 0.30
    return 0.10


def make_features(d, code):
    if len(d) < 35:
        return pd.DataFrame()
    x = d.copy()
    x["code"] = code
    x["r1"] = x.close.pct_change(1)
    x["r3"] = x.close.pct_change(3)
    x["r5"] = x.close.pct_change(5)
    x["r10"] = x.close.pct_change(10)
    x["r20"] = x.close.pct_change(20)
    x["relvol5"] = x.volume / x.volume.rolling(5).mean()
    x["relvol20"] = x.volume / x.volume.rolling(20).mean()
    x["turnover5"] = x.turnover.rolling(5).mean()
    x["close_pos"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
    x["maxret20"] = x.pct_chg.rolling(20).max() / 100

    # ✅ 修复：统一用 high 判定触板，与 target 口径一致
    limit_val = limit_rate(code)
    x["limitup20"] = ((x.high / x.close.shift(1) - 1) >= (limit_val - 0.002)).rolling(20).sum()

    x["range5"] = ((x.high - x.low) / x.close).rolling(5).mean()
    x["amp20"] = x.amplitude.rolling(20).mean()

    # ✅ 修复：clip 极端值，防止复牌/停牌导致爆量
    x["vol_chg5"] = x.volume.pct_change(5).clip(-1, 5)

    x["ma_bull"] = (x.close > x.close.rolling(5).mean()).astype(int)
    x["high_near20"] = ((x.high - x.close) / x.close).rolling(20).max()

    # ✅ 新增因子：放量突破 + 波动率
    x["vol_break"] = ((x.volume > x.volume.rolling(20).mean() * 1.8) & (x.r1 > 0.03)).astype(int)
    x["volatility20"] = x.r1.rolling(20).std()

    # 目标：次日最高价触及涨停
    x["target"] = (x.high.shift(-1) >= x.close * (1 + limit_val) * 0.995).astype(int)
    return x


# ✅ 更新特征列表（含新增因子）
FEATURES = [
    "r1", "r3", "r5", "r10", "r20",
    "relvol5", "relvol20",
    "turnover", "turnover5",
    "close_pos", "maxret20", "limitup20",
    "range5", "amp20", "vol_chg5",
    "ma_bull", "high_near20",
    "vol_break", "volatility20",
]


# ==================== 模型训练 ====================

def train_model(n, boards, min_price, min_amt):
    s = spot()
    s = s[~s["f14"].astype(str).str.contains("ST|退", regex=True, na=False)].copy()

    # ✅ 修复：00开头的主板股用 5 位数字
    s = s[s["f12"].astype(str).str.match(r"^(0\d{5}|3\d{5}|60\d{4}|68\d{4})$", na=False)]

    main_board = s["f12"].astype(str).str.startswith(("60", "00"))
    chi_next = s["f12"].astype(str).str.startswith(("30", "301"))
    star_market = s["f12"].astype(str).str.startswith(("68", "689"))
    bse = s["f12"].astype(str).str.startswith(("43", "83", "87", "4"))

    mask = pd.Series(False, index=s.index)
    if "主板" in boards: mask |= main_board
    if "创业板" in boards: mask |= chi_next
    if "科创板" in boards: mask |= star_market
    if "北交所" in boards: mask |= bse
    s = s[mask]

    s["price"] = pd.to_numeric(s["f2"], errors="coerce")

    # ✅ 修复：f6 单位是"元"，转换为"万元"
    s["amt"] = pd.to_numeric(s["f6"], errors="coerce").fillna(0) / 1e4

    s = s[(s["price"] >= min_price) & (s["amt"] >= min_amt)]
    s = s.sort_values("amt", ascending=False).head(n)

    all_rows, latest = [], []
    progress = st.progress(0)
    status = st.empty()

    for i, code in enumerate(s.f12.astype(str)):
        try:
            d = hist(code)
            z = make_features(d, code)
            if len(z):
                # ✅ 过滤：最后一行 K 线必须是最近 10 天内的（排除停牌股）
                last_date = pd.to_datetime(z["date"].iloc[-1])
                if last_date < pd.Timestamp.today() - pd.Timedelta(days=10):
                    continue
                all_rows.append(z.iloc[:-1].copy())
                latest.append(z.iloc[-1:].copy())
        except Exception:
            pass
        progress.progress((i + 1) / len(s))
        status.write(f"正在读取 {i+1}/{len(s)}：{code}")

        # ✅ 修复：请求间隔从 0.15-0.35s 加大到 1.2-2.5s，大幅降低封IP概率
        time.sleep(random.uniform(1.2, 2.5))

    progress.empty()
    status.empty()

    if not all_rows:
        raise RuntimeError("没有取得历史数据，请检查网络或稍后重试。")

    df = pd.concat(all_rows, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in FEATURES + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + ["target"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    cutoff = df.date.max()
    train_end = cutoff - pd.Timedelta(days=365)

    # ✅ 修复：先切分，再用训练集中位数填补（消除数据泄露）
    tr = df[df["date"] < train_end].copy()
    te = df[df["date"] >= train_end].copy()

    if len(tr) < 500 or te["target"].nunique() < 2:
        tr = df.iloc[: int(len(df) * 0.8)].copy()
        te = df.iloc[int(len(df) * 0.8):].copy()

    med = tr[FEATURES].median()
    tr[FEATURES] = tr[FEATURES].fillna(med)
    te[FEATURES] = te[FEATURES].fillna(med)

    # 类别平衡
    pos = tr[tr.target == 1]
    neg = tr[tr.target == 0]

    if len(pos) > 0 and len(neg) > len(pos) * 2:
        neg = neg.sample(n=len(pos) * 2, random_state=42)

    tr_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    base_model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=15,
        min_samples_leaf=30, learning_rate=0.06,
        l2_regularization=1.5, random_state=42,
    )
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(tr_balanced[FEATURES], tr_balanced.target)

    if te["target"].nunique() == 2:
        auc = roc_auc_score(te.target, model.predict_proba(te[FEATURES])[:, 1])
        brier = brier_score_loss(te.target, model.predict_proba(te[FEATURES])[:, 1])
    else:
        auc, brier = np.nan, np.nan

    # 最新一行预测
    latest_df = pd.concat(latest, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for col in FEATURES:
        latest_df[col] = pd.to_numeric(latest_df[col], errors="coerce")

    # ✅ 修复：最新一行也用训练集中位数填补
    latest_df[FEATURES] = latest_df[FEATURES].fillna(med)
    latest_df = latest_df.dropna(subset=FEATURES)

    latest_df["触板概率"] = model.predict_proba(latest_df[FEATURES])[:, 1]
    latest_df["历史样本"] = latest_df["code"].map(df.groupby("code").size())
    latest_df["近5日涨幅"] = latest_df.r5
    latest_df["量比5日"] = latest_df.relvol5
    latest_df["换手率"] = latest_df.turnover
    latest_df["收盘位置"] = latest_df.close_pos
    latest_df["股票代码"] = latest_df.code
    latest_df["最新价"] = latest_df.close

    return latest_df.sort_values("触板概率", ascending=False), auc, brier, len(df)


# ==================== 界面 ====================

st.sidebar.header("筛选参数")
boards = st.sidebar.multiselect("选择板块", ["主板", "创业板", "科创板", "北交所"], default=["主板", "创业板"])
n = st.sidebar.slider("训练股票数量", 50, 800, 200, 50)
topk = st.sidebar.slider("显示前几名", 5, 30, 10)
threshold = st.sidebar.slider("最低触板概率", 0.05, 0.50, 0.15, 0.01)
min_price = st.sidebar.number_input("最低股价(元)", 0.0, 1000.0, 2.0)
min_amt = st.sidebar.number_input("最低日成交额(万元)", 0.0, 100000.0, 5000.0)

if st.button("🚀 开始筛选", type="primary", use_container_width=True):
    with st.spinner("正在抓取历史数据并训练模型，第一次可能需要几分钟……"):
        try:
            result, auc, brier, samples = train_model(n, boards, min_price, min_amt)
            result = result[result["触板概率"] >= threshold].head(topk)
            st.session_state["result"] = result
            st.session_state["auc"] = auc
            st.session_state["brier"] = brier
            st.session_state["samples"] = samples
        except Exception as e:
            st.error(f"运行失败：{e}")
            st.info("如果持续失败，请尝试：① 稍等几分钟再点；② 减少训练股票数量；③ 更换网络环境。")

if "result" in st.session_state:
    r = st.session_state["result"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本外 AUC", "—" if pd.isna(st.session_state["auc"]) else f'{st.session_state["auc"]:.3f}')
    c2.metric("Brier分数", "—" if pd.isna(st.session_state["brier"]) else f'{st.session_state["brier"]:.3f}')
    c3.metric("训练历史样本", f'{st.session_state["samples"]:,}')
    c4.metric("候选数量", len(r))

    show = r[["股票代码", "最新价", "触板概率", "近5日涨幅", "量比5日",
              "换手率", "收盘位置", "历史样本"]].copy()
    show["触板概率"] = (show["触板概率"] * 100).round(1).astype(str) + "%"
    show["近5日涨幅"] = (show["近5日涨幅"] * 100).round(1).astype(str) + "%"
    show["量比5日"] = show["量比5日"].round(2)
    show["换手率"] = show["换手率"].round(2).astype(str) + "%"
    show["收盘位置"] = (show["收盘位置"] * 100).round(1).astype(str) + "%"

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "下载候选股票 CSV",
        r.to_csv(index=False).encode("utf-8-sig"),
        "A股次日候选.csv", "text/csv",
    )

st.divider()
st.caption(
    "说明：这是研究型原型。概率来自模型历史样本，不代表未来收益。"
    "已修复：代码正则、成交额单位、数据泄露、特征口径、反爬对抗等问题。"
)
