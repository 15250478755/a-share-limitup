import streamlit as st
import pandas as pd
import numpy as np
import requests, time, random
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

st.set_page_config(page_title="A股次日涨停概率筛选器", page_icon="📈", layout="wide")
st.title("📈 A股次日涨停概率筛选器")
st.caption("T-1 → T：根据历史量价特征估计次日触及涨停的概率。研究工具，不构成收益保证。")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.eastmoney.com/"
}

EAST = "https://82.push2.eastmoney.com/api/qt/clist/get"
HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

@st.cache_data(ttl=300)
def spot():
    p = {
        "pn":1,"pz":5000,"po":1,"np":1,"ut":"bd1d9ddb04089700cf9c27f6f7426281",
        "fltt":2,"invt":2,"fid":"f3","fs":"m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields":"f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f20,f21,f23"
    }
    r=requests.get(EAST,params=p,headers=HEADERS,timeout=30)
    r.raise_for_status()
    d=r.json()["data"]["diff"]
    return pd.DataFrame(d)

@st.cache_data(ttl=86400)
def hist(code, beg="20180101", end="20991231"):
    p={"secid":("1."+code if code.startswith(("6","68")) else "0."+code),
       "ut":"fa5fd1943c7b386f172d6893dbfba10b","fields1":"f1,f2,f3,f4",
       "fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
       "klt":101,"fqt":1,"beg":beg,"end":end}
    r=requests.get(HIS,params=p,headers=HEADERS,timeout=30)
    r.raise_for_status()
    kl=r.json().get("data",{}).get("klines",[])
    if not kl: return pd.DataFrame()
    rows=[x.split(",") for x in kl]
    cols=["date","open","close","high","low","volume","amount","amplitude","pct_chg","turnover","extra"]
    d=pd.DataFrame(rows,columns=cols)
    for c in cols[1:]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d

def limit_rate(code):
    if code.startswith(("300","301","688","689")): return .20
    if code.startswith(("430","440","830","8","4")): return .30
    return .10

def make_features(d, code):
    if len(d)<35: return pd.DataFrame()
    x=d.copy()
    x["code"]=code
    x["r1"]=x.close.pct_change(1)
    x["r3"]=x.close.pct_change(3)
    x["r5"]=x.close.pct_change(5)
    x["r10"]=x.close.pct_change(10)
    x["r20"]=x.close.pct_change(20)
    x["relvol5"]=x.volume/x.volume.rolling(5).mean()
    x["relvol20"]=x.volume/x.volume.rolling(20).mean()
    x["turnover"]=x.turnover
    x["turnover5"]=x.turnover.rolling(5).mean()
    x["close_pos"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x["maxret20"]=x.pct_chg.rolling(20).max()/100
    x["limitup20"]=(x.pct_chg/100 >= (limit_rate(code)-.002)).rolling(20).sum()
    x["range5"]=((x.high-x.low)/x.close).rolling(5).mean()
    x["amp20"]=x.amplitude.rolling(20).mean()
    x["vol_chg5"]=x.volume.pct_change(5)
    x["ma_bull"]=(x.close > x.close.rolling(5).mean()).astype(int)
    x["high_near20"]=((x.high-x.close)/x.close).rolling(20).max()
    x["target"]=(x.high.shift(-1) >= x.close*(1+limit_rate(code))*.995).astype(int)
    return x

FEATURES=["r1","r3","r5","r10","r20","relvol5","relvol20","turnover","turnover5","close_pos","maxret20","limitup20","range5","amp20","vol_chg5","ma_bull","high_near20"]

def train_model(n, boards, min_price, min_amt):
    s=spot()
    s=s[~s["f14"].astype(str).str.contains("ST|退",regex=True,na=False)].copy()
    s=s[s["f12"].astype(str).str.match(r"^(0\d{4}|3\d{5}|60\d{4}|68\d{4})$",na=False)]
    
    main_board = s["f12"].astype(str).str.startswith(("60","00"))
    chi_next = s["f12"].astype(str).str.startswith(("30","301"))
    star_market = s["f12"].astype(str).str.startswith(("68","689"))
    bse = s["f12"].astype(str).str.startswith(("43","83","87","4"))
    
    mask = pd.Series(False, index=s.index)
    if "主板" in boards: mask |= main_board
    if "创业板" in boards: mask |= chi_next
    if "科创板" in boards: mask |= star_market
    if "北交所" in boards: mask |= bse
    s = s[mask]
    
    s["price"]=pd.to_numeric(s["f2"],errors="coerce")
    s["amt"]=pd.to_numeric(s["f6"],errors="coerce").fillna(0)
    s=s[(s["price"]>=min_price) & (s["amt"]>=min_amt)]
    s=s.sort_values("amt",ascending=False).head(n)

    all_rows=[]
    latest=[]
    progress=st.progress(0)
    status=st.empty()
    for i,code in enumerate(s.f12.astype(str)):
        try:
            d=hist(code)
            z=make_features(d,code)
            if len(z):
                all_rows.append(z.iloc[:-1].copy())
                latest.append(z.iloc[-1:].copy())
        except Exception:
            pass
        progress.progress((i+1)/len(s))
        status.write(f"正在读取 {i+1}/{len(s)}：{code}")
        time.sleep(random.uniform(0.15, 0.35))

    progress.empty(); status.empty()
    if not all_rows: raise RuntimeError("没有取得历史数据，请检查网络或稍后重试。")
    
    df=pd.concat(all_rows,ignore_index=True)
    df=df.replace([np.inf,-np.inf],np.nan)
    for col in FEATURES + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES+["target"]] = df[FEATURES+["target"]].fillna(df.median(numeric_only=True))
    df=df.dropna(subset=FEATURES+["target"])
    df=df.sort_values("date")
    
    cutoff=df.date.max()
    train_end=pd.to_datetime(cutoff)-pd.Timedelta(days=365)
    tr=df[pd.to_datetime(df.date)<train_end]
    te=df[pd.to_datetime(df.date)>=train_end]
    
    if len(tr)<500 or te.target.nunique()<2:
        tr=df.iloc[:int(len(df)*.8)]
        te=df.iloc[int(len(df)*.8):]
        
    pos = tr[tr.target==1]
    neg = tr[tr.target==0]
    if len(pos) > 0 and len(neg) > len(pos)*2:
        neg = neg.sample(n=len(pos)*2, random_state=42)
    tr_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    
    base_model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,min_samples_leaf=30,learning_rate=.06,l2_regularization=1.5,random_state=42)
    model=CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    model.fit(tr_balanced[FEATURES],tr_balanced.target)
    
    auc=roc_auc_score(te.target,model.predict_proba(te[FEATURES])[:,1]) if te.target.nunique()==2 else np.nan
    brier = brier_score_loss(te.target, model.predict_proba(te[FEATURES])[:,1]) if te.target.nunique()==2 else np.nan

    latest_df=pd.concat(latest,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    for col in FEATURES:
        latest_df[col] = pd.to_numeric(latest_df[col], errors="coerce")
    latest_df[FEATURES] = latest_df[FEATURES].fillna(latest_df.median(numeric_only=True))
    latest_df=latest_df.dropna(subset=FEATURES)
    latest_df["触板概率"]=model.predict_proba(latest_df[FEATURES])[:,1]
    latest_df["历史样本"]=latest_df["code"].map(df.groupby("code").size())
    latest_df["近5日涨幅"]=latest_df.r5
    latest_df["量比5日"]=latest_df.relvol5
    latest_df["换手率"]=latest_df.turnover
    latest_df["收盘位置"]=latest_df.close_pos
    latest_df["股票代码"]=latest_df.code
    latest_df["最新价"]=latest_df.close
    return latest_df.sort_values("触板概率",ascending=False), auc, brier, len(df)

st.sidebar.header("筛选参数")
boards=st.sidebar.multiselect("选择板块",["主板","创业板","科创板","北交所"],default=["主板","创业板"])
n=st.sidebar.slider("训练股票数量",100,800,300,50)
topk=st.sidebar.slider("显示前几名",5,30,10)
threshold=st.sidebar.slider("最低触板概率",0.05,0.50,0.15,0.01)
min_price = st.sidebar.number_input("最低股价(元)", 0.0, 1000.0, 2.0)
min_amt = st.sidebar.number_input("最低日成交额(万)", 0.0, 100000.0, 5000.0)

if st.button("🚀 开始筛选",type="primary",use_container_width=True):
    with st.spinner("正在抓取历史数据并训练模型，第一次可能需要几分钟……"):
        try:
            result,auc,brier,samples=train_model(n, boards, min_price, min_amt)
            result=result[result["触板概率"]>=threshold].head(topk)
            st.session_state["result"]=result
            st.session_state["auc"]=auc
            st.session_state["brier"]=brier
            st.session_state["samples"]=samples
        except Exception as e:
            st.error(f"运行失败：{e}")

if "result" in st.session_state:
    r=st.session_state["result"]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("样本外 AUC", "—" if pd.isna(st.session_state["auc"]) else f'{st.session_state["auc"]:.3f}')
    c2.metric("Brier分数", "—" if pd.isna(st.session_state["brier"]) else f'{st.session_state["brier"]:.3f}')
    c3.metric("训练历史样本", f'{st.session_state["samples"]:,}')
    c4.metric("候选数量", len(r))
    
    show=r[["股票代码","最新价","触板概率","近5日涨幅","量比5日","换手率","收盘位置","历史样本"]].copy()
    show["触板概率"]=(show["触板概率"]*100).round(1).astype(str)+"%"
    show["近5日涨幅"]=(show["近5日涨幅"]*100).round(1).astype(str)+"%"
    show["量比5日"]=show["量比5日"].round(2)
    show["换手率"]=show["换手率"].round(2).astype(str)+"%"
    show["收盘位置"]=(show["收盘位置"]*100).round(1).astype(str)+"%"
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.download_button("下载候选股票 CSV",r.to_csv(index=False).encode("utf-8-sig"),"A股次日候选.csv","text/csv")

st.divider()
st.caption("说明：这是研究型原型。概率来自模型历史样本，不代表未来收益。已加入类别平衡与概率校准提升准确率。")
