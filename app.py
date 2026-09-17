import streamlit as st
import pandas as pd
import numpy as np
import requests, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="A股次日涨停概率筛选器", page_icon="📈", layout="wide")

st.title("📈 A股次日涨停概率筛选器")
st.caption("T-1 → T：根据历史量价特征估计次日触及涨停的概率。研究工具，不构成收益保证。")

EAST = "https://82.push2.eastmoney.com/api/qt/clist/get"
HIS = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

@st.cache_data(ttl=300)
def spot():
    p = {
        "pn":1,"pz":5000,"po":1,"np":1,"ut":"bd1d9ddb04089700cf9c27f6f7426281",
        "fltt":2,"invt":2,"fid":"f3","fs":"m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields":"f12,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f20,f21,f23"
    }
    r=requests.get(EAST,params=p,timeout=15)
    r.raise_for_status()
    d=r.json()["data"]["diff"]
    return pd.DataFrame(d)

@st.cache_data(ttl=86400)
def hist(code, beg="20180101", end="20991231"):
    p={"secid":("1."+code if code.startswith(("6","68")) else "0."+code),
       "ut":"fa5fd1943c7b386f172d6893dbfba10b","fields1":"f1,f2,f3,f4",
       "fields2":"f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
       "klt":101,"fqt":1,"beg":beg,"end":end}
    r=requests.get(HIS,params=p,timeout=15)
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
    x["r3"]=x.close.pct_change(3)
    x["r5"]=x.close.pct_change(5)
    x["r10"]=x.close.pct_change(10)
    x["r20"]=x.close.pct_change(20)
    x["relvol5"]=x.volume/x.volume.rolling(5).mean()
    x["relvol20"]=x.volume/x.volume.rolling(20).mean()
    x["turnover"]=x.turnover
    x["close_pos"]=(x.close-x.low)/(x.high-x.low).replace(0,np.nan)
    x["maxret20"]=x.pct_chg.rolling(20).max()/100
    x["limitup20"]=(x.pct_chg/100 >= (limit_rate(code)-.002)).rolling(20).sum()
    x["range5"]=((x.high-x.low)/x.close).rolling(5).mean()
    x["target"]=(x.high.shift(-1) >= x.close*(1+limit_rate(code))*.995).astype(int)
    return x

FEATURES=["r3","r5","r10","r20","relvol5","relvol20","turnover","close_pos","maxret20","limitup20","range5"]

def train_model(n):
    s=spot()
    s=s[~s["f14"].astype(str).str.contains("ST|退",regex=True,na=False)].copy()
    s=s[s["f12"].astype(str).str.match(r"^(0|3|6|68)\d{4}$",na=False)]
    s["amt"]=pd.to_numeric(s["f6"],errors="coerce").fillna(0)
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
        time.sleep(.03)

    progress.empty(); status.empty()
    if not all_rows: raise RuntimeError("没有取得历史数据，请检查网络或稍后重试。")
    df=pd.concat(all_rows,ignore_index=True)
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])
    df=df.sort_values("date")
    cutoff=df.date.max()
    train_end=pd.to_datetime(cutoff)-pd.Timedelta(days=365)
    tr=df[pd.to_datetime(df.date)<train_end]
    te=df[pd.to_datetime(df.date)>=train_end]
    if len(tr)<500 or te.target.nunique()<2:
        tr=df.iloc[:int(len(df)*.8)]
        te=df.iloc[int(len(df)*.8):]
    model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1.0,random_state=42)
    model.fit(tr[FEATURES],tr.target)
    auc=roc_auc_score(te.target,model.predict_proba(te[FEATURES])[:,1]) if te.target.nunique()==2 else np.nan

    latest_df=pd.concat(latest,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    latest_df=latest_df.dropna(subset=FEATURES)
    latest_df["触板概率"]=model.predict_proba(latest_df[FEATURES])[:,1]
    latest_df["封板概率"]=np.nan
    latest_df["历史样本"]=latest_df["code"].map(df.groupby("code").size())
    latest_df["近5日涨幅"]=latest_df.r5
    latest_df["量比5日"]=latest_df.relvol5
    latest_df["换手率"]=latest_df.turnover
    latest_df["收盘位置"]=latest_df.close_pos
    latest_df["股票代码"]=latest_df.code
    latest_df["最新价"]=latest_df.close
    return latest_df.sort_values("触板概率",ascending=False), auc, len(df)

st.sidebar.header("筛选参数")
n=st.sidebar.slider("训练股票数量",100,800,300,50)
topk=st.sidebar.slider("显示前几名",5,30,10)
threshold=st.sidebar.slider("最低触板概率",0.05,0.50,0.15,0.01)

if st.button("🚀 开始筛选",type="primary",use_container_width=True):
    with st.spinner("正在抓取历史数据并训练模型，第一次可能需要几分钟……"):
        try:
            result,auc,samples=train_model(n)
            result=result[result["触板概率"]>=threshold].head(topk)
            st.session_state["result"]=result
            st.session_state["auc"]=auc
            st.session_state["samples"]=samples
        except Exception as e:
            st.error(f"运行失败：{e}")

if "result" in st.session_state:
    r=st.session_state["result"]
    c1,c2,c3=st.columns(3)
    c1.metric("样本外 AUC", "—" if pd.isna(st.session_state["auc"]) else f'{st.session_state["auc"]:.3f}')
    c2.metric("训练历史样本", f'{st.session_state["samples"]:,}')
    c3.metric("候选数量", len(r))
    show=r[["股票代码","最新价","触板概率","近5日涨幅","量比5日","换手率","收盘位置","历史样本"]].copy()
    show["触板概率"]=(show["触板概率"]*100).round(1).astype(str)+"%"
    show["近5日涨幅"]=(show["近5日涨幅"]*100).round(1).astype(str)+"%"
    show["量比5日"]=show["量比5日"].round(2)
    show["换手率"]=show["换手率"].round(2).astype(str)+"%"
    show["收盘位置"]=(show["收盘位置"]*100).round(1).astype(str)+"%"
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.download_button("下载候选股票 CSV",r.to_csv(index=False).encode("utf-8-sig"),"A股次日候选.csv","text/csv")
else:
    st.info("点上面的「开始筛选」即可。第一次运行会下载历史数据，请耐心等待。")

st.divider()
st.caption("说明：这是研究型原型。概率来自模型历史样本，不代表未来收益，也不保证次日涨停。当前版本重点验证 T-1→T 的触板预测框架；正式交易前应进行独立回测并加入交易成本、涨停买不到/卖不掉等执行约束。")
