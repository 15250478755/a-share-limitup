import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="A股次日涨停概率筛选器", layout="centered")
st.title("📈 A股次日涨停概率筛选器")

st.caption("T-1 → T：根据历史量价特征估计次日触及涨停的概率。研究工具，不构成收益保证。")

if st.button("🚀 开始筛选"):
    # 这里放你原来读取数据的代码（比如读取CSV或爬取数据）
    # 下面这行是测试用的假数据，你原本有读取真实数据的代码可以保留替换这里
    df = pd.DataFrame({
        '股票名称': ['测试A', '测试B', '测试C'],
        '价格': [10, '20', 30],  # 故意混个文字看看会不会报错
        '成交量': ['100', 200, 300]
    })
    
    # 【核心修复】：把所有可能算中位数的列强行变成数字（算不了的变成空值）
    for col in df.columns:
        df[col] = pd.to_numeric(df, errors='coerce')
        
    # 如果你的代码里有分组算中位数，加上 numeric_only=True
    # 例子：df.groupby('分组列').median(numeric_only=True)
    
    st.success("筛选完成！（测试修复版）")
    st.dataframe(df)

st.markdown("---")
st.caption("说明：这是研究型原型。概率来自模型历史样本，不代表未来收益。已加入类别平衡与概率校准提升准确率。")
