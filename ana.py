import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 페이지 설정
st.set_page_config(page_title="비트코인(BTC) 분석 대시보드", layout="wide")

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    # 데이터 로드 (세미콜론 구분자)
    df = pd.read_csv(file_path, sep=";")
    
    # 시간 관련 컬럼 변환
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # 날짜순 정렬
    df = df.sort_values('timeOpen')
    
    # 이동 평균선 계산
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 전일 대비 변동율 (%)
    df['pct_change'] = df['close'].pct_change() * 100
    
    return df

def predict_next_day(df):
    """선형회귀를 이용한 내일 가격 예측"""
    # 최근 30일 데이터를 학습 데이터로 사용
    model_df = df.tail(30).copy()
    
    # 날짜를 숫자로 변환 (학습용)
    model_df['days_from_start'] = (model_df['timeOpen'] - model_df['timeOpen'].min()).dt.days
    
    X = model_df[['days_from_start']].values
    y = model_df['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산
    next_day_num = np.array([[model_df['days_from_start'].max() + 1]])
    prediction = model.predict(next_day_num)[0]
    
    return prediction

def main():
    st.title("🪙 비트코인(BTC) 데이터 분석 및 예측 대시보드")
    
    file_name = "bc.csv"
    df = load_data(file_name)

    if df is None:
        st.error(f"'{file_name}' 파일을 찾을 수 없습니다. 파이썬 파일과 같은 폴더에 데이터 파일이 있는지 확인해주세요.")
        return

    # 사이드바 설정
    st.sidebar.header("📊 분석 옵션")
    
    # 기간 선택
    min_date = df['timeOpen'].min().to_pydatetime()
    max_date = df['timeOpen'].max().to_pydatetime()
    date_range = st.sidebar.date_input(
        "조회 기간",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    show_ma = st.sidebar.multiselect(
        "이동 평균선 표시",
        options=["MA7", "MA20"],
        default=["MA7"]
    )

    # 데이터 필터링
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)
        f_df = df.loc[mask].copy()
    else:
        f_df = df.copy()

    # 상단 Metrics 및 예측 결과
    latest_price = df.iloc[-1]['close']
    predicted_price = predict_next_day(df)
    price_diff = predicted_price - latest_price
    percent_diff = (price_diff / latest_price) * 100

    # 예측 섹션
    st.subheader("🔮 인공지능 내일 가격 예측 (Linear Regression)")
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    
    with p_col1:
        st.metric("내일 예상 가격", f"₩{predicted_price:,.0f}", f"{percent_diff:+.2f}%")
    
    with p_col2:
        if price_diff > 0:
            st.success("📈 상승 예측 (매수 추천)")
        else:
            st.error("📉 하락 예측 (관망 추천)")
            
    with p_col3:
        st.info("최근 30일간의 가격 추세를 선형회귀 모델로 분석한 결과입니다. 투자의 책임은 본인에게 있습니다.")

    st.divider()

    # 주요 지표
    latest = f_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 종가", f"₩{latest['close']:,.0f}", f"{latest['pct_change']:.2f}%")
    col2.metric("기간 내 최고가", f"₩{f_df['high'].max():,.0f}")
    col3.metric("평균 거래량", f"{f_df['volume'].mean():,.0e}")
    col4.metric("변동성 (표준편차)", f"{f_df['close'].std():,.0f}")

    st.divider()

    # 메인 차트 (Price + MA)
    st.subheader("가격 및 이동 평균선 추이")
    fig = go.Figure()
    
    # 종가 라인
    fig.add_trace(go.Scatter(x=f_df['timeOpen'], y=f_df['close'], name="Close", line=dict(color='#FF9900', width=2)))
    
    # 이동 평균선 추가
    if "MA7" in show_ma:
        fig.add_trace(go.Scatter(x=f_df['timeOpen'], y=f_df['MA7'], name="7일 평균", line=dict(dash='dot', color='#00CCFF')))
    if "MA20" in show_ma:
        fig.add_trace(go.Scatter(x=f_df['timeOpen'], y=f_df['MA20'], name="20일 평균", line=dict(dash='dash', color='#FF66CC')))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="날짜",
        yaxis_title="가격 (KRW)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 하단 2분할 차트
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("일별 거래량")
        fig_vol = px.bar(f_df, x='timeOpen', y='volume', color_discrete_sequence=['#444'])
        fig_vol.update_layout(template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with c2:
        st.subheader("수익률 분포")
        fig_hist = px.histogram(f_df, x='pct_change', nbins=50, title="Daily Return Distribution")
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    # 데이터 테이블
    with st.expander("데이터 상세 보기"):
        st.dataframe(f_df.sort_values('timeOpen', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
