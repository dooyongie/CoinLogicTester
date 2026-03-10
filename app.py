import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import time
import datetime

st.set_page_config(page_title="CoinLogicTester - Crypto Quant Sandbox", layout="wide")
st.title("🚀 Crypto Quant Sandbox (Global Edition)")
st.markdown("### 나만의 퀀트 전략을 조립하고 바이낸스 데이터로 검증해 보세요!")

# ==========================================
# 🌟 [NEW] 바이낸스 데이터 수집 엔진 (USDT 마켓)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_binance_data(symbol, start_date, end_date, interval="5m"):
    url = "https://api.binance.com/api/v3/klines"
    # 바이낸스는 밀리초(ms) 단위 timestamp 사용
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)).timestamp() * 1000) - 1

    all_klines = []
    limit = 1000 # 바이낸스 최대 한도
    current_ts = start_ts

    while current_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": end_ts,
            "limit": limit
        }
        res = requests.get(url, params=params)
        data = res.json()
        
        if type(data) is dict and 'msg' in data: # 에러 처리
            st.error(f"API Error: {data['msg']}")
            break
        if not data:
            break
            
        all_klines.extend(data)
        current_ts = data[-1][0] + 1 # 마지막 캔들 시간 + 1ms
        time.sleep(0.1)

    if not all_klines: return None

    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    # UTC 기준 시간으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# ==========================================
# 🌟 [NEW] 대중을 위한 다양한 기술적 지표 추가
# ==========================================
def add_indicators(df, p):
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # 2. 볼린저 밴드
    df['bb_mid'] = df['close'].rolling(window=p['bb_window']).mean()
    df['bb_std'] = df['close'].rolling(window=p['bb_window']).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (p['bb_mult'] * df['bb_std'])
    df['bb_upper'] = df['bb_mid'] + (p['bb_mult'] * df['bb_std'])
    
    # 3. 이동평균선 (10, 20)
    df['sma10'] = df['close'].rolling(window=10).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    
    # 4. 스토캐스틱 (5, 3, 3)
    ndays_high = df['high'].rolling(window=5).max()
    ndays_low = df['low'].rolling(window=5).min()
    df['stoch_k'] = 100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df

# ==========================================
# 코어 백테스트 엔진 (퍼블릭용 범용 엔진)
# ==========================================
def run_backtest_public(df, p):
    balance = 10_000 # 시작 자본 10,000 USDT
    position = 0
    entry_price = 0
    fee_rate = 0.0004 # 바이낸스/바이비트 평균 수수료 (0.04%)
    
    trades, buy_points, sell_points = [], [], []
    
    for i in range(25, len(df)-1):
        prev = df.iloc[i-1]    # 직전 캔들 (크로스 감지용)
        current = df.iloc[i]   # 현재 캔들 (조건 판별용)
        next_candle = df.iloc[i+1] # 진입/청산 캔들
        
        # --- [ENTRY 로직: 유저가 선택한 조건만 AND로 결합] ---
        if position == 0:
            buy_signal = True
            
            # 1. 이동평균선 골든크로스 (10MA가 20MA를 상향 돌파)
            if p['use_ma_cross']:
                if not (current['sma10'] > current['sma20'] and prev['sma10'] <= prev['sma20']):
                    buy_signal = False
                    
            # 2. 스토캐스틱 골든크로스 (%K가 %D를 상향 돌파)
            if p['use_stoch_cross']:
                if not (current['stoch_k'] > current['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']):
                    buy_signal = False
                    
            # 3. 볼린저밴드 하단 돌파
            if p['use_bb_touch']:
                if not (current['low'] <= current['bb_lower']):
                    buy_signal = False
                    
            # 4. RSI 과매도
            if p['use_rsi']:
                if not (current['rsi'] <= p['rsi_buy_limit']):
                    buy_signal = False

            # 유저가 아무것도 체크 안 했을 때 방지
            if not (p['use_ma_cross'] or p['use_stoch_cross'] or p['use_bb_touch'] or p['use_rsi']):
                buy_signal = False

            if buy_signal:
                entry_price = next_candle['open']
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))

        # --- [EXIT 로직: 단순화된 익절/손절] ---
        if position > 0:
            tp_price = entry_price * (1 + p['tp_pct'] / 100)
            sl_price = entry_price * (1 - p['sl_pct'] / 100)
            sell_price = 0
            
            if next_candle['low'] <= sl_price: sell_price = min(next_candle['open'], sl_price)
            elif next_candle['high'] >= tp_price: sell_price = max(next_candle['open'], tp_price)
            
            if sell_price > 0:
                balance = position * sell_price * (1 - fee_rate)
                profit_pct = (sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100
                trades.append(profit_pct)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0

    if position > 0: # 미청산 정산
        last_candle = df.iloc[-1]
        balance = position * last_candle['close'] * (1 - fee_rate)
        trades.append((last_candle['close'] / entry_price * (1 - fee_rate * 2) - 1) * 100)
        sell_points.append((last_candle['timestamp'], last_candle['close']))

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# UI 셋업
# ==========================================
if st.sidebar.button("🔄 바이낸스 데이터 캐시 초기화", width="stretch"):
    st.cache_data.clear()

st.sidebar.header("🌐 1. 바이낸스 마켓 설정")
ticker = st.sidebar.selectbox("코인 페어 (USDT)", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=30))
end_date = col2.date_input("종료일", datetime.date.today())

st.sidebar.markdown("---")
st.sidebar.header("🧩 2. 매수 전략 조립 (체크 시 AND 조건)")
p = {}
p['use_ma_cross'] = st.sidebar.checkbox("📈 이평선 골든크로스 (10MA가 20MA 돌파)")
p['use_stoch_cross'] = st.sidebar.checkbox("🌊 스토캐스틱 골든크로스 (%K가 %D 돌파)")

st.sidebar.markdown("##### 보조 지표 필터")
p['use_bb_touch'] = st.sidebar.checkbox("📉 볼린저밴드 하단선 터치")
p['bb_window'] = 20
p['bb_mult'] = 2.0

p['use_rsi'] = st.sidebar.checkbox("📊 RSI 과매도 필터")
p['rsi_period'] = 14
p['rsi_buy_limit'] = st.sidebar.slider("RSI 기준선", 10, 50, 30) if p['use_rsi'] else 100

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 3. 익절/손절 설정")
p['tp_pct'] = st.sidebar.number_input("✅ 목표 수익률 (%)", value=2.0, step=0.1)
p['sl_pct'] = st.sidebar.number_input("🛑 손절 라인 (%)", value=2.0, step=0.1)

run_btn = st.sidebar.button("▶️ 봇 시뮬레이션 실행", type="primary", width="stretch")

# ==========================================
# 실행 메인
# ==========================================
if run_btn:
    with st.spinner(f"Binance 서버에서 {ticker} 데이터를 로드 중입니다..."):
        df = fetch_binance_data(ticker, start_date, end_date)
        if df is None or len(df) < 50:
            st.error("데이터 로드 실패. 날짜 범위를 확인하세요.")
            st.stop()
            
    df = add_indicators(df, p)
    
    with st.spinner("선택하신 전략을 검증하고 있습니다..."):
        profit, trades, buys, sells, result_df = run_backtest_public(df, p)
        
    st.subheader(f"📊 백테스트 결과: {ticker} (Binance)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 계좌 수익률", f"{profit:.2f}%")
    col2.metric("총 거래 횟수", f"{len(trades)}회")
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    col3.metric("승률", f"{win_rate:.1f}%")
    col4.metric("평균 수익 (건당)", f"{np.mean(trades) if trades else 0:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
    
    if p['use_bb_touch']:
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name='BB Lower'))
    if p['use_ma_cross']:
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['sma10'], line=dict(color='orange'), name='SMA 10'))
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['sma20'], line=dict(color='blue'), name='SMA 20'))
        
    if buys:
        bt, bp = zip(*buys)
        fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
    if sells:
        st_times, sp = zip(*sells)
        fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
                                 
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")
    
    # ==========================================
    # 💰 [핵심] 바이낸스 레퍼럴 연결 섹션
    # ==========================================
    st.markdown("---")
    st.markdown("### 🔥 나만의 승률 높은 로직을 찾으셨나요?")
    st.info("실전 자동매매는 **수수료 절감**이 생명입니다. 일반 계정으로 봇을 돌리면 수익의 절반이 수수료로 증발합니다. 아래 버튼을 통해 **평생 수수료 20% 할인 VIP 계정**을 활성화하고 매매를 시작하세요.")
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.link_button(
            "🟡 바이낸스(Binance) 20% 할인 가입", 
            "https://accounts.binance.com/register?ref=총감독님_바이낸스_레퍼럴코드", 
            type="primary", 
            width="stretch"
        )
    with btn_col2:
        st.link_button(
            "⚫ 바이비트(Bybit) 최대 증정금 가입", 
            "https://www.bybit.com/register?affiliate_id=총감독님_바이비트_레퍼럴코드", 
            type="primary", 
            width="stretch"
        )
