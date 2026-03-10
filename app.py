import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import datetime

st.set_page_config(page_title="Crypto Quant Sandbox", layout="wide")
st.title("🚀 Crypto Quant Sandbox (Pro Edition)")
st.markdown("### 📊 전문 퀀트 트레이더처럼 나만의 지표와 익절/손절 로직을 조립해 보세요.")

# ==========================================
# 바이낸스 데이터 수집 엔진 (USDT 마켓)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_binance_data(symbol, start_date, end_date, interval="5m"):
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    all_klines = []
    
    current_ts = start_ts
    while current_ts < end_ts:
        res = requests.get(url, params={"symbol": symbol, "interval": interval, "startTime": current_ts, "endTime": end_ts, "limit": 1000})
        data = res.json()
        if type(data) is dict and 'msg' in data: 
            st.error(f"API Error: {data['msg']}")
            break
        if not data: break
        all_klines.extend(data)
        current_ts = data[-1][0] + 1
        time.sleep(0.1)

    if not all_klines: return None

    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# ==========================================
# 디테일한 기술적 지표 계산
# ==========================================
def add_indicators(df, p):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # 볼린저 밴드
    df['bb_mid'] = df['close'].rolling(window=p['bb_window']).mean()
    df['bb_std'] = df['close'].rolling(window=p['bb_window']).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (p['bb_mult'] * df['bb_std'])
    df['bb_upper'] = df['bb_mid'] + (p['bb_mult'] * df['bb_std'])
    
    # 이동평균선 (커스텀 기간)
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    
    # 스토캐스틱 (커스텀 기간 & 스무딩)
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    fast_k = 100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)
    df['stoch_k'] = fast_k.rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    
    return df

# ==========================================
# 프로페셔널 백테스트 엔진
# ==========================================
def run_backtest_pro(df, p):
    balance = 10_000 
    position = 0
    entry_price = 0
    entry_time = None
    fee_rate = 0.0004 # 바이낸스/바이비트 기준 (0.04%)
    trades, buy_points, sell_points = [], [], []
    
    for i in range(50, len(df)-1):
        prev = df.iloc[i-1]    
        current = df.iloc[i]   
        next_candle = df.iloc[i+1] 
        
        # --- [ENTRY 로직] ---
        if position == 0:
            buy_signal = True
            
            if p['use_ma_cross'] and not (current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']): buy_signal = False
            if p['use_stoch_cross'] and not (current['stoch_k'] > current['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']): buy_signal = False
            if p['use_bb_touch'] and not (current['low'] <= current['bb_lower']): buy_signal = False
            if p['use_rsi'] and not (current['rsi'] <= p['rsi_buy_limit']): buy_signal = False
            if not any([p['use_ma_cross'], p['use_stoch_cross'], p['use_bb_touch'], p['use_rsi']]): buy_signal = False

            if buy_signal:
                entry_price = next_candle['open']
                entry_time = next_candle['timestamp']
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))

        # --- [EXIT 로직: 다운시프트 및 복합 익절] ---
        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            
            # 1. 목표가 설정 (유저 선택 모드)
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            if p['exit_mode'] == "🎯 Max 모드 (TP와 BB중앙선 중 더 높은 가격에서 익절)":
                target_price = max(current['bb_mid'], base_tp)
            else: # OR 모드 (둘 중 하나라도 닿으면 즉시 익절)
                target_price = min(current['bb_mid'], base_tp) 
                
            # 2. 다운시프트 개입 (시간 경과 시 목표가 대폭 하향)
            if p['use_downshift'] and elapsed_mins >= p['downshift_mins']:
                downshift_tp = entry_price * (1 + p['downshift_tp_pct'] / 100)
                target_price = min(target_price, downshift_tp) # 목표가를 강제로 끌어내림
                
            sl_price = entry_price * (1 - p['sl_pct'] / 100)
            sell_price = 0
            
            # 매도 체결 확인
            if next_candle['low'] <= sl_price: sell_price = min(next_candle['open'], sl_price)
            elif next_candle['high'] >= target_price: sell_price = max(next_candle['open'], target_price)
            
            if sell_price > 0:
                balance = position * sell_price * (1 - fee_rate)
                profit_pct = (sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100
                trades.append(profit_pct)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0

    if position > 0: 
        last_candle = df.iloc[-1]
        balance = position * last_candle['close'] * (1 - fee_rate)
        trades.append((last_candle['close'] / entry_price * (1 - fee_rate * 2) - 1) * 100)
        sell_points.append((last_candle['timestamp'], last_candle['close']))

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# UI 셋업
# ==========================================
if st.sidebar.button("🔄 데이터 캐시 초기화", width="stretch"): st.cache_data.clear()

st.sidebar.header("🌐 1. 데이터 설정 (Binance)")
ticker = st.sidebar.selectbox("코인 페어", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=30))
end_date = col2.date_input("종료일", datetime.date.today())
timeframe = st.sidebar.select_slider("⏱️ 캔들 타임프레임", options=['5m', '15m', '30m', '1h', '4h'], value='5m')

st.sidebar.markdown("---")
st.sidebar.header("🧩 2. 매수 전략 (체크 시 AND 결합)")
p = {}

with st.sidebar.expander("📈 이동평균선 크로스 설정", expanded=False):
    p['use_ma_cross'] = st.checkbox("단기 MA가 장기 MA 상향 돌파 시 매수")
    p['ma_fast'] = st.number_input("단기 이평선 (Fast MA)", value=10, step=1)
    p['ma_slow'] = st.number_input("장기 이평선 (Slow MA)", value=20, step=1)

with st.sidebar.expander("🌊 스토캐스틱 설정", expanded=False):
    p['use_stoch_cross'] = st.checkbox("%K가 %D 상향 돌파 시 매수")
    c1, c2, c3 = st.columns(3)
    p['stoch_k_len'] = c1.number_input("%K 길이", value=5, step=1)
    p['stoch_k_smooth'] = c2.number_input("%K 스무딩", value=3, step=1)
    p['stoch_d_smooth'] = c3.number_input("%D 스무딩", value=3, step=1)

with st.sidebar.expander("📉 볼린저 밴드 & RSI 설정", expanded=True):
    p['use_bb_touch'] = st.checkbox("캔들 저가가 볼밴 하단 터치 시 매수", value=True)
    c4, c5 = st.columns(2)
    p['bb_window'] = c4.number_input("BB 기간", value=20, step=1)
    p['bb_mult'] = c5.number_input("BB 승수", value=2.0, step=0.1)
    
    st.markdown("---")
    p['use_rsi'] = st.checkbox("RSI 과매도 구간 매수", value=True)
    c6, c7 = st.columns(2)
    p['rsi_period'] = c6.number_input("RSI 기간", value=14, step=1)
    p['rsi_buy_limit'] = c7.number_input("RSI 진입선", value=30.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 3. 정밀 익절/손절 설정")
st.sidebar.caption("0.1% 단위로 세밀하게 조정하세요.")
c8, c9 = st.sidebar.columns(2)
p['tp_pct'] = c8.number_input("✅ 목표가 (TP) %", value=1.0, step=0.1, format="%.1f")
p['sl_pct'] = c9.number_input("🛑 손절가 (SL) %", value=2.0, step=0.1, format="%.1f")

p['exit_mode'] = st.sidebar.radio(
    "🔄 볼린저 밴드 중앙선 결합 모드",
    options=["🎯 Max 모드 (TP와 BB중앙선 중 더 높은 가격에서 익절)", "⚡ OR 모드 (TP나 BB중앙선 중 먼저 닿는 곳에서 즉시 익절)"],
    help="시장이 강하게 반등할 때 중앙선까지 수익을 극대화할지, 아니면 안전하게 먼저 닿는 선에서 팔고 나갈지 결정합니다."
)

st.sidebar.markdown("##### ⏳ 특수 기능: 다운시프트 (Downshift)")
p['use_downshift'] = st.sidebar.checkbox("다운시프트 활성화", value=True, help="단타(Scalping) 시 진입 후 가격이 오르지 않고 횡보하며 시간이 지체될 때, 자금이 묶이는 것을 방지하기 위해 목표가를 대폭 낮춰 본절 부근에서 빠르게 탈출하는 기관급 로직입니다.")
if p['use_downshift']:
    c10, c11 = st.sidebar.columns(2)
    p['downshift_mins'] = c10.number_input("발동 시간 (분)", value=60, step=15, help="진입 후 N분이 지나면 발동")
    p['downshift_tp_pct'] = c11.number_input("하향 목표가 %", value=0.2, step=0.1, format="%.1f", help="예: 60분 경과 시 목표가를 0.2%로 확 낮춤")

run_btn = st.sidebar.button("▶️ 나만의 퀀트 전략 백테스트 실행", type="primary", width="stretch")

# ==========================================
# 실행 메인
# ==========================================
if run_btn:
    with st.spinner(f"Binance 서버에서 {ticker} 데이터를 로드 중입니다..."):
        df = fetch_binance_data(ticker, start_date, end_date, interval=timeframe)
        if df is None or len(df) < 50:
            st.error("데이터 로드 실패. 날짜 범위를 확인하세요.")
            st.stop()
            
    df = add_indicators(df, p)
    
    with st.spinner("수만 개의 캔들을 분석하며 시뮬레이션을 진행합니다..."):
        profit, trades, buys, sells, result_df = run_backtest_pro(df, p)
        
    st.subheader(f"📊 백테스트 리포트: {ticker} ({timeframe})")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최종 계좌 수익률", f"{profit:.2f}%")
    col2.metric("총 거래 횟수", f"{len(trades)}회")
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    col3.metric("승률", f"{win_rate:.1f}%")
    col4.metric("평균 수익 (건당)", f"{np.mean(trades) if trades else 0:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
    
    if p['use_bb_touch']:
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['bb_mid'], line=dict(color='rgba(255,255,0,0.4)', dash='dash'), name='BB Mid'))
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['bb_lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dot'), name='BB Lower'))
    if p['use_ma_cross']:
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['sma_fast'], line=dict(color='orange'), name=f"MA {p['ma_fast']}"))
        fig.add_trace(go.Scatter(x=result_df['timestamp'], y=result_df['sma_slow'], line=dict(color='blue'), name=f"MA {p['ma_slow']}"))
        
    if buys:
        bt, bp = zip(*buys)
        fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
    if sells:
        st_times, sp = zip(*sells)
        fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
                                 
    fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width="stretch")
    
    # ==========================================
    # 💰 [레퍼럴 수익화 섹션] 
    # ==========================================
    st.markdown("---")
    st.markdown("### 🔥 나만의 승률 높은 로직을 찾으셨나요?")
    st.info("실전 자동매매는 **수수료 절감**이 생명입니다. 일반 계정으로 봇을 돌리면 수익의 절반이 수수료로 증발합니다. 아래 버튼을 통해 **평생 수수료 20% 할인 VIP 계정**을 활성화하고 매매를 시작하세요.")
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.link_button("🟡 바이낸스(Binance) 20% 할인 가입", "https://accounts.binance.com/register?ref=총감독님_바이낸스_코드", type="primary", width="stretch")
    with btn_col2:
        st.link_button("⚫ 바이비트(Bybit) 최대 증정금 가입", "https://www.bybit.com/register?affiliate_id=총감독님_바이비트_코드", type="primary", width="stretch")
