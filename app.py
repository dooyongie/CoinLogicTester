import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import time

st.set_page_config(page_title="Crypto Quant Sandbox", layout="wide")
st.title("🚀 Crypto Quant Sandbox (Fee-Control Edition)")
st.markdown("### 📊 수수료의 위력을 직접 체감하고, 최적의 거래 조건을 설계해 보세요.")

# ==========================================
# 데이터 수집 엔진 (Yahoo Finance + KST 보정)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_data_via_yf(symbol, start_date, end_date, interval="5m"):
    yf_symbol = symbol.replace("USDT", "-USD")
    try:
        data = yf.download(tickers=yf_symbol, start=start_date, end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval=interval, progress=False)
        if data.empty: return None
        df = data.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        rename_map = {'Datetime': 'timestamp', 'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=rename_map, inplace=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=9)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}"); return None

# ==========================================
# 기술적 지표 계산
# ==========================================
def add_indicators(df, p):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['bb_mid'] = df['close'].rolling(window=p['bb_window']).mean()
    df['bb_std'] = df['close'].rolling(window=p['bb_window']).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (p['bb_mult'] * df['bb_std'])
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    fast_k = 100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)
    df['stoch_k'] = fast_k.rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    return df

# 시간 필터 로직
def check_time_filter(dt, params):
    if not params['use_time_filter']: return True
    t = dt.time()
    if params['use_s1']:
        start, end = params['s1_start'], params['s1_end']
        if start <= end:
            if start <= t <= end: return True
        else:
            if t >= start or t <= end: return True
    if params['use_s2']:
        start, end = params['s2_start'], params['s2_end']
        if start <= end:
            if start <= t <= end: return True
        else:
            if t >= start or t <= end: return True
    return False

# ==========================================
# 🌟 수수료 조절형 백테스트 엔진
# ==========================================
def run_backtest_master(df, p):
    balance = 10_000 
    position = 0
    entry_price = 0
    entry_time = None
    
    # 🔴 사용자가 입력한 왕복 수수료를 편도로 변환 (0.1% 입력 시 0.0005)
    fee_rate = (p['fee_pct'] / 100) / 2
    
    trades, buy_points, sell_points = [], [], []
    
    for i in range(30, len(df)-1):
        prev = df.iloc[i-1]; current = df.iloc[i]; next_candle = df.iloc[i+1]
        
        if position == 0:
            buy_signal = True
            if p['use_time_filter'] and not check_time_filter(current['timestamp'], p): buy_signal = False
            if p['use_ma_cross'] and not (current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']): buy_signal = False
            if p['use_stoch_cross'] and not (current['stoch_k'] > current['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']): buy_signal = False
            if p['use_bb_touch'] and not (current['low'] <= current['bb_lower']): buy_signal = False
            if p['use_rsi'] and not (current['rsi'] <= p['rsi_buy_limit']): buy_signal = False
            if not any([p['use_ma_cross'], p['use_stoch_cross'], p['use_bb_touch'], p['use_rsi']]): buy_signal = False

            if buy_signal:
                entry_price = next_candle['open']
                entry_time = next_candle['timestamp']
                # 매수 시 수수료 차감
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))

        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            target_price = max(current['bb_mid'], base_tp) if p['exit_mode'] == "🎯 Max 모드" else min(current['bb_mid'], base_tp) 
            if p['use_downshift'] and elapsed_mins >= p['downshift_mins']:
                target_price = min(target_price, entry_price * (1 + p['downshift_tp_pct'] / 100))
                
            sl_price = entry_price * (1 - p['sl_pct'] / 100)
            sell_price = 0
            if next_candle['low'] <= sl_price: sell_price = min(next_candle['open'], sl_price)
            elif next_candle['high'] >= target_price: sell_price = max(next_candle['open'], target_price)
            elif p['use_dead_cross'] and (current['sma_slow'] > current['sma_fast'] and prev['sma_slow'] <= prev['sma_fast']):
                sell_price = next_candle['open']
            
            if sell_price > 0:
                # 매도 시 수수료 차감
                balance = position * sell_price * (1 - fee_rate)
                trades.append((sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# UI 셋업
# ==========================================
st.sidebar.header("🌐 1. 시장 및 수수료 설정")
ticker = st.sidebar.selectbox("대상 코인", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=29))
end_date = col2.date_input("종료일", datetime.date.today())
timeframe = st.sidebar.selectbox("⏱️ 타임프레임", options=['5m', '15m', '30m', '1h', '1d'], index=0)

# 🌟 [NEW] 수수료 조절 UI
p = {}
p['fee_pct'] = st.sidebar.slider("💸 왕복 매매 수수료 (%)", 0.00, 0.50, 0.10, step=0.01, 
                               help="거래 시 발생하는 총 수수료입니다. 업비트 보통 0.1%, 바이낸스 VIP 0.04~0.08% 수준입니다.")

st.sidebar.markdown("---")
st.sidebar.header("🕒 2. 시간대 필터 (KST)")
p['use_time_filter'] = st.sidebar.checkbox("시간 필터 적용", value=True)
if p['use_time_filter']:
    p['use_s1'] = st.sidebar.checkbox("☀️ 오전 세션 (08:30~12:00)", value=True)
    if p['use_s1']:
        c1, c2 = st.sidebar.columns(2)
        p['s1_start'] = c1.time_input("시작", datetime.time(8, 30), key="s1s")
        p['s1_end'] = c2.time_input("종료", datetime.time(12, 0), key="s1e")
    p['use_s2'] = st.sidebar.checkbox("🌙 야간 세션 (22:00~02:30)", value=True)
    if p['use_s2']:
        c3, c4 = st.sidebar.columns(2)
        p['s2_start'] = c3.time_input("시작", datetime.time(22, 0), key="s2s")
        p['s2_end'] = c4.time_input("종료", datetime.time(2, 30), key="s2e")

st.sidebar.markdown("---")
st.sidebar.header("🧩 3. 매수 전략 (Entry)")
with st.sidebar.expander("📈 지표 상세 설정"):
    p['use_ma_cross'] = st.checkbox("MA 골든크로스"); p['ma_fast'] = st.number_input("단기 이평", value=10); p['ma_slow'] = st.number_input("장기 이평", value=20)
    st.markdown("---")
    p['use_stoch_cross'] = st.checkbox("스토캐스틱 골든크로스")
    p['stoch_k_len'] = st.number_input("K 길이", value=5); p['stoch_k_smooth'] = st.number_input("K 스무딩", value=3); p['stoch_d_smooth'] = st.number_input("D 스무딩", value=3)
with st.sidebar.expander("📉 볼밴 & RSI", expanded=True):
    p['use_bb_touch'] = st.checkbox("볼밴 하단 터치", value=True); p['bb_window'] = st.number_input("BB 기간", value=20); p['bb_mult'] = st.number_input("BB 승수", value=2.0)
    p['use_rsi'] = st.checkbox("RSI 필터", value=True); p['rsi_period'] = st.number_input("RSI 기간", value=14); p['rsi_buy_limit'] = st.number_input("RSI 진입선", value=30.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 청산 전략 (Exit)")
c5, c6 = st.sidebar.columns(2)
p['tp_pct'] = c5.number_input("✅ 목표 수익 %", value=1.0, step=0.1)
p['sl_pct'] = c6.number_input("🛑 손절 라인 %", value=2.0, step=0.1)
p['exit_mode'] = st.sidebar.radio("익절 모드", ["🎯 Max 모드", "⚡ OR 모드"])
p['use_dead_cross'] = st.sidebar.checkbox("📉 이평선 데드크로스 탈출")
p['use_downshift'] = st.sidebar.checkbox("⏳ 다운시프트 활성화", value=True)
if p['use_downshift']:
    c7, c8 = st.sidebar.columns(2)
    p['downshift_mins'] = c7.number_input("발동(분)", value=60)
    p['downshift_tp_pct'] = c8.number_input("목표 %", value=0.2)

run_btn = st.sidebar.button("▶️ 퀀트 시뮬레이션 시작", type="primary", width="stretch")

# ==========================================
# 실행 메인
# ==========================================
if run_btn:
    with st.spinner("야후 파이낸스에서 데이터를 안전하게 가져오는 중..."):
        df = fetch_data_via_yf(ticker, start_date, end_date, interval=timeframe)
    if df is not None:
        df = add_indicators(df, p)
        profit, trades, buys, sells, result_df = run_backtest_master(df, p)
        st.subheader(f"📊 백테스트 리포트: {ticker} (수수료 {p['fee_pct']}% 반영)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 수익률", f"{profit:.2f}%"); c2.metric("거래 횟수", f"{len(trades)}회"); c3.metric("승률", f"{win_rate := (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0:.1f}%"); c4.metric("평균 수익", f"{np.mean(trades) if trades else 0:.2f}%")
        fig = go.Figure(); fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
        if buys: bt, bp = zip(*buys); fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        if sells: st_times, sp = zip(*sells); fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
        st.markdown("---")
        st.markdown("### 💰 수수료를 줄여야 수익이 극대화됩니다!")
        st.info(f"방금 테스트하신 전략은 수수료 **{p['fee_pct']}%** 조건에서 시뮬레이션되었습니다. 만약 바이비트 VIP 혜택을 받는다면 수익률은 더 올라갑니다.")
        st.link_button("⚫ 바이비트(Bybit) 수수료 할인 + 증정금 가입", "https://www.bybit.com/register?affiliate_id=총감독님_아이디", type="primary", width="stretch")
