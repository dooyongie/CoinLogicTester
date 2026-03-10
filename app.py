import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import time

# UI 설정
st.set_page_config(page_title="나만의 매매기법 테스트", layout="wide")
st.title("🚀 나만의 매매기법 테스트")
st.markdown("### 나만의 매매기법을 직접 설정하고 최적의 거래 조건을 찾아보세요!")

# ==========================================
# 데이터 수집 및 지표 계산
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

def add_indicators(df, p):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # 볼린저 밴드 계산
    df['bb_mid'] = df['close'].rolling(window=p['bb_window']).mean()
    df['bb_std'] = df['close'].rolling(window=p['bb_window']).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (p['bb_mult'] * df['bb_std'])
    df['bb_upper'] = df['bb_mid'] + (p['bb_mult'] * df['bb_std'])
    
    # 🌟 BB폭 계산 (Upper - Lower) / Mid * 100
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    df['stoch_k'] = (100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)).rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    return df

# 시간 필터
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

# 백테스트 엔진
def run_backtest_master(df, p):
    balance = 10_000 
    position, entry_price, entry_time = 0, 0, None
    fee_rate = (p['fee_pct'] / 100) / 2
    trades, buy_points, sell_points = [], [], []
    
    for i in range(50, len(df)-1):
        prev = df.iloc[i-1]; current = df.iloc[i]; next_candle = df.iloc[i+1]
        
        if position == 0:
            buy_signal = True
            if p['use_time_filter'] and not check_time_filter(current['timestamp'], p): buy_signal = False
            
            # 🌟 BB폭 제한 조건 체크
            if p['use_bb_width'] and not (current['bb_width'] >= p['bb_width_limit']): buy_signal = False
            
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
                balance = position * sell_price * (1 - fee_rate)
                trades.append((sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# 🎨 UI 레이아웃
# ==========================================
st.sidebar.header("📂 1. 기본 설정")
ticker = st.sidebar.selectbox("테스트 코인", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=29))
end_date = col_d2.date_input("종료일", datetime.date.today())
timeframe = st.sidebar.selectbox("캔들 주기", options=['5m', '15m', '30m', '1h', '1d'], index=0)
p = {}
p['fee_pct'] = st.sidebar.slider("💸 왕복 수수료 (%)", 0.00, 0.50, 0.10, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("🕒 2. 매매 시간대 (KST)")
p['use_time_filter'] = st.sidebar.checkbox("시간대 필터 사용", value=True)
if p['use_time_filter']:
    c_s1, c_e1 = st.sidebar.columns(2)
    p['use_s1'] = st.sidebar.checkbox("☀️ 오전 세션", value=True)
    p['s1_start'] = c_s1.time_input("시작", datetime.time(8, 30), key="t1")
    p['s1_end'] = c_e1.time_input("종료", datetime.time(12, 0), key="t2")
    c_s2, c_e2 = st.sidebar.columns(2)
    p['use_s2'] = st.sidebar.checkbox("🌙 야간 세션", value=True)
    p['s2_start'] = c_s2.time_input("시작", datetime.time(22, 0), key="t3")
    p['s2_end'] = c_e2.time_input("종료", datetime.time(2, 30), key="t4")

st.sidebar.markdown("---")
st.sidebar.header("🧩 3. 매수 기법 (Entry)")
with st.sidebar.expander("📊 볼린저 밴드 상세 설정", expanded=True):
    # 🌟 BB폭 상세 조절 기능 추가
    p['use_bb_width'] = st.checkbox("BB폭 하한선 제한 (변동성 필터)", value=True, help="박스권 횡보장에서 무분별한 진입을 막기 위해 밴드의 폭이 일정 이상 벌어졌을 때만 진입합니다.")
    p['bb_width_limit'] = st.number_input("최소 BB폭 (%)", value=1.2, step=0.1, format="%.1f")
    st.markdown("---")
    p['use_bb_touch'] = st.checkbox("밴드 하단 터치 시 매수", value=True)
    p['bb_window'] = st.number_input("볼밴 기간", value=20)
    p['bb_mult'] = st.number_input("볼밴 승수", value=2.0, step=0.1)

with st.sidebar.expander("📈 이평선 & RSI & 스토캐스틱"):
    p['use_ma_cross'] = st.checkbox("이평선 골든크로스")
    p['ma_fast'] = st.number_input("단기 이평", value=10); p['ma_slow'] = st.number_input("장기 이평", value=20)
    p['use_rsi'] = st.checkbox("RSI 과매도 필터")
    p['rsi_period'] = st.number_input("RSI 기간", value=14); p['rsi_buy_limit'] = st.number_input("진입선", value=30.0)
    p['use_stoch_cross'] = st.checkbox("스토캐스틱 골든크로스")
    p['stoch_k_len'] = 5; p['stoch_k_smooth'] = 3; p['stoch_d_smooth'] = 3

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 청산 전략 (Exit)")
col_tp, col_sl = st.sidebar.columns(2)
p['tp_pct'] = col_tp.number_input("✅ 목표 수익 %", value=1.0, step=0.1)
p['sl_pct'] = col_sl.number_input("🛑 손절 라인 %", value=2.0, step=0.1)
p['exit_mode'] = st.sidebar.radio("청산 모드", ["🎯 Max 모드", "⚡ OR 모드"], help="Max: 중앙선까지 홀딩 / OR: 먼저 닿는 곳 청산")
p['use_dead_cross'] = st.sidebar.checkbox("📉 이평선 데드크로스 탈출")

# 🌟 다운시프트 상세 설명 및 설정
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ 다운시프트(Downshift)")
st.sidebar.info("기관급 단타 필수 로직! 진입 후 가격이 오르지 않고 시간이 지체되면 자금이 묶이는 것을 방지하기 위해 목표가를 낮춰서 빠르게 탈출하는 기술입니다.")
p['use_downshift'] = st.sidebar.checkbox("다운시프트 활성화", value=True)
if p['use_downshift']:
    c_ds1, c_ds2 = st.sidebar.columns(2)
    p['downshift_mins'] = c_ds1.number_input("발동 시간(분)", value=60)
    p['downshift_tp_pct'] = c_ds2.number_input("하향 목표 %", value=0.2, step=0.1)

run_btn = st.sidebar.button("▶️ 기법 검증 시작", type="primary", width="stretch")

# 결과 출력
if run_btn:
    with st.spinner("과거 데이터를 정밀 분석 중입니다..."):
        df = fetch_data_via_yf(ticker, start_date, end_date, interval=timeframe)
    if df is not None:
        df = add_indicators(df, p)
        profit, trades, buys, sells, result_df = run_backtest_master(df, p)
        total_trades = len(trades)
        win_rate = (len([t for t in trades if t > 0]) / total_trades * 100) if total_trades > 0 else 0
        
        st.subheader(f"📊 백테스트 결과: {ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 수익률", f"{profit:.2f}%")
        c2.metric("총 거래 횟수", f"{total_trades}회")
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("평균 수익률", f"{np.mean(trades) if total_trades > 0 else 0:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
        if buys:
            bt, bp = zip(*buys); fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        if sells:
            st_times, sp = zip(*sells); fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("---")
        st.markdown("### 💰 실전 자동매매는 바이비트에서 시작하세요!")
        st.link_button("⚫ 바이비트(Bybit) 수수료 할인 및 혜택 가입", "https://www.bybit.com/register?affiliate_id=총감독님_아이디", type="primary", width="stretch")
