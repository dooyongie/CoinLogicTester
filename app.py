import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import pyupbit
import time
import datetime

# 2026년형 스트림릿 설정
st.set_page_config(page_title="CoinLogicTester v6.0", layout="wide")
st.title("🚀 CoinLogicTester v6.0 (2026 Next-Gen)")
st.markdown("### 2026년 최신 표준 문법과 타임존 무결성 로직이 적용된 최종 진화형입니다.")

# ==========================================
# 🌟 [무결점] 데이터 수집 엔진 (Timezone Stripper)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_upbit_data(ticker, start_date, end_date):
    df_list = []
    
    # 모든 날짜를 꼬리표 없는 Naive 상태로 통일
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    end_dt = end_dt.tz_localize(None)
    start_dt = pd.to_datetime(start_date).tz_localize(None)
    
    current_to = end_dt 
    max_iters = 500
    iter_count = 0

    while current_to > start_dt and iter_count < max_iters:
        iter_count += 1
        df = pyupbit.get_ohlcv(ticker, interval="minute5", to=current_to, count=200)
        
        if df is None or df.empty: break
        
        # 가져온 데이터 인덱스의 타임존 강제 제거
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        else:
            df.index = df.index.tz_localize(None)
            
        df_list.append(df)
        
        first_timestamp = df.index[0]
        if first_timestamp <= start_dt: break
            
        current_to = first_timestamp
        time.sleep(0.15) 
        
    if not df_list: return None
        
    final_df = pd.concat(df_list)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    final_df.sort_index(inplace=True)
    
    final_df = final_df.loc[start_dt:end_dt] 
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # timestamp 컬럼도 Naive인지 한 번 더 확인
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp']).dt.tz_localize(None)
    
    if 'value' not in final_df.columns:
        final_df['value'] = final_df['close'] * final_df['volume']
        
    return final_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'value']]

# ==========================================
# 캔들 변환 및 지표 계산
# ==========================================
def resample_data(df, timeframe):
    if timeframe == '5분봉': return df
    tf_map = {'10분봉': '10T', '15분봉': '15T', '30분봉': '30T', '1시간봉': '1H', '4시간봉': '4H', '일봉': '1D'}
    rule = tf_map.get(timeframe, '5T')
    df = df.set_index('timestamp')
    resampled = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    return resampled.reset_index()

def add_indicators(df, rsi_period, bb_window, bb_mult):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['bb_mid'] = df['close'].rolling(window=bb_window).mean()
    df['bb_std'] = df['close'].rolling(window=bb_window).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (bb_mult * df['bb_std'])
    df['bb_upper'] = df['bb_mid'] + (bb_mult * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    return df

# 시간 필터 로직
def check_time_filter(dt, params):
    if not params['use_time_filter']: return True
    t = dt.time()
    if params['use_session_1']:
        s1_start, s1_end = params['s1_start'], params['s1_end']
        if s1_start <= s1_end:
            if s1_start <= t <= s1_end: return True
        else:
            if t >= s1_start or t <= s1_end: return True
    if params['use_session_2']:
        s2_start, s2_end = params['s2_start'], params['s2_end']
        if s2_start <= s2_end:
            if s2_start <= t <= s2_end: return True
        else:
            if t >= s2_start or t <= s2_end: return True
    return False

# 백테스트 엔진
def run_backtest_ultimate(df, params):
    balance = 10_000_000
    position = 0
    entry_price = 0
    fee_rate = 0.0005  
    bars_since_entry = 0 
    last_sell_time = df['timestamp'].iloc[0] - pd.Timedelta(days=1)
    trades, buy_points, sell_points = [], [], []
    
    for i in range(20, len(df)-1):
        current = df.iloc[i]        
        next_candle = df.iloc[i+1]  
        just_bought = False
        seconds_since_sell = (next_candle['timestamp'] - last_sell_time).total_seconds()
        
        if position == 0 and seconds_since_sell >= params['cooldown_sec'] and check_time_filter(current['timestamp'], params):
            cond1 = current['bb_width'] >= (params['volatility_limit'] / 100)
            cond2 = current['low'] <= current['bb_lower']
            cond3 = current['rsi'] < params['rsi_buy_limit']
            if cond1 and cond2 and cond3:
                entry_price = next_candle['open']
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))
                bars_since_entry = 0
                just_bought = True

        if position > 0:
            if not just_bought: bars_since_entry += 1
            base_target = max(current['bb_mid'], entry_price * (1 + params['min_profit_pct'] / 100))
            if bars_since_entry >= params['downshift_bars']:
                pr_approx = (current['close'] - entry_price) / entry_price
                if pr_approx > (params['downshift_allow_pct'] / 100):
                    base_target = min(base_target, entry_price * (1 + params['downshift_target_pct'] / 100))
            tp_price, sl_price = base_target, entry_price * (1 - params['stop_loss_pct'] / 100)
            sell_price = 0
            if next_candle['low'] <= sl_price: sell_price = min(next_candle['open'], sl_price)
            elif next_candle['high'] >= tp_price: sell_price = max(next_candle['open'], tp_price)
            if sell_price > 0:
                balance = position * sell_price * (1 - fee_rate)
                trades.append((sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0
                last_sell_time = next_candle['timestamp']

    if position > 0: # 미청산 정산
        last_candle = df.iloc[-1]
        balance = position * last_candle['close'] * (1 - fee_rate)
        trades.append((last_candle['close'] / entry_price * (1 - fee_rate * 2) - 1) * 100)
        sell_points.append((last_candle['timestamp'], last_candle['close']))

    return (balance / 10_000_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# UI 셋업 (2026 최신 문법 적용)
# ==========================================
st.sidebar.header("📂 1. 데이터 소스 설정")

# 2026 표준: width="stretch" (기존 use_container_width=True 대체)
if st.sidebar.button("🔄 실시간 캐시 초기화", width="stretch"):
    st.cache_data.clear()
    st.sidebar.success("캐시 초기화 완료!")

data_source = st.sidebar.radio("데이터 불러오기 방식", ["🌐 업비트 실시간 다운로드", "📁 로컬 CSV 파일"])

if data_source == "🌐 업비트 실시간 다운로드":
    ticker = st.sidebar.text_input("코인 티커", value="KRW-XRP")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=14))
    end_date = col2.date_input("종료일", datetime.date.today())
    is_utc = False 
    selected_file = f"{ticker} ({start_date} ~ {end_date})"
else:
    available_files = [f for f in os.listdir() if f.endswith('.csv')]
    if not available_files: st.stop()
    selected_file = st.sidebar.selectbox("테스트 파일", available_files)
    is_utc = st.sidebar.checkbox("📡 UTC 기준 데이터입니까?", value=False)

selected_tf = st.sidebar.select_slider("⏱️ 타임프레임", options=['5분봉', '10분봉', '15분봉', '30분봉', '1시간봉'])

st.sidebar.markdown("---")
st.sidebar.header("🎯 2. 진입 조건")
p = {}
p['volatility_limit'] = st.sidebar.number_input("1️⃣ BB폭 하한선 (%)", value=1.2, format="%.2f")
p['bb_mult'] = st.sidebar.number_input("2️⃣ BB 승수", value=2.0)
p['rsi_period'] = st.sidebar.number_input("3️⃣ RSI 기간", value=4)
p['rsi_buy_limit'] = st.sidebar.number_input("4️⃣ RSI 기준", value=16.0)
p['bb_window'] = 20
p['cooldown_sec'] = st.sidebar.number_input("⏳ 쿨다운 (초)", value=300)

st.sidebar.markdown("---")
st.sidebar.header("🕒 3. 시간 필터")
p['use_time_filter'] = st.sidebar.checkbox("시간 필터 적용", value=True)
if p['use_time_filter']:
    p['use_session_1'] = st.sidebar.checkbox("☀️ 오전장", value=True)
    if p['use_session_1']:
        c1, c2 = st.sidebar.columns(2)
        p['s1_start'] = c1.time_input("시작", datetime.time(8, 30))
        p['s1_end'] = c2.time_input("종료", datetime.time(11, 59))
    p['use_session_2'] = st.sidebar.checkbox("🌙 야간장", value=True)
    if p['use_session_2']:
        c3, c4 = st.sidebar.columns(2)
        p['s2_start'] = c3.time_input("시작", datetime.time(22, 0))
        p['s2_end'] = c4.time_input("종료", datetime.time(2, 30))

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 청산 조건")
p['min_profit_pct'] = st.sidebar.number_input("✅ 목표 수익 (%)", value=0.7)
p['stop_loss_pct'] = st.sidebar.number_input("🛑 손절 라인 (%)", value=2.0)
p['downshift_bars'] = st.sidebar.number_input("⏳ 다운시프트 (캔들)", value=12) 
p['downshift_target_pct'] = st.sidebar.number_input("📉 하향 목표 (%)", value=0.3)
p['downshift_allow_pct'] = st.sidebar.number_input("🛡️ 허용 수익 (%)", value=-0.5)

# 2026 표준: width="stretch"
run_btn = st.sidebar.button("▶️ 시뮬레이션 실행", type="primary", width="stretch")

if run_btn:
    if data_source == "🌐 업비트 실시간 다운로드":
        df = fetch_upbit_data(ticker, start_date, end_date)
    else:
        df = pd.read_csv(selected_file)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        if is_utc: df['timestamp'] += pd.Timedelta(hours=9)
    
    if df is None: st.error("데이터 오류!"); st.stop()
    
    df = resample_data(df, selected_tf)
    df = add_indicators(df, p['rsi_period'], p['bb_window'], p['bb_mult'])
    
    with st.spinner("2026 엔진 가동 중..."):
        profit, trades, buys, sells, result_df = run_backtest_ultimate(df, p)
        
    st.subheader(f"📊 백테스트 결과: {selected_file}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("수익률", f"{profit:.2f}%")
    col2.metric("거래", f"{len(trades)}회")
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    col3.metric("승률", f"{win_rate:.1f}%")
    col4.metric("평균수익", f"{np.mean(trades) if trades else 0:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close']))
    if buys:
        bt, bp = zip(*buys)
        fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
    if sells:
        st_times, sp = zip(*sells)
        fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
                                 
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False)
    # 2026 표준: width="stretch"
    st.plotly_chart(fig, width="stretch")