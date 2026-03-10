import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import time

# 2026 Next-Gen 설정 및 한국어 UI 최적화
st.set_page_config(page_title="나만의 매매기법 테스트", layout="wide")
st.title("🚀 나만의 매매기법 테스트")
st.markdown("### 나만의 매매기법을 직접 설정하고 최적의 거래 조건을 찾아보세요!")

# ==========================================
# 🌟 데이터 수집 엔진 (KST 보정 및 안정화)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_data_via_yf(symbol, start_date, end_date, interval="5m"):
    yf_symbol = symbol.replace("USDT", "-USD")
    try:
        data = yf.download(
            tickers=yf_symbol, 
            start=start_date, 
            end=pd.to_datetime(end_date) + pd.Timedelta(days=1), 
            interval=interval, 
            progress=False
        )
        if data.empty: return None
        df = data.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        rename_map = {'Datetime': 'timestamp', 'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=rename_map, inplace=True)
        # UTC -> KST 보정
        df['timestamp'] = df['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=9)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# ==========================================
# 🛠️ 정교한 지표 계산 로직
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
    # 이동평균선
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    # 스토캐스틱
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    fast_k = 100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)
    df['stoch_k'] = fast_k.rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    return df

# 시간대 필터 로직
def check_time_filter(dt, params):
    if not params['use_time_filter']: return True
    t = dt.time()
    if params['use_s1']:
        start, end = params['s1_start'], params['s1_end']
        if start <= end:
            if start <= t <= end: return True
        else: # 자정 돌파
            if t >= start or t <= end: return True
    if params['use_s2']:
        start, end = params['s2_start'], params['s2_end']
        if start <= end:
            if start <= t <= end: return True
        else: # 자정 돌파
            if t >= start or t <= end: return True
    return False

# ==========================================
# 🏁 정교한 백테스트 엔진
# ==========================================
def run_backtest_master(df, p):
    balance = 10_000 
    position = 0
    entry_price = 0
    entry_time = None
    fee_rate = (p['fee_pct'] / 100) / 2
    trades, buy_points, sell_points = [], [], []
    
    for i in range(50, len(df)-1):
        prev = df.iloc[i-1]; current = df.iloc[i]; next_candle = df.iloc[i+1]
        
        # --- [매수 진입 로직] ---
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
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))

        # --- [매도 청산 로직] ---
        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            
            # 익절 모드
            if p['exit_mode'] == "🎯 Max 모드":
                target_price = max(current['bb_mid'], base_tp)
            else: # OR 모드
                target_price = min(current['bb_mid'], base_tp) 
                
            # 다운시프트
            if p['use_downshift'] and elapsed_mins >= p['downshift_mins']:
                target_price = min(target_price, entry_price * (1 + p['downshift_tp_pct'] / 100))
                
            sl_price = entry_price * (1 - p['sl_pct'] / 100)
            sell_price = 0
            
            # 체결 체크
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
# 🎨 UI: 정교한 사이드바 레이아웃
# ==========================================
st.sidebar.header("📂 1. 데이터 및 수수료")
ticker = st.sidebar.selectbox("테스트 코인", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=29))
end_date = col_d2.date_input("종료일", datetime.date.today())
timeframe = st.sidebar.selectbox("캔들 주기", options=['5m', '15m', '30m', '1h', '1d'], index=0)

p = {}
p['fee_pct'] = st.sidebar.slider("💸 왕복 수수료 (%)", 0.00, 0.50, 0.10, step=0.01, help="실제 수익과 직결됩니다. 신중하게 설정하세요.")

st.sidebar.markdown("---")
st.sidebar.header("🕒 2. 매매 시간대 설정 (KST)")
p['use_time_filter'] = st.sidebar.checkbox("시간대 필터 활성화", value=True)
if p['use_time_filter']:
    p['use_s1'] = st.sidebar.checkbox("☀️ 오전 집중 세션", value=True)
    if p['use_s1']:
        c_s1, c_e1 = st.sidebar.columns(2)
        p['s1_start'] = c_s1.time_input("시작", datetime.time(8, 30), key="k_s1s")
        p['s1_end'] = c_e1.time_input("종료", datetime.time(12, 0), key="k_s1e")
    p['use_s2'] = st.sidebar.checkbox("🌙 야간 집중 세션", value=True)
    if p['use_s2']:
        c_s2, c_e2 = st.sidebar.columns(2)
        p['s2_start'] = c_s2.time_input("시작", datetime.time(22, 0), key="k_s2s")
        p['s2_end'] = c_e2.time_input("종료", datetime.time(2, 30), key="k_s2e")

st.sidebar.markdown("---")
st.sidebar.header("🧩 3. 매수 기법 조립")
with st.sidebar.expander("📈 이동평균선 & 스토캐스틱", expanded=False):
    p['use_ma_cross'] = st.checkbox("이평선 골든크로스")
    p['ma_fast'] = st.number_input("단기 이평선", value=10); p['ma_slow'] = st.number_input("장기 이평선", value=20)
    st.markdown("---")
    p['use_stoch_cross'] = st.checkbox("스토캐스틱 골든크로스")
    p['stoch_k_len'] = st.number_input("K 기간", value=5); p['stoch_k_smooth'] = st.number_input("K 스무딩", value=3); p['stoch_d_smooth'] = st.number_input("D 스무딩", value=3)

with st.sidebar.expander("📉 볼린저 밴드 & RSI", expanded=True):
    p['use_bb_touch'] = st.checkbox("볼밴 하단 터치 시 매수", value=True)
    p['bb_window'] = st.number_input("볼밴 기간", value=20); p['bb_mult'] = st.number_input("볼밴 승수", value=2.0)
    st.markdown("---")
    p['use_rsi'] = st.checkbox("RSI 과매도 필터 사용", value=True)
    p['rsi_period'] = st.number_input("RSI 기간", value=14); p['rsi_buy_limit'] = st.number_input("RSI 진입 기준선", value=30.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 청산(매도) 전략")
col_tp, col_sl = st.sidebar.columns(2)
p['tp_pct'] = col_tp.number_input("✅ 목표 수익 %", value=1.0, step=0.1, format="%.1f")
p['sl_pct'] = col_sl.number_input("🛑 손절 라인 %", value=2.0, step=0.1, format="%.1f")
p['exit_mode'] = st.sidebar.radio("청산 모드", ["🎯 Max 모드", "⚡ OR 모드"], help="Max: 중앙선까지 수익 홀딩 / OR: TP 도달 시 즉시 매도")
p['use_dead_cross'] = st.sidebar.checkbox("📉 이평선 데드크로스 탈출")

with st.sidebar.expander("⏳ 다운시프트(Downshift) 설정", expanded=False):
    p['use_downshift'] = st.checkbox("다운시프트 활성화", value=True)
    p['downshift_mins'] = st.number_input("발동 시간(분)", value=60)
    p['downshift_tp_pct'] = st.number_input("하향된 목표 수익 %", value=0.2, step=0.1, format="%.1f")

run_btn = st.sidebar.button("▶️ 기법 검증 시작", type="primary", width="stretch")

# ==========================================
# 📊 결과 리포트 섹션
# ==========================================
if run_btn:
    with st.spinner("과거 데이터를 정밀 분석 중입니다..."):
        df = fetch_data_via_yf(ticker, start_date, end_date, interval=timeframe)
        
    if df is not None:
        df = add_indicators(df, p)
        profit, trades, buys, sells, result_df = run_backtest_master(df, p)
        
        # 🌟 [수정] NameError 방지를 위한 변수 선행 계산
        total_trades = len(trades)
        win_count = len([t for t in trades if t > 0])
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_profit = np.mean(trades) if total_trades > 0 else 0
        
        st.subheader(f"📊 백테스트 결과: {ticker} ({timeframe})")
        st.caption(f"수수료 {p['fee_pct']}%를 제외한 순수 자산 변동 결과입니다.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 수익률", f"{profit:.2f}%")
        c2.metric("총 거래 횟수", f"{total_trades}회")
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("평균 수익률", f"{avg_profit:.2f}%")
        
        # 차트 시각화
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
        if buys:
            bt, bp = zip(*buys); fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        if sells:
            st_times, sp = zip(*sells); fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
        
        # 💰 수익화 레퍼럴 버튼
        st.markdown("---")
        st.markdown("### 💰 찾으신 기법을 바이비트에서 실전 가동해 보세요!")
        st.info("💡 퀀트 매매는 수수료와의 싸움입니다. 아래 링크로 가입하셔야 평생 수수료 20% 할인 혜택을 받고 수익률을 극대화할 수 있습니다.")
        st.link_button("⚫ 바이비트(Bybit) 수수료 할인 및 VIP 혜택 가입", "https://www.bybit.com/register?affiliate_id=총감독님_아이디", type="primary", width="stretch")
