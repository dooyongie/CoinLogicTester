import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime

# ==========================================
# 1. UI 및 기본 설정
# ==========================================
st.set_page_config(page_title="나만의 퀀트 매매기법 테스트", layout="wide")
st.title("🚀 나만의 퀀트 매매기법 테스트")
st.markdown("### 📊 수십 가지 지표를 조합하여 나만의 최적 거래 조건을 설계하세요!")

# ==========================================
# 2. 데이터 수집 및 캔들 합성 엔진
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_base_data(symbol, start_date, end_date):
    yf_symbol = symbol.replace("USDT", "-USD")
    try:
        data = yf.download(tickers=yf_symbol, start=start_date, end=pd.to_datetime(end_date) + pd.Timedelta(days=1), interval="5m", progress=False)
        if data.empty: return None
        df = data.reset_index()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        rename_map = {'Datetime': 'timestamp', 'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=rename_map, inplace=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None) + pd.Timedelta(hours=9)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

def resample_dataframe(df, tf_rule):
    if tf_rule == '5T': return df
    df = df.set_index('timestamp')
    resampled = df.resample(tf_rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    return resampled.reset_index()

# ==========================================
# 3. 기술적 지표 계산
# ==========================================
def add_all_indicators(df, p):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/p['rsi_period'], adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    df['bb_mid'] = df['close'].rolling(window=p['bb_window']).mean()
    df['bb_std'] = df['close'].rolling(window=p['bb_window']).std(ddof=0)
    df['bb_lower'] = df['bb_mid'] - (p['bb_mult'] * df['bb_std'])
    df['bb_upper'] = df['bb_mid'] + (p['bb_mult'] * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    df['ema_fast'] = df['close'].ewm(span=p['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=p['ema_slow'], adjust=False).mean()
    
    macd_line = df['close'].ewm(span=p['macd_short'], adjust=False).mean() - df['close'].ewm(span=p['macd_long'], adjust=False).mean()
    df['macd_signal'] = macd_line.ewm(span=p['macd_signal_len'], adjust=False).mean()
    df['macd_hist'] = macd_line - df['macd_signal']
    
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    df['stoch_k'] = (100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)).rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    
    hl2 = (df['high'] + df['low']) / 2
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    atr = pd.Series(tr).rolling(p['st_atr']).mean()
    basic_ub = hl2 + (p['st_mult'] * atr)
    basic_lb = hl2 - (p['st_mult'] * atr)
    
    final_ub, final_lb, trend = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for i in range(p['st_atr'], len(df)):
        prev_ub = final_ub[i-1] if final_ub[i-1] != 0 else basic_ub.iloc[i-1]
        prev_lb = final_lb[i-1] if final_lb[i-1] != 0 else basic_lb.iloc[i-1]
        final_ub[i] = basic_ub.iloc[i] if basic_ub.iloc[i] < prev_ub or df['close'].iloc[i-1] > prev_ub else prev_ub
        final_lb[i] = basic_lb.iloc[i] if basic_lb.iloc[i] > prev_lb or df['close'].iloc[i-1] < prev_lb else prev_lb
        if df['close'].iloc[i] > final_ub[i]: trend[i] = 1
        elif df['close'].iloc[i] < final_lb[i]: trend[i] = -1
        else: trend[i] = trend[i-1]
    df['supertrend'] = trend
    return df

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
# 4. 백테스트 코어 엔진 (청산 로직 업그레이드)
# ==========================================
def run_backtest_ultimate(df, p):
    balance = 10_000 
    position, entry_price, entry_time = 0, 0, None
    fee_rate = (p['fee_pct'] / 100) / 2
    trades, buy_points, sell_points = [], [], []
    
    for i in range(50, len(df)-1):
        prev = df.iloc[i-1]
        current = df.iloc[i]
        next_candle = df.iloc[i+1]
        
        eval_candle = prev if p['entry_timing'] == "안전: 캔들 종가 마감 확인 후 진입" else current
        eval_prev = df.iloc[i-2] if p['entry_timing'] == "안전: 캔들 종가 마감 확인 후 진입" else prev
        
        if position == 0:
            if p['use_time_filter'] and not check_time_filter(current['timestamp'], p): continue
            
            signals = []
            if p['use_bb_width']: signals.append(eval_candle['bb_width'] >= p['bb_width_limit'])
            if p['use_bb_touch']: signals.append(eval_candle['low'] <= eval_candle['bb_lower'])
            if p['use_rsi']: signals.append(eval_candle['rsi'] <= p['rsi_buy_limit'])
            if p['use_ma_cross']: signals.append(eval_candle['sma_fast'] > eval_candle['sma_slow'] and eval_prev['sma_fast'] <= eval_prev['sma_slow'])
            if p['use_ema_cross']: signals.append(eval_candle['ema_fast'] > eval_candle['ema_slow'] and eval_prev['ema_fast'] <= eval_prev['ema_slow'])
            if p['use_stoch_cross']: signals.append(eval_candle['stoch_k'] > eval_candle['stoch_d'] and eval_prev['stoch_k'] <= eval_prev['stoch_d'])
            if p['use_macd']: signals.append(eval_candle['macd_hist'] > 0 and eval_prev['macd_hist'] <= 0)
            if p['use_supertrend']: signals.append(eval_candle['supertrend'] == 1 and eval_prev['supertrend'] == -1)
            
            if not signals: continue 
            buy_signal = all(signals) if p['logic_op'] == "모든 조건 동시 만족 (AND)" else any(signals)

            if buy_signal:
                entry_price = current['open'] if p['entry_timing'] == "안전: 캔들 종가 마감 확인 후 진입" else next_candle['open']
                entry_time = current['timestamp'] if p['entry_timing'] == "안전: 캔들 종가 마감 확인 후 진입" else next_candle['timestamp']
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((entry_time, entry_price))

        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            
            # 🌟 [신규] 볼린저 밴드 목표가 설정 로직
            if p['bb_exit_target'] == "사용 안 함 (오직 목표수익% 적용)":
                target_price = base_tp
            else:
                bb_target_price = current['bb_mid'] if p['bb_exit_target'] == "BB 중앙선 터치 시" else current['bb_upper']
                
                if p['exit_mode'] == "🎯 Max 모드 (수익 극대화)":
                    target_price = max(bb_target_price, base_tp)
                else: # OR 모드
                    target_price = min(bb_target_price, base_tp) 
            
            # 다운시프트 개입
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

    if position > 0:
        last_candle = df.iloc[-1]
        sell_price = last_candle['close']
        balance = position * sell_price * (1 - fee_rate)
        profit_pct = (sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100
        trades.append(profit_pct)
        sell_points.append((last_candle['timestamp'], sell_price))

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# 5. UI 사이드바
# ==========================================
if st.sidebar.button("🔄 실시간 데이터 캐시 초기화", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("캐시 초기화 완료! 최신 데이터를 불러옵니다.")

st.sidebar.header("📂 1. 데이터 및 기본 세팅")
ticker = st.sidebar.selectbox("테스트 코인", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
c_d1, c_d2 = st.sidebar.columns(2)

min_allowed_date = datetime.date.today() - datetime.timedelta(days=59)
start_date = c_d1.date_input("시작일", 
                             value=max(datetime.date.today() - datetime.timedelta(days=29), min_allowed_date),
                             min_value=min_allowed_date,
                             max_value=datetime.date.today())
end_date = c_d2.date_input("종료일", datetime.date.today(), max_value=datetime.date.today())

tf_options = {"5분": "5T", "10분": "10T", "15분": "15T", "30분": "30T", "45분": "45T", "1시간": "1H", "4시간": "4H", "1일": "1D"}
tf_choice = st.sidebar.selectbox("⏱️ 캔들 타임프레임", list(tf_options.keys()), index=0)
timeframe = tf_options[tf_choice]

p = {}
p['fee_pct'] = st.sidebar.slider("💸 왕복 수수료 (%)", 0.00, 0.50, 0.10, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("🕒 2. 매매 시간대 필터")
p['use_time_filter'] = st.sidebar.checkbox("시간대 필터 활성화", value=False)
if p['use_time_filter']:
    p['use_s1'] = st.sidebar.checkbox("☀️ 오전 세션", value=True)
    c_s1, c_e1 = st.sidebar.columns(2)
    p['s1_start'] = c_s1.time_input("시작", datetime.time(8, 30))
    p['s1_end'] = c_e1.time_input("종료", datetime.time(12, 0))
    p['use_s2'] = st.sidebar.checkbox("🌙 야간 세션", value=True)
    c_s2, c_e2 = st.sidebar.columns(2)
    p['s2_start'] = c_s2.time_input("시작", datetime.time(22, 0))
    p['s2_end'] = c_e2.time_input("종료", datetime.time(2, 30))

st.sidebar.markdown("---")
st.sidebar.header("🧩 3. 매수 기법 조립 (Entry)")
p['logic_op'] = st.sidebar.radio("지표 결합 방식", ["모든 조건 동시 만족 (AND)", "조건 중 하나라도 만족 (OR)"])
p['entry_timing'] = st.sidebar.radio("매수 타이밍", ["안전: 캔들 종가 마감 확인 후 진입", "공격: 조건 달성 시 실시간 즉시 진입"])

st.sidebar.markdown("##### 🔽 활성화할 지표를 체크하고 조율하세요")
with st.sidebar.expander("📉 볼린저 밴드 & 변동성", expanded=True):
    p['use_bb_width'] = st.checkbox("BB폭 하한선 제한", value=True)
    p['bb_width_limit'] = st.number_input("최소 BB폭 (%)", value=1.2, step=0.1)
    st.markdown("---")
    p['use_bb_touch'] = st.checkbox("밴드 하단 터치", value=True)
    p['bb_window'] = int(st.number_input("BB 기간", value=20))
    p['bb_mult'] = st.number_input("BB 승수", value=2.0, step=0.1)

with st.sidebar.expander("📈 이동평균 & MACD"):
    p['use_ma_cross'] = st.checkbox("단순이평(SMA) 골든크로스")
    p['ma_fast'] = int(st.number_input("SMA 단기", value=10)); p['ma_slow'] = int(st.number_input("SMA 장기", value=20))
    st.markdown("---")
    p['use_ema_cross'] = st.checkbox("지수이평(EMA) 골든크로스")
    p['ema_fast'] = int(st.number_input("EMA 단기", value=10)); p['ema_slow'] = int(st.number_input("EMA 장기", value=20))
    st.markdown("---")
    p['use_macd'] = st.checkbox("MACD 0선 돌파")
    p['macd_short'] = 12; p['macd_long'] = 26; p['macd_signal_len'] = 9

with st.sidebar.expander("🌊 오실레이터 & 슈퍼트렌드"):
    p['use_rsi'] = st.checkbox("RSI 과매도")
    p['rsi_period'] = int(st.number_input("RSI 기간", value=14)); p['rsi_buy_limit'] = st.number_input("진입선", value=30.0)
    st.markdown("---")
    p['use_stoch_cross'] = st.checkbox("스토캐스틱 골든크로스")
    p['stoch_k_len'] = int(st.number_input("K 기간", value=5)); p['stoch_k_smooth'] = 3; p['stoch_d_smooth'] = 3
    st.markdown("---")
    p['use_supertrend'] = st.checkbox("슈퍼트렌드 상승 전환")
    p['st_atr'] = int(st.number_input("ATR 기간", value=10)); p['st_mult'] = st.number_input("승수", value=3.0)

# 🌟 [신규] 청산 전략 UI 대폭 업그레이드
st.sidebar.markdown("---")
st.sidebar.header("🛡️ 4. 청산 전략 (Exit)")

c_tp, c_sl = st.sidebar.columns(2)
p['tp_pct'] = c_tp.number_input("✅ 목표수익 %", value=1.0, step=0.1)
p['sl_pct'] = c_sl.number_input("🛑 손절 %", value=2.0, step=0.1)

st.sidebar.markdown("##### 🎯 볼린저 밴드 연동 익절")
p['bb_exit_target'] = st.sidebar.radio(
    "어느 선에서 익절하시겠습니까?", 
    ["BB 중앙선 터치 시", "BB 상단선 터치 시", "사용 안 함 (오직 목표수익% 적용)"], 
    index=0
)

st.sidebar.markdown("##### ⚖️ 익절 결합 모드")
p['exit_mode'] = st.sidebar.radio(
    "목표수익% 와 볼밴 라인이 겹칠 때", 
    ["🎯 Max 모드 (수익 극대화)", "⚡ OR 모드 (빠른 청산)"],
    help="• Max 모드: 정해둔 %와 볼밴 라인 중 더 '높은' 가격을 기다려 크게 먹습니다.\n• OR 모드: 정해둔 %나 볼밴 라인 중 아무거나 먼저 닿으면 미련 없이 팝니다."
)

st.sidebar.markdown("---")
p['use_dead_cross'] = st.sidebar.checkbox("📉 SMA 데드크로스 비상 탈출", help="목표가에 안 왔어도 단기 이평선이 장기 이평선을 뚫고 내려가면 즉시 던집니다.")
p['use_downshift'] = st.sidebar.checkbox("⏳ 다운시프트 (시간 지체 시 탈출)", value=True, help="물려서 시간이 오래 지나면 목표가를 강제로 낮춰서 본절 부근에서 탈출합니다.")
if p['use_downshift']:
    c_dsm, c_dsp = st.sidebar.columns(2)
    p['downshift_mins'] = int(c_dsm.number_input("발동(분)", value=60))
    p['downshift_tp_pct'] = c_dsp.number_input("하향 %", value=0.2, step=0.1)

run_btn = st.sidebar.button("▶️ 퀀트 전략 백테스트 실행", type="primary", use_container_width=True)

# ==========================================
# 6. 결과 출력
# ==========================================
if run_btn:
    if (end_date - start_date).days > 59:
        st.error("⚠️ 야후 파이낸스의 5분봉 데이터는 최근 60일까지만 제공됩니다. 날짜 범위를 줄여주세요.")
        st.stop()

    with st.spinner("⏳ 거래소 서버에서 과거 데이터를 불러와 매매 로직을 시뮬레이션 중입니다... (데이터 크기에 따라 3~5초 소요)"):
        base_df = fetch_base_data(ticker, start_date, end_date)
        
    if base_df is None:
        st.error("⚠️ 데이터를 불러오지 못했습니다. 티커나 선택하신 날짜 범위를 다시 확인해 주세요.")
        st.stop()

    df = resample_dataframe(base_df, timeframe)
    df = add_all_indicators(df, p)
    profit, trades, buys, sells, result_df = run_backtest_ultimate(df, p)
    
    total_trades = len(trades)
    win_count = len([t for t in trades if t > 0])
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    avg_profit = np.mean(trades) if total_trades > 0 else 0
    
    st.subheader(f"📊 백테스트 리포트: {ticker} ({tf_choice})")
    st.caption(f"진입 조건: {p['logic_op']} | 매수 타이밍: {p['entry_timing']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최종 수익률", f"{profit:.2f}%")
    c2.metric("총 거래 횟수", f"{total_trades}회")
    c3.metric("승률", f"{win_rate:.1f}%")
    c4.metric("평균 수익률", f"{avg_profit:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
    if buys: bt, bp = zip(*buys); fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
    if sells: st_times, sp = zip(*sells); fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
    fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 🌟 [신규] VIP 텔레그램 시그널 방 비즈니스 모델로 교체
    st.markdown("---")
    st.markdown("### 🔔 백테스트에서 찾은 완벽한 타점, 이제 실시간으로 받아보세요!")
    st.info("💡 하루 종일 차트만 들여다볼 수는 없습니다. 총감독의 VIP 봇이 이 사이트의 필승 로직이 겹치는 정확한 순간, 여러분의 텔레그램으로 실시간 매수/매도 알림을 쏴드립니다.")
    
    # 총감독님의 텔레그램 주소나 결제 링크를 아래에 넣으세요
    st.link_button("🚀 VIP 텔레그램 시그널 방 입장하기 (선착순 마감)", "https://t.me/총감독님_텔레그램_주소", type="primary", use_container_width=True)
