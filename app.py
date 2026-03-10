import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import datetime

st.set_page_config(page_title="Crypto Quant Sandbox - Bybit", layout="wide")
st.title("🚀 Crypto Quant Sandbox (Bybit Edition)")
st.markdown("### 📊 Bybit 실시간 데이터를 활용하여 나만의 퀀트 전략을 검증해 보세요.")

# ==========================================
# 🌟 [수정] Bybit 데이터 수집 엔진 (V5 API)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_bybit_data(symbol, start_date, end_date, interval="5"):
    url = "https://api.bybit.com/v5/market/kline"
    # Bybit는 밀리초(ms) 단위 timestamp 사용
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)).timestamp() * 1000)

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_ts,
        "end": end_ts,
        "limit": 1000
    }
    
    try:
        res = requests.get(url, params=params)
        data = res.json()
        
        if data['retCode'] != 0:
            st.error(f"Bybit API Error: {data['retMsg']}")
            return None
            
        klines = data['result']['list']
        if not klines:
            return None
            
        # Bybit는 최신순(Reverse)으로 데이터를 주므로 시간순으로 뒤집어야 함
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df = df.iloc[::-1] 
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"연결 오류: {e}")
        return None

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
    df['bb_upper'] = df['bb_mid'] + (p['bb_mult'] * df['bb_std'])
    
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    
    ndays_high = df['high'].rolling(window=p['stoch_k_len']).max()
    ndays_low = df['low'].rolling(window=p['stoch_k_len']).min()
    fast_k = 100 * (df['close'] - ndays_low) / (ndays_high - ndays_low)
    df['stoch_k'] = fast_k.rolling(window=p['stoch_k_smooth']).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=p['stoch_d_smooth']).mean()
    
    return df

# ==========================================
# 백테스트 엔진
# ==========================================
def run_backtest_pro(df, p):
    balance = 10_000 
    position = 0
    entry_price = 0
    entry_time = None
    fee_rate = 0.0004 
    trades, buy_points, sell_points = [], [], []
    
    for i in range(50, len(df)-1):
        prev = df.iloc[i-1]    
        current = df.iloc[i]   
        next_candle = df.iloc[i+1] 
        
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

        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            
            if p['exit_mode'] == "🎯 Max 모드":
                target_price = max(current['bb_mid'], base_tp)
            else:
                target_price = min(current['bb_mid'], base_tp) 
                
            if p['use_downshift'] and elapsed_mins >= p['downshift_mins']:
                target_price = min(target_price, entry_price * (1 + p['downshift_tp_pct'] / 100))
                
            sl_price = entry_price * (1 - p['sl_pct'] / 100)
            sell_price = 0
            
            if next_candle['low'] <= sl_price: sell_price = min(next_candle['open'], sl_price)
            elif next_candle['high'] >= target_price: sell_price = max(next_candle['open'], target_price)
            
            if sell_price > 0:
                balance = position * sell_price * (1 - fee_rate)
                trades.append((sell_price / entry_price * (1 - fee_rate * 2) - 1) * 100)
                sell_points.append((next_candle['timestamp'], sell_price))
                position = 0

    return (balance / 10_000 - 1) * 100, trades, buy_points, sell_points, df

# ==========================================
# UI 셋업
# ==========================================
if st.sidebar.button("🔄 데이터 캐시 초기화", width='stretch'): st.cache_data.clear()

st.sidebar.header("🌐 1. 데이터 설정 (Bybit)")
ticker = st.sidebar.selectbox("코인 페어", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=30))
end_date = col2.date_input("종료일", datetime.date.today())

# 🌟 [수정] Bybit 타임프레임 규격 (5, 15, 60 등 숫자형태)
tf_display = st.sidebar.select_slider("⏱️ 캔들 타임프레임", options=['5분', '15분', '30분', '1시간', '4시간'], value='5분')
tf_map = {'5분': '5', '15분': '15', '30분': '30', '1시간': '60', '4시간': '240'}
selected_tf = tf_map[tf_display]

st.sidebar.markdown("---")
st.sidebar.header("🧩 2. 매수 전략 설정")
p = {}

with st.sidebar.expander("📈 이동평균선 설정"):
    p['use_ma_cross'] = st.checkbox("MA 골든크로스")
    p['ma_fast'] = st.number_input("단기 이평", value=10)
    p['ma_slow'] = st.number_input("장기 이평", value=20)

with st.sidebar.expander("🌊 스토캐스틱 설정"):
    p['use_stoch_cross'] = st.checkbox("Stoch 골든크로스")
    p['stoch_k_len'] = st.number_input("K 길이", value=5)
    p['stoch_k_smooth'] = st.number_input("K 스무딩", value=3)
    p['stoch_d_smooth'] = st.number_input("D 스무딩", value=3)

with st.sidebar.expander("📉 볼린저 밴드 & RSI", expanded=True):
    p['use_bb_touch'] = st.checkbox("볼밴 하단 터치", value=True)
    p['bb_window'] = st.number_input("BB 기간", value=20)
    p['bb_mult'] = st.number_input("BB 승수", value=2.0)
    p['use_rsi'] = st.checkbox("RSI 필터", value=True)
    p['rsi_period'] = st.number_input("RSI 기간", value=14)
    p['rsi_buy_limit'] = st.number_input("RSI 진입선", value=30.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 3. 청산 로직 설정")
p['tp_pct'] = st.sidebar.number_input("✅ 목표가 %", value=1.0, step=0.1)
p['sl_pct'] = st.sidebar.number_input("🛑 손절가 %", value=2.0, step=0.1)
p['exit_mode'] = st.sidebar.radio("익절 모드", ["🎯 Max 모드", "⚡ OR 모드"])

st.sidebar.markdown("##### ⏳ 다운시프트 (Downshift)")
p['use_downshift'] = st.sidebar.checkbox("다운시프트 활성화", value=True)
if p['use_downshift']:
    p['downshift_mins'] = st.sidebar.number_input("발동 시간 (분)", value=60)
    p['downshift_tp_pct'] = st.sidebar.number_input("하향 목표가 %", value=0.2)

run_btn = st.sidebar.button("▶️ 전략 백테스트 실행", type="primary", width='stretch')

# ==========================================
# 실행
# ==========================================
if run_btn:
    with st.spinner(f"Bybit 서버에서 {ticker} 데이터를 로드 중입니다..."):
        df = fetch_bybit_data(ticker, start_date, end_date, interval=selected_tf)
        
    if df is not None:
        df = add_indicators(df, p)
        profit, trades, buys, sells, result_df = run_backtest_pro(df, p)
        
        st.subheader(f"📊 백테스트 리포트: {ticker} ({tf_display})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 수익률", f"{profit:.2f}%")
        c2.metric("거래 횟수", f"{len(trades)}회")
        win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("평균 수익", f"{np.mean(trades) if trades else 0:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        
        if buys:
            bt, bp = zip(*buys)
            fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        if sells:
            st_times, sp = zip(*sells)
            fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        
        st.plotly_chart(fig, width='stretch')
        
        # 🌟 [수정] 바이비트 레퍼럴 버튼 강조
        st.markdown("---")
        st.markdown("### 💰 이 전략을 실전 봇으로 가동하고 싶다면?")
        st.info("실전 매매는 **수수료 싸움**입니다. 아래 버튼을 통해 바이비트 VIP 계정을 활성화하고 증정금 혜택과 수수료 할인을 동시에 받으세요.")
        
        st.link_button(
            "⚫ 바이비트(Bybit) 수수료 할인 및 최대 증정금 가입하기", 
            "https://www.bybit.com/register?affiliate_id=총감독님_아이디", 
            type="primary", 
            width='stretch'
        )
