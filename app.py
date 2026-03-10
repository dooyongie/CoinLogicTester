import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import time

# 2026 Next-Gen 설정
st.set_page_config(page_title="Crypto Quant Sandbox", layout="wide")
st.title("🚀 Crypto Quant Sandbox (Global Stability)")
st.markdown("### 📊 전 세계 어디서나 끊김 없는 데이터로 나만의 퀀트 전략을 검증하세요.")

# ==========================================
# 🌟 [엔진 교체] Yahoo Finance 기반 데이터 수집 (IP 차단 해결)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_data_via_yf(symbol, start_date, end_date, interval="5m"):
    # 바이비트 티커(BTCUSDT)를 야후 규격(BTC-USD)으로 변환
    yf_symbol = symbol.replace("USDT", "-USD")
    
    try:
        # 야후 파이낸스는 5분/15분 데이터의 경우 최근 60일까지만 제공하는 제약이 있음
        data = yf.download(
            tickers=yf_symbol,
            start=start_date,
            end=pd.to_datetime(end_date) + pd.Timedelta(days=1),
            interval=interval,
            progress=False
        )
        
        if data.empty:
            return None
            
        df = data.reset_index()
        # 멀티인덱스 컬럼 정리 (yfinance 최신버전 대응)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 컬럼명 표준화
        rename_map = {'Datetime': 'timestamp', 'Date': 'timestamp', 'Open': 'open', 
                      'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=rename_map, inplace=True)
        
        # 타임존 제거 및 필요한 컬럼만 추출
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# ==========================================
# 지표 및 백테스트 로직 (총감독님 지시사항 반영)
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
    df['sma_fast'] = df['close'].rolling(window=p['ma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(window=p['ma_slow']).mean()
    return df

def run_backtest_pro(df, p):
    balance = 10_000 
    position = 0
    entry_price = 0
    entry_time = None
    fee_rate = 0.0004 
    trades, buy_points, sell_points = [], [], []
    
    for i in range(30, len(df)-1):
        prev = df.iloc[i-1]; current = df.iloc[i]; next_candle = df.iloc[i+1]
        
        if position == 0:
            buy_signal = True
            if p['use_ma_cross'] and not (current['sma_fast'] > current['sma_slow'] and prev['sma_fast'] <= prev['sma_slow']): buy_signal = False
            if p['use_bb_touch'] and not (current['low'] <= current['bb_lower']): buy_signal = False
            if p['use_rsi'] and not (current['rsi'] <= p['rsi_buy_limit']): buy_signal = False
            if not any([p['use_ma_cross'], p['use_bb_touch'], p['use_rsi']]): buy_signal = False

            if buy_signal:
                entry_price = next_candle['open']
                entry_time = next_candle['timestamp']
                position = (balance * (1 - fee_rate)) / entry_price 
                buy_points.append((next_candle['timestamp'], entry_price))

        elif position > 0:
            elapsed_mins = (next_candle['timestamp'] - entry_time).total_seconds() / 60
            base_tp = entry_price * (1 + p['tp_pct'] / 100)
            
            # 🌟 [설명 반영] 익절 모드 결정
            if p['exit_mode'] == "🎯 Max 모드":
                target_price = max(current['bb_mid'], base_tp)
            else: # OR 모드 (먼저 닿는 쪽)
                target_price = min(current['bb_mid'], base_tp) 
                
            # 🌟 [설명 반영] 다운시프트 개입
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
st.sidebar.header("🌐 1. 시장 설정 (Global)")
ticker = st.sidebar.selectbox("대상 코인", ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"])
col1, col2 = st.sidebar.columns(2)
# 야후 5분봉 제약(60일)을 고려하여 기본값 설정
start_date = col1.date_input("시작일", datetime.date.today() - datetime.timedelta(days=29))
end_date = col2.date_input("종료일", datetime.date.today())
timeframe = st.sidebar.selectbox("⏱️ 타임프레임", options=['5m', '15m', '30m', '1h', '1d'], index=0)

st.sidebar.markdown("---")
st.sidebar.header("🧩 2. 매수 전략 (Entry)")
p = {}
with st.sidebar.expander("📈 이동평균선 설정"):
    p['use_ma_cross'] = st.checkbox("MA 골든크로스 사용")
    p['ma_fast'] = st.number_input("단기(Fast)", value=10)
    p['ma_slow'] = st.number_input("장기(Slow)", value=20)
with st.sidebar.expander("📉 볼린저 밴드 & RSI", expanded=True):
    p['use_bb_touch'] = st.checkbox("볼밴 하단 터치", value=True)
    p['bb_window'] = st.number_input("BB 기간", value=20)
    p['bb_mult'] = st.number_input("BB 승수", value=2.0, step=0.1)
    p['use_rsi'] = st.checkbox("RSI 과매도 필터", value=True)
    p['rsi_period'] = st.number_input("RSI 기간", value=14)
    p['rsi_buy_limit'] = st.number_input("RSI 진입선", value=30.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 3. 청산 로직 (Exit)")
c8, c9 = st.sidebar.columns(2)
p['tp_pct'] = c8.number_input("✅ 목표 수익 %", value=1.0, step=0.1, format="%.1f")
p['sl_pct'] = c9.number_input("🛑 손절 라인 %", value=2.0, step=0.1, format="%.1f")

# 🌟 [설명 추가] 익절 모드 설명 유저 친화적 배치
st.sidebar.markdown("**익절 모드 선택**")
p['exit_mode'] = st.sidebar.radio(
    "익절 방식", ["🎯 Max 모드", "⚡ OR 모드"],
    help="🎯 Max 모드: 목표 수익률에 도달해도 볼밴 중앙선이 더 높다면 중앙선까지 기다려 수익을 극대화합니다. \n\n⚡ OR 모드: 목표 수익률이나 중앙선 중 먼저 닿는 곳에서 즉시 팔아 자금 회전율을 높입니다."
)

st.sidebar.markdown("---")
# 🌟 [설명 추가] 다운시프트 상세 설명
st.sidebar.markdown("##### ⏳ 특수 로직: 다운시프트(Downshift)")
p['use_downshift'] = st.sidebar.checkbox("다운시프트 활성화", value=True)
if p['use_downshift']:
    st.sidebar.caption("💡 진입 후 가격이 오르지 않고 시간이 지체될 때, 목표가를 낮춰서 본절 부근에서 빠르게 탈출하는 전문 단타 로직입니다.")
    c10, c11 = st.sidebar.columns(2)
    p['downshift_mins'] = c10.number_input("발동 시간(분)", value=60, step=15)
    p['downshift_tp_pct'] = c11.number_input("하향 목표 %", value=0.2, step=0.1, format="%.1f")

run_btn = st.sidebar.button("▶️ 퀀트 시뮬레이션 실행", type="primary", width="stretch")

# ==========================================
# 실행 메인
# ==========================================
if run_btn:
    with st.spinner("야후 파이낸스에서 데이터를 안전하게 긁어오는 중..."):
        df = fetch_data_via_yf(ticker, start_date, end_date, interval=timeframe)
        
    if df is not None:
        df = add_indicators(df, p)
        profit, trades, buys, sells, result_df = run_backtest_pro(df, p)
        
        st.subheader(f"📊 백테스트 결과: {ticker}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 수익률", f"{profit:.2f}%")
        c2.metric("거래 횟수", f"{len(trades)}회")
        win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("평균 수익", f"{np.mean(trades) if trades else 0:.2f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=result_df['timestamp'], open=result_df['open'], high=result_df['high'], low=result_df['low'], close=result_df['close'], name='Price'))
        if buys:
            bt, bp = zip(*buys); fig.add_trace(go.Scatter(x=bt, y=bp, mode='markers', marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        if sells:
            st_times, sp = zip(*sells); fig.add_trace(go.Scatter(x=st_times, y=sp, mode='markers', marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")
        
        # 💰 [레퍼럴 수익화]
        st.markdown("---")
        st.markdown("### 💰 이 전략으로 바이비트에서 자동매매를 시작하세요!")
        st.info("💡 본 사이트의 데이터는 검증용입니다. 실전 매매는 업계 최저 수수료를 제공하는 **바이비트(Bybit)**에서 진행하셔야 수익을 보존할 수 있습니다.")
        st.link_button("⚫ 바이비트(Bybit) 수수료 할인 + 증정금 혜택 가입", "https://www.bybit.com/register?affiliate_id=총감독님_아이디", type="primary", width="stretch")
    else:
        st.error("데이터 로드에 실패했습니다. (5분/15분봉은 최근 60일 이내만 가능합니다)")
