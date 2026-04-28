# pip install streamlit yfinance pandas numpy matplotlib requests plotly

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from turtle_strategy import OriginalTurtleTrading


st.set_page_config(page_title="Turtle Backtest", layout="wide")

st.title("🐢 Original Turtle Trading Backtest")
st.caption("S1: 20봉 돌파 / 10봉 청산 · S2: 55봉 돌파 / 20봉 청산 · N=20EMA(TR)")

# ─────────────────────────────────────────
# 탭: 시장 선택
# ─────────────────────────────────────────
tab_us, tab_kr, tab_crypto = st.tabs(["🇺🇸 미국 주식", "🇰🇷 한국 주식", "₿ 비트코인/코인"])


def render_results(turtle, market_label):
    """백테스트 결과 출력 공용 함수"""
    df = turtle.df
    eq = pd.Series(turtle.equity_curve, index=df.index, dtype=float)
    final_equity = float(eq.iloc[-1])
    total_return = (final_equity / turtle.initial_capital - 1) * 100
    mdd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

    trades_df = pd.DataFrame(turtle.trades)
    exits = trades_df[trades_df['type'] == 'EXIT'] if len(trades_df) else pd.DataFrame()
    entries = trades_df[trades_df['type'] == 'ENTRY'] if len(trades_df) else pd.DataFrame()

    bnh_return = (df.iloc[-1]['Close'] / df.iloc[0]['Close'] - 1) * 100

    # 핵심 지표
    st.subheader(f"📊 {market_label} · {turtle.symbol} · {turtle.interval}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("초기 자본", f"{turtle.initial_capital:,.0f}")
    c2.metric("최종 자산", f"{final_equity:,.0f}", f"{total_return:.2f}%")
    c3.metric("최대낙폭(MDD)", f"{mdd:.2f}%")
    c4.metric("Buy & Hold", f"{bnh_return:.2f}%")

    # 트레이드 통계
    if len(exits) > 0:
        wins = exits[exits['pnl'] > 0]
        losses = exits[exits['pnl'] <= 0]
        win_rate = len(wins) / len(exits) * 100

        total_win = wins['pnl'].sum() if len(wins) else 0
        total_loss = losses['pnl'].sum() if len(losses) else 0
        pf = abs(total_win / total_loss) if total_loss != 0 else float('inf')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 진입", f"{len(entries)}회")
        c2.metric("총 청산", f"{len(exits)}회")
        c3.metric("승률", f"{win_rate:.2f}%")
        c4.metric("Profit Factor", f"{pf:.2f}" if pf != float('inf') else "∞")

        c1, c2 = st.columns(2)
        if len(wins) > 0:
            c1.metric("평균 수익", f"{wins['pnl'].mean():,.0f}")
        if len(losses) > 0:
            c2.metric("평균 손실", f"{losses['pnl'].mean():,.0f}")

    # 자산 곡선 (Plotly)
    st.subheader("📈 자산 곡선")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=eq.values,
        mode='lines', name='Equity',
        line=dict(color='#2E86AB', width=2)
    ))
    fig.add_hline(y=turtle.initial_capital, line_dash="dash",
                  line_color="gray", annotation_text="초기자본")
    fig.update_layout(
        height=400,
        xaxis_title="Date", yaxis_title="Equity",
        hovermode='x unified',
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 가격 + 진입/청산 표시
    st.subheader("💹 가격 차트와 매매 기록")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', name='Close', line=dict(color='#888', width=1)
    ))

    if len(entries) > 0:
        fig2.add_trace(go.Scatter(
            x=entries['date'], y=entries['price'],
            mode='markers', name='Entry',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))
    if len(exits) > 0:
        fig2.add_trace(go.Scatter(
            x=exits['date'], y=exits['price'],
            mode='markers', name='Exit',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))
    fig2.update_layout(
        height=400, hovermode='x unified',
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 청산 사유 / 시스템별 통계
    if len(exits) > 0:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**청산 사유별**")
            by_reason = exits.groupby('reason')['pnl'].agg(['count', 'sum']).round(2)
            st.dataframe(by_reason, use_container_width=True)
        with c2:
            st.write("**시스템별 (S1/S2)**")
            by_system = exits.groupby('system')['pnl'].agg(['count', 'sum']).round(2)
            st.dataframe(by_system, use_container_width=True)

    # 거래내역 테이블
    with st.expander("📋 전체 거래내역 보기"):
        st.dataframe(trades_df, use_container_width=True)


def session_input(market_type, key_prefix):
    """
    market_type: 'us', 'kr', 'crypto'
    """
    if market_type == 'us':
        options = {
            "정규장 (09:30~16:00 ET)": ("regular", None, "America/New_York"),
            "프리+정규+애프터 (04:00~20:00 ET)": ("extended", None, "America/New_York"),
            "프리장만 (04:00~09:30 ET)": ("premarket", None, "America/New_York"),
            "애프터장만 (16:00~20:00 ET)": ("postmarket", None, "America/New_York"),
            "시간 지정 (한국시간 기준)": ("custom_kr", None, "Asia/Seoul"),
            "시간 지정 (미국시간 기준)": ("custom_us", None, "America/New_York"),
        }
    elif market_type == 'kr':
        options = {
            "정규장 (09:00~15:30 KST)": ("custom_kr_regular", ("09:00", "15:30"), "Asia/Seoul"),
            "시간 지정 (한국시간 기준)": ("custom_kr", None, "Asia/Seoul"),
        }
    else:  # crypto
        options = {
            "24시간 전체 (필터 없음)": ("none", None, "UTC"),
            "시간 지정 (한국시간 기준)": ("custom_kr", None, "Asia/Seoul"),
            "시간 지정 (UTC 기준)": ("custom_utc", None, "UTC"),
        }


    choice = st.selectbox(
        "백테스트 시간대",
        list(options.keys()),
        key=f"{key_prefix}_session_choice"
    )
    mode_key, preset, tz = options[choice]

    custom_session = preset
    if mode_key in ("custom_kr", "custom_us", "custom_utc"):
        c1, c2 = st.columns(2)
        start_t = c1.text_input("시작 시간 (HH:MM)", "09:30",
                                key=f"{key_prefix}_start_t")
        end_t = c2.text_input("종료 시간 (HH:MM)", "16:00",
                            key=f"{key_prefix}_end_t")
        custom_session = (start_t, end_t)
        session_mode = "custom"
    elif mode_key == "none":
        # 시간 필터 없음 = regular 모드 (코인은 어차피 session 컬럼이 'crypto_24x7'이라 필터 안 걸림)
        session_mode = "regular"
        custom_session = None
    elif mode_key.startswith("custom_kr_") or mode_key == "custom_kr_24h":
        session_mode = "custom"
    else:
        session_mode = mode_key


    return session_mode, custom_session, tz


def common_inputs(key_prefix, default_capital=100_000, default_fee=0.0005):
    """공통 백테스트 파라미터 입력"""
    c1, c2 = st.columns(2)
    start_date = c1.date_input(
        "시작일", value=date.today() - timedelta(days=365),
        key=f"{key_prefix}_start"
    )
    end_date = c2.date_input(
        "종료일", value=date.today(),
        key=f"{key_prefix}_end"
    )

    c1, c2, c3 = st.columns(3)
    interval = c1.selectbox(
        "봉 단위", ["1d", "1h", "30m", "15m", "5m", "1wk"],
        index=0, key=f"{key_prefix}_interval"
    )
    initial_capital = c2.number_input(
        "초기 자본", min_value=1000.0, value=float(default_capital),
        step=1000.0, key=f"{key_prefix}_capital"
    )
    risk_per_unit = c3.number_input(
        "유닛당 리스크 (%)", min_value=0.1, max_value=10.0, value=1.0,
        step=0.1, key=f"{key_prefix}_risk"
    ) / 100

    c1, c2, c3 = st.columns(3)
    max_units = c1.number_input(
        "최대 유닛 수", min_value=1, max_value=10, value=4,
        key=f"{key_prefix}_units"
    )
    fee_rate = c2.number_input(
        "수수료 (%)", min_value=0.0, max_value=1.0,
        value=default_fee * 100, step=0.01,
        key=f"{key_prefix}_fee"
    ) / 100
    slippage_rate = c3.number_input(
        "슬리피지 (%)", min_value=0.0, max_value=1.0, value=0.05,
        step=0.01, key=f"{key_prefix}_slip"
    ) / 100

    allow_short = st.checkbox("공매도 허용", value=False,
                              key=f"{key_prefix}_short")

    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "interval": interval,
        "initial_capital": initial_capital,
        "risk_per_unit": risk_per_unit,
        "max_units": max_units,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "allow_short": allow_short,
    }


def run_and_show(symbol, source, params, session_mode, custom_session, tz, market_label):
    """실행 + 결과 표시"""
    try:
        with st.spinner(f"{symbol} 데이터 로드 및 백테스트 진행 중..."):
            turtle = OriginalTurtleTrading(
                symbol=symbol,
                source=source,
                session_mode=session_mode,
                custom_session=custom_session,
                exchange_tz=tz,
                **params,
            )
            turtle.fetch_data().calculate_indicators().run_backtest()

        render_results(turtle, market_label)
        st.success("✅ 백테스트 완료")
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
        st.exception(e)


# ─────────────────────────────────────────
# 미국 주식 탭
# ─────────────────────────────────────────
with tab_us:
    st.markdown("### 미국 주식/ETF 백테스트")
    symbol = st.text_input("티커", value="AAPL", key="us_symbol",
                           help="예: AAPL, TSLA, NVDA, SPY, QQQ")

    params = common_inputs("us", default_capital=100_000, default_fee=0.0005)
    session_mode, custom_session, tz = session_input("us", "us")

    if st.button("🚀 백테스트 실행", key="us_run", type="primary"):
        run_and_show(symbol.strip().upper(), "yfinance", params,
                     session_mode, custom_session, tz, "🇺🇸 미국 주식")


# ─────────────────────────────────────────
# 한국 주식 탭
# ─────────────────────────────────────────
with tab_kr:
    st.markdown("### 한국 주식 백테스트")
    st.info("티커 형식: 코스피는 `.KS`, 코스닥은 `.KQ` (예: 삼성전자 `005930.KS`)")

    symbol = st.text_input("티커", value="005930.KS", key="kr_symbol",
                           help="예: 005930.KS (삼성전자), 035720.KS (카카오), 247540.KQ (에코프로비엠)")

    params = common_inputs("kr", default_capital=100_000_000, default_fee=0.0015)
    session_mode, custom_session, tz = session_input("kr", "kr")

    if st.button("🚀 백테스트 실행", key="kr_run", type="primary"):
        run_and_show(symbol.strip(), "yfinance", params,
                     session_mode, custom_session, tz, "🇰🇷 한국 주식")


# ─────────────────────────────────────────
# 코인 탭
# ─────────────────────────────────────────
with tab_crypto:
    st.markdown("### 코인 백테스트 (Binance)")
    st.info("심볼 형식: `BTCUSDT`, `ETHUSDT` 등 (USDT 페어)")

    symbol = st.text_input("심볼", value="BTCUSDT", key="cr_symbol",
                           help="예: BTCUSDT, ETHUSDT, SOLUSDT")

    params = common_inputs("cr", default_capital=10_000, default_fee=0.001)
    session_mode, custom_session, tz = session_input("crypto", "cr")

    if st.button("🚀 백테스트 실행", key="cr_run", type="primary"):
        run_and_show(symbol.strip().upper(), "binance", params,
                     session_mode, custom_session, tz, "₿ 코인")


# ─────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ 본 백테스트 결과는 과거 데이터 기반 시뮬레이션이며 실제 투자 성과를 보장하지 않습니다.")
