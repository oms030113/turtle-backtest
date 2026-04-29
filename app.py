import io
import contextlib
from datetime import date, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from turtle_engine import OriginalTurtleTrading


# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Original Turtle Trading Backtest",
    page_icon="🐢",
    layout="wide",
)


# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #7b7b7b;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.5rem;
        margin-bottom: 1.0rem;
    }
    .metric-card {
        background-color: #f7f8fa;
        padding: 1.0rem;
        border-radius: 0.8rem;
        border: 1px solid #eeeeee;
    }
    .small-help {
        color: #808080;
        font-size: 0.8rem;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.55rem 1.1rem;
        font-weight: 700;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# Utility
# ─────────────────────────────────────────
def parse_money(value, default=100000.0):
    """
    '100,000', '100000', '100,000.50' 입력을 float으로 변환.
    """
    try:
        if value is None:
            return float(default)
        value = str(value).replace(",", "").strip()
        if value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def format_money(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "-"


def format_pct(value, decimals=2):
    try:
        return f"{float(value):,.{decimals}f}%"
    except Exception:
        return "-"


def capture_print(func):
    """
    print_results() 같은 콘솔 출력 캡처용.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func()
    return buf.getvalue()


def get_summary(turtle):
    """
    Streamlit metric 출력용 요약 계산.
    """
    if turtle.df is None or len(turtle.df) == 0 or not turtle.equity_curve:
        return {}

    final_equity = float(turtle.equity_curve[-1])
    total_return = (final_equity / turtle.initial_capital - 1.0) * 100.0

    eq = pd.Series(turtle.equity_curve, index=turtle.df.index, dtype=float)
    mdd = ((eq - eq.cummax()) / eq.cummax() * 100.0).min()

    bnh = None
    if hasattr(turtle, "_calculate_buy_and_hold"):
        try:
            bnh = turtle._calculate_buy_and_hold()
        except Exception:
            bnh = None

    trades_df = turtle.get_trades_df()

    if len(trades_df) > 0 and "type" in trades_df.columns:
        entries = trades_df[trades_df["type"] == "ENTRY"]
        adds = trades_df[trades_df["type"] == "ADD"]
        exits = trades_df[trades_df["type"] == "EXIT"]
    else:
        entries = pd.DataFrame()
        adds = pd.DataFrame()
        exits = pd.DataFrame()

    win_rate = np.nan
    profit_factor = np.nan
    total_realized_pnl = np.nan

    if len(exits) > 0 and "pnl" in exits.columns:
        wins = exits[exits["pnl"] > 0]
        losses = exits[exits["pnl"] <= 0]

        win_rate = len(wins) / max(len(exits), 1) * 100.0

        total_win = wins["pnl"].sum() if len(wins) else 0.0
        total_loss = losses["pnl"].sum() if len(losses) else 0.0

        if total_loss != 0:
            profit_factor = abs(total_win / total_loss)

        total_realized_pnl = exits["pnl"].sum()

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "mdd": mdd,
        "bnh": bnh,
        "entries_count": len(entries),
        "adds_count": len(adds),
        "exits_count": len(exits),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_realized_pnl": total_realized_pnl,
        "trades_df": trades_df,
    }


def plot_equity_matplotlib(turtle):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(turtle.df.index, turtle.df["equity"], linewidth=1.3, label="Strategy Equity")
    ax.axhline(turtle.initial_capital, linestyle="--", alpha=0.5, color="gray", label="Initial Capital")
    ax.set_title(f"Equity Curve - {turtle.symbol}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_price_matplotlib(turtle):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(turtle.df.index, turtle.df["Close"], linewidth=1.1, label="Close")

    trades = turtle.get_trades_df()

    if len(trades) > 0 and "type" in trades.columns:
        entries = trades[trades["type"] == "ENTRY"]
        adds = trades[trades["type"] == "ADD"]
        exits = trades[trades["type"] == "EXIT"]

        if len(entries) > 0:
            ax.scatter(entries["date"], entries["price"], marker="^", s=70, label="ENTRY", zorder=5)

        if len(adds) > 0:
            ax.scatter(adds["date"], adds["price"], marker="o", s=45, label="ADD", zorder=5)

        if len(exits) > 0:
            ax.scatter(exits["date"], exits["price"], marker="v", s=70, label="EXIT", zorder=5)

    ax.set_title(f"Price & Trades - {turtle.symbol}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def render_summary(turtle):
    summary = get_summary(turtle)

    if not summary:
        st.warning("결과 요약을 계산할 수 없습니다.")
        return

    bnh = summary["bnh"]

    st.markdown("### 📊 결과 요약")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("초기 자본", format_money(turtle.initial_capital, 0))
    c2.metric("전략 최종자산", format_money(summary["final_equity"], 0))
    c3.metric("전략 수익률", format_pct(summary["total_return"]))
    c4.metric("전략 MDD", format_pct(summary["mdd"]))

    if bnh is not None:
        excess_return = summary["total_return"] - bnh["return_pct"]
        excess_value = summary["final_equity"] - bnh["final_value"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("B&H 최종자산", format_money(bnh["final_value"], 0))
        c2.metric("B&H 수익률", format_pct(bnh["return_pct"]))
        c3.metric("초과수익률", f"{excess_return:,.2f}%p")
        c4.metric("초과금액", format_money(excess_value, 0))

        with st.expander("Buy & Hold 기준 확인", expanded=False):
            st.write(f"**시작:** {bnh['start_time']} / Open = {bnh['start_price']:,.4f}")
            st.write(f"**종료:** {bnh['end_time']} / Close = {bnh['end_price']:,.4f}")
    else:
        st.info("Buy & Hold 계산 불가")

    st.markdown("### 🧾 거래 통계")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("신규 진입", f"{summary['entries_count']:,}회")
    c2.metric("추가 매수", f"{summary['adds_count']:,}회")
    c3.metric("청산", f"{summary['exits_count']:,}회")

    if pd.notna(summary["win_rate"]):
        c4.metric("승률", format_pct(summary["win_rate"]))
    else:
        c4.metric("승률", "-")

    c1, c2, c3, c4 = st.columns(4)

    if pd.notna(summary["profit_factor"]):
        c1.metric("Profit Factor", f"{summary['profit_factor']:,.2f}")
    else:
        c1.metric("Profit Factor", "-")

    if pd.notna(summary["total_realized_pnl"]):
        c2.metric("총 실현 PnL", format_money(summary["total_realized_pnl"], 0))
    else:
        c2.metric("총 실현 PnL", "-")

    if turtle.position_side != 0:
        c3.metric("미청산 포지션", "LONG" if turtle.position_side == 1 else "SHORT")
        c4.metric("보유 유닛", f"{len(turtle.units)}")
    else:
        c3.metric("미청산 포지션", "없음")
        c4.metric("보유 유닛", "0")


def run_backtest_with_config(config):
    turtle = OriginalTurtleTrading(**config)

    with st.spinner("데이터 로드 및 백테스트 실행 중..."):
        turtle.fetch_data().calculate_indicators().run_backtest()

    return turtle


def common_inputs(
    market_type,
    default_symbol,
    default_capital,
    default_fee,
    default_slippage,
    default_interval,
    interval_options,
):
    """
    공통 입력 UI.
    """
    symbol = st.text_input("티커", value=default_symbol)

    c1, c2 = st.columns(2)
    start_date = c1.date_input("시작일", value=date.today() - timedelta(days=365))
    end_date = c2.date_input("종료일", value=date.today())

    c1, c2, c3 = st.columns(3)

    interval = c1.selectbox(
        "봉 단위",
        options=interval_options,
        index=interval_options.index(default_interval) if default_interval in interval_options else 0,
    )

    initial_capital_str = c2.text_input(
        "초기 자본",
        value=f"{default_capital:,}",
        help="예: 100,000 또는 100000",
    )

    risk_per_unit = c3.number_input(
        "유닛당 리스크 (%)",
        min_value=0.01,
        max_value=100.0,
        value=1.00,
        step=0.10,
        format="%.2f",
    )

    c1, c2, c3 = st.columns(3)

    max_units = c1.number_input(
        "최대 유닛 수",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
    )

    fee_rate = c2.number_input(
        "수수료 (%)",
        min_value=0.0,
        max_value=10.0,
        value=default_fee,
        step=0.01,
        format="%.3f",
    )

    slippage_rate = c3.number_input(
        "슬리피지 (%)",
        min_value=0.0,
        max_value=10.0,
        value=default_slippage,
        step=0.01,
        format="%.3f",
    )

    initial_capital = parse_money(initial_capital_str, default=default_capital)

    return {
        "symbol": symbol.strip(),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "interval": interval,
        "initial_capital": initial_capital,
        "risk_per_unit": risk_per_unit / 100.0,
        "max_units": int(max_units),
        "fee_rate": fee_rate / 100.0,
        "slippage_rate": slippage_rate / 100.0,
    }


def render_result_tabs(turtle):
    render_summary(turtle)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity", "📉 Price & Trades", "🧾 Trades", "🛠 Debug Logs"])

    with tab1:
        st.pyplot(plot_equity_matplotlib(turtle), use_container_width=True)

    with tab2:
        st.pyplot(plot_price_matplotlib(turtle), use_container_width=True)

    with tab3:
        trades = turtle.get_trades_df()
        if len(trades) > 0:
            st.dataframe(trades, use_container_width=True)
            st.download_button(
                "거래내역 CSV 다운로드",
                data=trades.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{turtle.symbol}_trades.csv",
                mime="text/csv",
            )
        else:
            st.info("거래내역이 없습니다.")

    with tab4:
        logs = turtle.get_debug_log_df()
        if len(logs) > 0:
            st.dataframe(logs, use_container_width=True)
            st.download_button(
                "디버그 로그 CSV 다운로드",
                data=logs.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{turtle.symbol}_debug_logs.csv",
                mime="text/csv",
            )
        else:
            st.info("디버그 로그가 없습니다.")


# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.markdown('<div class="main-title">🐢 Original Turtle Trading Backtest</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">S1: 20봉 돌파 / 10봉 청산 · S2: 55봉 돌파 / 20봉 청산 · N=20EMA(TR) · 종가 확정 후 다음 봉 시가 체결</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────
tab_us, tab_kr, tab_crypto = st.tabs(["🇺🇸 미국 주식/ETF", "🇰🇷 한국 주식", "₿ 비트코인/코인"])


# ─────────────────────────────────────────
# US Stock Tab
# ─────────────────────────────────────────
with tab_us:
    st.markdown('<div class="section-title">미국 주식/ETF 백테스트</div>', unsafe_allow_html=True)

    inputs = common_inputs(
        market_type="us",
        default_symbol="AAPL",
        default_capital=100000,
        default_fee=0.05,
        default_slippage=0.05,
        default_interval="1d",
        interval_options=["1d", "1h", "30m", "15m", "5m"],
    )

    st.markdown("#### 백테스트 시간대")

    session_label = st.selectbox(
        "세션",
        options=[
            "정규장 (09:30~16:00 ET)",
            "프리+정규+애프터",
            "프리마켓",
            "애프터마켓",
            "커스텀 시간",
        ],
        index=0,
    )

    session_mode = "regular"
    custom_session = None
    custom_execution_control = True
    custom_data_session = "regular"

    if session_label == "정규장 (09:30~16:00 ET)":
        session_mode = "regular"
        custom_execution_control = False

    elif session_label == "프리+정규+애프터":
        session_mode = "extended"
        custom_execution_control = False

    elif session_label == "프리마켓":
        session_mode = "premarket"
        custom_execution_control = False

    elif session_label == "애프터마켓":
        session_mode = "postmarket"
        custom_execution_control = False

    elif session_label == "커스텀 시간":
        session_mode = "custom"
        c1, c2 = st.columns(2)
        custom_start = c1.text_input("커스텀 시작 시간 ET", value="09:30")
        custom_end = c2.text_input("커스텀 종료 시간 ET", value="12:30")
        custom_session = (custom_start, custom_end)

        custom_data_session = st.selectbox(
            "지표 계산용 데이터",
            options=["regular", "extended", "raw"],
            index=0,
            help="보통 regular 추천. custom 시간은 매매 가능 시간 제어로만 사용합니다.",
        )

        custom_execution_control = True

    config = {
        "symbol": inputs["symbol"],
        "start_date": inputs["start_date"],
        "end_date": inputs["end_date"],
        "interval": inputs["interval"],
        "source": "yfinance",
        "initial_capital": inputs["initial_capital"],
        "risk_per_unit": inputs["risk_per_unit"],
        "max_units": inputs["max_units"],
        "fee_rate": inputs["fee_rate"],
        "slippage_rate": inputs["slippage_rate"],
        "allow_short": False,
        "session_mode": session_mode,
        "custom_session": custom_session,
        "exchange_tz": "America/New_York",
        "custom_execution_control": custom_execution_control,
        "custom_data_session": custom_data_session,
    }

    if st.button("🚀 미국 주식 백테스트 실행", key="run_us"):
        try:
            turtle = run_backtest_with_config(config)
            st.session_state["last_turtle_us"] = turtle
            st.success("백테스트 완료")
            render_result_tabs(turtle)
        except Exception as e:
            st.error(f"오류 발생: {e}")

    if "last_turtle_us" in st.session_state:
        with st.expander("이전 미국 주식 결과 다시 보기", expanded=False):
            render_result_tabs(st.session_state["last_turtle_us"])


# ─────────────────────────────────────────
# Korean Stock Tab
# ─────────────────────────────────────────
with tab_kr:
    st.markdown('<div class="section-title">한국 주식 백테스트</div>', unsafe_allow_html=True)

    inputs = common_inputs(
        market_type="kr",
        default_symbol="005930.KS",
        default_capital=100000000,
        default_fee=0.15,
        default_slippage=0.05,
        default_interval="1d",
        interval_options=["1d", "1h", "30m", "15m", "5m"],
    )

    st.markdown("#### 한국장 시간")

    use_custom_kr_session = st.checkbox(
        "한국장 시간 필터 사용",
        value=True,
        help="보통 09:00~15:30 사용",
    )

    if use_custom_kr_session:
        c1, c2 = st.columns(2)
        kr_start = c1.text_input("장 시작 시간", value="09:00")
        kr_end = c2.text_input("장 종료 시간", value="15:30")

        session_mode = "custom"
        custom_session = (kr_start, kr_end)
        custom_data_session = "custom_filter"
        custom_execution_control = False
    else:
        session_mode = "regular"
        custom_session = None
        custom_data_session = "regular"
        custom_execution_control = False

    st.info("한국 주식 탭에서는 공매도 기능을 사용하지 않습니다.")

    config = {
        "symbol": inputs["symbol"],
        "start_date": inputs["start_date"],
        "end_date": inputs["end_date"],
        "interval": inputs["interval"],
        "source": "yfinance",
        "initial_capital": inputs["initial_capital"],
        "risk_per_unit": inputs["risk_per_unit"],
        "max_units": inputs["max_units"],
        "fee_rate": inputs["fee_rate"],
        "slippage_rate": inputs["slippage_rate"],
        "allow_short": False,
        "session_mode": session_mode,
        "custom_session": custom_session,
        "exchange_tz": "Asia/Seoul",
        "custom_execution_control": custom_execution_control,
        "custom_data_session": custom_data_session,
    }

    if st.button("🚀 한국 주식 백테스트 실행", key="run_kr"):
        try:
            turtle = run_backtest_with_config(config)
            st.session_state["last_turtle_kr"] = turtle
            st.success("백테스트 완료")
            render_result_tabs(turtle)
        except Exception as e:
            st.error(f"오류 발생: {e}")

    if "last_turtle_kr" in st.session_state:
        with st.expander("이전 한국 주식 결과 다시 보기", expanded=False):
            render_result_tabs(st.session_state["last_turtle_kr"])


# ─────────────────────────────────────────
# Crypto Tab
# ─────────────────────────────────────────
with tab_crypto:
    st.markdown('<div class="section-title">비트코인/코인 백테스트</div>', unsafe_allow_html=True)

    inputs = common_inputs(
        market_type="crypto",
        default_symbol="BTCUSDT",
        default_capital=100000,
        default_fee=0.05,
        default_slippage=0.05,
        default_interval="1d",
        interval_options=["1d", "4h", "1h", "30m", "15m", "5m", "3m", "1m"],
    )

    c1, c2 = st.columns(2)

    allow_short = c1.checkbox(
        "숏 허용",
        value=True,
        help="코인 탭에서만 숏 포지션 기능을 제공합니다.",
    )

    crypto_session_mode = c2.selectbox(
        "세션",
        options=["24시간 전체", "커스텀 시간"],
        index=0,
    )

    session_mode = "regular"
    custom_session = None
    custom_execution_control = False
    custom_data_session = "raw"

    if crypto_session_mode == "커스텀 시간":
        session_mode = "custom"
        c1, c2 = st.columns(2)
        custom_start = c1.text_input("커스텀 시작 시간", value="00:00", key="crypto_custom_start")
        custom_end = c2.text_input("커스텀 종료 시간", value="23:59", key="crypto_custom_end")
        custom_session = (custom_start, custom_end)
        custom_execution_control = False
        custom_data_session = "custom_filter"

    st.info("코인은 Binance spot API 기준입니다. 예: BTCUSDT, ETHUSDT, SOLUSDT")

    config = {
        "symbol": inputs["symbol"].upper(),
        "start_date": inputs["start_date"],
        "end_date": inputs["end_date"],
        "interval": inputs["interval"],
        "source": "binance",
        "initial_capital": inputs["initial_capital"],
        "risk_per_unit": inputs["risk_per_unit"],
        "max_units": inputs["max_units"],
        "fee_rate": inputs["fee_rate"],
        "slippage_rate": inputs["slippage_rate"],
        "allow_short": allow_short,
        "session_mode": session_mode,
        "custom_session": custom_session,
        "exchange_tz": "UTC",
        "custom_execution_control": custom_execution_control,
        "custom_data_session": custom_data_session,
    }

    if st.button("🚀 코인 백테스트 실행", key="run_crypto"):
        try:
            turtle = run_backtest_with_config(config)
            st.session_state["last_turtle_crypto"] = turtle
            st.success("백테스트 완료")
            render_result_tabs(turtle)
        except Exception as e:
            st.error(f"오류 발생: {e}")

    if "last_turtle_crypto" in st.session_state:
        with st.expander("이전 코인 결과 다시 보기", expanded=False):
            render_result_tabs(st.session_state["last_turtle_crypto"])
