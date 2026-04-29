# turtle_engine.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from datetime import timedelta
import yfinance as yf


class OriginalTurtleTrading:
    """
    Close-confirmed Turtle Trading Backtester

    핵심 규칙:
    - S1: 이전 20봉 고점 돌파 진입 / 이전 10봉 저점 trailing stop
    - S2: 이전 55봉 고점 돌파 진입 / 이전 20봉 저점 trailing stop
    - S1과 S2가 동시에 발생하면 S2 우선
    - 진입 신호: 종가가 돌파선 위에서 마감 + MA200 위에서 마감
    - 실제 진입: 다음 봉 시가
    - ATR/N: 신호 발생 봉 기준으로 고정
    - 최초 손절선: 실제 진입가 - 2N
    - 추가매수: 최초 진입가 +0.5N, +1.0N, +1.5N
    - 추가매수 신호: 종가가 add level 위에서 마감
    - 실제 추가매수: 다음 봉 시가
    - 손절선 업데이트: exit channel low가 기존 stop보다 위로 올라오면 stop 상향
    - 손절 신호: 종가가 stop 아래에서 마감
    - 실제 청산: 다음 봉 시가
    - 포지션 보유 중에는 신규 진입 신호 무시

    Custom trading window:
    - 미국 주식 custom 사용 시 exchange_tz='America/New_York' 권장
    - custom_session=('09:30','12:30')이면 미국 현지시간 기준
    - DST는 America/New_York 타임존이 자동 처리
    - custom window 밖에서 발생한 신호는 다음날로 이월하지 않음
    - custom 종료 봉 마감 후 다음 봉 시가에 전량 강제청산
    - custom 종료 시각에는 신규진입/추가매수 금지, 청산만 허용
    """

    def __init__(
        self,
        symbol,
        start_date,
        end_date,
        interval="1d",
        source="auto",                    # 'auto' | 'yfinance' | 'binance'
        initial_capital=100_000.0,
        risk_per_unit=0.01,
        max_units=4,
        fee_rate=0.0005,
        slippage_rate=0.0005,
        allow_short=False,
        session_mode="regular",           # 'regular'|'extended'|'premarket'|'postmarket'|'custom'
        custom_session=None,              # ('HH:MM','HH:MM')
        exchange_tz="America/New_York",
        use_current_ma=True,

        # custom window를 데이터 필터가 아니라 매매 가능 시간 제어로 사용할지
        custom_execution_control=True,

        # custom_execution_control=True일 때 지표 계산용 데이터 범위
        # 미국 주식이면 보통 'regular' 추천
        # 'regular' | 'extended' | 'raw' | 'custom_filter'
        custom_data_session="regular",
    ):
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.interval = interval
        self.source = source

        self.initial_capital = float(initial_capital)
        self.risk_per_unit = float(risk_per_unit)
        self.max_units = int(max_units)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
        self.allow_short = bool(allow_short)

        self.session_mode = str(session_mode).lower()
        self.custom_session = custom_session
        self.exchange_tz = exchange_tz
        self.use_current_ma = bool(use_current_ma)

        self.custom_execution_control = bool(custom_execution_control)
        self.custom_data_session = str(custom_data_session).lower()

        self.resolved_source = None
        self.raw_df = None
        self.df = None

        valid_modes = ["regular", "extended", "premarket", "postmarket", "custom"]
        if self.session_mode not in valid_modes:
            raise ValueError(f"session_mode는 {valid_modes} 중 하나여야 합니다.")

        valid_custom_data_sessions = ["regular", "extended", "raw", "custom_filter"]
        if self.custom_data_session not in valid_custom_data_sessions:
            raise ValueError(
                f"custom_data_session은 {valid_custom_data_sessions} 중 하나여야 합니다."
            )

        if self.session_mode == "custom":
            if not isinstance(custom_session, (tuple, list)) or len(custom_session) != 2:
                raise ValueError(
                    "session_mode='custom'일 때는 custom_session=('HH:MM','HH:MM') 형식이어야 합니다."
                )

            try:
                for t in custom_session:
                    hh, mm = t.split(":")
                    int(hh)
                    int(mm)
            except Exception:
                raise ValueError("custom_session 시간 형식이 잘못되었습니다. 예: ('09:30','12:30')")

        # regular/premarket/postmarket/extended는 미국장 기준이므로 NY로 고정
        # custom은 사용자가 지정한 exchange_tz를 유지함.
        # 다만 미국 주식 custom은 exchange_tz='America/New_York' 권장.
        if self.session_mode in ["regular", "premarket", "postmarket", "extended"]:
            if self.exchange_tz != "America/New_York":
                print(
                    f"  [알림] session_mode='{self.session_mode}'는 미국시간 기준입니다. "
                    f"exchange_tz를 'America/New_York'로 자동 설정합니다."
                )
                self.exchange_tz = "America/New_York"

        self._reset_backtest_state()

    # ─────────────────────────────────────────
    # State / utility
    # ─────────────────────────────────────────
    def _reset_backtest_state(self):
        self.cash = self.initial_capital

        self.units = []
        self.position_side = 0
        self.current_system = None
        self.current_unit_qty = 0.0
        self.entry_N = np.nan

        self.base_entry_price = np.nan
        self.initial_stop_price = np.nan
        self.stop_price = np.nan

        self.add_levels = []
        self.next_add_index = 0

        self.pending_entry = None
        self.pending_add_count = 0
        self.pending_exit = None

        self.trades = []
        self.debug_logs = []
        self.equity_curve = []

    def _log(self, date, event, reason, **kwargs):
        row = {"date": date, "event": event, "reason": reason}
        row.update(kwargs)
        self.debug_logs.append(row)

    def get_debug_log_df(self):
        return pd.DataFrame(self.debug_logs)

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

    def print_debug_logs(self, last_n=50):
        if not self.debug_logs:
            print("디버깅 로그가 없습니다.")
            return

        df = pd.DataFrame(self.debug_logs)
        print(df.tail(last_n).to_string(index=False))

    def _net_position_qty(self):
        if not self.units:
            return 0.0
        return float(sum(u["qty"] for u in self.units))

    def _equity(self, mark_price):
        return float(self.cash + self._net_position_qty() * float(mark_price))

    def _buy_fill(self, price):
        return float(price) * (1.0 + self.slippage_rate)

    def _sell_fill(self, price):
        return float(price) * (1.0 - self.slippage_rate)

    def _entry_fill(self, price, side):
        return self._buy_fill(price) if side == 1 else self._sell_fill(price)

    def _exit_fill(self, price, side):
        return self._sell_fill(price) if side == 1 else self._buy_fill(price)

    def _apply_fee(self, notional):
        fee = abs(float(notional)) * self.fee_rate
        self.cash -= fee
        return fee

    def _is_intraday(self):
        return self.interval in [
            "1m", "2m", "3m", "5m", "15m", "30m", "60m", "90m", "1h", "4h"
        ]

    def _is_crypto_symbol(self):
        s = str(self.symbol).upper()
        return s.endswith(("USDT", "BUSD", "FDUSD", "USDC"))

    def _allows_fractional_qty(self):
        if self.resolved_source == "binance":
            return True
        if self.resolved_source == "yfinance":
            return False
        return self._is_crypto_symbol()

    def _unit_size(self, equity, N, price_hint=None):
        if pd.isna(N) or N <= 0 or equity <= 0:
            return 0.0

        qty = (equity * self.risk_per_unit) / N

        if self._allows_fractional_qty():
            qty = np.floor(qty * 10000) / 10000
        else:
            qty = np.floor(qty)

        return max(float(qty), 0.0)

    def _to_exchange_tz(self, ts):
        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            return ts.tz_localize(self.exchange_tz)

        return ts.tz_convert(self.exchange_tz)

    def _buffer_days(self):
        """
        워밍업 버퍼.
        MA200, S2 55봉, N 계산을 위해 넉넉히 확보.
        """
        if self._is_intraday():
            return 120

        bars_per_day = {
            "1d": 1,
            "5d": 1 / 5,
            "1wk": 1 / 7,
            "1mo": 1 / 30,
        }

        bpd = bars_per_day.get(self.interval, 1)
        return int((220 / max(bpd, 0.05)) * 1.5) + 30

    # ─────────────────────────────────────────
    # Custom window helpers
    # ─────────────────────────────────────────
    def _custom_control_enabled(self):
        return (
            self.session_mode == "custom"
            and self.custom_session is not None
            and self.custom_execution_control
            and self._is_intraday()
        )

    def _interval_timedelta(self):
        mapping = {
            "1m": pd.Timedelta(minutes=1),
            "2m": pd.Timedelta(minutes=2),
            "3m": pd.Timedelta(minutes=3),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "60m": pd.Timedelta(hours=1),
            "1h": pd.Timedelta(hours=1),
            "90m": pd.Timedelta(minutes=90),
            "4h": pd.Timedelta(hours=4),
        }

        if self.interval not in mapping:
            raise ValueError(f"지원하지 않는 intraday interval: {self.interval}")

        return mapping[self.interval]

    def _localize_custom_time(self, date_obj, time_str):
        hh, mm = map(int, time_str.split(":"))

        ts = pd.Timestamp(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            hour=hh,
            minute=mm,
        )

        return ts.tz_localize(self.exchange_tz)

    def _custom_session_bounds_for_date(self, date_obj):
        """
        custom_session을 exchange_tz 기준 현지시간으로 해석.

        미국 주식이면:
            exchange_tz='America/New_York'
            custom_session=('09:30','12:30')

        이 경우 DST 보정은 timezone이 자동 처리.
        """
        start_str, end_str = self.custom_session

        start = self._localize_custom_time(date_obj, start_str)
        end = self._localize_custom_time(date_obj, end_str)

        # 자정 넘어가는 custom window 지원
        if end <= start:
            end = end + pd.Timedelta(days=1)

        return start, end

    def _custom_bounds_candidates(self, ts):
        """
        overnight custom session 때문에 현재 날짜와 전날 날짜 후보 둘 다 확인.
        """
        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize(self.exchange_tz)
        else:
            ts = ts.tz_convert(self.exchange_tz)

        d0 = ts.date()
        d1 = (ts - pd.Timedelta(days=1)).date()

        return [
            self._custom_session_bounds_for_date(d0),
            self._custom_session_bounds_for_date(d1),
        ]

    def _bar_close_time(self, bar_start_ts):
        """
        대부분의 yfinance intraday bar index는 bar 시작 시각.
        따라서 bar close time은 start + interval로 가정.
        """
        ts = pd.Timestamp(bar_start_ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize(self.exchange_tz)
        else:
            ts = ts.tz_convert(self.exchange_tz)

        return ts + self._interval_timedelta()

    def _is_time_inside_custom_session(self, ts, include_end=True):
        if not self._custom_control_enabled():
            return True

        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize(self.exchange_tz)
        else:
            ts = ts.tz_convert(self.exchange_tz)

        for start, end in self._custom_bounds_candidates(ts):
            if include_end:
                if start <= ts <= end:
                    return True
            else:
                if start <= ts < end:
                    return True

        return False

    def _is_entry_execution_time(self, ts):
        """
        신규진입/추가매수 실행 가능한 시가인지 확인.

        예:
        custom_session=('09:30','12:30')이면

        09:30 가능
        10:30 가능
        11:30 가능
        12:30 불가

        12:30은 청산만 허용.
        """
        if not self._custom_control_enabled():
            return True

        ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize(self.exchange_tz)
        else:
            ts = ts.tz_convert(self.exchange_tz)

        for start, end in self._custom_bounds_candidates(ts):
            if start <= ts < end:
                return True

        return False

    def _is_entry_signal_allowed_for_next_open(self, signal_bar_start_ts, next_open_ts):
        """
        현재 봉 종가에서 발생한 진입/추가매수 신호를
        다음 봉 시가에 실행해도 되는지 확인.

        조건:
        1. 신호 봉의 종가 시간이 custom window 안이어야 함.
        2. 다음 봉 시가가 신규진입/추가매수 가능한 시간이어야 함.
        3. custom 종료 시각에는 신규진입/추가매수 금지.
        """
        if not self._custom_control_enabled():
            return True

        if next_open_ts is None:
            return False

        signal_close_ts = self._bar_close_time(signal_bar_start_ts)

        signal_time_ok = self._is_time_inside_custom_session(
            signal_close_ts,
            include_end=True,
        )

        execution_time_ok = self._is_entry_execution_time(next_open_ts)

        return signal_time_ok and execution_time_ok

    def _is_force_exit_signal_bar(self, bar_start_ts):
        """
        custom session 종료 봉이면 True.

        예:
        custom_session=('09:30','12:30'), interval='1h'

        11:30~12:30 봉의 종가 확인 후
        다음 봉 시가 12:30에 전량 청산.
        """
        if not self._custom_control_enabled():
            return False

        close_ts = self._bar_close_time(bar_start_ts)

        for start, end in self._custom_bounds_candidates(close_ts):
            if abs((close_ts - end).total_seconds()) < 1:
                return True

        return False

    # ─────────────────────────────────────────
    # Session filter
    # ─────────────────────────────────────────
    def _classify_us_session(self, dt_index):
        idx_local = dt_index.tz_convert("America/New_York")
        hhmm = idx_local.hour * 100 + idx_local.minute

        session = np.where(
            (hhmm >= 400) & (hhmm < 930),
            "premarket",
            np.where(
                (hhmm >= 930) & (hhmm < 1600),
                "regular",
                np.where(
                    (hhmm >= 1600) & (hhmm < 2000),
                    "postmarket",
                    "other",
                ),
            ),
        )

        return pd.Series(session, index=dt_index)

    def _apply_session_filter(self, df):
        if self.session_mode == "custom":
            start_str, end_str = self.custom_session

            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))

            start_hm = sh * 100 + sm
            end_hm = eh * 100 + em

            idx_local = df.index.tz_convert(self.exchange_tz)
            hhmm = idx_local.hour * 100 + idx_local.minute

            if start_hm <= end_hm:
                mask = (hhmm >= start_hm) & (hhmm < end_hm)
            else:
                mask = (hhmm >= start_hm) | (hhmm < end_hm)

            return df[mask].copy()

        if "session" not in df.columns:
            return df

        if self.session_mode == "extended":
            return df[df["session"].isin(["premarket", "regular", "postmarket"])].copy()

        if self.session_mode == "regular":
            return df[df["session"] == "regular"].copy()

        if self.session_mode == "premarket":
            return df[df["session"] == "premarket"].copy()

        if self.session_mode == "postmarket":
            return df[df["session"] == "postmarket"].copy()

        return df

    def _max_yf_lookback_days(self):
        if not self._is_intraday():
            return None

        if self.interval in ["60m", "1h"]:
            return 730

        if self.interval == "1m":
            return 7

        return 60

    # ─────────────────────────────────────────
    # Data loader
    # ─────────────────────────────────────────
    def fetch_data(self):
        interval_map_binance = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        interval_map_yf = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "60m",
            "90m": "90m",
            "1h": "60m",
            "1d": "1d",
            "5d": "5d",
            "1wk": "1wk",
            "1mo": "1mo",
        }

        if self.interval not in set(interval_map_binance.keys()).union(interval_map_yf.keys()):
            raise ValueError(f"지원하지 않는 interval: {self.interval}")

        buffer_days = self._buffer_days()
        buffer_start = self.start_date - timedelta(days=buffer_days)
        end_dt = self.end_date + timedelta(days=1)

        if self.source == "auto":
            source = "binance" if self._is_crypto_symbol() else "yfinance"
        else:
            source = str(self.source).lower()

        self.resolved_source = source

        session_label = self.session_mode
        if self.session_mode == "custom":
            session_label = f"custom({self.custom_session[0]}~{self.custom_session[1]})"

        print(
            f"[데이터 로드] {self.symbol} | source={source} | "
            f"interval={self.interval} | session_mode={session_label}"
        )

        # ─────────────────────────────────────
        # Binance
        # ─────────────────────────────────────
        if source == "binance":
            if self.interval not in interval_map_binance:
                raise ValueError(f"바이낸스에서 지원하지 않는 interval: {self.interval}")

            if not str(self.symbol).upper().endswith(("USDT", "BUSD", "FDUSD", "USDC")):
                print(
                    "  [주의] Binance spot 심볼은 보통 ETHUSDT, BTCUSDT 같은 형식입니다. "
                    f"현재 symbol={self.symbol}"
                )

            all_data = []
            start_ms = int(buffer_start.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            while start_ms < end_ms:
                params = {
                    "symbol": str(self.symbol).upper(),
                    "interval": interval_map_binance[self.interval],
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                }

                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and data.get("code") is not None:
                    raise RuntimeError(f"바이낸스 API 오류: {data}")

                if not data:
                    break

                all_data.extend(data)
                start_ms = data[-1][6] + 1

                print(f"\r  {len(all_data)}봉 로드중...", end="")

                if len(data) < 1000:
                    break

                time.sleep(0.08)

            if not all_data:
                raise ValueError(f"데이터를 가져올 수 없습니다: {self.symbol}")

            df = pd.DataFrame(
                all_data,
                columns=[
                    "open_time",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "close_time",
                    "quote_vol",
                    "trades_count",
                    "taker_buy_vol",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].astype(float)

            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("datetime", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df["session"] = "crypto_24x7"

            if self.session_mode == "custom" and not self.custom_execution_control:
                df = self._apply_session_filter(df)

        # ─────────────────────────────────────
        # yfinance
        # ─────────────────────────────────────
        elif source == "yfinance":
            if self.interval not in interval_map_yf:
                raise ValueError(f"yfinance에서 지원하지 않는 interval: {self.interval}")

            yf_start = buffer_start
            yf_end = end_dt

            if self._is_intraday():
                max_days = self._max_yf_lookback_days()

                if max_days is not None:
                    max_lookback_start = yf_end - timedelta(days=max_days)

                    if yf_start < max_lookback_start:
                        print(
                            f"  [주의] yfinance {self.interval} 데이터는 최근 약 {max_days}일 범위로 자동 조정합니다."
                        )
                        yf_start = max_lookback_start

            prepost = self._is_intraday()

            df = yf.download(
                self.symbol,
                start=yf_start.strftime("%Y-%m-%d"),
                end=yf_end.strftime("%Y-%m-%d"),
                interval=interval_map_yf[self.interval],
                auto_adjust=True,
                progress=False,
                prepost=prepost,
            )

            if df is None or len(df) == 0:
                raise ValueError(f"주식/ETF 데이터를 가져올 수 없습니다: {self.symbol}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            needed_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in needed_cols:
                if col not in df.columns:
                    raise ValueError(f"필수 컬럼 누락: {col}")

            df = df[needed_cols].copy()

            if self._is_intraday():
                if df.index.tz is None:
                    df.index = df.index.tz_localize(self.exchange_tz)
                else:
                    df.index = df.index.tz_convert(self.exchange_tz)

                df["session"] = self._classify_us_session(df.index)

                if self.session_mode == "custom" and self.custom_execution_control:
                    # 중요:
                    # custom은 데이터 필터가 아니라 매매 가능 시간 제어로 사용.
                    # 따라서 custom 시간으로 바로 자르지 않는다.
                    if self.custom_data_session == "regular":
                        df = df[df["session"] == "regular"].copy()

                    elif self.custom_data_session == "extended":
                        df = df[
                            df["session"].isin(["premarket", "regular", "postmarket"])
                        ].copy()

                    elif self.custom_data_session == "raw":
                        df = df.copy()

                    elif self.custom_data_session == "custom_filter":
                        df = self._apply_session_filter(df)

                else:
                    df = self._apply_session_filter(df)

            else:
                df["session"] = "daily"

        else:
            raise ValueError("source는 'auto', 'yfinance', 'binance' 중 하나여야 합니다.")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        if len(df) == 0:
            raise ValueError("세션 필터 적용 후 데이터가 없습니다.")

        self.raw_df = df.copy()

        print(f"\n  완료: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")
        return self

    # ─────────────────────────────────────────
    # Indicators
    # ─────────────────────────────────────────
    def calculate_indicators(self):
        if self.raw_df is None or len(self.raw_df) == 0:
            raise ValueError("먼저 fetch_data()를 실행하세요.")

        df = self.raw_df.copy()

        df["prev_close"] = df["Close"].shift(1)

        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["prev_close"]).abs()
        tr3 = (df["Low"] - df["prev_close"]).abs()

        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR/N: 신호 발생 봉 종가 기준으로 확정되는 값
        df["N"] = df["TR"].ewm(alpha=1 / 20, adjust=False).mean()

        # 현재 봉 제외 이전 봉 기준 channel
        df["s1_entry_high"] = df["High"].rolling(20).max().shift(1)
        df["s1_entry_low"] = df["Low"].rolling(20).min().shift(1)

        df["s2_entry_high"] = df["High"].rolling(55).max().shift(1)
        df["s2_entry_low"] = df["Low"].rolling(55).min().shift(1)

        df["s1_exit_low"] = df["Low"].rolling(10).min().shift(1)
        df["s1_exit_high"] = df["High"].rolling(10).max().shift(1)

        df["s2_exit_low"] = df["Low"].rolling(20).min().shift(1)
        df["s2_exit_high"] = df["High"].rolling(20).max().shift(1)

        if self.use_current_ma:
            # 종가 확정 후 판단이므로 현재 봉 종가 포함 MA200 사용 가능
            df["MA200"] = df["Close"].rolling(200).mean()
        else:
            # 더 보수적 방식
            df["MA200"] = df["Close"].rolling(200).mean().shift(1)

        # 분석 기간으로 자르기
        if df.index.tz is not None:
            if self.resolved_source == "binance":
                start_ts = (
                    pd.Timestamp(self.start_date).tz_localize("UTC")
                    if pd.Timestamp(self.start_date).tzinfo is None
                    else pd.Timestamp(self.start_date).tz_convert("UTC")
                )

                end_ts = (
                    pd.Timestamp(self.end_date + timedelta(days=1)).tz_localize("UTC")
                    if pd.Timestamp(self.end_date).tzinfo is None
                    else pd.Timestamp(self.end_date + timedelta(days=1)).tz_convert("UTC")
                )

            else:
                start_ts = self._to_exchange_tz(self.start_date)
                end_ts = self._to_exchange_tz(self.end_date + timedelta(days=1))

            self.df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

        else:
            self.df = df.loc[self.start_date:self.end_date].copy()

        if len(self.df) == 0:
            raise ValueError("분석 기간에 데이터가 없습니다.")

        print(f"  지표 계산 완료: {self.df.index[0]} ~ {self.df.index[-1]} ({len(self.df)}봉)")
        return self

    # ─────────────────────────────────────────
    # Trading actions
    # ─────────────────────────────────────────
    def _enter_position(self, date, side, system_name, entry_price, signal_N, mark_price, signal_info=None):
        equity = self._equity(mark_price)
        qty = self._unit_size(equity, signal_N, entry_price)

        if qty <= 0:
            self._log(
                date,
                "SKIP",
                "unit size <= 0",
                system=system_name,
                N=signal_N,
                equity=equity,
            )
            return False

        fill = self._entry_fill(entry_price, side)
        signed_qty = side * qty

        self.cash -= signed_qty * fill
        fee = self._apply_fee(signed_qty * fill)

        self.units = [
            {
                "qty": signed_qty,
                "price": fill,
                "fee": fee,
                "type": "ENTRY",
            }
        ]

        self.position_side = side
        self.current_system = system_name
        self.current_unit_qty = qty
        self.entry_N = float(signal_N)

        self.base_entry_price = fill

        if side == 1:
            self.initial_stop_price = fill - 2.0 * self.entry_N
            self.stop_price = self.initial_stop_price
            self.add_levels = [
                fill + 0.5 * self.entry_N,
                fill + 1.0 * self.entry_N,
                fill + 1.5 * self.entry_N,
            ]
        else:
            self.initial_stop_price = fill + 2.0 * self.entry_N
            self.stop_price = self.initial_stop_price
            self.add_levels = [
                fill - 0.5 * self.entry_N,
                fill - 1.0 * self.entry_N,
                fill - 1.5 * self.entry_N,
            ]

        self.next_add_index = 0

        trade = {
            "date": date,
            "type": "ENTRY",
            "system": system_name,
            "side": "LONG" if side == 1 else "SHORT",
            "price": fill,
            "raw_open_price": entry_price,
            "qty": qty,
            "units_after": len(self.units),
            "fee": fee,
            "N": self.entry_N,
            "initial_stop": self.initial_stop_price,
            "stop_price": self.stop_price,
        }

        if signal_info:
            trade.update(
                {
                    "signal_date": signal_info.get("signal_date"),
                    "signal_close": signal_info.get("signal_close"),
                    "signal_level": signal_info.get("signal_level"),
                }
            )

        self.trades.append(trade)

        self._log(
            date=date,
            event="ENTRY",
            reason=f"{system_name} signal executed at next open",
            side="LONG" if side == 1 else "SHORT",
            raw_open_price=entry_price,
            fill_price=fill,
            qty=qty,
            N=self.entry_N,
            initial_stop=self.initial_stop_price,
            stop_price=self.stop_price,
            add_levels=self.add_levels,
            signal_info=signal_info,
        )

        return True

    def _add_unit(self, date, entry_price):
        if self.position_side == 0:
            return False

        if len(self.units) >= self.max_units:
            return False

        fill = self._entry_fill(entry_price, self.position_side)
        signed_qty = self.position_side * self.current_unit_qty

        self.cash -= signed_qty * fill
        fee = self._apply_fee(signed_qty * fill)

        self.units.append(
            {
                "qty": signed_qty,
                "price": fill,
                "fee": fee,
                "type": "ADD",
            }
        )

        self.trades.append(
            {
                "date": date,
                "type": "ADD",
                "system": self.current_system,
                "side": "LONG" if self.position_side == 1 else "SHORT",
                "price": fill,
                "raw_open_price": entry_price,
                "qty": self.current_unit_qty,
                "units_after": len(self.units),
                "fee": fee,
                "N": self.entry_N,
                "stop_price": self.stop_price,
            }
        )

        self._log(
            date=date,
            event="ADD",
            reason="add signal executed at next open",
            raw_open_price=entry_price,
            fill_price=fill,
            qty=self.current_unit_qty,
            units_after=len(self.units),
            stop_price=self.stop_price,
        )

        return True

    def _update_trailing_stop(self, date, row):
        if self.position_side == 0:
            return

        if self.current_system == "S1":
            candidate = row["s1_exit_low"] if self.position_side == 1 else row["s1_exit_high"]
        elif self.current_system == "S2":
            candidate = row["s2_exit_low"] if self.position_side == 1 else row["s2_exit_high"]
        else:
            return

        if pd.isna(candidate):
            return

        old_stop = self.stop_price

        if self.position_side == 1:
            self.stop_price = max(float(self.stop_price), float(candidate))
        else:
            self.stop_price = min(float(self.stop_price), float(candidate))

        if self.stop_price != old_stop:
            self._log(
                date=date,
                event="STOP_UPDATE",
                reason="exit channel moved beyond previous stop",
                system=self.current_system,
                old_stop=old_stop,
                new_stop=self.stop_price,
                channel_level=float(candidate),
            )

    def _exit_all(self, date, exit_price, reason, close_price=None):
        if not self.units:
            return

        side = self.position_side
        signed_total = self._net_position_qty()

        fill = self._exit_fill(exit_price, side)

        self.cash += signed_total * fill
        exit_fee = self._apply_fee(signed_total * fill)

        gross_pnl = sum((fill - u["price"]) * u["qty"] for u in self.units)
        entry_fees = sum(u.get("fee", 0.0) for u in self.units)
        net_pnl = gross_pnl - entry_fees - exit_fee

        self.trades.append(
            {
                "date": date,
                "type": "EXIT",
                "system": self.current_system,
                "side": "LONG" if side == 1 else "SHORT",
                "price": fill,
                "raw_open_price": exit_price,
                "qty": abs(signed_total),
                "units_before": len(self.units),
                "gross_pnl": gross_pnl,
                "entry_fees": entry_fees,
                "exit_fee": exit_fee,
                "fee": entry_fees + exit_fee,
                "pnl": net_pnl,
                "reason": reason,
                "stop_price": self.stop_price,
                "close_price": close_price,
            }
        )

        self._log(
            date=date,
            event="EXIT",
            reason=reason,
            raw_open_price=exit_price,
            fill_price=fill,
            gross_pnl=gross_pnl,
            entry_fees=entry_fees,
            exit_fee=exit_fee,
            pnl=net_pnl,
            units_before=len(self.units),
            stop_price=self.stop_price,
            close_price=close_price,
        )

        self.units = []
        self.position_side = 0
        self.current_system = None
        self.current_unit_qty = 0.0
        self.entry_N = np.nan

        self.base_entry_price = np.nan
        self.initial_stop_price = np.nan
        self.stop_price = np.nan

        self.add_levels = []
        self.next_add_index = 0

        self.pending_add_count = 0
        self.pending_exit = None

    # ─────────────────────────────────────────
    # Signal helpers
    # ─────────────────────────────────────────
    def _indicators_ready(self, row):
        needed = [
            row["s1_entry_high"],
            row["s1_entry_low"],
            row["s1_exit_low"],
            row["s1_exit_high"],
            row["s2_entry_high"],
            row["s2_entry_low"],
            row["s2_exit_low"],
            row["s2_exit_high"],
            row["N"],
            row["MA200"],
        ]

        return not any(pd.isna(x) for x in needed)

    def _make_entry_signal(self, date, row):
        close = float(row["Close"])
        ma200 = float(row["MA200"])
        N = float(row["N"])

        s2_long_ok = close > row["s2_entry_high"] and close > ma200
        s1_long_ok = close > row["s1_entry_high"] and close > ma200

        s2_short_ok = self.allow_short and close < row["s2_entry_low"] and close < ma200
        s1_short_ok = self.allow_short and close < row["s1_entry_low"] and close < ma200

        # S2 우선
        if s2_long_ok:
            return {
                "side": 1,
                "system": "S2",
                "N": N,
                "signal_date": date,
                "signal_close": close,
                "signal_level": float(row["s2_entry_high"]),
                "reason": "S2 long breakout close confirmed",
            }

        if s2_short_ok:
            return {
                "side": -1,
                "system": "S2",
                "N": N,
                "signal_date": date,
                "signal_close": close,
                "signal_level": float(row["s2_entry_low"]),
                "reason": "S2 short breakout close confirmed",
            }

        if s1_long_ok:
            return {
                "side": 1,
                "system": "S1",
                "N": N,
                "signal_date": date,
                "signal_close": close,
                "signal_level": float(row["s1_entry_high"]),
                "reason": "S1 long breakout close confirmed",
            }

        if s1_short_ok:
            return {
                "side": -1,
                "system": "S1",
                "N": N,
                "signal_date": date,
                "signal_close": close,
                "signal_level": float(row["s1_entry_low"]),
                "reason": "S1 short breakout close confirmed",
            }

        return None

    # ─────────────────────────────────────────
    # Backtest
    # ─────────────────────────────────────────
    def run_backtest(self):
        if self.df is None or len(self.df) == 0:
            raise ValueError("먼저 calculate_indicators()를 실행하세요.")

        df = self.df.copy()

        self._reset_backtest_state()

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            next_date = df.index[i + 1] if i + 1 < len(df) else None

            open_price = float(row["Open"])
            close = float(row["Close"])

            indicators_ready = self._indicators_ready(row)

            # --------------------------------------------------
            # 1) 이전 봉에서 예약된 청산을 현재 봉 시가에 먼저 실행
            # --------------------------------------------------
            if self.pending_exit is not None and self.position_side != 0:
                px = self.pending_exit

                self._exit_all(
                    date=date,
                    exit_price=open_price,
                    reason=px.get("reason", "PENDING_EXIT"),
                    close_price=px.get("signal_close"),
                )

                self.pending_exit = None
                self.pending_add_count = 0
                self.pending_entry = None

            # --------------------------------------------------
            # 2) 이전 봉에서 예약된 신규진입을 현재 봉 시가에 실행
            # --------------------------------------------------
            if self.pending_entry is not None and self.position_side == 0:
                if self._is_entry_execution_time(date):
                    pe = self.pending_entry

                    self._enter_position(
                        date=date,
                        side=pe["side"],
                        system_name=pe["system"],
                        entry_price=open_price,
                        signal_N=pe["N"],
                        mark_price=open_price,
                        signal_info=pe,
                    )
                else:
                    self._log(
                        date=date,
                        event="PENDING_CANCELLED",
                        reason="pending entry execution time is outside custom trading window",
                        pending_entry=self.pending_entry,
                    )

                self.pending_entry = None

            # --------------------------------------------------
            # 3) 이전 봉에서 예약된 추가매수를 현재 봉 시가에 실행
            # --------------------------------------------------
            if self.position_side != 0 and self.pending_add_count > 0:
                if self._is_entry_execution_time(date):
                    for _ in range(self.pending_add_count):
                        if len(self.units) >= self.max_units:
                            break

                        self._add_unit(date=date, entry_price=open_price)
                else:
                    self._log(
                        date=date,
                        event="PENDING_CANCELLED",
                        reason="pending add execution time is outside custom trading window",
                        pending_add_count=self.pending_add_count,
                    )

                self.pending_add_count = 0

            # --------------------------------------------------
            # 4) 현재 봉 종가 기준 trailing stop 업데이트
            # --------------------------------------------------
            if self.position_side != 0 and indicators_ready:
                self._update_trailing_stop(date=date, row=row)

            # --------------------------------------------------
            # 5) 현재 봉 종가 기준 손절 청산 신호 생성
            #    실제 청산은 다음 봉 시가
            # --------------------------------------------------
            if self.position_side != 0:
                stop_triggered = False

                if self.position_side == 1 and close < self.stop_price:
                    stop_triggered = True

                elif self.position_side == -1 and close > self.stop_price:
                    stop_triggered = True

                if stop_triggered:
                    self.pending_exit = {
                        "reason": "TRAILING_STOP_CLOSE_CONFIRMED",
                        "signal_date": date,
                        "signal_close": close,
                        "stop_price": self.stop_price,
                    }

                    self._log(
                        date=date,
                        event="EXIT_SIGNAL",
                        reason="close crossed stop; execute next open",
                        side="LONG" if self.position_side == 1 else "SHORT",
                        close=close,
                        stop_price=self.stop_price,
                        execute="next open",
                    )

            # --------------------------------------------------
            # 6) custom session 종료 봉이면 강제청산 예약
            #
            # 예:
            # custom_session=('09:30','12:30'), interval='1h'
            # 11:30~12:30 봉 종가 확인 후
            # 다음 봉 시가 12:30에 전량 청산
            # --------------------------------------------------
            if self.position_side != 0 and self.pending_exit is None:
                if self._is_force_exit_signal_bar(date):
                    self.pending_exit = {
                        "reason": "CUSTOM_SESSION_FORCE_EXIT",
                        "signal_date": date,
                        "signal_close": close,
                        "stop_price": self.stop_price,
                    }

                    self._log(
                        date=date,
                        event="EXIT_SIGNAL",
                        reason="custom session end; force exit next open",
                        side="LONG" if self.position_side == 1 else "SHORT",
                        close=close,
                        stop_price=self.stop_price,
                        execute="next open",
                    )

            # --------------------------------------------------
            # 7) 현재 봉 종가 기준 추가매수 신호 생성
            #    실제 추가매수는 다음 봉 시가
            # --------------------------------------------------
            if self.position_side != 0 and self.pending_exit is None:
                add_signal_allowed = self._is_entry_signal_allowed_for_next_open(
                    signal_bar_start_ts=date,
                    next_open_ts=next_date,
                )

                if add_signal_allowed:
                    add_count = 0

                    while (
                        self.next_add_index < len(self.add_levels)
                        and len(self.units) + add_count < self.max_units
                    ):
                        level = self.add_levels[self.next_add_index]

                        if self.position_side == 1:
                            crossed = close >= level
                        else:
                            crossed = close <= level

                        if crossed:
                            add_count += 1
                            self.next_add_index += 1
                        else:
                            break

                    if add_count > 0:
                        self.pending_add_count = add_count

                        self._log(
                            date=date,
                            event="ADD_SIGNAL",
                            reason="close crossed add level; execute next open",
                            side="LONG" if self.position_side == 1 else "SHORT",
                            close=close,
                            add_count=add_count,
                            next_add_index=self.next_add_index,
                            execute="next open",
                        )

            # --------------------------------------------------
            # 8) 현재 봉 종가 기준 신규 진입 신호 생성
            #    실제 진입은 다음 봉 시가
            # --------------------------------------------------
            if (
                self.position_side == 0
                and self.pending_entry is None
                and self.pending_exit is None
                and indicators_ready
            ):
                entry_signal_allowed = self._is_entry_signal_allowed_for_next_open(
                    signal_bar_start_ts=date,
                    next_open_ts=next_date,
                )

                if entry_signal_allowed:
                    signal = self._make_entry_signal(date=date, row=row)

                    if signal is not None:
                        self.pending_entry = signal

                        self._log(
                            date=date,
                            event="ENTRY_SIGNAL",
                            reason=signal["reason"],
                            side="LONG" if signal["side"] == 1 else "SHORT",
                            system=signal["system"],
                            close=signal["signal_close"],
                            level=signal["signal_level"],
                            N=signal["N"],
                            execute="next open",
                        )

            # --------------------------------------------------
            # 9) Equity 기록
            # --------------------------------------------------
            self.equity_curve.append(self._equity(close))

        last_date = df.index[-1]

        if self.pending_entry is not None:
            self._log(
                date=last_date,
                event="PENDING_IGNORED",
                reason="backtest ended before pending entry execution",
                pending_entry=self.pending_entry,
            )

        if self.pending_add_count > 0:
            self._log(
                date=last_date,
                event="PENDING_IGNORED",
                reason="backtest ended before pending add execution",
                pending_add_count=self.pending_add_count,
            )

        if self.pending_exit is not None:
            self._log(
                date=last_date,
                event="PENDING_IGNORED",
                reason="backtest ended before pending exit execution",
                pending_exit=self.pending_exit,
            )

        df["equity"] = self.equity_curve
        self.df = df

        return self

    def _calculate_buy_and_hold(self):
        """
        Buy & Hold 계산.

        원칙:
        1. 미국 주식 intraday:
            분석 시작일 이후 첫 미국 정규장 봉 Open
            분석 종료일 이전 마지막 미국 정규장 봉 Close

        2. 한국 주식 등 비미국 intraday:
            session_mode='custom'이면 custom_session을 정규 거래시간으로 간주
            해당 시간 안의 첫 봉 Open, 마지막 봉 Close

        3. daily 또는 crypto:
            분석 기간 첫 봉 Open, 마지막 봉 Close
        """
        # raw_df가 있으면 raw_df 기준, 없으면 self.df 기준
        if self.raw_df is not None and len(self.raw_df) > 0:
            df = self.raw_df.copy()
        elif self.df is not None and len(self.df) > 0:
            df = self.df.copy()
        else:
            return None

        # --------------------------------------------------
        # 1) 분석 기간으로 제한
        # --------------------------------------------------
        if df.index.tz is not None:
            if self.resolved_source == "binance":
                start_ts = (
                    pd.Timestamp(self.start_date).tz_localize("UTC")
                    if pd.Timestamp(self.start_date).tzinfo is None
                    else pd.Timestamp(self.start_date).tz_convert("UTC")
                )

                end_ts = (
                    pd.Timestamp(self.end_date + timedelta(days=1)).tz_localize("UTC")
                    if pd.Timestamp(self.end_date).tzinfo is None
                    else pd.Timestamp(self.end_date + timedelta(days=1)).tz_convert("UTC")
                )
            else:
                start_ts = self._to_exchange_tz(self.start_date)
                end_ts = self._to_exchange_tz(self.end_date + timedelta(days=1))

            df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()

        else:
            df = df.loc[self.start_date:self.end_date].copy()

        if len(df) == 0:
            return None

        # --------------------------------------------------
        # 2) intraday yfinance 주식 처리
        # --------------------------------------------------
        if self.resolved_source == "yfinance" and self._is_intraday():

            # 미국 주식: 미국 regular session 기준
            if self.exchange_tz == "America/New_York":
                if "session" in df.columns:
                    regular_df = df[df["session"] == "regular"].copy()

                    if len(regular_df) > 0:
                        df = regular_df
                    else:
                        # fallback: 혹시 session 분류가 없거나 이상하면 09:30~16:00 직접 필터
                        idx_local = df.index.tz_convert("America/New_York")
                        hhmm = idx_local.hour * 100 + idx_local.minute
                        mask = (hhmm >= 930) & (hhmm < 1600)
                        df = df[mask].copy()
                else:
                    idx_local = df.index.tz_convert("America/New_York")
                    hhmm = idx_local.hour * 100 + idx_local.minute
                    mask = (hhmm >= 930) & (hhmm < 1600)
                    df = df[mask].copy()

            # 한국 주식 등 비미국 주식
            else:
                # custom_session이 있으면 그 시간을 정규 거래시간으로 간주
                if self.session_mode == "custom" and self.custom_session is not None:
                    start_str, end_str = self.custom_session

                    sh, sm = map(int, start_str.split(":"))
                    eh, em = map(int, end_str.split(":"))

                    start_hm = sh * 100 + sm
                    end_hm = eh * 100 + em

                    idx_local = df.index.tz_convert(self.exchange_tz)
                    hhmm = idx_local.hour * 100 + idx_local.minute

                    if start_hm <= end_hm:
                        mask = (hhmm >= start_hm) & (hhmm < end_hm)
                    else:
                        mask = (hhmm >= start_hm) | (hhmm < end_hm)

                    filtered = df[mask].copy()

                    # 필터 결과가 있으면 사용, 없으면 기존 df 유지
                    # 한국 yfinance 데이터가 이미 custom_filter 적용된 경우도 있으므로 fallback 필요
                    if len(filtered) > 0:
                        df = filtered

        if len(df) == 0:
            return None

        # --------------------------------------------------
        # 3) 시작가/종료가
        # --------------------------------------------------
        start_price = float(df.iloc[0]["Open"])
        end_price = float(df.iloc[-1]["Close"])

        start_time = df.index[0]
        end_time = df.index[-1]

        return_pct = (end_price / start_price - 1.0) * 100.0
        final_value = self.initial_capital * (end_price / start_price)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "start_price": start_price,
            "end_price": end_price,
            "return_pct": return_pct,
            "final_value": final_value,
        }



    # ─────────────────────────────────────────
    # Result
    # ─────────────────────────────────────────
    def print_results(self):
        print("\n" + "=" * 100)
        print(f"Close-Confirmed Turtle Backtest | {self.symbol} ({self.interval})")
        print("S1: 20-bar breakout / 10-bar trailing stop")
        print("S2: 55-bar breakout / 20-bar trailing stop")
        print("Signal: close confirmed | Execution: next bar open")
        print("Priority: S2 over S1")
        print("N: 20EMA(TR), fixed at signal bar")
        print(f"Unit Risk: {self.risk_per_unit * 100:.2f}% | Max Units: {self.max_units}")
        print(f"Fee: {self.fee_rate * 100:.3f}% | Slippage: {self.slippage_rate * 100:.3f}%")
        print(f"Short: {self.allow_short}")
        print(f"Source: {self.resolved_source} | Session Mode: {self.session_mode}")

        if self.session_mode == "custom":
            print(f"Custom Session: {self.custom_session} | TZ: {self.exchange_tz}")
            print(f"Custom Execution Control: {self.custom_execution_control}")
            print(f"Custom Data Session: {self.custom_data_session}")

        print("=" * 100)

        if self.df is None or len(self.df) == 0:
            print("결과 데이터가 없습니다.")
            return

        if not self.equity_curve:
            print("equity_curve가 없습니다. 먼저 run_backtest()를 실행하세요.")
            return

        # ─────────────────────────────────────────
        # 전략 성과
        # ─────────────────────────────────────────
        final_equity = float(self.equity_curve[-1])
        total_return = (final_equity / self.initial_capital - 1.0) * 100.0

        eq = pd.Series(self.equity_curve, index=self.df.index, dtype=float)
        mdd = ((eq - eq.cummax()) / eq.cummax() * 100.0).min()

        # ─────────────────────────────────────────
        # Buy & Hold 성과
        # ─────────────────────────────────────────
        bnh = self._calculate_buy_and_hold()

        print(f"\n초기 자본:        {self.initial_capital:>15,.2f}")
        print(f"전략 최종자산:    {final_equity:>15,.2f}")
        print(f"전략 수익률:      {total_return:>14.2f} %")
        print(f"전략 MDD:         {mdd:>14.2f} %")

        if bnh is not None:
            excess_return = total_return - bnh["return_pct"]
            excess_value = final_equity - bnh["final_value"]

            print("\n" + "-" * 100)
            print("Buy & Hold 비교")
            print("-" * 100)
            print(f"B&H 최종자산:     {bnh['final_value']:>15,.2f}")
            print(f"B&H 수익률:       {bnh['return_pct']:>14.2f} %")
            print(f"초과수익률:        {excess_return:>14.2f} %p")
            print(f"초과금액:          {excess_value:>15,.2f}")
            print(f"B&H 시작:         {str(bnh['start_time']):>25} | Open={bnh['start_price']:,.4f}")
            print(f"B&H 종료:         {str(bnh['end_time']):>25} | Close={bnh['end_price']:,.4f}")
        else:
            print("\nBuy & Hold:       계산 불가")

        # ─────────────────────────────────────────
        # 거래 통계
        # ─────────────────────────────────────────
        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) > 0:
            entries = trades_df[trades_df["type"] == "ENTRY"].copy()
            adds = trades_df[trades_df["type"] == "ADD"].copy()
            exits = trades_df[trades_df["type"] == "EXIT"].copy()
        else:
            entries = pd.DataFrame()
            adds = pd.DataFrame()
            exits = pd.DataFrame()

        print("\n" + "-" * 100)
        print("거래 통계")
        print("-" * 100)
        print(f"총 신규 진입:     {len(entries):>14} 회")
        print(f"총 추가 매수:     {len(adds):>14} 회")
        print(f"총 청산:          {len(exits):>14} 회")

        if self.position_side != 0:
            print("\n[주의] 백테스트 종료 시점에 미청산 포지션이 있습니다.")
            print(f"보유 방향:        {'LONG' if self.position_side == 1 else 'SHORT'}")
            print(f"보유 유닛 수:     {len(self.units)}")
            print(f"현재 Stop:        {self.stop_price:,.4f}")

        if len(exits) > 0:
            wins = exits[exits["pnl"] > 0]
            losses = exits[exits["pnl"] <= 0]

            win_rate = len(wins) / max(len(exits), 1) * 100.0

            print(f"\n승률:             {win_rate:>14.2f} %")

            if len(wins) > 0:
                print(f"평균 수익:        {wins['pnl'].mean():>15,.2f}")
                print(f"최대 수익:        {wins['pnl'].max():>15,.2f}")
                print(f"총 수익:          {wins['pnl'].sum():>15,.2f}")

            if len(losses) > 0:
                print(f"평균 손실:        {losses['pnl'].mean():>15,.2f}")
                print(f"최대 손실:        {losses['pnl'].min():>15,.2f}")
                print(f"총 손실:          {losses['pnl'].sum():>15,.2f}")

            total_win = wins["pnl"].sum() if len(wins) else 0.0
            total_loss = losses["pnl"].sum() if len(losses) else 0.0

            if total_loss != 0:
                pf = abs(total_win / total_loss)
                print(f"Profit Factor:    {pf:>14.2f}")

            avg_pnl = exits["pnl"].mean()
            total_pnl = exits["pnl"].sum()

            print(f"평균 청산 PnL:    {avg_pnl:>15,.2f}")
            print(f"총 실현 PnL:      {total_pnl:>15,.2f}")

            if "reason" in exits.columns:
                by_reason = exits.groupby("reason")["pnl"].agg(["count", "sum", "mean"])
                print("\n청산 사유별:")
                print(by_reason)

            if "system" in exits.columns:
                by_system = exits.groupby("system")["pnl"].agg(["count", "sum", "mean"])
                print("\n시스템별:")
                print(by_system)

        print("\n" + "=" * 100)


    # ─────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────
    def plot_equity(self, show=True):
        if self.df is None or "equity" not in self.df.columns:
            print("먼저 run_backtest()를 실행하세요.")
            return

        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df["equity"], label="Turtle Equity", linewidth=1.2)
        plt.axhline(self.initial_capital, color="gray", linestyle="--", alpha=0.5)
        plt.title(
            f"Close-Confirmed Turtle Equity Curve - "
            f"{self.symbol} ({self.interval}, {self.session_mode})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()

    def plot_price_with_trades(self, show=True):
        if self.df is None:
            print("먼저 run_backtest()를 실행하세요.")
            return

        trades = self.get_trades_df()

        plt.figure(figsize=(14, 6))
        plt.plot(self.df.index, self.df["Close"], label="Close", linewidth=1.0)

        if len(trades) > 0:
            entries = trades[trades["type"] == "ENTRY"]
            adds = trades[trades["type"] == "ADD"]
            exits = trades[trades["type"] == "EXIT"]

            if len(entries) > 0:
                plt.scatter(
                    entries["date"],
                    entries["price"],
                    marker="^",
                    s=80,
                    label="ENTRY",
                    zorder=5,
                )

            if len(adds) > 0:
                plt.scatter(
                    adds["date"],
                    adds["price"],
                    marker="o",
                    s=50,
                    label="ADD",
                    zorder=5,
                )

            if len(exits) > 0:
                plt.scatter(
                    exits["date"],
                    exits["price"],
                    marker="v",
                    s=80,
                    label="EXIT",
                    zorder=5,
                )

        plt.title(f"Price & Trades - {self.symbol}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()