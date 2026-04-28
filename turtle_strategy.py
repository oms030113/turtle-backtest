# turtle_strategy.py
# 필요 패키지: pip install yfinance pandas numpy matplotlib requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from datetime import timedelta
import yfinance as yf


class OriginalTurtleTrading:
    def __init__(self, symbol, start_date, end_date,
                 interval='1d',
                 source='auto',              # 'auto' | 'yfinance' | 'binance'
                 initial_capital=100_000.0,
                 risk_per_unit=0.01,         # 1 Unit = equity의 1% 리스크
                 max_units=4,
                 fee_rate=0.0005,
                 slippage_rate=0.0005,
                 allow_short=False,
                 session_mode='regular',     # 'regular'|'extended'|'premarket'|'postmarket'|'custom'
                 custom_session=None,        # ('HH:MM','HH:MM')  session_mode='custom'일 때만 사용
                 exchange_tz='America/New_York'):
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
        self.allow_short = allow_short

        self.session_mode = str(session_mode).lower()
        self.custom_session = custom_session
        self.exchange_tz = exchange_tz
        self.resolved_source = None

        valid_modes = ['regular', 'extended', 'premarket', 'postmarket', 'custom']
        if self.session_mode not in valid_modes:
            raise ValueError(f"session_mode는 {valid_modes} 중 하나여야 합니다.")

        if self.session_mode == 'custom':
            if (not isinstance(custom_session, (tuple, list))
                    or len(custom_session) != 2):
                raise ValueError(
                    "session_mode='custom'일 때는 custom_session=('HH:MM','HH:MM') 형식으로 지정해야 합니다."
                )
            try:
                for t in custom_session:
                    hh, mm = t.split(':')
                    int(hh); int(mm)
            except Exception:
                raise ValueError("custom_session 시간 형식이 잘못되었습니다. 예: ('09:45','11:00')")

        self.raw_df = None
        self.df = None

        # backtest state
        self.cash = self.initial_capital
        self.units = []
        self.position_side = 0
        self.current_system = None
        self.current_unit_qty = 0.0
        self.entry_N = np.nan
        self.next_add_price = np.nan
        self.stop_price = np.nan

        # S1 skip rule
        self.skip_next_s1 = False

        self.trades = []
        self.debug_logs = []
        self.equity_curve = []

        # 미국시간 기준 모드일 때 exchange_tz 자동 보정
        if self.session_mode in ['regular', 'premarket', 'postmarket', 'extended']:
            # 인트라데이일 때만 미국 기준 강제 (일봉은 세션 필터 안 타니 무관)
            if self._is_intraday() and self.exchange_tz != 'America/New_York':
                # 한국/일본/유럽 주식은 custom으로 쓰라고 안내
                if not str(self.symbol).upper().endswith(('.KS', '.KQ', '.T', '.HK', '.L')):
                    print(f"  [알림] session_mode='{self.session_mode}'는 미국시간 기준입니다. "
                          f"exchange_tz를 'America/New_York'로 자동 설정합니다.")
                    self.exchange_tz = 'America/New_York'

    # ─────────────────────────────────────────
    # Debug / util
    # ─────────────────────────────────────────
    def _log(self, date, event, reason, **kwargs):
        row = {'date': date, 'event': event, 'reason': reason}
        row.update(kwargs)
        self.debug_logs.append(row)

    def get_debug_log_df(self):
        return pd.DataFrame(self.debug_logs)

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

    def print_debug_logs(self, last_n=30):
        if not self.debug_logs:
            print("디버깅 로그가 없습니다.")
            return
        df = pd.DataFrame(self.debug_logs)
        print(df.tail(last_n).to_string(index=False))

    def _net_position_qty(self):
        if not self.units:
            return 0.0
        return sum(u['qty'] for u in self.units)

    def _equity(self, mark_price):
        return self.cash + self._net_position_qty() * mark_price

    def _buy_fill(self, price):
        return price * (1 + self.slippage_rate)

    def _sell_fill(self, price):
        return price * (1 - self.slippage_rate)

    def _entry_fill(self, price, side):
        return self._buy_fill(price) if side == 1 else self._sell_fill(price)

    def _exit_fill(self, price, side):
        return self._sell_fill(price) if side == 1 else self._buy_fill(price)

    def _apply_fee(self, notional):
        fee = abs(notional) * self.fee_rate
        self.cash -= fee
        return fee

    def _is_intraday(self):
        return self.interval in ['1m', '2m', '3m', '5m', '15m', '30m', '60m', '90m', '1h', '4h']

    def _is_crypto_symbol(self):
        s = str(self.symbol).upper()
        return s.endswith('USDT') or s.endswith('BUSD')

    def _allows_fractional_qty(self):
        if self.resolved_source == 'binance':
            return True
        if self.resolved_source == 'yfinance':
            return False
        return self._is_crypto_symbol()

    def _to_exchange_tz(self, ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize(self.exchange_tz)
        return ts.tz_convert(self.exchange_tz)

    def _buffer_days(self):
        if self._is_intraday():
            return 15

        bars_per_day = {
            '1d': 1,
            '5d': 1 / 5,
            '1wk': 1 / 7,
            '1mo': 1 / 30
        }
        bpd = bars_per_day.get(self.interval, 1)
        return int((55 / max(bpd, 0.1)) * 2.0) + 30

    def _unit_size(self, equity, N, price_hint=None):
        """
        Unit = (equity * risk_per_unit) / N
        주식/ETF/코인용 단순화 버전 (1포인트=1가격단위).
        """
        if pd.isna(N) or N <= 0 or equity <= 0:
            return 0.0

        qty = (equity * self.risk_per_unit) / N

        if self._allows_fractional_qty():
            qty = np.floor(qty * 10000) / 10000
        else:
            qty = np.floor(qty)

        return max(float(qty), 0.0)

    def _choose_breakout_side(self, open_, long_level, short_level, high, low):
        hit_long = pd.notna(long_level) and high >= long_level
        hit_short = pd.notna(short_level) and low <= short_level

        if hit_long and not hit_short:
            return 1, long_level
        if hit_short and not hit_long:
            return -1, short_level
        if hit_long and hit_short:
            if abs(open_ - long_level) <= abs(open_ - short_level):
                return 1, long_level
            return -1, short_level
        return 0, np.nan

    def _classify_us_session(self, dt_index):
        """
        미국 주식용 세션 분류
        - premarket : 04:00 ~ 09:29
        - regular   : 09:30 ~ 15:59
        - postmarket: 16:00 ~ 19:59
        - other     : 그 외
        """
        idx_local = dt_index.tz_convert(self.exchange_tz)
        hhmm = idx_local.hour * 100 + idx_local.minute

        session = np.where(
            (hhmm >= 400) & (hhmm < 930), 'premarket',
            np.where(
                (hhmm >= 930) & (hhmm < 1600), 'regular',
                np.where((hhmm >= 1600) & (hhmm < 2000), 'postmarket', 'other')
            )
        )
        return pd.Series(session, index=dt_index)

    def _max_yf_lookback_days(self):
        if not self._is_intraday():
            return None
        if self.interval in ['60m', '1h']:
            return 730
        if self.interval == '1m':
            return 7
        return 60

    def _apply_session_filter(self, df):
        # custom 시간 범위
        if self.session_mode == 'custom':
            start_str, end_str = self.custom_session
            sh, sm = map(int, start_str.split(':'))
            eh, em = map(int, end_str.split(':'))
            start_hm = sh * 100 + sm
            end_hm = eh * 100 + em

            idx_local = df.index.tz_convert(self.exchange_tz)
            hhmm = idx_local.hour * 100 + idx_local.minute

            if start_hm <= end_hm:
                mask = (hhmm >= start_hm) & (hhmm < end_hm)
            else:
                # 자정을 넘는 범위 (예: 22:00 ~ 04:00)
                mask = (hhmm >= start_hm) | (hhmm < end_hm)

            return df[mask].copy()

        # 사전 정의 모드
        if 'session' not in df.columns:
            return df

        if self.session_mode == 'extended':
            return df[df['session'].isin(['premarket', 'regular', 'postmarket'])].copy()
        elif self.session_mode == 'regular':
            return df[df['session'] == 'regular'].copy()
        elif self.session_mode == 'premarket':
            return df[df['session'] == 'premarket'].copy()
        elif self.session_mode == 'postmarket':
            return df[df['session'] == 'postmarket'].copy()

        return df

    # ─────────────────────────────────────────
    # Data loader: yfinance / binance
    # ─────────────────────────────────────────
    def fetch_data(self):
        interval_map_binance = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
            '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1d'
        }

        interval_map_yf = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '90m': '90m',
            '1h': '60m',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo'
        }

        if self.interval not in set(interval_map_binance.keys()).union(interval_map_yf.keys()):
            raise ValueError(f"지원하지 않는 interval: {self.interval}")

        buffer_days = self._buffer_days()
        buffer_start = self.start_date - timedelta(days=buffer_days)
        end_dt = self.end_date + timedelta(days=1)

        # source auto 판별
        if self.source == 'auto':
            source = 'binance' if self._is_crypto_symbol() else 'yfinance'
        else:
            source = self.source.lower()

        self.resolved_source = source

        session_label = self.session_mode
        if self.session_mode == 'custom' and self.custom_session:
            session_label = f"custom({self.custom_session[0]}~{self.custom_session[1]})"
        print(f"[데이터 로드] {self.symbol} | source={source} | interval={self.interval} | session_mode={session_label}")

        # ─────────────────────────────────────
        # Binance
        # ─────────────────────────────────────
        if source == 'binance':
            if self.interval not in interval_map_binance:
                raise ValueError(f"바이낸스에서 지원하지 않는 interval: {self.interval}")

            all_data = []
            start_ms = int(buffer_start.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            while start_ms < end_ms:
                params = {
                    'symbol': self.symbol,
                    'interval': interval_map_binance[self.interval],
                    'startTime': start_ms,
                    'endTime': end_ms,
                    'limit': 1000
                }

                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params=params,
                    timeout=10
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

            df = pd.DataFrame(all_data, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_vol', 'trades_count',
                'taker_buy_vol', 'taker_buy_quote', 'ignore'
            ])

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)

            df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df.set_index('datetime', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['session'] = 'crypto_24x7'

            # 코인은 항상 UTC tz-aware로 통일
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            if self.session_mode == 'custom':
                df = self._apply_session_filter(df)

        # ─────────────────────────────────────
        # yfinance
        # ─────────────────────────────────────
        elif source == 'yfinance':
            if self.interval not in interval_map_yf:
                raise ValueError(f"yfinance에서 지원하지 않는 interval: {self.interval}")

            yf_start = buffer_start
            yf_end = end_dt

            # interval별 Yahoo lookback 제한 반영
            if self._is_intraday():
                max_days = self._max_yf_lookback_days()
                if max_days is not None:
                    max_lookback_start = yf_end - timedelta(days=max_days)
                    if yf_start < max_lookback_start:
                        print(f"  [주의] yfinance {self.interval} 데이터는 최근 약 {max_days}일 범위로 자동 조정합니다.")
                        yf_start = max_lookback_start

            # intraday는 pre/post 포함으로 받아놓고 session_mode로 필터링
            prepost = self._is_intraday()

            df = yf.download(
                self.symbol,
                start=yf_start.strftime('%Y-%m-%d'),
                end=yf_end.strftime('%Y-%m-%d'),
                interval=interval_map_yf[self.interval],
                auto_adjust=True,
                progress=False,
                prepost=prepost
            )

            if df is None or len(df) == 0:
                raise ValueError(f"주식/ETF 데이터를 가져올 수 없습니다: {self.symbol}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            needed_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in needed_cols:
                if col not in df.columns:
                    raise ValueError(f"필수 컬럼 누락: {col}")

            df = df[needed_cols].copy()

            if self._is_intraday():
                if df.index.tz is None:
                    df.index = df.index.tz_localize(self.exchange_tz)
                else:
                    df.index = df.index.tz_convert(self.exchange_tz)

                df['session'] = self._classify_us_session(df.index)
                df = self._apply_session_filter(df)
            else:
                df['session'] = 'daily'

        else:
            raise ValueError("source는 'auto', 'yfinance', 'binance' 중 하나여야 합니다.")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        if len(df) == 0:
            raise ValueError("세션 필터 적용 후 데이터가 없습니다. session_mode/custom_session 또는 기간을 확인하세요.")

        self.raw_df = df.copy()
        print(f"\n  완료: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")
        return self

    # ─────────────────────────────────────────
    # Indicator
    # ─────────────────────────────────────────
    def calculate_indicators(self):
        if self.raw_df is None or len(self.raw_df) == 0:
            raise ValueError("먼저 fetch_data()를 실행하세요.")

        df = self.raw_df.copy()

        df['prev_close'] = df['Close'].shift(1)

        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['prev_close']).abs()
        tr3 = (df['Low'] - df['prev_close']).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Original Turtle: N = 20EMA(TR)
        df['N_raw'] = df['TR'].ewm(alpha=1/20, adjust=False).mean()
        df['N'] = df['N_raw'].shift(1)

        # System 1: 20-bar breakout / 10-bar exit
        df['s1_entry_high'] = df['High'].rolling(20).max().shift(1)
        df['s1_entry_low'] = df['Low'].rolling(20).min().shift(1)
        df['s1_exit_low'] = df['Low'].rolling(10).min().shift(1)
        df['s1_exit_high'] = df['High'].rolling(10).max().shift(1)

        # System 2: 55-bar breakout / 20-bar exit
        df['s2_entry_high'] = df['High'].rolling(55).max().shift(1)
        df['s2_entry_low'] = df['Low'].rolling(55).min().shift(1)
        df['s2_exit_low'] = df['Low'].rolling(20).min().shift(1)
        df['s2_exit_high'] = df['High'].rolling(20).max().shift(1)

        # 실제 분석 구간만 잘라서 self.df에 저장
        if df.index.tz is not None:
            # tz-aware (intraday yfinance, custom 모드의 binance 등)
            if self.resolved_source == 'binance':
                # 코인은 UTC 기준으로 비교
                start_ts = pd.Timestamp(self.start_date).tz_localize('UTC') \
                    if pd.Timestamp(self.start_date).tzinfo is None \
                    else pd.Timestamp(self.start_date)
                end_ts = pd.Timestamp(self.end_date + timedelta(days=1)).tz_localize('UTC') \
                    if pd.Timestamp(self.end_date).tzinfo is None \
                    else pd.Timestamp(self.end_date + timedelta(days=1))
            else:
                start_ts = self._to_exchange_tz(self.start_date)
                end_ts = self._to_exchange_tz(self.end_date + timedelta(days=1))

            self.df = df[(df.index >= start_ts) & (df.index < end_ts)].copy()
        else:
            # tz-naive (일봉 등)
            self.df = df.loc[self.start_date:self.end_date].copy()

        if len(self.df) == 0:
            raise ValueError("분석 기간에 데이터가 없습니다.")

        print(f"  지표 계산 완료: {self.df.index[0]} ~ {self.df.index[-1]} ({len(self.df)}봉)")
        return self

    # ─────────────────────────────────────────
    # Trading actions
    # ─────────────────────────────────────────
    def _enter_position(self, date, side, system_name, breakout_price, N, close_price):
        equity = self._equity(close_price)
        qty = self._unit_size(equity, N, breakout_price)

        if qty <= 0:
            self._log(date, 'SKIP', 'unit size <= 0', system=system_name, N=N, equity=equity)
            return False

        fill = self._entry_fill(breakout_price, side)
        signed_qty = side * qty

        self.cash -= signed_qty * fill
        fee = self._apply_fee(signed_qty * fill)

        self.units = [{'qty': signed_qty, 'price': fill}]
        self.position_side = side
        self.current_system = system_name
        self.current_unit_qty = qty
        self.entry_N = N
        self.next_add_price = fill + side * (0.5 * N)
        self.stop_price = fill - side * (2.0 * N)

        self.trades.append({
            'date': date,
            'type': 'ENTRY',
            'system': system_name,
            'side': 'LONG' if side == 1 else 'SHORT',
            'price': fill,
            'qty': qty,
            'units_after': len(self.units),
            'fee': fee,
            'N': N
        })

        self._log(
            date=date,
            event='ENTRY',
            reason=f'{system_name} breakout',
            side='LONG' if side == 1 else 'SHORT',
            breakout_price=breakout_price,
            fill_price=fill,
            qty=qty,
            N=N,
            stop_price=self.stop_price,
            next_add_price=self.next_add_price
        )
        return True

    def _add_unit(self, date, add_price):
        if len(self.units) >= self.max_units:
            return False

        fill = self._entry_fill(add_price, self.position_side)
        signed_qty = self.position_side * self.current_unit_qty

        self.cash -= signed_qty * fill
        fee = self._apply_fee(signed_qty * fill)

        self.units.append({'qty': signed_qty, 'price': fill})

        self.stop_price = fill - self.position_side * (2.0 * self.entry_N)
        self.next_add_price = fill + self.position_side * (0.5 * self.entry_N)

        self.trades.append({
            'date': date,
            'type': 'ADD',
            'system': self.current_system,
            'side': 'LONG' if self.position_side == 1 else 'SHORT',
            'price': fill,
            'qty': self.current_unit_qty,
            'units_after': len(self.units),
            'fee': fee,
            'N': self.entry_N
        })

        self._log(
            date=date,
            event='ADD',
            reason='0.5N pyramid',
            fill_price=fill,
            qty=self.current_unit_qty,
            units_after=len(self.units),
            stop_price=self.stop_price,
            next_add_price=self.next_add_price
        )
        return True

    def _exit_all(self, date, exit_price, reason, close_price):
        if not self.units:
            return

        signed_total = self._net_position_qty()
        fill = self._exit_fill(exit_price, self.position_side)

        self.cash += signed_total * fill
        fee = self._apply_fee(signed_total * fill)

        gross_pnl = sum((fill - u['price']) * u['qty'] for u in self.units)
        net_pnl = gross_pnl - fee

        self.trades.append({
            'date': date,
            'type': 'EXIT',
            'system': self.current_system,
            'side': 'LONG' if self.position_side == 1 else 'SHORT',
            'price': fill,
            'qty': abs(signed_total),
            'units_before': len(self.units),
            'gross_pnl': gross_pnl,
            'pnl': net_pnl,
            'fee': fee,
            'reason': reason
        })

        self._log(
            date=date,
            event='EXIT',
            reason=reason,
            fill_price=fill,
            gross_pnl=gross_pnl,
            pnl=net_pnl,
            units_before=len(self.units),
            close_price=close_price
        )

        # S1 skip rule 근사 구현
        if self.current_system == 'S1':
            if reason == 'STOP':
                self.skip_next_s1 = False
            else:
                self.skip_next_s1 = (gross_pnl > 0)

        self.units = []
        self.position_side = 0
        self.current_system = None
        self.current_unit_qty = 0.0
        self.entry_N = np.nan
        self.next_add_price = np.nan
        self.stop_price = np.nan

    # ─────────────────────────────────────────
    # Backtest
    # ─────────────────────────────────────────
    def run_backtest(self):
        if self.df is None or len(self.df) == 0:
            raise ValueError("먼저 calculate_indicators()를 실행하세요.")

        df = self.df.copy()

        # reset state
        self.cash = self.initial_capital
        self.units = []
        self.position_side = 0
        self.current_system = None
        self.current_unit_qty = 0.0
        self.entry_N = np.nan
        self.next_add_price = np.nan
        self.stop_price = np.nan
        self.skip_next_s1 = False

        self.trades = []
        self.debug_logs = []
        self.equity_curve = []

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]

            open_ = row['Open']
            high = row['High']
            low = row['Low']
            close = row['Close']
            N = row['N']

            needed = [
                row['s1_entry_high'], row['s1_entry_low'],
                row['s1_exit_low'], row['s1_exit_high'],
                row['s2_entry_high'], row['s2_entry_low'],
                row['s2_exit_low'], row['s2_exit_high'],
                N
            ]

            if any(pd.isna(x) for x in needed):
                self.equity_curve.append(self._equity(close))
                continue

            # 1) 기존 포지션 관리
            if self.position_side != 0:
                # stop 우선
                if self.position_side == 1 and low <= self.stop_price:
                    self._exit_all(date, self.stop_price, 'STOP', close)
                elif self.position_side == -1 and high >= self.stop_price:
                    self._exit_all(date, self.stop_price, 'STOP', close)

                # channel exit
                if self.position_side != 0:
                    if self.current_system == 'S1':
                        exit_level = row['s1_exit_low'] if self.position_side == 1 else row['s1_exit_high']
                    else:
                        exit_level = row['s2_exit_low'] if self.position_side == 1 else row['s2_exit_high']

                    if self.position_side == 1 and low <= exit_level:
                        self._exit_all(date, exit_level, 'CHANNEL_EXIT', close)
                    elif self.position_side == -1 and high >= exit_level:
                        self._exit_all(date, exit_level, 'CHANNEL_EXIT', close)

                # 피라미딩
                while self.position_side != 0 and len(self.units) < self.max_units:
                    if self.position_side == 1 and high >= self.next_add_price:
                        self._add_unit(date, self.next_add_price)
                    elif self.position_side == -1 and low <= self.next_add_price:
                        self._add_unit(date, self.next_add_price)
                    else:
                        break

            # 2) 신규 진입
            if self.position_side == 0:
                # System 1
                s1_side, s1_level = self._choose_breakout_side(
                    open_, row['s1_entry_high'], row['s1_entry_low'], high, low
                )

                if s1_side != 0 and (self.allow_short or s1_side == 1):
                    if self.skip_next_s1:
                        self._log(
                            date=date,
                            event='SKIP',
                            reason='skip next S1 after prior S1 winner',
                            side='LONG' if s1_side == 1 else 'SHORT',
                            level=s1_level
                        )
                        self.skip_next_s1 = False
                    else:
                        self._enter_position(date, s1_side, 'S1', s1_level, N, close)

                # System 2 failsafe
                if self.position_side == 0:
                    s2_side, s2_level = self._choose_breakout_side(
                        open_, row['s2_entry_high'], row['s2_entry_low'], high, low
                    )

                    if s2_side != 0 and (self.allow_short or s2_side == 1):
                        self._enter_position(date, s2_side, 'S2', s2_level, N, close)

            self.equity_curve.append(self._equity(close))

        df['equity'] = self.equity_curve
        self.df = df
        return self

    # ─────────────────────────────────────────
    # Result
    # ─────────────────────────────────────────
    def print_results(self):
        print("\n" + "=" * 100)
        print(f"Original Turtle Backtest | {self.symbol} ({self.interval})")
        print("S1: 20-bar breakout / 10-bar exit | S2: 55-bar breakout / 20-bar exit")
        print(f"N: 20EMA(TR) | Unit Risk: {self.risk_per_unit*100:.2f}% | Max Units: {self.max_units}")
        print(f"Fee: {self.fee_rate*100:.3f}% | Slippage: {self.slippage_rate*100:.3f}% | Short: {self.allow_short}")

        session_label = self.session_mode
        if self.session_mode == 'custom' and self.custom_session:
            session_label = f"custom({self.custom_session[0]}~{self.custom_session[1]}, tz={self.exchange_tz})"
        print(f"Source: {self.resolved_source} | Session Mode: {session_label}")
        print("=" * 100)

        if self.df is None or len(self.df) == 0:
            print("결과 데이터가 없습니다.")
            return

        final_equity = float(self.equity_curve[-1])
        total_return = (final_equity / self.initial_capital - 1) * 100

        eq = pd.Series(self.equity_curve, index=self.df.index, dtype=float)
        mdd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

        trades_df = pd.DataFrame(self.trades)
        exits = trades_df[trades_df['type'] == 'EXIT'].copy() if len(trades_df) else pd.DataFrame()
        entries = trades_df[trades_df['type'] == 'ENTRY'].copy() if len(trades_df) else pd.DataFrame()

        print(f"\n초기 자본:      {self.initial_capital:>15,.2f}")
        print(f"최종 자산:      {final_equity:>15,.2f}")
        print(f"총 수익률:      {total_return:>14.2f} %")
        print(f"최대낙폭 MDD:   {mdd:>14.2f} %")

        first_close = self.df.iloc[0]['Close']
        last_close = self.df.iloc[-1]['Close']
        bnh_return = (last_close / first_close - 1) * 100
        print(f"Buy & Hold:     {bnh_return:>14.2f} %")

        if len(exits) > 0:
            wins = exits[exits['pnl'] > 0]
            losses = exits[exits['pnl'] <= 0]

            print(f"\n총 진입:        {len(entries)}회")
            print(f"총 청산:        {len(exits)}회")
            print(f"승률:           {len(wins) / max(len(exits), 1) * 100:>14.2f} %")

            if len(wins) > 0:
                print(f"평균 수익:      {wins['pnl'].mean():>15,.2f}")
            if len(losses) > 0:
                print(f"평균 손실:      {losses['pnl'].mean():>15,.2f}")

            total_win = wins['pnl'].sum() if len(wins) else 0.0
            total_loss = losses['pnl'].sum() if len(losses) else 0.0
            if total_loss != 0:
                pf = abs(total_win / total_loss)
                print(f"Profit Factor:  {pf:>14.2f}")

            if 'reason' in exits.columns:
                by_reason = exits.groupby('reason')['pnl'].agg(['count', 'sum'])
                print("\n청산 사유별:")
                print(by_reason)

            if 'system' in exits.columns:
                by_system = exits.groupby('system')['pnl'].agg(['count', 'sum'])
                print("\n시스템별:")
                print(by_system)

        print("\n" + "=" * 100)

    # ─────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────
    def plot_equity(self):
        if self.df is None or 'equity' not in self.df.columns:
            print("먼저 run_backtest()를 실행하세요.")
            return

        plt.figure(figsize=(14, 5))
        plt.plot(self.df.index, self.df['equity'], label='Turtle Equity', linewidth=1.2)
        plt.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        plt.title(f'Original Turtle Equity Curve - {self.symbol} ({self.interval}, {self.session_mode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
