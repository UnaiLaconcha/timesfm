import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=1000.0, fee_rate=0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.trades = []

    def run_backtest(self, df: pd.DataFrame, model_predictor, horizon_len=24, stop_loss_pct=0.02, threshold_pct=0.01, step_size=6, take_profit_pct=0.04, overlapping=False, max_positions=5, trailing_sl=False, trailing_sl_pct=0.02, break_even=False, break_even_trigger_pct=0.015, dynamic_sizing=False, confidence_multiplier=1.5, uncertainty_filter=False, max_uncertainty_pct=0.05, adaptive_sl=False, volatility_multiplier=2.0):
        """
        Runs a walk-forward backtest using TimesFM predictions.
        df: DataFrame with at least 'close' prices.
        model_predictor: initialized TimesFMPredictor.
        stop_loss_pct: 0.02 means 2% initial stop loss.
        take_profit_pct: 0.04 means 4% take profit. Set to 0 to disable.
        threshold_pct: 0.01 means 1% expected move to enter trade.
        step_size: step size for walk-forward evaluation.
        overlapping: if True, allow multiple simultaneous positions.
        max_positions: maximum number of simultaneous positions (only used when overlapping=True).
        trailing_sl: if True, use dynamic Trailing Stop Loss instead of static TP.
        trailing_sl_pct: distance percentage for trailing stop loss (e.g., 0.02 for 2%).
        break_even: if True, move stop loss to entry price once trigger profit is reached.
        break_even_trigger_pct: profit percentage required to trigger Break-Even (e.g., 0.015 for 1.5%).
        dynamic_sizing: if True, scale position size based on quantile prediction confidence.
        confidence_multiplier: multiplier factor for high-confidence trades (e.g., 1.5 for 150%).
        uncertainty_filter: if True, skip entries when prediction quantile spread exceeds max_uncertainty_pct.
        max_uncertainty_pct: maximum allowed spread between q90 and q10 relative to price (e.g., 0.05 for 5%).
        adaptive_sl: if True, dynamically scale stop loss percentage based on recent market return volatility stddev.
        volatility_multiplier: multiplier factor k for stddev volatility SL (e.g., 2.0 for 2*stddev).
        """
        context_len = model_predictor.context_len
        if len(df) < context_len + horizon_len:
            raise ValueError("Dataset is too small for the given context_len and horizon_len.")

        if overlapping:
            return self._run_backtest_overlapping(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier)
        else:
            return self._run_backtest_single(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier)

    def _precompute_forecasts(self, df, model_predictor, context_len, horizon_len, step_size):
        """Precompute batch forecasts for ultra-fast walk-forward iteration."""
        step_indices = list(range(context_len, len(df) - horizon_len, step_size))
        if not step_indices:
            return {}
        
        if hasattr(model_predictor, 'predict_batch'):
            try:
                contexts = [df['close'].iloc[idx - context_len : idx].values.astype(np.float32) for idx in step_indices]
                pt_batch, qt_batch = model_predictor.predict_batch(contexts)
                return {step_indices[k]: (pt_batch[k], qt_batch[k] if len(qt_batch) > k else None) for k in range(len(step_indices))}
            except Exception:
                pass
        return {}

    def _run_backtest_single(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier):
        """Single-position backtest logic supporting all risk management modes and institutional metrics."""
        capital = self.initial_capital
        position = 0 # 0: flat, 1: long, -1: short
        entry_price = 0
        entry_cap_allocated = 0
        extreme_price = 0 # peak for Long, trough for Short
        be_activated = False
        size_multiplier = 1.0
        active_sl_pct = stop_loss_pct
        filtered_count = 0
        
        equity_curve = []
        forecast_cache = self._precompute_forecasts(df, model_predictor, context_len, horizon_len, step_size)

        for i in range(context_len, len(df) - horizon_len, step_size):
            current_idx = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Record equity
            current_equity = capital
            if position == 1:
                current_equity = capital * (1 + size_multiplier * (current_price - entry_price) / entry_price)
            elif position == -1:
                current_equity = capital * (1 + size_multiplier * (entry_price - current_price) / entry_price)
            
            equity_curve.append((current_idx, current_equity))
            
            # Helper to record closed trade with PnL
            def record_close(trade_type, close_p):
                nonlocal capital, position
                pnl_p = size_multiplier * (close_p - entry_price) / entry_price if position == 1 else size_multiplier * (entry_price - close_p) / entry_price
                pnl_u = entry_cap_allocated * pnl_p
                capital = current_equity * (1 - self.fee_rate)
                self.trades.append({
                    'time': current_idx, 'type': trade_type, 'price': close_p,
                    'capital': capital, 'confidence': 'High' if size_multiplier > 1.0 else 'Standard',
                    'pnl_pct': pnl_p, 'pnl_usd': pnl_u
                })
                position = 0

            # Check Take Profit & Stop Loss & Break-Even
            if position == 1:
                if break_even and not be_activated:
                    if current_price >= entry_price * (1 + break_even_trigger_pct):
                        be_activated = True

                base_sl = entry_price if be_activated else entry_price * (1 - active_sl_pct)

                if trailing_sl:
                    extreme_price = max(extreme_price, current_price)
                    dynamic_sl = max(base_sl, extreme_price * (1 - trailing_sl_pct))
                    if current_price <= dynamic_sl:
                        if be_activated and dynamic_sl == entry_price:
                            tt = 'CLOSE_LONG_BE'
                        elif dynamic_sl > base_sl:
                            tt = 'CLOSE_LONG_TSL'
                        else:
                            tt = 'CLOSE_LONG_SL'
                        record_close(tt, current_price)
                        continue
                else:
                    if take_profit_pct > 0 and current_price >= entry_price * (1 + take_profit_pct):
                        record_close('CLOSE_LONG_TP', current_price)
                        continue
                    elif current_price <= base_sl:
                        tt = 'CLOSE_LONG_BE' if be_activated else 'CLOSE_LONG_SL'
                        record_close(tt, current_price)
                        continue

            elif position == -1:
                if break_even and not be_activated:
                    if current_price <= entry_price * (1 - break_even_trigger_pct):
                        be_activated = True

                base_sl = entry_price if be_activated else entry_price * (1 + active_sl_pct)

                if trailing_sl:
                    extreme_price = min(extreme_price, current_price)
                    dynamic_sl = min(base_sl, extreme_price * (1 + trailing_sl_pct))
                    if current_price >= dynamic_sl:
                        if be_activated and dynamic_sl == entry_price:
                            tt = 'CLOSE_SHORT_BE'
                        elif dynamic_sl < base_sl:
                            tt = 'CLOSE_SHORT_TSL'
                        else:
                            tt = 'CLOSE_SHORT_SL'
                        record_close(tt, current_price)
                        continue
                else:
                    if take_profit_pct > 0 and current_price <= entry_price * (1 - take_profit_pct):
                        record_close('CLOSE_SHORT_TP', current_price)
                        continue
                    elif current_price >= base_sl:
                        tt = 'CLOSE_SHORT_BE' if be_activated else 'CLOSE_SHORT_SL'
                        record_close(tt, current_price)
                        continue
            
            # If we are already in a position, we do not evaluate new entries
            if position != 0:
                continue

            # Get forecast from precomputed batch or fallback to single prediction
            if i in forecast_cache:
                point_fc, quant_fc = forecast_cache[i]
            else:
                context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
                point_fc, quant_fc = model_predictor.predict(context_data)
                
            expected_price = point_fc[-1]
            expected_move = (expected_price - current_price) / current_price
            
            # Decision logic (only for position == 0)
            if expected_move > threshold_pct or expected_move < -threshold_pct:
                if uncertainty_filter and quant_fc is not None:
                    q10_price = quant_fc[-1, 0]
                    q90_price = quant_fc[-1, 8] if quant_fc.shape[1] > 8 else quant_fc[-1, -1]
                    uncertainty_spread = (q90_price - q10_price) / current_price
                    if uncertainty_spread > max_uncertainty_pct:
                        filtered_count += 1
                        continue

                active_sl_pct = stop_loss_pct
                if adaptive_sl:
                    context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
                    returns = np.diff(context_data) / context_data[:-1]
                    vol_std = np.std(returns) if len(returns) > 0 else 0.0
                    calc_sl = vol_std * float(volatility_multiplier)
                    active_sl_pct = max(stop_loss_pct, calc_sl)

                if expected_move > threshold_pct:
                    position = 1
                    entry_price = current_price
                    extreme_price = current_price
                    be_activated = False
                    size_multiplier = 1.0
                    
                    if dynamic_sizing and quant_fc is not None:
                        q10_price = quant_fc[-1, 0]
                        if q10_price > current_price:
                            size_multiplier = float(confidence_multiplier)

                    capital *= (1 - self.fee_rate)
                    entry_cap_allocated = capital
                    self.trades.append({'time': current_idx, 'type': 'ENTER_LONG', 'price': current_price, 'capital': capital, 'confidence': 'High' if size_multiplier > 1.0 else 'Standard'})
                elif expected_move < -threshold_pct:
                    position = -1
                    entry_price = current_price
                    extreme_price = current_price
                    be_activated = False
                    size_multiplier = 1.0

                    if dynamic_sizing and quant_fc is not None:
                        q90_price = quant_fc[-1, 8] if quant_fc.shape[1] > 8 else quant_fc[-1, -1]
                        if q90_price < current_price:
                            size_multiplier = float(confidence_multiplier)

                    capital *= (1 - self.fee_rate)
                    entry_cap_allocated = capital
                    self.trades.append({'time': current_idx, 'type': 'ENTER_SHORT', 'price': current_price, 'capital': capital, 'confidence': 'High' if size_multiplier > 1.0 else 'Standard'})

        # Close any open positions at the end
        final_price = df['close'].iloc[-1]
        final_idx = df.index[-1]
        if position == 1:
            pnl_p = size_multiplier * (final_price - entry_price) / entry_price
            pnl_u = entry_cap_allocated * pnl_p
            capital = capital * (1 + pnl_p) * (1 - self.fee_rate)
            self.trades.append({'time': final_idx, 'type': 'CLOSE_LONG_END', 'price': final_price, 'capital': capital, 'confidence': 'High' if size_multiplier > 1.0 else 'Standard', 'pnl_pct': pnl_p, 'pnl_usd': pnl_u})
        elif position == -1:
            pnl_p = size_multiplier * (entry_price - final_price) / entry_price
            pnl_u = entry_cap_allocated * pnl_p
            capital = capital * (1 + pnl_p) * (1 - self.fee_rate)
            self.trades.append({'time': final_idx, 'type': 'CLOSE_SHORT_END', 'price': final_price, 'capital': capital, 'confidence': 'High' if size_multiplier > 1.0 else 'Standard', 'pnl_pct': pnl_p, 'pnl_usd': pnl_u})

        equity_curve.append((final_idx, capital))
        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        
        metrics = self._calculate_institutional_metrics(equity_df, self.trades, capital, step_size, filtered_count)
        
        return {
            'equity_df': equity_df,
            'trades': pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame(columns=['time', 'type', 'price', 'capital', 'confidence', 'pnl_pct', 'pnl_usd']),
            'metrics': metrics
        }

    def _run_backtest_overlapping(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier):
        """Overlapping-positions backtest with pre-batched forecasting and institutional metrics."""
        available_capital = self.initial_capital
        active_positions = []
        filtered_count = 0
        equity_curve = []
        forecast_cache = self._precompute_forecasts(df, model_predictor, context_len, horizon_len, step_size)

        for i in range(context_len, len(df) - horizon_len, step_size):
            current_idx = df.index[i]
            current_price = df['close'].iloc[i]

            # ── Step 1: Check SL/TP/BE for all active positions ──
            positions_to_close = []
            for pos_idx, pos in enumerate(active_positions):
                pos_sl_pct = pos.get('sl_pct', stop_loss_pct)
                if pos['direction'] == 1:  # Long
                    if break_even and not pos['be_activated']:
                        if current_price >= pos['entry_price'] * (1 + break_even_trigger_pct):
                            pos['be_activated'] = True

                    base_sl = pos['entry_price'] if pos['be_activated'] else pos['entry_price'] * (1 - pos_sl_pct)

                    if trailing_sl:
                        pos['extreme_price'] = max(pos['extreme_price'], current_price)
                        dynamic_sl = max(base_sl, pos['extreme_price'] * (1 - trailing_sl_pct))
                        if current_price <= dynamic_sl:
                            if pos['be_activated'] and dynamic_sl == pos['entry_price']:
                                tt = 'CLOSE_LONG_BE'
                            elif dynamic_sl > base_sl:
                                tt = 'CLOSE_LONG_TSL'
                            else:
                                tt = 'CLOSE_LONG_SL'
                            positions_to_close.append((pos_idx, tt))
                    else:
                        if take_profit_pct > 0 and current_price >= pos['entry_price'] * (1 + take_profit_pct):
                            positions_to_close.append((pos_idx, 'CLOSE_LONG_TP'))
                        elif current_price <= base_sl:
                            tt = 'CLOSE_LONG_BE' if pos['be_activated'] else 'CLOSE_LONG_SL'
                            positions_to_close.append((pos_idx, tt))

                elif pos['direction'] == -1:  # Short
                    if break_even and not pos['be_activated']:
                        if current_price <= pos['entry_price'] * (1 - break_even_trigger_pct):
                            pos['be_activated'] = True

                    base_sl = pos['entry_price'] if pos['be_activated'] else pos['entry_price'] * (1 + pos_sl_pct)

                    if trailing_sl:
                        pos['extreme_price'] = min(pos['extreme_price'], current_price)
                        dynamic_sl = min(base_sl, pos['extreme_price'] * (1 + trailing_sl_pct))
                        if current_price >= dynamic_sl:
                            if pos['be_activated'] and dynamic_sl == pos['entry_price']:
                                tt = 'CLOSE_SHORT_BE'
                            elif dynamic_sl < base_sl:
                                tt = 'CLOSE_SHORT_TSL'
                            else:
                                tt = 'CLOSE_SHORT_SL'
                            positions_to_close.append((pos_idx, tt))
                    else:
                        if take_profit_pct > 0 and current_price <= pos['entry_price'] * (1 - take_profit_pct):
                            positions_to_close.append((pos_idx, 'CLOSE_SHORT_TP'))
                        elif current_price >= base_sl:
                            tt = 'CLOSE_SHORT_BE' if pos['be_activated'] else 'CLOSE_SHORT_SL'
                            positions_to_close.append((pos_idx, tt))

            # Close positions in reverse order to preserve indices
            for pos_idx, close_type in sorted(positions_to_close, key=lambda x: x[0], reverse=True):
                pos = active_positions.pop(pos_idx)
                size_mult = pos.get('size_multiplier', 1.0)
                if pos['direction'] == 1:
                    pnl_p = size_mult * (current_price - pos['entry_price']) / pos['entry_price']
                else:
                    pnl_p = size_mult * (pos['entry_price'] - current_price) / pos['entry_price']
                pnl_u = pos['entry_capital'] * pnl_p
                realized = pos['entry_capital'] * (1 + pnl_p) * (1 - self.fee_rate)
                available_capital += realized
                self.trades.append({
                    'time': current_idx, 'type': close_type, 'price': current_price,
                    'capital': available_capital + self._total_mtm(active_positions, current_price),
                    'confidence': 'High' if size_mult > 1.0 else 'Standard',
                    'pnl_pct': pnl_p, 'pnl_usd': pnl_u
                })

            # ── Step 2: Record equity ──
            total_equity = available_capital + self._total_mtm(active_positions, current_price)
            equity_curve.append((current_idx, total_equity))

            # ── Step 3: Evaluate new entry if slots available ──
            if len(active_positions) < max_positions and available_capital > 0:
                if i in forecast_cache:
                    point_fc, quant_fc = forecast_cache[i]
                else:
                    context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
                    point_fc, quant_fc = model_predictor.predict(context_data)
                    
                expected_price = point_fc[-1]
                expected_move = (expected_price - current_price) / current_price

                if expected_move > threshold_pct or expected_move < -threshold_pct:
                    if uncertainty_filter and quant_fc is not None:
                        q10_price = quant_fc[-1, 0]
                        q90_price = quant_fc[-1, 8] if quant_fc.shape[1] > 8 else quant_fc[-1, -1]
                        uncertainty_spread = (q90_price - q10_price) / current_price
                        if uncertainty_spread > max_uncertainty_pct:
                            filtered_count += 1
                            continue

                    remaining_slots = max_positions - len(active_positions)
                    alloc_capital = available_capital / remaining_slots
                    alloc_capital_after_fee = alloc_capital * (1 - self.fee_rate)
                    available_capital -= alloc_capital

                    direction = 1 if expected_move > threshold_pct else -1
                    trade_type = 'ENTER_LONG' if direction == 1 else 'ENTER_SHORT'

                    size_multiplier = 1.0
                    if dynamic_sizing and quant_fc is not None:
                        if direction == 1:
                            q10_price = quant_fc[-1, 0]
                            if q10_price > current_price:
                                size_multiplier = float(confidence_multiplier)
                        else:
                            q90_price = quant_fc[-1, 8] if quant_fc.shape[1] > 8 else quant_fc[-1, -1]
                            if q90_price < current_price:
                                size_multiplier = float(confidence_multiplier)

                    active_sl_pct = stop_loss_pct
                    if adaptive_sl:
                        context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
                        returns = np.diff(context_data) / context_data[:-1]
                        vol_std = np.std(returns) if len(returns) > 0 else 0.0
                        calc_sl = vol_std * float(volatility_multiplier)
                        active_sl_pct = max(stop_loss_pct, calc_sl)

                    active_positions.append({
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_capital': alloc_capital_after_fee,
                        'entry_time': current_idx,
                        'extreme_price': current_price,
                        'be_activated': False,
                        'size_multiplier': size_multiplier,
                        'sl_pct': active_sl_pct
                    })

                    self.trades.append({
                        'time': current_idx, 'type': trade_type, 'price': current_price,
                        'capital': available_capital + self._total_mtm(active_positions, current_price),
                        'confidence': 'High' if size_multiplier > 1.0 else 'Standard'
                    })

        # ── Close all remaining positions at end ──
        final_price = df['close'].iloc[-1]
        final_idx = df.index[-1]

        for pos in active_positions:
            close_type = 'CLOSE_LONG_END' if pos['direction'] == 1 else 'CLOSE_SHORT_END'
            size_mult = pos.get('size_multiplier', 1.0)
            if pos['direction'] == 1:
                pnl_p = size_mult * (final_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_p = size_mult * (pos['entry_price'] - final_price) / pos['entry_price']
            pnl_u = pos['entry_capital'] * pnl_p
            realized = pos['entry_capital'] * (1 + pnl_p) * (1 - self.fee_rate)
            available_capital += realized
            self.trades.append({
                'time': final_idx, 'type': close_type, 'price': final_price,
                'capital': available_capital, 'confidence': 'High' if size_mult > 1.0 else 'Standard',
                'pnl_pct': pnl_p, 'pnl_usd': pnl_u
            })
        active_positions.clear()

        equity_curve.append((final_idx, available_capital))
        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        
        metrics = self._calculate_institutional_metrics(equity_df, self.trades, available_capital, step_size, filtered_count)

        return {
            'equity_df': equity_df,
            'trades': pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame(columns=['time', 'type', 'price', 'capital', 'confidence', 'pnl_pct', 'pnl_usd']),
            'metrics': metrics
        }

    def _calculate_institutional_metrics(self, equity_df, trades_list, final_capital, step_size, filtered_count):
        """Compute full institutional-grade risk metrics including Win Rate, Profit Factor, Sortino Ratio and Expectancy."""
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(365 * 24 / step_size) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() != 0 else 0.0

        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(365 * 24 / step_size) * returns.mean() / downside_returns.std() if len(downside_returns) > 1 and downside_returns.std() != 0 else (sharpe_ratio * 1.2 if sharpe_ratio > 0 else 0.0)

        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        closed_trades = [t for t in trades_list if 'pnl_usd' in t]
        total_closed = len(closed_trades)
        winning_trades = [t for t in closed_trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl_usd'] < 0]

        win_rate = len(winning_trades) / total_closed if total_closed > 0 else 0.0
        gross_profit = sum(t['pnl_usd'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl_usd'] for t in losing_trades))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 99.0
        else:
            profit_factor = 0.0

        expectancy = sum(t['pnl_usd'] for t in closed_trades) / total_closed if total_closed > 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_trades': len(trades_list),
            'closed_trades': total_closed,
            'filtered_trades': filtered_count
        }

    @staticmethod
    def _calc_position_value(pos, current_price):
        """Calculate current mark-to-market value of a single position accounting for size multiplier."""
        size_mult = pos.get('size_multiplier', 1.0)
        if pos['direction'] == 1:  # Long
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        else:  # Short
            pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price']
        return pos['entry_capital'] * (1 + size_mult * pnl_pct)

    @staticmethod
    def _total_mtm(active_positions, current_price):
        """Sum mark-to-market value of all active positions."""
        return sum(
            BacktestEngine._calc_position_value(pos, current_price)
            for pos in active_positions
        )
