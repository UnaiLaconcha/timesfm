import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=1000.0, fee_rate=0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.trades = []

    def run_backtest(self, df: pd.DataFrame, model_predictor, horizon_len=24, stop_loss_pct=0.02, threshold_pct=0.01, step_size=6, take_profit_pct=0.04, overlapping=False, max_positions=5, trailing_sl=False, trailing_sl_pct=0.02, break_even=False, break_even_trigger_pct=0.015, dynamic_sizing=False, confidence_multiplier=1.5, uncertainty_filter=False, max_uncertainty_pct=0.05, adaptive_sl=False, volatility_multiplier=2.0, trade_direction="Long & Short", progress_callback=None, symbol_name=""):
        """
        Runs a walk-forward backtest using TimesFM predictions for a single asset.
        trade_direction: 'Long & Short', 'Solo Long', or 'Solo Short'. Default is 'Long & Short'.
        """
        context_len = model_predictor.context_len
        if len(df) < context_len + horizon_len:
            raise ValueError("Dataset is too small for the given context_len and horizon_len.")

        if overlapping:
            return self._run_backtest_overlapping(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier, trade_direction, progress_callback=progress_callback, symbol_name=symbol_name)
        else:
            return self._run_backtest_single(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier, trade_direction, progress_callback=progress_callback, symbol_name=symbol_name)

    def run_portfolio_backtest(self, df_dict: dict, model_predictor, horizon_len=24, stop_loss_pct=0.02, threshold_pct=0.01, step_size=6, take_profit_pct=0.04, overlapping=False, max_positions=5, trailing_sl=False, trailing_sl_pct=0.02, break_even=False, break_even_trigger_pct=0.015, dynamic_sizing=False, confidence_multiplier=1.5, uncertainty_filter=False, max_uncertainty_pct=0.05, adaptive_sl=False, volatility_multiplier=2.0, trade_direction="Long & Short", progress_callback=None):
        """
        Runs a multi-asset portfolio backtest. Proportionally divides initial capital among assets,
        executes strategy per asset, and consolidates global portfolio metrics & equity curve.
        """
        if not df_dict:
            raise ValueError("df_dict is empty. Provide at least one asset DataFrame.")

        num_assets = len(df_dict)
        asset_initial_cap = self.initial_capital / num_assets
        
        asset_results = {}
        all_trades_list = []
        total_filtered = 0
        equity_series_list = []

        for asset_idx, (symbol, df_asset) in enumerate(df_dict.items()):
            def make_asset_cb(a_idx, a_sym):
                def asset_cb(curr, tot, msg):
                    if progress_callback:
                        sub_pct = (curr / max(1, tot))
                        overall_step = a_idx + sub_pct
                        progress_callback(overall_step, num_assets, f"Evaluando {a_sym} ({a_idx+1}/{num_assets}): {msg}")
                return asset_cb

            asset_engine = BacktestEngine(initial_capital=asset_initial_cap, fee_rate=self.fee_rate)
            res = asset_engine.run_backtest(
                df_asset, model_predictor, horizon_len=horizon_len, stop_loss_pct=stop_loss_pct,
                threshold_pct=threshold_pct, step_size=step_size, take_profit_pct=take_profit_pct,
                overlapping=overlapping, max_positions=max_positions, trailing_sl=trailing_sl,
                trailing_sl_pct=trailing_sl_pct, break_even=break_even, break_even_trigger_pct=break_even_trigger_pct,
                dynamic_sizing=dynamic_sizing, confidence_multiplier=confidence_multiplier,
                uncertainty_filter=uncertainty_filter, max_uncertainty_pct=max_uncertainty_pct,
                adaptive_sl=adaptive_sl, volatility_multiplier=volatility_multiplier, trade_direction=trade_direction,
                progress_callback=make_asset_cb(asset_idx, symbol), symbol_name=symbol
            )
            
            # Tag trades with asset symbol
            trades_df = res['trades'].copy()
            if not trades_df.empty:
                trades_df['symbol'] = symbol
                all_trades_list.append(trades_df)
                
            total_filtered += res['metrics'].get('filtered_trades', 0)
            asset_results[symbol] = res
            equity_series_list.append(res['equity_df']['equity'].rename(symbol))

        # Combine equity curves across aligned timestamps
        combined_equity_df = pd.concat(equity_series_list, axis=1).ffill().bfill()
        portfolio_equity = combined_equity_df.sum(axis=1)
        portfolio_equity_df = pd.DataFrame({'equity': portfolio_equity})

        # Combined trade log
        if all_trades_list:
            combined_trades_df = pd.concat(all_trades_list, ignore_index=True).sort_values('time').reset_index(drop=True)
        else:
            combined_trades_df = pd.DataFrame(columns=['time', 'type', 'price', 'capital', 'confidence', 'pnl_pct', 'pnl_usd', 'symbol'])

        # Calculate consolidated portfolio metrics
        raw_trades_dicts = combined_trades_df.to_dict('records') if not combined_trades_df.empty else []
        final_portfolio_cap = portfolio_equity.iloc[-1] if not portfolio_equity.empty else self.initial_capital
        portfolio_metrics = self._calculate_institutional_metrics(portfolio_equity_df, raw_trades_dicts, final_portfolio_cap, step_size, total_filtered)

        return {
            'equity_df': portfolio_equity_df,
            'trades': combined_trades_df,
            'metrics': portfolio_metrics,
            'asset_results': asset_results,
            'combined_equity_grid': combined_equity_df
        }

    def _precompute_forecasts(self, df, model_predictor, context_len, horizon_len, step_size, progress_callback=None, symbol_name=""):
        """Precompute batch forecasts for ultra-fast walk-forward iteration."""
        step_indices = list(range(context_len, len(df) - horizon_len, step_size))
        total_steps = len(step_indices)
        if not step_indices:
            return {}
        
        forecast_map = {}
        if hasattr(model_predictor, 'predict_batch'):
            try:
                contexts = [df['close'].iloc[idx - context_len : idx].values.astype(np.float32) for idx in step_indices]
                batch_size = 64
                for b_start in range(0, total_steps, batch_size):
                    b_end = min(b_start + batch_size, total_steps)
                    b_contexts = contexts[b_start:b_end]
                    pt_batch, qt_batch = model_predictor.predict_batch(b_contexts)
                    for k in range(len(b_contexts)):
                        forecast_map[step_indices[b_start + k]] = (pt_batch[k], qt_batch[k] if len(qt_batch) > k else None)
                    
                    if progress_callback:
                        msg = f"Inferencia batch TimesFM ({b_end}/{total_steps} velas)"
                        progress_callback(b_end, total_steps, msg)
                return forecast_map
            except Exception:
                pass
        return {}

    def _run_backtest_single(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier, trade_direction, progress_callback=None, symbol_name=""):
        """Single-position backtest logic supporting trade direction filtering."""
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
        forecast_cache = self._precompute_forecasts(df, model_predictor, context_len, horizon_len, step_size, progress_callback=progress_callback, symbol_name=symbol_name)

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
            
            # Check direction filtering
            is_long_signal = expected_move > threshold_pct and trade_direction in ["Long & Short", "Solo Long"]
            is_short_signal = expected_move < -threshold_pct and trade_direction in ["Long & Short", "Solo Short"]

            if is_long_signal or is_short_signal:
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

                if is_long_signal:
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
                elif is_short_signal:
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

    def _run_backtest_overlapping(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions, trailing_sl, trailing_sl_pct, break_even, break_even_trigger_pct, dynamic_sizing, confidence_multiplier, uncertainty_filter, max_uncertainty_pct, adaptive_sl, volatility_multiplier, trade_direction, progress_callback=None, symbol_name=""):
        """Overlapping-positions backtest supporting trade direction filtering."""
        available_capital = self.initial_capital
        active_positions = []
        filtered_count = 0
        equity_curve = []
        forecast_cache = self._precompute_forecasts(df, model_predictor, context_len, horizon_len, step_size, progress_callback=progress_callback, symbol_name=symbol_name)

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

                is_long_signal = expected_move > threshold_pct and trade_direction in ["Long & Short", "Solo Long"]
                is_short_signal = expected_move < -threshold_pct and trade_direction in ["Long & Short", "Solo Short"]

                if is_long_signal or is_short_signal:
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

                    direction = 1 if is_long_signal else -1
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

        closed_trades = [t for t in trades_list if isinstance(t, dict) and 'pnl_usd' in t and pd.notna(t['pnl_usd'])]
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
