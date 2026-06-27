import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=1000.0, fee_rate=0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.trades = []

    def run_backtest(self, df: pd.DataFrame, model_predictor, horizon_len=24, stop_loss_pct=0.02, threshold_pct=0.01, step_size=6, take_profit_pct=0.04, overlapping=False, max_positions=5):
        """
        Runs a walk-forward backtest using TimesFM predictions.
        df: DataFrame with at least 'close' prices.
        model_predictor: initialized TimesFMPredictor.
        stop_loss_pct: 0.02 means 2% stop loss.
        take_profit_pct: 0.04 means 4% take profit. Set to 0 to disable.
        threshold_pct: 0.01 means 1% expected move to enter trade.
        step_size: step size for walk-forward evaluation.
        overlapping: if True, allow multiple simultaneous positions.
        max_positions: maximum number of simultaneous positions (only used when overlapping=True).
        """
        context_len = model_predictor.context_len
        if len(df) < context_len + horizon_len:
            raise ValueError("Dataset is too small for the given context_len and horizon_len.")

        if overlapping:
            return self._run_backtest_overlapping(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions)
        else:
            return self._run_backtest_single(df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct)

    def _run_backtest_single(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct):
        """Original single-position backtest logic — unchanged."""
        capital = self.initial_capital
        position = 0 # 0: flat, 1: long, -1: short
        entry_price = 0
        
        equity_curve = []
        
        for i in range(context_len, len(df) - horizon_len, step_size):
            current_idx = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Record equity
            current_equity = capital
            if position == 1:
                current_equity = capital * (1 + (current_price - entry_price) / entry_price)
            elif position == -1:
                current_equity = capital * (1 + (entry_price - current_price) / entry_price)
            
            equity_curve.append((current_idx, current_equity))
            
            # Check Take Profit & Stop Loss
            if position == 1:
                if take_profit_pct > 0 and current_price >= entry_price * (1 + take_profit_pct):
                    # Close Long TP
                    capital = current_equity * (1 - self.fee_rate)
                    self.trades.append({'time': current_idx, 'type': 'CLOSE_LONG_TP', 'price': current_price, 'capital': capital})
                    position = 0
                    continue
                elif current_price <= entry_price * (1 - stop_loss_pct):
                    # Close Long SL
                    capital = current_equity * (1 - self.fee_rate)
                    self.trades.append({'time': current_idx, 'type': 'CLOSE_LONG_SL', 'price': current_price, 'capital': capital})
                    position = 0
                    continue
            elif position == -1:
                if take_profit_pct > 0 and current_price <= entry_price * (1 - take_profit_pct):
                    # Close Short TP
                    capital = current_equity * (1 - self.fee_rate)
                    self.trades.append({'time': current_idx, 'type': 'CLOSE_SHORT_TP', 'price': current_price, 'capital': capital})
                    position = 0
                    continue
                elif current_price >= entry_price * (1 + stop_loss_pct):
                    # Close Short SL
                    capital = current_equity * (1 - self.fee_rate)
                    self.trades.append({'time': current_idx, 'type': 'CLOSE_SHORT_SL', 'price': current_price, 'capital': capital})
                    position = 0
                    continue
            
            # If we are already in a position, we do not evaluate new entries
            if position != 0:
                continue

            # Prepare context
            context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
            
            # Predict
            point_fc, quant_fc = model_predictor.predict(context_data)
            expected_price = point_fc[-1] # End of horizon
            expected_move = (expected_price - current_price) / current_price
            
            # Decision logic (only for position == 0)
            if expected_move > threshold_pct:
                # Enter Long
                position = 1
                entry_price = current_price
                capital *= (1 - self.fee_rate) # Deduct fee on entry
                self.trades.append({'time': current_idx, 'type': 'ENTER_LONG', 'price': current_price, 'capital': capital})
            elif expected_move < -threshold_pct:
                # Enter Short
                position = -1
                entry_price = current_price
                capital *= (1 - self.fee_rate)
                self.trades.append({'time': current_idx, 'type': 'ENTER_SHORT', 'price': current_price, 'capital': capital})

        # Close any open positions at the end
        final_price = df['close'].iloc[-1]
        final_idx = df.index[-1]
        if position == 1:
            capital = capital * (1 + (final_price - entry_price) / entry_price) * (1 - self.fee_rate)
            self.trades.append({'time': final_idx, 'type': 'CLOSE_LONG_END', 'price': final_price, 'capital': capital})
        elif position == -1:
            capital = capital * (1 + (entry_price - final_price) / entry_price) * (1 - self.fee_rate)
            self.trades.append({'time': final_idx, 'type': 'CLOSE_SHORT_END', 'price': final_price, 'capital': capital})

        equity_curve.append((final_idx, capital))
        
        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')
        
        # Calculate metrics
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (capital - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(365 * 24 / step_size) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() != 0 else 0
        
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'equity_df': equity_df,
            'trades': pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame(columns=['time', 'type', 'price', 'capital']),
            'metrics': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.trades)
            }
        }

    def _run_backtest_overlapping(self, df, model_predictor, context_len, horizon_len, stop_loss_pct, threshold_pct, step_size, take_profit_pct, max_positions):
        """
        Overlapping-positions backtest: multiple positions can be open simultaneously.
        
        Capital model:
        - `available_capital` holds the uninvested cash pool.
        - Each new position is allocated `available_capital / remaining_slots` of capital.
        - When a position closes, its realized P&L (after fees) is returned to available_capital.
        - Total equity = available_capital + sum of mark-to-market of all active positions.
        """
        available_capital = self.initial_capital
        active_positions = []  # list of dicts: {direction, entry_price, entry_capital, entry_time}

        equity_curve = []

        for i in range(context_len, len(df) - horizon_len, step_size):
            current_idx = df.index[i]
            current_price = df['close'].iloc[i]

            # ── Step 1: Check SL/TP for all active positions ──
            positions_to_close = []
            for pos_idx, pos in enumerate(active_positions):
                if pos['direction'] == 1:  # Long
                    if take_profit_pct > 0 and current_price >= pos['entry_price'] * (1 + take_profit_pct):
                        positions_to_close.append((pos_idx, 'CLOSE_LONG_TP'))
                    elif current_price <= pos['entry_price'] * (1 - stop_loss_pct):
                        positions_to_close.append((pos_idx, 'CLOSE_LONG_SL'))
                elif pos['direction'] == -1:  # Short
                    if take_profit_pct > 0 and current_price <= pos['entry_price'] * (1 - take_profit_pct):
                        positions_to_close.append((pos_idx, 'CLOSE_SHORT_TP'))
                    elif current_price >= pos['entry_price'] * (1 + stop_loss_pct):
                        positions_to_close.append((pos_idx, 'CLOSE_SHORT_SL'))

            # Close positions in reverse order to preserve indices
            for pos_idx, close_type in sorted(positions_to_close, key=lambda x: x[0], reverse=True):
                pos = active_positions.pop(pos_idx)
                realized = self._calc_position_value(pos, current_price) * (1 - self.fee_rate)
                available_capital += realized
                self.trades.append({
                    'time': current_idx,
                    'type': close_type,
                    'price': current_price,
                    'capital': available_capital + self._total_mtm(active_positions, current_price),
                })

            # ── Step 2: Record equity ──
            total_equity = available_capital + self._total_mtm(active_positions, current_price)
            equity_curve.append((current_idx, total_equity))

            # ── Step 3: Evaluate new entry if slots available ──
            if len(active_positions) < max_positions and available_capital > 0:
                # Prepare context & predict
                context_data = df['close'].iloc[i - context_len : i].values.astype(np.float32)
                point_fc, quant_fc = model_predictor.predict(context_data)
                expected_price = point_fc[-1]
                expected_move = (expected_price - current_price) / current_price

                if expected_move > threshold_pct or expected_move < -threshold_pct:
                    # Allocate capital: fair share of available capital for remaining slots
                    remaining_slots = max_positions - len(active_positions)
                    alloc_capital = available_capital / remaining_slots
                    alloc_capital_after_fee = alloc_capital * (1 - self.fee_rate)
                    available_capital -= alloc_capital  # Remove full amount from pool

                    direction = 1 if expected_move > threshold_pct else -1
                    trade_type = 'ENTER_LONG' if direction == 1 else 'ENTER_SHORT'

                    active_positions.append({
                        'direction': direction,
                        'entry_price': current_price,
                        'entry_capital': alloc_capital_after_fee,
                        'entry_time': current_idx,
                    })

                    self.trades.append({
                        'time': current_idx,
                        'type': trade_type,
                        'price': current_price,
                        'capital': available_capital + self._total_mtm(active_positions, current_price),
                    })

        # ── Close all remaining positions at end ──
        final_price = df['close'].iloc[-1]
        final_idx = df.index[-1]

        for pos in active_positions:
            close_type = 'CLOSE_LONG_END' if pos['direction'] == 1 else 'CLOSE_SHORT_END'
            realized = self._calc_position_value(pos, final_price) * (1 - self.fee_rate)
            available_capital += realized
            self.trades.append({
                'time': final_idx,
                'type': close_type,
                'price': final_price,
                'capital': available_capital,
            })
        active_positions.clear()

        equity_curve.append((final_idx, available_capital))

        equity_df = pd.DataFrame(equity_curve, columns=['timestamp', 'equity']).set_index('timestamp')

        # Calculate metrics
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (available_capital - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(365 * 24 / step_size) * returns.mean() / returns.std() if len(returns) > 1 and returns.std() != 0 else 0

        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            'equity_df': equity_df,
            'trades': pd.DataFrame(self.trades) if len(self.trades) > 0 else pd.DataFrame(columns=['time', 'type', 'price', 'capital']),
            'metrics': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(self.trades)
            }
        }

    @staticmethod
    def _calc_position_value(pos, current_price):
        """Calculate current mark-to-market value of a single position."""
        if pos['direction'] == 1:  # Long
            return pos['entry_capital'] * (1 + (current_price - pos['entry_price']) / pos['entry_price'])
        else:  # Short
            return pos['entry_capital'] * (1 + (pos['entry_price'] - current_price) / pos['entry_price'])

    @staticmethod
    def _total_mtm(active_positions, current_price):
        """Sum mark-to-market value of all active positions."""
        return sum(
            BacktestEngine._calc_position_value(pos, current_price)
            for pos in active_positions
        )
