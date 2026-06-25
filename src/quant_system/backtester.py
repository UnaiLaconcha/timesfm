import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=1000.0, fee_rate=0.0004):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.trades = []

    def run_backtest(self, df: pd.DataFrame, model_predictor, horizon_len=24, stop_loss_pct=0.02, threshold_pct=0.01, step_size=6, take_profit_pct=0.04):
        """
        Runs a walk-forward backtest using TimesFM predictions.
        df: DataFrame with at least 'close' prices.
        model_predictor: initialized TimesFMPredictor.
        stop_loss_pct: 0.02 means 2% stop loss.
        take_profit_pct: 0.04 means 4% take profit. Set to 0 to disable.
        threshold_pct: 0.01 means 1% expected move to enter trade.
        step_size: step size for walk-forward evaluation.
        """
        context_len = model_predictor.context_len
        if len(df) < context_len + horizon_len:
            raise ValueError("Dataset is too small for the given context_len and horizon_len.")

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
