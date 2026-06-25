import pandas as pd
from binance.client import Client

class BinanceLoader:
    def __init__(self):
        self.client = Client()  # Public client, no API keys
        
    def fetch_historical_data(self, symbol: str, interval: str, start_str: str, end_str: str = None) -> pd.DataFrame:
        """
        Fetch historical klines from Binance and return as a Pandas DataFrame.
        """
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
