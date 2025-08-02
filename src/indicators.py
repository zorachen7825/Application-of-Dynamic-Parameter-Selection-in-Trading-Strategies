"""
技術指標計算模組
包含移動平均線、威廉指標等技術指標的計算
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def add_indicators(df: pd.DataFrame, ma_short: int, ma_long: int, 
                   will_period_1: int, will_period_2: int) -> pd.DataFrame:
    """
    為 DataFrame 添加技術指標
    
    Args:
        df: 原始價格數據
        ma_short: 短期移動平均線週期
        ma_long: 長期移動平均線週期
        will_period_1: 威廉指標買進信號週期
        will_period_2: 威廉指標賣出信號週期
        
    Returns:
        添加技術指標後的 DataFrame
    """
    df = df.copy()
    
    # 計算移動平均線
    df[f"MA_Short_{ma_short}"] = df['Close'].rolling(window=ma_short).mean()
    df[f"MA_Long_{ma_long}"] = df['Close'].rolling(window=ma_long).mean()
    
    # 計算威廉指標 (買進信號)
    highest_high_buy = df['High'].rolling(window=will_period_1).max()
    lowest_low_buy = df['Low'].rolling(window=will_period_1).min()
    df[f"Williams_1_{will_period_1}"] = -100 * (highest_high_buy - df['Close']) / (highest_high_buy - lowest_low_buy)
    df[f"Williams_1_{will_period_1}"] = df[f"Williams_1_{will_period_1}"].where(highest_high_buy != lowest_low_buy, 0)
    
    # 計算威廉指標 (賣出信號)
    highest_high_sell = df['High'].rolling(window=will_period_2).max()
    lowest_low_sell = df['Low'].rolling(window=will_period_2).min()
    df[f"Williams_2_{will_period_2}"] = -100 * (highest_high_sell - df['Close']) / (highest_high_sell - lowest_low_sell)
    df[f"Williams_2_{will_period_2}"] = df[f"Williams_2_{will_period_2}"].where(highest_high_sell != lowest_low_sell, 0)
    
    return df


def calculate_ma_crossover(df: pd.DataFrame, ma_short_col: str, ma_long_col: str) -> pd.Series:
    """
    計算移動平均線交叉信號
    
    Args:
        df: 包含移動平均線的 DataFrame
        ma_short_col: 短期均線欄位名稱
        ma_long_col: 長期均線欄位名稱
        
    Returns:
        黃金交叉信號 (True 表示短期均線上穿長期均線)
    """
    golden_cross = (df[ma_short_col] > df[ma_long_col]) & (df[ma_short_col].shift(1) <= df[ma_long_col].shift(1))
    death_cross = (df[ma_short_col] < df[ma_long_col]) & (df[ma_short_col].shift(1) >= df[ma_long_col].shift(1))
    
    return golden_cross, death_cross


def calculate_williams_signals(df: pd.DataFrame, will_col: str, 
                              buy_threshold: float, sell_threshold: float) -> tuple:
    """
    計算威廉指標信號
    
    Args:
        df: 包含威廉指標的 DataFrame
        will_col: 威廉指標欄位名稱
        buy_threshold: 買進閾值
        sell_threshold: 賣出閾值
        
    Returns:
        (超賣信號, 超買信號)
    """
    oversold = df[will_col] < buy_threshold
    overbought = df[will_col] > sell_threshold
    
    return oversold, overbought


def calculate_price_ma_signals(df: pd.DataFrame, price_col: str, ma_col: str) -> tuple:
    """
    計算價格與移動平均線的關係信號
    
    Args:
        df: 包含價格和均線的 DataFrame
        price_col: 價格欄位名稱
        ma_col: 移動平均線欄位名稱
        
    Returns:
        (價格突破均線, 價格跌破均線)
    """
    price_above_ma = df[price_col] > df[ma_col]
    price_below_ma = df[price_col] < df[ma_col]
    
    return price_above_ma, price_below_ma


def calculate_stop_loss(entry_price: float, current_price: float, stop_loss_ratio: float) -> bool:
    """
    計算停損信號
    
    Args:
        entry_price: 進場價格
        current_price: 當前價格
        stop_loss_ratio: 停損比例
        
    Returns:
        是否觸發停損
    """
    return (entry_price - current_price) / entry_price >= stop_loss_ratio


def get_indicator_columns(ma_short: int, ma_long: int, 
                         will_period_1: int, will_period_2: int) -> Dict[str, str]:
    """
    獲取技術指標欄位名稱
    
    Args:
        ma_short: 短期移動平均線週期
        ma_long: 長期移動平均線週期
        will_period_1: 威廉指標買進信號週期
        will_period_2: 威廉指標賣出信號週期
        
    Returns:
        指標欄位名稱字典
    """
    return {
        'ma_short': f"MA_Short_{ma_short}",
        'ma_long': f"MA_Long_{ma_long}",
        'williams_buy': f"Williams_1_{will_period_1}",
        'williams_sell': f"Williams_2_{will_period_2}"
    }


def validate_parameters(ma_short: int, ma_long: int, 
                       will_period_1: int, will_period_2: int) -> bool:
    """
    驗證參數組合是否有效
    
    Args:
        ma_short: 短期移動平均線週期
        ma_long: 長期移動平均線週期
        will_period_1: 威廉指標買進信號週期
        will_period_2: 威廉指標賣出信號週期
        
    Returns:
        參數組合是否有效
    """
    # 檢查移動平均線參數
    if ma_short >= ma_long:
        return False
    
    # 檢查週期參數是否為正數
    if any(x <= 0 for x in [ma_short, ma_long, will_period_1, will_period_2]):
        return False
    
    return True 