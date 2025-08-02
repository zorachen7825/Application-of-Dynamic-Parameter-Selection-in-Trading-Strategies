"""
交易策略實作模組
包含四種不同的交易策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from indicators import add_indicators, calculate_stop_loss


def execute_trade(trades: List[Dict], capital: float, entry_price: float, 
                  exit_price: float, entry_time, exit_time, entry_index: int, 
                  exit_index: int, volume: int = 200, fee_rate: float = 0.00002, 
                  exit_reason: str = '') -> Tuple[List[Dict], float]:
    """
    執行交易並記錄交易結果
    
    Args:
        trades: 交易記錄列表
        capital: 當前資金
        entry_price: 進場價格
        exit_price: 出場價格
        entry_time: 進場時間
        exit_time: 出場時間
        entry_index: 進場索引
        exit_index: 出場索引
        volume: 交易量
        fee_rate: 手續費率
        exit_reason: 出場原因
        
    Returns:
        (更新後的交易記錄, 更新後的資金)
    """
    entry_fee = entry_price * volume * fee_rate
    exit_fee = exit_price * volume * fee_rate
    
    capital -= entry_fee
    profit = (exit_price - entry_price) * volume
    net_profit = profit - exit_fee
    capital += net_profit
    
    trade_record = {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'entry_index': entry_index,
        'exit_index': exit_index,
        'profit': net_profit,
        'capital': capital,
        'entry_fee': entry_fee,
        'exit_fee': exit_fee,
        'exit_reason': exit_reason
    }
    
    trades.append(trade_record)
    
    return trades, capital


def simulate_strategy_1(df: pd.DataFrame, ma_short: int, ma_long: int, 
                       will_period_1: int, will_buy_threshold_1: int,
                       will_period_2: int, will_sell_threshold_2: int, 
                       stop_loss: float, initial_capital: float = 1000000) -> Tuple[List[Dict], pd.Series, float]:
    """
    策略 1：均線買 + 均線賣
    
    進場：均線黃金交叉 AND 威廉指標超賣
    出場：均線死亡交叉 OR 威廉指標超買 OR 停損
    """
    df = add_indicators(df, ma_short, ma_long, will_period_1, will_period_2)
    position = 0
    capital = initial_capital
    trades = []
    entry_price = 0
    entry_time = None
    entry_index = 0
    
    for i in range(1, len(df) - 1):
        if position == 0 and i >= max(ma_long, will_period_1):
            # 進場條件：均線黃金交叉 AND 威廉指標超賣
            golden_cross = (df[f"MA_Short_{ma_short}"].iloc[i] > df[f"MA_Long_{ma_long}"].iloc[i] and
                           df[f"MA_Short_{ma_short}"].iloc[i-1] <= df[f"MA_Long_{ma_long}"].iloc[i-1])
            williams_oversold = df[f"Williams_1_{will_period_1}"].iloc[i] < will_buy_threshold_1
            
            if golden_cross and williams_oversold:
                position = 1
                entry_price = df['Open'].iloc[i+1]
                entry_time = df['Datetime'].iloc[i+1]
                entry_index = i + 1
                
        elif position == 1:
            current_price = df['Close'].iloc[i]
            stop_loss_triggered = calculate_stop_loss(entry_price, current_price, stop_loss)
            death_cross = (df[f"MA_Short_{ma_short}"].iloc[i] < df[f"MA_Long_{ma_long}"].iloc[i])
            williams_overbought = df[f"Williams_2_{will_period_2}"].iloc[i] > will_sell_threshold_2
            
            if death_cross or williams_overbought or stop_loss_triggered:
                position = 0
                exit_price = df['Open'].iloc[i+1]
                exit_time = df['Datetime'].iloc[i+1]
                exit_index = i + 1
                
                if stop_loss_triggered:
                    exit_reason = 'stop_loss'
                elif death_cross:
                    exit_reason = 'death_cross'
                else:
                    exit_reason = 'williams_overbought'
                    
                trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                              entry_time, exit_time, entry_index, exit_index, 
                                              exit_reason=exit_reason)
    
    # 處理最後一筆未平倉的交易
    if position == 1 and len(df) > 1:
        exit_price = df['Close'].iloc[-1]
        exit_time = df['Datetime'].iloc[-1]
        exit_index = len(df) - 1
        trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                      entry_time, exit_time, entry_index, exit_index, 
                                      exit_reason='end_of_data')
    
    if not trades:
        return trades, pd.Series([], dtype=float), capital
    
    returns = pd.Series([t['profit'] for t in trades], index=[t['exit_time'] for t in trades])
    return trades, returns, capital


def simulate_strategy_2(df: pd.DataFrame, ma_short: int, ma_long: int, 
                       will_period_1: int, will_buy_threshold_1: int,
                       will_period_2: int, will_sell_threshold_2: int, 
                       stop_loss: float, initial_capital: float = 1000000) -> Tuple[List[Dict], pd.Series, float]:
    """
    策略 2：均線買 + 價格賣
    
    進場：短期均線上穿長期均線 AND 威廉指標超賣
    出場：價格跌破長期均線 OR 威廉指標超買 OR 停損
    """
    df = add_indicators(df, ma_short, ma_long, will_period_1, will_period_2)
    position = 0
    capital = initial_capital
    trades = []
    entry_price = 0
    entry_time = None
    entry_index = 0
    
    for i in range(1, len(df) - 1):
        if position == 0 and i >= max(ma_long, will_period_1):
            # 進場條件：均線黃金交叉 AND 威廉指標超賣
            golden_cross = (df[f"MA_Short_{ma_short}"].iloc[i] > df[f"MA_Long_{ma_long}"].iloc[i] and
                           df[f"MA_Short_{ma_short}"].iloc[i-1] <= df[f"MA_Long_{ma_long}"].iloc[i-1])
            williams_oversold = df[f"Williams_1_{will_period_1}"].iloc[i] < will_buy_threshold_1
            
            if golden_cross and williams_oversold:
                position = 1
                entry_price = df['Open'].iloc[i+1]
                entry_time = df['Datetime'].iloc[i+1]
                entry_index = i + 1
                
        elif position == 1:
            current_price = df['Close'].iloc[i]
            stop_loss_triggered = calculate_stop_loss(entry_price, current_price, stop_loss)
            price_below_long_ma = current_price < df[f"MA_Long_{ma_long}"].iloc[i]
            williams_overbought = df[f"Williams_2_{will_period_2}"].iloc[i] > will_sell_threshold_2
            
            if price_below_long_ma or williams_overbought or stop_loss_triggered:
                position = 0
                exit_price = df['Open'].iloc[i+1]
                exit_time = df['Datetime'].iloc[i+1]
                exit_index = i + 1
                
                if stop_loss_triggered:
                    exit_reason = 'stop_loss'
                elif price_below_long_ma:
                    exit_reason = 'price_below_long_ma'
                else:
                    exit_reason = 'williams_overbought'
                    
                trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                              entry_time, exit_time, entry_index, exit_index, 
                                              exit_reason=exit_reason)
    
    # 處理最後一筆未平倉的交易
    if position == 1 and len(df) > 1:
        exit_price = df['Close'].iloc[-1]
        exit_time = df['Datetime'].iloc[-1]
        exit_index = len(df) - 1
        trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                      entry_time, exit_time, entry_index, exit_index, 
                                      exit_reason='end_of_data')
    
    if not trades:
        return trades, pd.Series([], dtype=float), capital
    
    returns = pd.Series([t['profit'] for t in trades], index=[t['exit_time'] for t in trades])
    return trades, returns, capital


def simulate_strategy_3(df: pd.DataFrame, ma_short: int, ma_long: int, 
                       will_period_1: int, will_buy_threshold_1: int,
                       will_period_2: int, will_sell_threshold_2: int, 
                       stop_loss: float, initial_capital: float = 1000000) -> Tuple[List[Dict], pd.Series, float]:
    """
    策略 3：價格買 + 均線賣
    
    進場：價格突破短期均線 AND 威廉指標超賣
    出場：短期均線下穿長期均線 OR 威廉指標超買 OR 停損
    """
    df = add_indicators(df, ma_short, ma_long, will_period_1, will_period_2)
    position = 0
    capital = initial_capital
    trades = []
    entry_price = 0
    entry_time = None
    entry_index = 0
    
    for i in range(1, len(df) - 1):
        if position == 0 and i >= max(ma_long, will_period_1):
            # 進場條件：價格突破短期均線 AND 威廉指標超賣
            price_above_short_ma = df['Close'].iloc[i] > df[f"MA_Short_{ma_short}"].iloc[i]
            williams_oversold = df[f"Williams_1_{will_period_1}"].iloc[i] < will_buy_threshold_1
            
            if price_above_short_ma and williams_oversold:
                position = 1
                entry_price = df['Open'].iloc[i+1]
                entry_time = df['Datetime'].iloc[i+1]
                entry_index = i + 1
                
        elif position == 1:
            current_price = df['Close'].iloc[i]
            stop_loss_triggered = calculate_stop_loss(entry_price, current_price, stop_loss)
            death_cross = (df[f"MA_Short_{ma_short}"].iloc[i] < df[f"MA_Long_{ma_long}"].iloc[i])
            williams_overbought = df[f"Williams_2_{will_period_2}"].iloc[i] > will_sell_threshold_2
            
            if death_cross or williams_overbought or stop_loss_triggered:
                position = 0
                exit_price = df['Open'].iloc[i+1]
                exit_time = df['Datetime'].iloc[i+1]
                exit_index = i + 1
                
                if stop_loss_triggered:
                    exit_reason = 'stop_loss'
                elif death_cross:
                    exit_reason = 'death_cross'
                else:
                    exit_reason = 'williams_overbought'
                    
                trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                              entry_time, exit_time, entry_index, exit_index, 
                                              exit_reason=exit_reason)
    
    # 處理最後一筆未平倉的交易
    if position == 1 and len(df) > 1:
        exit_price = df['Close'].iloc[-1]
        exit_time = df['Datetime'].iloc[-1]
        exit_index = len(df) - 1
        trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                      entry_time, exit_time, entry_index, exit_index, 
                                      exit_reason='end_of_data')
    
    if not trades:
        return trades, pd.Series([], dtype=float), capital
    
    returns = pd.Series([t['profit'] for t in trades], index=[t['exit_time'] for t in trades])
    return trades, returns, capital


def simulate_strategy_4(df: pd.DataFrame, ma_short: int, ma_long: int, 
                       will_period_1: int, will_buy_threshold_1: int,
                       will_period_2: int, will_sell_threshold_2: int, 
                       stop_loss: float, initial_capital: float = 1000000) -> Tuple[List[Dict], pd.Series, float]:
    """
    策略 4：價格買 + 價格賣
    
    進場：價格突破短期均線 AND 威廉指標超賣
    出場：價格跌破長期均線 OR 威廉指標超買 OR 停損
    """
    df = add_indicators(df, ma_short, ma_long, will_period_1, will_period_2)
    position = 0
    capital = initial_capital
    trades = []
    entry_price = 0
    entry_time = None
    entry_index = 0
    
    for i in range(1, len(df) - 1):
        if position == 0 and i >= max(ma_long, will_period_1):
            # 進場條件：價格突破短期均線 AND 威廉指標超賣
            price_above_short_ma = df['Close'].iloc[i] > df[f"MA_Short_{ma_short}"].iloc[i]
            williams_oversold = df[f"Williams_1_{will_period_1}"].iloc[i] < will_buy_threshold_1
            
            if price_above_short_ma and williams_oversold:
                position = 1
                entry_price = df['Open'].iloc[i+1]
                entry_time = df['Datetime'].iloc[i+1]
                entry_index = i + 1
                
        elif position == 1:
            current_price = df['Close'].iloc[i]
            stop_loss_triggered = calculate_stop_loss(entry_price, current_price, stop_loss)
            price_below_long_ma = current_price < df[f"MA_Long_{ma_long}"].iloc[i]
            williams_overbought = df[f"Williams_2_{will_period_2}"].iloc[i] > will_sell_threshold_2
            
            if price_below_long_ma or williams_overbought or stop_loss_triggered:
                position = 0
                exit_price = df['Open'].iloc[i+1]
                exit_time = df['Datetime'].iloc[i+1]
                exit_index = i + 1
                
                if stop_loss_triggered:
                    exit_reason = 'stop_loss'
                elif price_below_long_ma:
                    exit_reason = 'price_below_long_ma'
                else:
                    exit_reason = 'williams_overbought'
                    
                trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                              entry_time, exit_time, entry_index, exit_index, 
                                              exit_reason=exit_reason)
    
    # 處理最後一筆未平倉的交易
    if position == 1 and len(df) > 1:
        exit_price = df['Close'].iloc[-1]
        exit_time = df['Datetime'].iloc[-1]
        exit_index = len(df) - 1
        trades, capital = execute_trade(trades, capital, entry_price, exit_price, 
                                      entry_time, exit_time, entry_index, exit_index, 
                                      exit_reason='end_of_data')
    
    if not trades:
        return trades, pd.Series([], dtype=float), capital
    
    returns = pd.Series([t['profit'] for t in trades], index=[t['exit_time'] for t in trades])
    return trades, returns, capital


# 策略函數字典
STRATEGY_FUNCTIONS = {
    1: simulate_strategy_1,
    2: simulate_strategy_2,
    3: simulate_strategy_3,
    4: simulate_strategy_4
}


def get_strategy_function(strategy_id: int):
    """
    獲取策略函數
    
    Args:
        strategy_id: 策略編號 (1-4)
        
    Returns:
        策略函數
    """
    if strategy_id not in STRATEGY_FUNCTIONS:
        raise ValueError(f"不支援的策略編號：{strategy_id}")
    
    return STRATEGY_FUNCTIONS[strategy_id] 