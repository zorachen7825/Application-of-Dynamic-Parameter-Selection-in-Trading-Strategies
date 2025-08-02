#!/usr/bin/env python3
"""
測試策略腳本
"""

import sys
import os
from pathlib import Path

# 添加 src 目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from strategies import get_strategy_function
from evaluation import generate_performance_report, print_performance_summary


def test_single_strategy():
    """測試單一策略"""
    print("=== 測試策略 1 ===")
    
    # 載入數據
    data_file = "data/full_15K.csv"
    if not os.path.exists(data_file):
        print(f"錯誤：找不到數據檔案 {data_file}")
        return
    
    loader = DataLoader(data_file)
    data = loader.load_data()
    train_data, val_data, test_data = loader.split_data()
    
    # 測試參數
    test_params = {
        'ma_short': 15,
        'ma_long': 20,
        'will_period_1': 40,
        'will_buy_threshold_1': -50,
        'will_period_2': 60,
        'will_sell_threshold_2': -50,
        'stop_loss': 0.003
    }
    
    # 執行策略 1
    strategy_func = get_strategy_function(1)
    trades, returns, final_capital = strategy_func(
        val_data.copy(),
        **test_params
    )
    
    # 生成報告
    report = generate_performance_report(trades, returns, "策略 1")
    print_performance_summary(report)
    
    print(f"\n交易記錄數量：{len(trades)}")
    if trades:
        print(f"第一筆交易：{trades[0]}")
        print(f"最後一筆交易：{trades[-1]}")


if __name__ == "__main__":
    test_single_strategy() 