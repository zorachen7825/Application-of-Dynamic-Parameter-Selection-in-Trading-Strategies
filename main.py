#!/usr/bin/env python3
"""
Gork 量化交易策略主程式
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加 src 目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from strategies import get_strategy_function
from evaluation import generate_performance_report, print_performance_summary, compare_strategies


def main():
    """主程式"""
    print("=== Gork 量化交易策略分析 ===")
    
    # 載入數據
    data_file = "data/full_15K.csv"
    if not os.path.exists(data_file):
        print(f"錯誤：找不到數據檔案 {data_file}")
        return
    
    print(f"載入數據檔案：{data_file}")
    loader = DataLoader(data_file)
    data = loader.load_data()
    
    # 分割數據
    train_data, val_data, test_data = loader.split_data()
    
    # 定義測試參數
    test_params = {
        'ma_short': 15,
        'ma_long': 20,
        'will_period_1': 40,
        'will_buy_threshold_1': -50,
        'will_period_2': 60,
        'will_sell_threshold_2': -50,
        'stop_loss': 0.003
    }
    
    print(f"\n測試參數：{test_params}")
    
    # 測試四種策略
    strategies = {
        1: "均線買 + 均線賣",
        2: "均線買 + 價格賣", 
        3: "價格買 + 均線賣",
        4: "價格買 + 價格賣"
    }
    
    results = []
    
    for strategy_id, strategy_name in strategies.items():
        print(f"\n正在測試 {strategy_name}...")
        
        # 獲取策略函數
        strategy_func = get_strategy_function(strategy_id)
        
        # 執行回測
        trades, returns, final_capital = strategy_func(
            val_data.copy(),
            **test_params
        )
        
        # 生成績效報告
        report = generate_performance_report(trades, returns, strategy_name)
        results.append(report)
        
        # 打印摘要
        print_performance_summary(report)
    
    # 比較策略
    print("\n=== 策略比較 ===")
    comparison_df = compare_strategies(results)
    print(comparison_df.to_string(index=False))
    
    # 找出最佳策略
    best_strategy = max(results, key=lambda x: x['msr'])
    print(f"\n最佳策略（按穩健夏普比率）：{best_strategy['strategy_name']}")
    print(f"穩健夏普比率：{best_strategy['msr']:.4f}")
    
    print("\n分析完成！")


if __name__ == "__main__":
    main() 