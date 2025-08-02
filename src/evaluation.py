"""
績效評估模組
包含各種績效指標的計算
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    計算勝率
    
    Args:
        trades: 交易記錄列表
        
    Returns:
        勝率 (0-1)
    """
    if not trades:
        return 0.0
    
    wins = sum(1 for trade in trades if trade['profit'] > 0)
    return wins / len(trades)


def calculate_msr(returns: pd.Series, rf: float = 0.04/240) -> float:
    """
    計算穩健夏普比率 (Median Sharpe Ratio)
    
    Args:
        returns: 回報序列
        rf: 無風險利率
        
    Returns:
        穩健夏普比率
    """
    if len(returns) < 2:
        return 0.0
    
    median_return = np.median(returns)
    mad = np.median(np.abs(returns - median_return))
    
    if mad == 0:
        return 0.0
    
    return (median_return - rf) / mad


def calculate_sharpe_ratio(returns: pd.Series, rf: float = 0.04/240) -> float:
    """
    計算夏普比率
    
    Args:
        returns: 回報序列
        rf: 無風險利率
        
    Returns:
        夏普比率
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - rf) / std_return


def calculate_additional_metrics(trades: List[Dict]) -> Tuple[float, Counter, float, float]:
    """
    計算額外的績效指標
    
    Args:
        trades: 交易記錄列表
        
    Returns:
        (平均交易時長, 出場原因統計, 5筆最短持倉平均, 5筆最長持倉平均)
    """
    if not trades:
        return 0, Counter(), 0, 0
    
    # 計算每筆交易的持倉時間（K 棒數）
    durations = [trade['exit_index'] - trade['entry_index'] for trade in trades]
    avg_duration = np.mean(durations) if durations else 0
    
    # 統計出場原因
    exit_reasons = Counter(trade['exit_reason'] for trade in trades)
    
    # 計算 5 筆最短和最長持倉時間的平均
    sorted_durations = sorted(durations)
    shortest_5_avg = np.mean(sorted_durations[:5]) if len(sorted_durations) >= 5 else 0
    longest_5_avg = np.mean(sorted_durations[-5:]) if len(sorted_durations) >= 5 else 0
    
    return avg_duration, exit_reasons, shortest_5_avg, longest_5_avg


def calculate_returns_stats(returns: pd.Series) -> Tuple[float, float, float, float]:
    """
    計算回報分佈統計
    
    Args:
        returns: 回報序列
        
    Returns:
        (均值, 標準差, 最小值, 最大值)
    """
    if len(returns) == 0:
        return 0, 0, 0, 0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    return mean_return, std_return, min_return, max_return


def calculate_median_and_mad(returns: pd.Series) -> Tuple[float, float]:
    """
    計算中位數回報和 MAD
    
    Args:
        returns: 回報序列
        
    Returns:
        (中位數回報, MAD)
    """
    if len(returns) == 0:
        return 0, 0
    
    median_return = np.median(returns)
    mad = np.median(np.abs(returns - median_return))
    
    return median_return, mad


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    計算最大回撤
    
    Args:
        returns: 回報序列
        
    Returns:
        最大回撤
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return abs(drawdown.min())


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    計算獲利因子
    
    Args:
        trades: 交易記錄列表
        
    Returns:
        獲利因子
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    gross_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(trades: List[Dict]) -> float:
    """
    計算期望值
    
    Args:
        trades: 交易記錄列表
        
    Returns:
        期望值
    """
    if not trades:
        return 0.0
    
    return np.mean([trade['profit'] for trade in trades])


def calculate_risk_reward_ratio(trades: List[Dict]) -> float:
    """
    計算風險報酬比
    
    Args:
        trades: 交易記錄列表
        
    Returns:
        風險報酬比
    """
    if not trades:
        return 0.0
    
    profits = [trade['profit'] for trade in trades if trade['profit'] > 0]
    losses = [trade['profit'] for trade in trades if trade['profit'] < 0]
    
    if not profits or not losses:
        return 0.0
    
    avg_profit = np.mean(profits)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return float('inf') if avg_profit > 0 else 0.0
    
    return avg_profit / avg_loss


def generate_performance_report(trades: List[Dict], returns: pd.Series, 
                              strategy_name: str = "策略") -> Dict[str, Any]:
    """
    生成完整的績效報告
    
    Args:
        trades: 交易記錄列表
        returns: 回報序列
        strategy_name: 策略名稱
        
    Returns:
        績效報告字典
    """
    if not trades:
        return {
            'strategy_name': strategy_name,
            'trade_count': 0,
            'win_rate': 0.0,
            'final_capital': 1000000,
            'total_return': 0.0,
            'msr': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'risk_reward_ratio': 0.0,
            'avg_duration': 0.0,
            'exit_reasons': {},
            'returns_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'mad': 0.0
            }
        }
    
    # 基本指標
    trade_count = len(trades)
    win_rate = calculate_win_rate(trades)
    final_capital = trades[-1]['capital'] if trades else 1000000
    total_return = (final_capital - 1000000) / 1000000
    
    # 風險調整指標
    msr = calculate_msr(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(returns)
    
    # 交易品質指標
    profit_factor = calculate_profit_factor(trades)
    expectancy = calculate_expectancy(trades)
    risk_reward_ratio = calculate_risk_reward_ratio(trades)
    
    # 交易時長和出場原因
    avg_duration, exit_reasons, shortest_5_avg, longest_5_avg = calculate_additional_metrics(trades)
    
    # 回報統計
    mean_return, std_return, min_return, max_return = calculate_returns_stats(returns)
    median_return, mad = calculate_median_and_mad(returns)
    
    report = {
        'strategy_name': strategy_name,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'final_capital': final_capital,
        'total_return': total_return,
        'msr': msr,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'risk_reward_ratio': risk_reward_ratio,
        'avg_duration': avg_duration,
        'shortest_5_avg': shortest_5_avg,
        'longest_5_avg': longest_5_avg,
        'exit_reasons': dict(exit_reasons),
        'returns_stats': {
            'mean': mean_return,
            'std': std_return,
            'min': min_return,
            'max': max_return,
            'median': median_return,
            'mad': mad
        }
    }
    
    return report


def print_performance_summary(report: Dict[str, Any]):
    """
    打印績效摘要
    
    Args:
        report: 績效報告字典
    """
    print(f"\n=== {report['strategy_name']} 績效摘要 ===")
    print(f"交易次數: {report['trade_count']}")
    print(f"勝率: {report['win_rate']:.4f}")
    print(f"最終資金: {report['final_capital']:.2f}")
    print(f"總報酬率: {report['total_return']:.4f}")
    print(f"穩健夏普比率: {report['msr']:.4f}")
    print(f"夏普比率: {report['sharpe_ratio']:.4f}")
    print(f"最大回撤: {report['max_drawdown']:.4f}")
    print(f"獲利因子: {report['profit_factor']:.4f}")
    print(f"期望值: {report['expectancy']:.2f}")
    print(f"風險報酬比: {report['risk_reward_ratio']:.4f}")
    print(f"平均交易時長: {report['avg_duration']:.2f}")
    print(f"出場原因統計: {report['exit_reasons']}")
    
    stats = report['returns_stats']
    print(f"\n回報分佈統計:")
    print(f"  均值: {stats['mean']:.2f}")
    print(f"  標準差: {stats['std']:.2f}")
    print(f"  最小值: {stats['min']:.2f}")
    print(f"  最大值: {stats['max']:.2f}")
    print(f"  中位數: {stats['median']:.2f}")
    print(f"  MAD: {stats['mad']:.2f}")


def compare_strategies(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    比較多個策略的績效
    
    Args:
        reports: 績效報告列表
        
    Returns:
        比較結果 DataFrame
    """
    comparison_data = []
    
    for report in reports:
        comparison_data.append({
            '策略': report['strategy_name'],
            '交易次數': report['trade_count'],
            '勝率': report['win_rate'],
            '最終資金': report['final_capital'],
            '總報酬率': report['total_return'],
            '穩健夏普比率': report['msr'],
            '夏普比率': report['sharpe_ratio'],
            '最大回撤': report['max_drawdown'],
            '獲利因子': report['profit_factor'],
            '期望值': report['expectancy'],
            '風險報酬比': report['risk_reward_ratio'],
            '平均交易時長': report['avg_duration']
        })
    
    return pd.DataFrame(comparison_data) 