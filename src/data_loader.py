"""
數據載入模組
負責載入和預處理歷史價格數據
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """數據載入器類別"""
    
    def __init__(self, file_path: str):
        """
        初始化數據載入器
        
        Args:
            file_path: CSV 檔案路徑
        """
        self.file_path = file_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """
        載入數據並進行基本預處理
        
        Returns:
            處理後的 DataFrame
        """
        # 載入 CSV 檔案
        self.data = pd.read_csv(self.file_path)
        
        # 合併日期和時間欄位
        self.data['Datetime'] = pd.to_datetime(
            self.data['Date'] + ' ' + self.data['Time']
        )
        
        # 按時間排序
        self.data = self.data.sort_values('Datetime').reset_index(drop=True)
        
        return self.data
    
    def split_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分割數據為訓練、驗證和測試集
        
        Args:
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            
        Returns:
            (訓練集, 驗證集, 測試集)
        """
        if self.data is None:
            raise ValueError("請先載入數據")
        
        total_len = len(self.data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        train_data = self.data[:train_end].copy()
        val_data = self.data[train_end:val_end].copy()
        test_data = self.data[val_end:].copy()
        
        print(f"數據分割完成：")
        print(f"  訓練集：{len(train_data)} 筆")
        print(f"  驗證集：{len(val_data)} 筆")
        print(f"  測試集：{len(test_data)} 筆")
        
        return train_data, val_data, test_data
    
    def get_data_info(self) -> dict:
        """
        獲取數據基本資訊
        
        Returns:
            數據資訊字典
        """
        if self.data is None:
            raise ValueError("請先載入數據")
        
        info = {
            'total_rows': len(self.data),
            'columns': list(self.data.columns),
            'date_range': {
                'start': self.data['Datetime'].min(),
                'end': self.data['Datetime'].max()
            },
            'price_range': {
                'min': self.data[['Open', 'High', 'Low', 'Close']].min().min(),
                'max': self.data[['Open', 'High', 'Low', 'Close']].max().max()
            }
        }
        
        return info


def load_and_prepare_data(file_path: str, split: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    便捷函數：載入並準備數據
    
    Args:
        file_path: CSV 檔案路徑
        split: 是否分割數據
        
    Returns:
        如果 split=True，返回 (訓練集, 驗證集, 測試集)
        如果 split=False，返回 (完整數據, None, None)
    """
    loader = DataLoader(file_path)
    data = loader.load_data()
    
    if split:
        train_data, val_data, test_data = loader.split_data()
        return train_data, val_data, test_data
    else:
        return data, None, None 