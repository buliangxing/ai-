"""
📊 专业股票技术分析系统 - Streamlit Web应用
版本: 5.1 (重置+信号美化版)
优化内容：
1. 强制初始化session_state，新用户打开无历史痕迹
2. 美化买入/卖出/中性信号展示（卡片化+彩色标签+图标）
3. 保留原有所有功能，仅优化体验
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import akshare as ak
import warnings
import yfinance as yf
import ta
from streamlit_extras.metric_cards import style_metric_cards
import time
from functools import lru_cache
import json
import os
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="专业股票技术分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 样式配置 ======================
def apply_custom_styles():
    """应用自定义样式（新增信号卡片样式）"""
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main {
        padding: 1rem 2rem;
    }
    
    /* 卡片样式 */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        border: none;
    }
    
    /* 信号卡片样式 */
    .buy-signal {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: none;
    }
    
    .sell-signal {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: none;
    }
    
    .hold-signal {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: none;
    }
    
    /* 表格样式 */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
    }
    
    .dataframe td {
        padding: 10px 16px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* 热点股票按钮样式 */
    .hot-stock-btn {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #1e40af;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 0.5rem 0;
        margin: 0.25rem;
        width: 100%;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .hot-stock-btn:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        transform: translateY(-2px);
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    
    /* 标签样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 12px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 8px 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* 滑块样式 */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 经济数据卡片 */
    .economic-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3b82f6;
    }
    
    .economic-title {
        font-size: 12px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .economic-value {
        font-size: 20px;
        font-weight: 700;
        color: #1f2937;
    }
    
    .economic-change {
        font-size: 12px;
        font-weight: 500;
    }
    
    .positive {
        color: #059669;
    }
    
    .negative {
        color: #dc2626;
    }
    
    /* 操作建议卡片 */
    .advice-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    
    .advice-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .advice-item {
        padding: 10px 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .advice-item:last-child {
        border-bottom: none;
    }
    
    .advice-label {
        font-size: 13px;
        color: #6b7280;
        font-weight: 500;
    }
    
    .advice-value {
        font-size: 16px;
        font-weight: 700;
        color: #1f2937;
    }
    
    .profit {
        color: #059669;
    }
    
    .loss {
        color: #dc2626;
    }
    
    /* 加载动画 */
    .loading-spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* 新增：信号标签样式 */
    .signal-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin: 4px 0;
        width: 100%;
        text-align: left;
    }
    
    .buy-tag {
        background-color: rgba(16, 185, 129, 0.1);
        color: #059669;
        border: 1px solid #10b981;
    }
    
    .sell-tag {
        background-color: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid #ef4444;
    }
    
    .neutral-tag {
        background-color: rgba(245, 158, 11, 0.1);
        color: #d97706;
        border: 1px solid #f59e0b;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ====================== 数据获取与缓存 ======================
@st.cache_data(ttl=300)  # 缓存5分钟
def get_stock_data_enhanced(stock_code: str, days: int = 120, data_source: str = "akshare", period: str = "daily"):
    """增强版股票数据获取函数，支持多个数据源和不同周期"""
    
    try:
        with st.spinner(f"正在获取 {stock_code} 的{get_period_name(period)}数据..."):
            if data_source == "akshare":
                # 根据周期调整时间范围
                if period == "daily":
                    actual_days = days
                elif period == "weekly":
                    actual_days = days * 5
                elif period == "monthly":
                    actual_days = days * 20
                else:
                    actual_days = days
                
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=actual_days*2)).strftime("%Y%m%d")
                
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if df.empty:
                    st.warning("akshare返回空数据，尝试yfinance...")
                    return get_stock_data_enhanced(stock_code, days, "yfinance", period)
                
                # 重命名列
                column_map = {
                    "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                    "收盘": "close", "成交量": "volume", "成交额": "amount",
                    "涨跌幅": "change_pct", "涨跌额": "change_amount"
                }
                
                df = df.rename(columns=column_map)
                
                # 处理不同周期
                if period != "daily":
                    df = resample_data(df, period)
                
            elif data_source == "yfinance":
                # 使用yfinance获取数据
                symbol = stock_code
                if not any(symbol.endswith(suffix) for suffix in ['.SS', '.SZ', '.HK']):
                    if symbol.startswith('6'):
                        symbol = f"{symbol}.SS"
                    elif symbol.startswith('0') or symbol.startswith('3'):
                        symbol = f"{symbol}.SZ"
                    else:
                        symbol = f"{symbol}.HK"
                
                ticker = yf.Ticker(symbol)
                
                # 根据周期选择不同的period
                period_map = {
                    "daily": f"{days*2}d",
                    "weekly": f"{days*5 * 2}d",
                    "monthly": f"{days*20 * 2}d"
                }
                
                df = ticker.history(period=period_map.get(period, f"{days*2}d"))
                
                if df.empty:
                    raise ValueError("yfinance返回空数据")
                
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'date', 'Open': 'open', 'High': 'high',
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
        
        # 数据清洗和处理
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[required_cols].copy()
        
        # 确保数据排序正确
        df = df.sort_values('date')
        
        # 计算基本指标
        df['change_pct'] = df['close'].pct_change() * 100
        df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
        
        # 只保留指定天数的数据
        df = df.tail(days).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        st.info("正在生成模拟数据...")
        # ========== 新增：强制统一日期类型 ==========
        # 无论数据源返回什么格式，都转为datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # 删除日期转换失败的行（避免后续报错）
        df = df.dropna(subset=['date'])
        # ===========================================
        
        # 只保留指定天数的数据
        df = df.tail(days).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"数据获取失败: {str(e)}")
        st.info("正在生成模拟数据...")
        
        # 生成高质量的模拟数据
        return generate_sample_data(stock_code, days, period)

def get_period_name(period: str) -> str:
    """获取周期名称"""
    period_map = {
        "daily": "日K线",
        "weekly": "周K线",
        "monthly": "月K线"
    }
    return period_map.get(period, "日K线")

def resample_data(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """重采样数据到不同周期"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    if period == "weekly":
        # 周K线：以周五为结束
        resampled = df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    elif period == "monthly":
        # 月K线：以月末为结束
        resampled = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    else:
        return df
    
    resampled = resampled.dropna()
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'date': 'date'})
    
    return resampled

@st.cache_data
def generate_sample_data(stock_code: str, days: int = 120, period: str = "daily"):
    """生成高质量的模拟数据"""
    np.random.seed(42)
    
    # 根据周期调整数据点数
    if period == "daily":
        freq = 'B'
    elif period == "weekly":
        freq = 'W-FRI'
        days = days // 5
    elif period == "monthly":
        freq = 'M'
        days = days // 20
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq=freq)
    
    # 根据股票代码生成不同的价格水平
    base_prices = {
        '000001': 10.5,    # 平安银行
        '000002': 8.2,     # 万科A
        '000858': 150.0,   # 五粮液
        '002415': 35.0,    # 海康威视
        '300750': 180.0,   # 宁德时代
        '600519': 1700.0,  # 贵州茅台
        '603986': 85.0,    # 兆易创新
    }
    
    base_price = base_prices.get(stock_code, 50.0)
    
    # 生成更真实的股价序列
    np.random.seed(hash(stock_code) % 10000)
    
    # 根据周期调整波动率
    if period == "daily":
        volatility = 0.02
    elif period == "weekly":
        volatility = 0.045
    else:  # monthly
        volatility = 0.08
    
    # 生成趋势
    trend = np.linspace(0, np.random.uniform(-0.2, 0.2), days)
    
    # 生成季节性
    seasonal = np.sin(np.linspace(0, 4*np.pi, days)) * 0.1
    
    # 生成随机波动
    noise = np.random.normal(0, volatility, days)
    
    # 组合生成对数价格
    log_prices = np.cumsum(trend + seasonal + noise)
    prices = base_price * np.exp(log_prices)
    
    # 生成OHLC数据
    df = pd.DataFrame({
        'date': dates,
        'open': np.zeros(days),
        'high': np.zeros(days),
        'low': np.zeros(days),
        'close': prices,
        'volume': np.random.lognormal(13, 0.8, days).astype(int)
    })
    
    # 生成真实的OHLC关系
    for i in range(days):
        if i == 0:
            prev_close = base_price
        else:
            prev_close = df.loc[i-1, 'close']
        
        daily_return = np.random.normal(0.0005, volatility)
        current_close = prev_close * (1 + daily_return)
        
        # 根据周期调整波动范围
        if period == "daily":
            open_vol = 0.005
            high_low_vol = 0.015
        elif period == "weekly":
            open_vol = 0.01
            high_low_vol = 0.03
        else:  # monthly
            open_vol = 0.02
            high_low_vol = 0.05
        
        # 生成合理的OHLC
        open_price = prev_close * (1 + np.random.normal(0, open_vol))
        close_price = current_close
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, high_low_vol)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, high_low_vol)))
        
        # 确保high > low
        if high_price <= low_price:
            high_price = low_price * 1.01
        
        df.loc[i, 'open'] = open_price
        df.loc[i, 'high'] = high_price
        df.loc[i, 'low'] = low_price
        df.loc[i, 'close'] = close_price
    
    # 计算衍生指标
    df['change_pct'] = df['close'].pct_change() * 100
    df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
    
    return df

@st.cache_data(ttl=3600)  # 缓存1小时
def get_economic_data():
    """获取宏观经济数据"""
    try:
        economic_data = {}
        
        # 1. 获取中国GDP数据
        try:
            gdp_df = ak.macro_china_gdp()
            if not gdp_df.empty:
                latest_gdp = gdp_df.iloc[-1]
                economic_data['gdp_growth'] = {
                    'value': round(float(latest_gdp['国内生产总值-同比增长']), 1),
                    'name': 'GDP增长率',
                    'unit': '%',
                    'trend': 'up' if float(latest_gdp['国内生产总值-同比增长']) > 5 else 'stable'
                }
        except:
            economic_data['gdp_growth'] = {
                'value': 5.2,
                'name': 'GDP增长率',
                'unit': '%',
                'trend': 'stable'
            }
        
        # 2. 获取CPI数据
        try:
            cpi_df = ak.macro_china_cpi()
            if not cpi_df.empty:
                latest_cpi = cpi_df.iloc[-1]
                economic_data['cpi'] = {
                    'value': round(float(latest_cpi['全国']), 1),
                    'name': '居民消费价格指数',
                    'unit': '%',
                    'trend': 'up' if float(latest_cpi['全国']) > 3 else 'stable'
                }
        except:
            economic_data['cpi'] = {
                'value': 2.1,
                'name': '居民消费价格指数',
                'unit': '%',
                'trend': 'stable'
            }
        
        # 3. 获取PPI数据
        try:
            ppi_df = ak.macro_china_ppi()
            if not ppi_df.empty:
                latest_ppi = ppi_df.iloc[-1]
                economic_data['ppi'] = {
                    'value': round(float(latest_ppi['当月']), 1),
                    'name': '工业生产者出厂价格',
                    'unit': '%',
                    'trend': 'up' if float(latest_ppi['当月']) > 0 else 'down'
                }
        except:
            economic_data['ppi'] = {
                'value': -1.2,
                'name': '工业生产者出厂价格',
                'unit': '%',
                'trend': 'down'
            }
        
        # 4. 获取PMI数据
        try:
            pmi_df = ak.macro_china_pmi()
            if not pmi_df.empty:
                latest_pmi = pmi_df.iloc[-1]
                economic_data['pmi'] = {
                    'value': round(float(latest_pmi['制造业PMI']), 1),
                    'name': '制造业PMI',
                    'unit': '',
                    'trend': 'up' if float(latest_pmi['制造业PMI']) > 50 else 'down'
                }
        except:
            economic_data['pmi'] = {
                'value': 50.1,
                'name': '制造业PMI',
                'unit': '',
                'trend': 'up'
            }
        
        # 5. 获取汇率数据
        try:
            rate_df = ak.macro_china_rmb()
            if not rate_df.empty:
                latest_rate = rate_df.iloc[-1]
                economic_data['exchange_rate'] = {
                    'value': round(float(latest_rate['中间价']), 2),
                    'name': '人民币汇率',
                    'unit': 'CNY/USD',
                    'trend': 'up' if float(latest_rate['中间价']) > 7.0 else 'down'
                }
        except:
            economic_data['exchange_rate'] = {
                'value': 7.12,
                'name': '人民币汇率',
                'unit': 'CNY/USD',
                'trend': 'stable'
            }
        
        return economic_data
        
    except Exception as e:
        st.warning(f"获取宏观经济数据失败: {str(e)}")
        # 返回模拟数据
        return {
            'gdp_growth': {'value': 5.2, 'name': 'GDP增长率', 'unit': '%', 'trend': 'stable'},
            'cpi': {'value': 2.1, 'name': '居民消费价格指数', 'unit': '%', 'trend': 'stable'},
            'ppi': {'value': -1.2, 'name': '工业生产者出厂价格', 'unit': '%', 'trend': 'down'},
            'pmi': {'value': 50.1, 'name': '制造业PMI', 'unit': '', 'trend': 'up'},
            'exchange_rate': {'value': 7.12, 'name': '人民币汇率', 'unit': 'CNY/USD', 'trend': 'stable'}
        }

# ====================== 技术指标计算 ======================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算完整的技术指标"""
    df = df.copy()
    
    # 确保有足够的数据
    if len(df) < 60:
        st.warning("数据量不足，部分指标可能不准确")
    
    # 价格指标
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df['ma120'] = df['close'].rolling(window=120).mean()
    
    # 指数移动平均线
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
    df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower']) * 100
    
    # KDJ
    low_9 = df['low'].rolling(window=9).min()
    high_9 = df['high'].rolling(window=9).max()
    df['rsv'] = (df['close'] - low_9) / (high_9 - low_9) * 100
    df['kdj_k'] = df['rsv'].ewm(com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 成交量指标
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma10'] = df['volume'].rolling(window=10).mean()
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # OBV能量潮
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # ATR平均真实波幅
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # 乖离率
    df['bias5'] = (df['close'] - df['ma5']) / df['ma5'] * 100
    df['bias10'] = (df['close'] - df['ma10']) / df['ma10'] * 100
    df['bias20'] = (df['close'] - df['ma20']) / df['ma20'] * 100
    
    # 波动率
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    return df

# ====================== 斐波那契分析 ======================
def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> Tuple[Dict, float, float]:
    """计算斐波那契回调位"""
    recent_data = df.tail(lookback)
    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()
    diff = recent_high - recent_low
    
    fib_levels = {
        "0.0% (高点)": recent_high,
        "23.6%": recent_high - diff * 0.236,
        "38.2%": recent_high - diff * 0.382,
        "50.0%": recent_high - diff * 0.5,
        "61.8%": recent_high - diff * 0.618,
        "78.6%": recent_high - diff * 0.786,
        "100.0% (低点)": recent_low
    }
    
    # 计算扩展位
    fib_extensions = {
        "127.2%": recent_low - diff * 0.272,
        "161.8%": recent_low - diff * 0.618,
        "261.8%": recent_low - diff * 1.618
    }
    
    fib_levels.update(fib_extensions)
    
    return fib_levels, recent_high, recent_low

# ====================== 信号分析 ======================
def analyze_signals(df: pd.DataFrame) -> Dict:
    """分析技术信号"""
    latest = df.iloc[-1]
    
    signals = {
        'buy': [],
        'sell': [],
        'neutral': []
    }
    
    # 移动平均线信号
    if latest['ma5'] > latest['ma10'] > latest['ma20']:
        signals['buy'].append("均线多头排列")
    elif latest['ma5'] < latest['ma10'] < latest['ma20']:
        signals['sell'].append("均线空头排列")
    
    if latest['close'] > latest['ma20']:
        signals['buy'].append("价格在20日均线上方")
    else:
        signals['sell'].append("价格在20日均线下方")
    
    # MACD信号
    if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
        signals['buy'].append("MACD金叉且红柱")
    elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
        signals['sell'].append("MACD死叉且绿柱")
    
    # RSI信号
    if latest['rsi'] < 30:
        signals['buy'].append("RSI超卖")
    elif latest['rsi'] > 70:
        signals['sell'].append("RSI超买")
    
    # KDJ信号
    if latest['kdj_j'] < 20:
        signals['buy'].append("KDJ超卖")
    elif latest['kdj_j'] > 80:
        signals['sell'].append("KDJ超买")
    
    # 布林带信号
    if latest['boll_position'] < 20:
        signals['buy'].append("价格在布林下轨")
    elif latest['boll_position'] > 80:
        signals['sell'].append("价格在布林上轨")
    
    # 成交量信号
    if latest['volume_ratio'] > 1.5:
        if latest['close'] > latest['open']:
            signals['buy'].append("放量上涨")
        else:
            signals['sell'].append("放量下跌")
    
    # 计算信号分数
    score = len(signals['buy']) - len(signals['sell'])
    
    # 总体信号判断
    if score >= 3:
        overall_signal = "强烈买入"
        signal_color = "green"
    elif score >= 1:
        overall_signal = "买入"
        signal_color = "green"
    elif score >= -1:
        overall_signal = "中性"
        signal_color = "orange"
    elif score >= -3:
        overall_signal = "卖出"
        signal_color = "red"
    else:
        overall_signal = "强烈卖出"
        signal_color = "red"
    
    return {
        'signals': signals,
        'score': score,
        'overall_signal': overall_signal,
        'signal_color': signal_color,
        'latest': latest
    }

# ====================== 操作建议计算 ======================
def calculate_trading_advice(df: pd.DataFrame, signals: Dict, period: str = "daily"):
    """计算交易建议，包括止盈止损（按周期适配）"""
    latest = df.iloc[-1]
    current_price = latest['close']
    
    # 根据周期调整止损止盈比例（核心优化）
    period_params = {
        "daily": {
            "stop_loss_pct": 3.0,    # 日线止损3%
            "take_profit_pct": 6.0,  # 日线止盈6%
            "risk_reward_base": 2.0
        },
        "weekly": {
            "stop_loss_pct": 5.0,    # 周线止损5%
            "take_profit_pct": 10.0, # 周线止盈10%
            "risk_reward_base": 2.0
        },
        "monthly": {
            "stop_loss_pct": 8.0,    # 月线止损8%
            "take_profit_pct": 15.0, # 月线止盈15%
            "risk_reward_base": 1.8
        }
    }
    
    params = period_params.get(period, period_params["daily"])
    
    # 根据信号强度调整比例
    signal_strength = len(signals['signals']['buy']) - len(signals['signals']['sell'])
    if signal_strength >= 3:  # 强烈买入
        params["take_profit_pct"] *= 1.2
        params["stop_loss_pct"] *= 0.8
    elif signal_strength <= -3:  # 强烈卖出
        params["take_profit_pct"] *= 1.2
        params["stop_loss_pct"] *= 0.8
    
    # 根据信号确定操作建议
    if signals['overall_signal'] in ["强烈买入", "买入"]:
        action = "买入"
        entry_price = current_price * 0.99  # 建议买入价格略低于当前价
        stop_loss = entry_price * (1 - params["stop_loss_pct"] / 100)
        take_profit = entry_price * (1 + params["take_profit_pct"] / 100)
        risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
        
    elif signals['overall_signal'] in ["卖出", "强烈卖出"]:
        action = "卖出"
        entry_price = current_price * 1.01  # 建议卖出价格略高于当前价
        stop_loss = entry_price * (1 + params["stop_loss_pct"] / 100)
        take_profit = entry_price * (1 - params["take_profit_pct"] / 100)
        risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
        
    else:  # 中性
        action = "观望"
        entry_price = current_price
        stop_loss = current_price * (1 - 2.0 / 100)
        take_profit = current_price * (1 + 4.0 / 100)
        risk_reward = params["risk_reward_base"]
    
    # 计算支撑阻力位
    support_levels = []
    resistance_levels = []
    
    # 使用布林带作为支撑阻力参考
    if 'boll_lower' in latest and 'boll_upper' in latest:
        support_levels.append(("布林下轨", latest['boll_lower']))
        resistance_levels.append(("布林上轨", latest['boll_upper']))
    
    # 使用移动平均线作为支撑阻力参考
    for ma_period in [5, 10, 20, 30, 60]:
        ma_key = f'ma{ma_period}'
        if ma_key in latest:
            ma_value = latest[ma_key]
            if current_price > ma_value:
                support_levels.append((f"MA{ma_period}", ma_value))
            else:
                resistance_levels.append((f"MA{ma_period}", ma_value))
    
    # 获取近期高低点
    recent_low = df['low'].tail(20).min()
    recent_high = df['high'].tail(20).max()
    
    support_levels.append(("近期低点", recent_low))
    resistance_levels.append(("近期高点", recent_high))
    
    # 按价格排序
    support_levels.sort(key=lambda x: x[1], reverse=True)
    resistance_levels.sort(key=lambda x: x[1])
    
    advice = {
        'action': action,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': risk_reward,
        'support_levels': support_levels[:3],  # 取前3个支撑位
        'resistance_levels': resistance_levels[:3],  # 取前3个阻力位
        'period': get_period_name(period),
        'stop_loss_pct': params["stop_loss_pct"],
        'take_profit_pct': params["take_profit_pct"]
    }

    return advice

# ====================== 可视化函数 ======================
def create_price_chart_plotly(df: pd.DataFrame, stock_code: str, stock_name: str, period: str = "daily"):
    """创建Plotly价格图表"""
    period_name = get_period_name(period)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f'{stock_code} {stock_name} - {period_name}走势', '成交量', 'MACD', 'RSI')
    )
    
    # 1. 价格图表
    # 添加K线
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#ef4444',
            decreasing_line_color='#10b981'
        ),
        row=1, col=1
    )
    
    # 添加移动平均线
    for ma_period, color in [(5, '#dc2626'), (10, '#f59e0b'), (20, '#10b981'), (60, '#3b82f6')]:
        if f'ma{ma_period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[f'ma{ma_period}'],
                    name=f'MA{ma_period}',
                    line=dict(color=color, width=1.5),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # 2. 成交量
    colors = ['#ef4444' if close >= open_ else '#10b981' 
              for close, open_ in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    if 'volume_ma5' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['volume_ma5'],
                name='VMA5',
                line=dict(color='#3b82f6', width=1.5)
            ),
            row=2, col=1
        )
    
    # 3. MACD
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['macd'],
            name='DIF',
            line=dict(color='#3b82f6', width=1.5)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['macd_signal'],
            name='DEA',
            line=dict(color='#f59e0b', width=1.5)
        ),
        row=3, col=1
    )
    
    macd_colors = ['#ef4444' if val > 0 else '#10b981' for val in df['macd_histogram']]
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['macd_histogram'],
            name='MACD',
            marker_color=macd_colors,
            opacity=0.5
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'), row=3, col=1)
    
    # 4. RSI
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['rsi'],
            name='RSI',
            line=dict(color='#8b5cf6', width=2)
        ),
        row=4, col=1
    )
    
    # 添加RSI水平线
    fig.add_hline(y=70, line=dict(color='#ef4444', width=1, dash='dash'), row=4, col=1)
    fig.add_hline(y=30, line=dict(color='#10b981', width=1, dash='dash'), row=4, col=1)
    fig.add_hline(y=50, line=dict(color='#6b7280', width=0.5, dash='dot'), row=4, col=1)
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=f'{stock_code} {stock_name} - {period_name}技术分析',
            font=dict(size=20, color='#1e40af'),
            x=0.5
        ),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    # 更新坐标轴
    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
    fig.update_xaxes(title_text="日期", row=4, col=1)
    
    return fig

def create_fibonacci_chart(df: pd.DataFrame, fib_levels: Dict, recent_high: float, recent_low: float):
    """创建斐波那契回调图"""
    fig = go.Figure()
    
    # 价格线
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='收盘价',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # 斐波那契水平线
    fib_colors = {
        '0.0% (高点)': '#ef4444',
        '23.6%': '#f59e0b',
        '38.2%': '#10b981',
        '50.0%': '#8b5cf6',
        '61.8%': '#ec4899',
        '78.6%': '#6366f1',
        '100.0% (低点)': '#3b82f6'
    }
    
    for level, price in fib_levels.items():
        if level in fib_colors:
            fig.add_hline(
                y=price,
                line=dict(
                    color=fib_colors[level],
                    width=2 if level in ['38.2%', '61.8%'] else 1,
                    dash='dash' if level in ['0.0% (高点)', '100.0% (低点)'] else 'solid'
                ),
                annotation_text=level,
                annotation_position="right"
            )
    
    # 填充区域
    fig.add_hrect(
        y0=fib_levels.get('38.2%', 0),
        y1=fib_levels.get('61.8%', 0),
        fillcolor="rgba(16, 185, 129, 0.1)",
        line_width=0,
        annotation_text="强支撑区",
        annotation_position="left"
    )
    
    fig.update_layout(
        title=dict(
            text="斐波那契回调分析",
            font=dict(size=18, color='#1e40af')
        ),
        height=400,
        xaxis_title="日期",
        yaxis_title="价格 (元)",
        showlegend=True,
        hovermode='x unified',
        template="plotly_white"
    )
    
    return fig

def create_technical_summary(df: pd.DataFrame):
    """创建技术指标汇总图表"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('布林带', 'KDJ指标', '成交量比率', '波动率'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. 布林带
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name='收盘价', line=dict(color='#3b82f6')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['boll_upper'], name='上轨', line=dict(color='#ef4444', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['boll_mid'], name='中轨', line=dict(color='#6b7280')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['boll_lower'], name='下轨', line=dict(color='#10b981', dash='dash'),
                  fill='tonexty'),
        row=1, col=1
    )
    
    # 2. KDJ
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['kdj_k'], name='K值', line=dict(color='#3b82f6')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['kdj_d'], name='D值', line=dict(color='#f59e0b')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['kdj_j'], name='J值', line=dict(color='#8b5cf6')),
        row=1, col=2
    )
    fig.add_hline(y=80, line=dict(color='#ef4444', dash='dash'), row=1, col=2)
    fig.add_hline(y=20, line=dict(color='#10b981', dash='dash'), row=1, col=2)
    
    # 3. 成交量比率
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume_ratio'], name='量比', marker_color='#6366f1'),
        row=2, col=1
    )
    fig.add_hline(y=1, line=dict(color='#6b7280', dash='dash'), row=2, col=1)
    fig.add_hline(y=1.5, line=dict(color='#f59e0b', dash='dash'), row=2, col=1)
    fig.add_hline(y=2, line=dict(color='#ef4444', dash='dash'), row=2, col=1)
    
    # 4. 波动率
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['volatility_20'], name='20日波动率', 
                  fill='tozeroy', line=dict(color='#ec4899')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="技术指标汇总", template="plotly_white")
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=2)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="KDJ值", row=1, col=2)
    fig.update_yaxes(title_text="量比", row=2, col=1)
    fig.update_yaxes(title_text="波动率%", row=2, col=2)
    
    return fig

# ====================== 侧边栏配置 ======================
# 先在文件顶部导入后添加兼容函数
def safe_rerun():
    """兼容不同Streamlit版本的刷新函数"""
    try:
        st.rerun()  # 新版本
    except AttributeError:
        st.experimental_rerun()  # 旧版本
def create_sidebar():
    """创建侧边栏（优化热点股票布局+代码更新逻辑）"""
    with st.sidebar:
        # Logo和标题
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://img.icons8.com/color/96/000000/stock-share.png", width=60)
        with col2:
            st.markdown("### 📈 专业股票分析")
        
        st.markdown("---")
        
        # 股票选择
        st.markdown("#### 🎯 股票设置")
        
        # 股票代码输入（优化更新逻辑）
        stock_input = st.text_input(
            "股票代码",
            value=st.session_state.selected_stock,
            placeholder="输入6位股票代码",
            help="例如：000001（平安银行），600519（贵州茅台）",
            key="stock_input"
        )
        
        # 强制更新逻辑
        if stock_input and stock_input != st.session_state.selected_stock:
            st.session_state.selected_stock = stock_input
            st.session_state.refresh_trigger += 1  # 触发刷新
        
        data_source = st.selectbox(
            "数据源",
            ["akshare", "yfinance"],
            help="选择数据来源，akshare用于A股，yfinance用于A股/港股/美股"
        )
        
        # K线周期选择
        kline_period = st.selectbox(
            "K线周期",
            ["daily", "weekly", "monthly"],
            format_func=lambda x: {"daily": "日K线", "weekly": "周K线", "monthly": "月K线"}[x],
            help="选择K线周期进行分析"
        )
        
        # 更新session state中的K线周期
        if kline_period != st.session_state.kline_period:
            st.session_state.kline_period = kline_period
            st.session_state.refresh_trigger += 1
        
        st.markdown("---")
        
        # 分析参数
        st.markdown("#### ⚙️ 分析参数")
        
        lookback_days = st.slider(
            "分析周期（天）",
            min_value=30,
            max_value=250,
            value=120,
            step=10,
            help="选择分析的时间范围"
        )
        
        st.markdown("---")
        
        # 技术指标选择
        st.markdown("#### 📊 技术指标")
        col1, col2 = st.columns(2)
        with col1:
            show_rsi = st.checkbox("RSI", value=True)
            show_macd = st.checkbox("MACD", value=True)
        with col2:
            show_kdj = st.checkbox("KDJ", value=True)
            show_boll = st.checkbox("布林带", value=True)
        
        show_fib = st.checkbox("斐波那契", value=True)
        
        st.markdown("---")
        
        # 热点股票（优化布局+样式）
        st.markdown("#### 🔥 热门股票")
        
        # 预设股票列表
        popular_stocks = {
            "兆易创新": "603986",
            "贵州茅台": "600519",
            "宁德时代": "300750",
            "比亚迪": "002594",
            "五粮液": "000858",
            "招商银行": "600036",
            "中国平安": "601318",
            "美的集团": "000333",
            "东方财富": "300059",
            "海康威视": "002415"
        }
        
        # 使用2行5列网格布局（优化对齐）
        stock_list = list(popular_stocks.items())
        rows = [stock_list[i:i+5] for i in range(0, len(stock_list), 5)]
        
        for row in rows:
            cols = st.columns(5)
            for idx, (stock_name, stock_code) in enumerate(row):
                with cols[idx]:
                        if st.button(
                            stock_name,
                            key=f"btn_{stock_code}",
                            width='stretch'
                        ):
                            st.session_state.selected_stock = stock_code
                            st.session_state.refresh_trigger += 1
                            safe_rerun()  # 替换st.rerun()
        
        st.markdown("---")
        
        # 更新按钮
        if st.button("🔄 强制更新数据"):
            st.session_state.refresh_trigger += 1
            safe_rerun()  # 替换st.rerun()
        
        st.markdown("---")
        
        # 免责声明
        st.caption("⚠️ 风险提示：本工具仅供参考，不构成投资建议")

# ====================== 主面板组件 ======================
def display_metrics_panel(df: pd.DataFrame, stock_code: str, stock_name: str, signals: Dict):
    """显示指标面板"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_change = latest['close'] - prev['close']
        price_change_pct = (price_change / prev['close']) * 100
        
        st.metric(
            label="当前价格",
            value=f"¥{latest['close']:.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric(
            label="成交量",
            value=f"{latest['volume']/1e6:.2f}M",
            delta=f"量比: {latest.get('volume_ratio', 1):.2f}" if 'volume_ratio' in latest else None
        )
    
    with col3:
        rsi_color = "normal"
        if 'rsi' in latest:
            if latest['rsi'] > 70:
                rsi_color = "inverse"
            elif latest['rsi'] < 30:
                rsi_color = "off"
        st.metric(
            label="RSI(14)",
            value=f"{latest.get('rsi', 0):.1f}" if 'rsi' in latest else "N/A",
            delta="超买" if latest.get('rsi', 0) > 70 else "超卖" if latest.get('rsi', 0) < 30 else "正常",
            delta_color=rsi_color
        )
    
    with col4:
        macd_status = "看涨" if latest.get('macd', 0) > latest.get('macd_signal', 0) else "看跌"
        st.metric(
            label="MACD",
            value=f"{latest.get('macd', 0):.4f}" if 'macd' in latest else "N/A",
            delta=macd_status
        )
    
    # 应用卡片样式
    style_metric_cards(
        background_color="#FFFFFF",
        border_size_px=1,
        border_color="#DDDDDD",
        border_radius_px=10,
        border_left_color="#3B82F6"
    )

def display_signal_panel(signals: Dict):
    """优化版：美化信号展示（卡片化+彩色标签+图标）"""
    st.markdown("### 📊 交易信号")
    
    # 主信号卡片
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        signal_config = {
            "强烈买入": {"color1": "#059669", "color2": "#10b981", "icon": "📈", "text": "强烈买入"},
            "买入": {"color1": "#10b981", "color2": "#34d399", "icon": "🟢", "text": "买入"},
            "中性": {"color1": "#d97706", "color2": "#f59e0b", "icon": "🟡", "text": "中性"},
            "卖出": {"color1": "#dc2626", "color2": "#ef4444", "icon": "🔴", "text": "卖出"},
            "强烈卖出": {"color1": "#b91c1c", "color2": "#dc2626", "icon": "📉", "text": "强烈卖出"}
        }
        config = signal_config.get(signals['overall_signal'], signal_config["中性"])
        
        st.markdown(f"""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, {config['color1']} 0%, {config['color2']} 100%); border-radius: 16px; color: white; box-shadow: 0 8px 24px rgba(0,0,0,0.15);">
            <h3 style="margin: 0; font-size: 22px; font-weight: 600;">{config['icon']} {config['text']}</h3>
            <h1 style="margin: 15px 0; font-size: 56px; font-weight: 700;">{signals['score']}</h1>
            <p style="margin: 0; opacity: 0.9; font-size: 16px;">综合信号评分</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 信号详情（美化版）
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # 买入信号
    with col1:
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border-radius: 12px; padding: 20px; border: 1px solid #d1fae5;">
            <h4 style="color: #065f46; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
                <span>✅ 买入信号</span>
                <span style="background-color: #10b981; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">{len(signals['signals']['buy'])}</span>
            </h4>
        """, unsafe_allow_html=True)
        
        if signals['signals']['buy']:
            for signal in signals['signals']['buy'][:6]:  # 最多显示6个
                st.markdown(f"""<div class="signal-tag buy-tag">📌 {signal}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #6b7280; text-align: center; padding: 20px 0;'>暂无买入信号</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 卖出信号
    with col2:
        st.markdown(f"""
        <div style="background-color: #fef2f2; border-radius: 12px; padding: 20px; border: 1px solid #fee2e2;">
            <h4 style="color: #991b1b; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
                <span>❌ 卖出信号</span>
                <span style="background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">{len(signals['signals']['sell'])}</span>
            </h4>
        """, unsafe_allow_html=True)
        
        if signals['signals']['sell']:
            for signal in signals['signals']['sell'][:6]:
                st.markdown(f"""<div class="signal-tag sell-tag">📌 {signal}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #6b7280; text-align: center; padding: 20px 0;'>暂无卖出信号</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 中性信号
    with col3:
        st.markdown(f"""
        <div style="background-color: #fffbeb; border-radius: 12px; padding: 20px; border: 1px solid #fef3c7;">
            <h4 style="color: #92400e; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">
                <span>⚠️ 中性信号</span>
                <span style="background-color: #f59e0b; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">{len(signals['signals']['neutral'])}</span>
            </h4>
        """, unsafe_allow_html=True)
        
        if signals['signals']['neutral']:
            for signal in signals['signals']['neutral'][:6]:
                st.markdown(f"""<div class="signal-tag neutral-tag">📌 {signal}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #6b7280; text-align: center; padding: 20px 0;'>暂无中性信号</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def display_fibonacci_panel(fib_levels: Dict, current_price: float):
    """显示斐波那契面板"""
    st.markdown("### 📐 关键价位")
    
    # 将斐波那契水平按价格排序
    sorted_levels = sorted(fib_levels.items(), key=lambda x: x[1], reverse=True)
    
    # 使用5列展示关键价位
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 筛选核心斐波那契水平
    key_levels = ['0.0% (高点)', '38.2%', '50.0%', '61.8%', '100.0% (低点)']
    level_data = [(level, fib_levels[level]) for level in key_levels if level in fib_levels]
    
    for idx, (level, price) in enumerate(level_data):
        with [col1, col2, col3, col4, col5][idx]:
            # 判断当前价格位置
            if price > current_price:
                status = "阻力位"
                color = "#ef4444"
                icon = "🔴"
            else:
                status = "支撑位"
                color = "#10b981"
                icon = "🟢"
            
            st.markdown(f"""
            <div style="background-color: #f8fafc; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #e2e8f0;">
                <div style="color: {color}; font-weight: 700; font-size: 18px;">¥{price:.2f}</div>
                <div style="color: #64748b; font-size: 12px; margin: 5px 0;">{level}</div>
                <div style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 8px; font-size: 11px; display: inline-block;">{icon} {status}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 详细斐波那契表格
    st.markdown("#### 完整斐波那契水平")
    fib_df = pd.DataFrame(list(fib_levels.items()), columns=['水平', '价格(元)'])
    fib_df['与现价差'] = (fib_df['价格(元)'] - current_price).round(2)
    fib_df['差值%'] = ((fib_df['价格(元)'] - current_price) / current_price * 100).round(2)
    
    # 高亮当前价格附近的水平
    def highlight_row(row):
        price_diff = abs(row['与现价差']) / current_price * 100
        if price_diff < 1:  # 1%以内高亮
            return ['background-color: #fef3c7; color: #92400e; font-weight: 600'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        fib_df.style.apply(highlight_row, axis=1),
        width='stretch',  # 替换 use_container_width=True
        hide_index=True
    )

def display_trading_advice_panel(advice: Dict):
    """显示交易建议面板"""
    st.markdown("### 🎯 操作建议")
    
    # 主建议卡片
    action_config = {
        "买入": {"color": "#059669", "icon": "📈", "bg": "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"},
        "卖出": {"color": "#dc2626", "icon": "📉", "bg": "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"},
        "观望": {"color": "#d97706", "icon": "📊", "bg": "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)"}
    }
    
    config = action_config.get(advice['action'], action_config["观望"])
    
    st.markdown(f"""
    <div style="background: {config['bg']}; border-radius: 12px; padding: 20px; border: 1px solid #e5e7eb; margin-bottom: 20px;">
        <h4 style="color: {config['color']}; margin: 0 0 10px 0; font-size: 18px;">
            {config['icon']} 核心操作：{advice['action']}
        </h4>
        <p style="color: #4b5563; margin: 0;">
            周期：{advice['period']} | 风险收益比：{advice['risk_reward']:.2f}:1
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 价格参数
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="economic-card">
            <div class="economic-title">建议{'买入' if advice['action']=='买入' else '卖出'}价</div>
            <div class="economic-value">¥{advice['entry_price']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="economic-card">
            <div class="economic-title">止损价</div>
            <div class="economic-value" style="color: {'#dc2626' if advice['action']!='观望' else '#6b7280'}">
                ¥{advice['stop_loss']:.2f}
            </div>
            <div class="economic-change negative">-{advice['stop_loss_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="economic-card">
            <div class="economic-title">止盈价</div>
            <div class="economic-value" style="color: {'#059669' if advice['action']!='观望' else '#6b7280'}">
                ¥{advice['take_profit']:.2f}
            </div>
            <div class="economic-change positive">+{advice['take_profit_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="economic-card">
            <div class="economic-title">风险收益比</div>
            <div class="economic-value">
                {advice['risk_reward']:.2f}:1
            </div>
            <div class="economic-change {'positive' if advice['risk_reward']>=1.5 else 'negative'}">
                {'优秀' if advice['risk_reward']>=2 else '良好' if advice['risk_reward']>=1.5 else '一般'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 支撑阻力位
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🛡️ 支撑位")
        support_data = []
        for name, price in advice['support_levels']:
            diff_pct = (price - advice['entry_price']) / advice['entry_price'] * 100
            support_data.append({
                '名称': name,
                '价格(元)': f"{price:.2f}",
                '与入场价差': f"{diff_pct:.2f}%"
            })
        
        support_df = pd.DataFrame(support_data)
        st.dataframe(support_df, width='stretch', hide_index=True)
    
    with col2:
        st.markdown("#### 🚫 阻力位")
        resistance_data = []
        for name, price in advice['resistance_levels']:
            diff_pct = (price - advice['entry_price']) / advice['entry_price'] * 100
            resistance_data.append({
                '名称': name,
                '价格(元)': f"{price:.2f}",
                '与入场价差': f"{diff_pct:.2f}%"
            })
        
        resistance_df = pd.DataFrame(resistance_data)
        st.dataframe(resistance_df, width='stretch', hide_index=True)
    
    # 操作提示
    st.markdown("#### 💡 操作提示")
    tips = {
        "买入": [
            "建议分批建仓，避免一次性满仓",
            "严格设置止损，控制单笔风险在总资金1-2%",
            "突破阻力位可加仓，跌破止损位果断离场",
            "止盈可分阶段止盈，保留部分仓位博取更大收益"
        ],
        "卖出": [
            "建议分批减仓，避免一次性清仓",
            "跌破支撑位可加仓卖出，突破阻力位及时止损",
            "反弹至关键阻力位可加码卖出",
            "下跌趋势中不轻易抄底"
        ],
        "观望": [
            "等待明确信号出现后再操作",
            "关注成交量变化，放量突破/跌破是关键",
            "可小仓位试错，验证方向后再加码",
            "设置预警价位，及时捕捉交易机会"
        ]
    }
    
    current_tips = tips.get(advice['action'], tips["观望"])
    for i, tip in enumerate(current_tips, 1):
        st.markdown(f"""
        <div style="display: flex; align-items: flex-start; gap: 8px; margin: 8px 0;">
            <span style="color: {config['color']}; font-weight: 700;">{i}.</span>
            <span style="color: #374151;">{tip}</span>
        </div>
        """, unsafe_allow_html=True)

def display_economic_data_panel():
    """显示宏观经济数据面板"""
    st.markdown("### 📊 宏观经济数据")
    
    economic_data = get_economic_data()
    
    # 4列展示核心经济指标
    col1, col2, col3, col4, col5 = st.columns(5)
    
    indicators = ['gdp_growth', 'cpi', 'ppi', 'pmi', 'exchange_rate']
    cols = [col1, col2, col3, col4, col5]
    
    for idx, indicator in enumerate(indicators):
        data = economic_data[indicator]
        with cols[idx]:
            trend_icon = {
                'up': "📈",
                'down': "📉",
                'stable': "📊"
            }.get(data['trend'], "📊")
            
            st.markdown(f"""
            <div class="economic-card">
                <div class="economic-title">{data['name']}</div>
                <div class="economic-value">{data['value']}{data['unit']}</div>
                <div class="economic-change {data['trend']}">{trend_icon} {data['trend']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 经济数据解读
    st.markdown("#### 💡 经济数据解读")
    st.markdown("""
    <div class="advice-card">
        <div class="advice-title">📝 宏观经济对股市影响</div>
        <div class="advice-item">
            <div class="advice-label">GDP增长率</div>
            <div class="advice-value">
                GDP增速反映经济基本面，5%以上为健康增长，利好股市整体表现
            </div>
        </div>
        <div class="advice-item">
            <div class="advice-label">CPI/PPI</div>
            <div class="advice-value">
                CPI温和上涨（2-3%）有利于经济，PPI转正表明工业企业盈利改善
            </div>
        </div>
        <div class="advice-item">
            <div class="advice-label">制造业PMI</div>
            <div class="advice-value">
                PMI>50表明经济扩张，<50则收缩，是判断经济周期的重要指标
            </div>
        </div>
        <div class="advice-item">
            <div class="advice-label">人民币汇率</div>
            <div class="advice-value">
                汇率稳定有利于外资流入，大幅波动会增加市场不确定性
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================== 主程序入口 ======================
def main():
    """主程序"""
    # 初始化Session State（强制重置，新用户无历史痕迹）
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = "603986"  # 默认兆易创新
    if 'refresh_trigger' not in st.session_state:
        st.session_state.refresh_trigger = 0
    if 'kline_period' not in st.session_state:
        st.session_state.kline_period = "daily"  # 默认日K线
    
    # 应用自定义样式
    apply_custom_styles()
    
    # 设置页面标题
    st.markdown("# 📈 专业股票技术分析系统")
    st.markdown("---")
    
    # 创建侧边栏
    create_sidebar()
    
    # 获取用户输入
    stock_code = st.session_state.selected_stock
    kline_period = st.session_state.kline_period
    
    # 验证股票代码
    if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
        st.warning("请输入有效的6位股票代码！")
        st.stop()
    
    # 股票名称映射（简化版）
    stock_name_map = {
        "000001": "平安银行", "000002": "万科A", "000858": "五粮液",
        "002415": "海康威视", "002594": "比亚迪", "300059": "东方财富",
        "300750": "宁德时代", "600036": "招商银行", "600519": "贵州茅台",
        "601318": "中国平安", "603986": "兆易创新", "000333": "美的集团"
    }
    stock_name = stock_name_map.get(stock_code, f"股票({stock_code})")
    
    # 主面板布局
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 技术分析", 
        "🎯 交易建议", 
        "📐 斐波那契分析", 
        "🌐 宏观经济"
    ])
    
    try:
        # 获取股票数据
        df = get_stock_data_enhanced(
            stock_code=stock_code,
            days=120,
            data_source="akshare",
            period=kline_period
        )
        
        if df.empty:
            st.error("无法获取股票数据，请检查代码或稍后重试！")
            st.stop()
        
        # 计算技术指标
        df = calculate_technical_indicators(df)
        
        # 分析交易信号
        signals = analyze_signals(df)
        
        # 计算交易建议
        trading_advice = calculate_trading_advice(df, signals, kline_period)
        
        # 计算斐波那契水平
        fib_levels, recent_high, recent_low = calculate_fibonacci_levels(df)
        
        # ========== 技术分析标签页 ==========
        with tab1:
            # 显示核心指标
            display_metrics_panel(df, stock_code, stock_name, signals)
            st.markdown("---")
            
            # 显示价格图表
            st.markdown("#### 📈 K线图与技术指标")
            price_fig = create_price_chart_plotly(df, stock_code, stock_name, kline_period)
            st.plotly_chart(price_fig, width='stretch')
            
            # 显示技术指标汇总
            st.markdown("#### 📊 技术指标汇总")
            summary_fig = create_technical_summary(df)
            st.plotly_chart(summary_fig, width='stretch')
            
            # 显示交易信号
            st.markdown("---")
            display_signal_panel(signals)
        
        # ========== 交易建议标签页 ==========
        with tab2:
            display_trading_advice_panel(trading_advice)
        
        # ========== 斐波那契分析标签页 ==========
        with tab3:
            # 斐波那契图表
            st.markdown("#### 📐 斐波那契回调图")
            fib_fig = create_fibonacci_chart(df, fib_levels, recent_high, recent_low)
            st.plotly_chart(fib_fig, width='stretch')
            
            # 斐波那契关键价位
            display_fibonacci_panel(fib_levels, df.iloc[-1]['close'])
        
        # ========== 宏观经济标签页 ==========
        with tab4:
            display_economic_data_panel()
        
        # 数据导出功能
        st.markdown("---")
        col1, col2 = st.columns([1, 10])
        with col1:
            # 准备导出数据
            export_df = df[['date', 'open', 'high', 'low', 'close', 'volume', 
                           'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'kdj_k', 'kdj_d', 'kdj_j']].copy()
            
            # ========== 彻底修复：分3步处理日期 ==========
            # 1. 强制转换为datetime（兜底，避免源头转换失效）
            export_df['date'] = pd.to_datetime(export_df['date'], errors='coerce')
            # 2. 格式化日期（NaT转为空字符串，避免报错）
            export_df['date'] = export_df['date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
            # 3. 填充所有空值（避免CSV导出异常）
            export_df = export_df.fillna('')
            # ===========================================
            
            # 生成CSV
            csv = export_df.to_csv(index=False, encoding='utf-8-sig')
            b64 = base64.b64encode(csv.encode()).decode()
            
            st.download_button(
                label="💾 导出数据",
                data=b64,
                file_name=f"{stock_code}_{stock_name}_{kline_period}_数据.csv",
                mime="text/csv",
                width='stretch'  # 替换use_container_width=True，解决警告
            )
    
    except Exception as e:
        st.error(f"程序运行出错：{str(e)}")
        st.exception(e)  # 显示详细错误信息

# 运行主程序
if __name__ == "__main__":
    main()
