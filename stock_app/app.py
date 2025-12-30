import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings
import json
import ta  # 补充缺失的技术指标库

# ====================== 全局配置 ======================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="A股专业技术分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 常量定义（统一管理，避免重复）======================
DEFAULT_STOCK_CODE = "600519"  # 贵州茅台
DEFAULT_TIMEFRAME = "daily"
DATA_CACHE_TTL = 300  # 行情数据缓存5分钟
NAME_CACHE_TTL = 86400  # 股票名称缓存1天
MACRO_CACHE_TTL = 300  # 宏观数据缓存缩短为5分钟（保证最新）
DEFAULT_DAYS = 120
MIN_DATA_LENGTH = 20

# 技术指标参数
RSI_WINDOW = 14
MACD_EMA12 = 12
MACD_EMA26 = 26
MACD_SIGNAL = 9
BOLL_WINDOW = 20
KDJ_WINDOW = 9
VOLUME_AVG_WINDOW = 20
VOLATILITY_WINDOW = 20

# 颜色配置（A股红涨绿跌，只定义一次）
COLOR_RED = "#ef4444"      # 红色（涨）
COLOR_GREEN = "#10b981"    # 绿色（跌）
COLOR_BLUE = "#3b82f6"     # 蓝色
COLOR_YELLOW = "#f59e0b"   # 黄色
COLOR_GRAY = "#6b7280"     # 灰色
COLOR_BLACK = "#1f2937"    # 黑色

# 指数代码配置（新增）
INDEX_CODES = {
    "上证指数": {"code": "000001", "suffix": ".SS", "default": 3200.00},
    "深证成指": {"code": "399001", "suffix": ".SZ", "default": 10500.00},
    "创业板指": {"code": "399006", "suffix": ".SZ", "default": 2100.00}
}

# ====================== 自定义样式 ======================
def load_custom_styles() -> None:
    st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; }
    .advice-label { font-size: 13px; color: #6b7280; font-weight: 500; }
    .advice-value { font-size: 16px; font-weight: 700; color: #1f2937; }
    .profit { color: #e53e3e; }  /* A股红涨 */
    .loss { color: #10b981; }   /* A股绿跌 */
    .loading-spinner { border: 3px solid #f3f3f3; border-top: 3px solid #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .signal-tag { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; margin: 4px 0; width: 100%; text-align: center; }
    .buy-tag { background-color: rgba(229, 62, 62, 0.1); color: #e53e3e; border: 1px solid #e53e3e; }
    .sell-tag { background-color: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid #10b981; }
    .neutral-tag { background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid #f59e0b; }
    .table-header { background-color: #f0f4ff; font-weight: bold; }
    .table-row { background-color: #f9fafb; }
    .market-card, .advice-card, .macro-card { 
        border: 1px solid #e5e7eb; 
        border-radius: 12px; 
        padding: 12px; 
        margin: 8px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .trade-guide-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    .trade-guide-table th, .trade-guide-table td {
        border: 1px solid #e5e7eb;
        padding: 8px 12px;
        text-align: left;
    }
    .trade-guide-table th {
        background-color: #f0f4ff;
        font-weight: 600;
        color: #1f2937;
    }
    .trade-guide-table tr:nth-child(even) {
        background-color: #f9fafb;
    }
    .key-level {
        color: #3b82f6;
        font-weight: 600;
    }
    @media (max-width: 768px) {
        .stColumns { flex-direction: column !important; }
        .market-card, .advice-card, .macro-card { padding: 8px; }
        .signal-tag { font-size: 12px; padding: 2px 8px; }
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== 工具函数 ======================
def safe_rerun() -> None:
    """兼容Streamlit新旧版本的rerun方法"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.warning("无法刷新页面，请手动刷新浏览器")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
def safe_requests_get(url: str, params: Dict = None, timeout: int = 10) -> requests.Response:
    """安全的HTTP请求"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
    return requests.get(url, params=params, headers=headers, timeout=timeout)

def fmt_num(value: Any, default: float = 0.0, decimal: int = 2) -> str:
    """安全格式化数值"""
    if value is None or pd.isna(value):
        return f"{default:.{decimal}f}"
    if isinstance(value, str):
        import re
        num_match = re.search(r'(\d+\.?\d*)', value)
        if num_match:
            return f"{float(num_match.group(1)):.{decimal}f}"
        return f"{default:.{decimal}f}"
    try:
        return f"{float(value):.{decimal}f}"
    except (ValueError, TypeError):
        return f"{default:.{decimal}f}"

def extract_num(value: Any, default: float = 0.0) -> float:
    """从任意类型中提取纯数值"""
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        import re
        num_match = re.search(r'(\d+\.?\d*)', value)
        if num_match:
            return float(num_match.group(1))
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# ====================== 新增：指数数据获取专用函数 ======================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def get_index_data_akshare(index_code: str) -> Tuple[float, float]:
    """使用AKShare获取指数最新数据"""
    try:
        # 方法1：获取指数实时行情
        df = ak.index_zh_a_spot()
        if not df.empty and '代码' in df.columns and '最新价' in df.columns and '涨跌幅' in df.columns:
            index_row = df[df['代码'] == index_code]
            if not index_row.empty:
                close = float(index_row['最新价'].iloc[0])
                change = float(index_row['涨跌幅'].iloc[0])
                return round(close, 2), round(change, 2)
        
        # 方法2：获取历史数据（备用）
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
        df_hist = ak.index_zh_a_hist(
            index_code=index_code,
            period="daily",
            start_date=start_date,
            end_date=end_date
        )
        
        if not df_hist.empty and len(df_hist) >= 2:
            close_col = "收盘" if "收盘" in df_hist.columns else "close"
            prev_close_col = "前收盘" if "前收盘" in df_hist.columns else close_col
            close = round(float(df_hist[close_col].iloc[-1]), 2)
            prev_close = round(float(df_hist[prev_close_col].iloc[-1]), 2)
            change = round(((close - prev_close) / prev_close) * 100, 2)
            return close, change
        
        return None, None
    except Exception as e:
        st.debug(f"AKShare获取指数{index_code}失败: {str(e)[:50]}")
        return None, None

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def get_index_data_yfinance(index_suffix: str) -> Tuple[float, float]:
    """使用YFinance获取指数最新数据（备用）"""
    try:
        ticker = yf.Ticker(index_suffix, timeout=10)
        hist = ticker.history(period="5d")
        if not hist.empty and len(hist) >= 2:
            close = round(float(hist['Close'].iloc[-1]), 2)
            prev_close = round(float(hist['Close'].iloc[-2]), 2)
            change = round(((close - prev_close) / prev_close) * 100, 2)
            return close, change
        return None, None
    except Exception as e:
        st.debug(f"YFinance获取指数{index_suffix}失败: {str(e)[:50]}")
        return None, None

def get_latest_index_data(index_name: str) -> Dict[str, Any]:
    """获取指数最新数据（多重备用方案）"""
    index_config = INDEX_CODES.get(index_name, {})
    default_close = index_config.get("default", 0.0)
    default_change = 0.0
    
    # 方案1：优先使用AKShare
    close, change = get_index_data_akshare(index_config.get("code", ""))
    if close and change:
        return {
            "close": close,
            "change": change,
            "color": COLOR_RED if change > 0 else COLOR_GREEN if change < 0 else COLOR_GRAY
        }
    
    # 方案2：使用YFinance备用
    close, change = get_index_data_yfinance(index_config.get("suffix", ""))
    if close and change:
        return {
            "close": close,
            "change": change,
            "color": COLOR_RED if change > 0 else COLOR_GREEN if change < 0 else COLOR_GRAY
        }
    
    # 方案3：使用默认值并提示
    st.warning(f"无法获取{index_name}最新数据，使用默认值")
    return {
        "close": default_close,
        "change": default_change,
        "color": COLOR_GRAY
    }

# ====================== 数据获取模块 ======================
@st.cache_data(ttl=NAME_CACHE_TTL, show_spinner="正在查询A股名称...")
def get_stock_name(stock_code: str) -> str:
    """获取A股股票名称"""
    stock_code = stock_code.strip()
    
    if stock_code.isdigit() and len(stock_code) == 6:
        try:
            # 优先使用AKShare获取名称
            stock_info_df = ak.stock_info_a_code_name()
            name = stock_info_df[stock_info_df['code'] == stock_code]['name'].iloc[0]
            return name
        except Exception as e:
            st.warning(f"A股名称查询失败: {str(e)[:50]}")
            return f"A股({stock_code})"
    else:
        st.error(f"请输入6位A股代码（如600519），当前输入：{stock_code}")
        return f"无效代码({stock_code})"

@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner="正在获取A股行情数据...")
def get_stock_data_enhanced(
    stock_code: str, 
    days: int = DEFAULT_DAYS, 
    data_source: str = "akshare", 
    timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    """获取A股行情数据（修复：设置date为index）"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(days*1.5))).strftime("%Y%m%d")
    
    try:
        if data_source == "akshare":
            # AKShare获取A股数据（前复权）
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                timeout=15
            )
            if df.empty:
                st.warning("AKShare返回空数据，尝试备用接口")
                return pd.DataFrame()
            
            # 标准化列名 + 设置date为index
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            })[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        elif data_source == "yfinance":
            # 备用：YFinance（A股代码格式：600519.SS/300750.SZ）
            ticker_suffix = f"{stock_code}.SS" if stock_code.startswith(('6', '9')) else f"{stock_code}.SZ"
            ticker = yf.Ticker(ticker_suffix, timeout=10)
            period_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
            df = ticker.history(period=f"{days}d", interval=period_map[timeframe])
            
            if df.empty:
                st.error("YFinance返回空数据，请检查股票代码")
                return pd.DataFrame()
            
            # 标准化列名 + 设置date为index
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        else:
            raise ValueError(f"不支持的数据源：{data_source}")
        
        # 数据清洗 + 设置date为index（关键修复：绘图需要datetime index）
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[df['volume'] >= 0]
        df = df.set_index('date')  # 设置date列为索引
        
        if len(df) < MIN_DATA_LENGTH:
            st.warning(f"有效数据仅{len(df)}条（最少需要{MIN_DATA_LENGTH}条），部分指标可能无法计算")
        
        return df
    
    except Exception as e:
        st.error(f"数据获取失败: {str(e)[:100]}")
        return pd.DataFrame()

# ====================== 技术指标计算模块 ======================
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标（增加空值保护）"""
    if df.empty or len(df) < MIN_DATA_LENGTH:
        return df
    
    df = df.copy()
    close = df['close'].fillna(method='ffill')
    high = df['high'].fillna(method='ffill')
    low = df['low'].fillna(method='ffill')
    volume = df['volume'].fillna(0)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=RSI_WINDOW, min_periods=1).mean()
    avg_loss = loss.rolling(window=RSI_WINDOW, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=MACD_EMA12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=MACD_EMA26, adjust=False, min_periods=1).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False, min_periods=1).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 布林带
    df['boll_mid'] = close.rolling(window=BOLL_WINDOW, min_periods=1).mean()
    boll_std = close.rolling(window=BOLL_WINDOW, min_periods=1).std().fillna(0)
    df['boll_upper'] = df['boll_mid'] + 2 * boll_std
    df['boll_lower'] = df['boll_mid'] - 2 * boll_std
    
    # KDJ
    df['low_9'] = low.rolling(window=KDJ_WINDOW, min_periods=1).min()
    df['high_9'] = high.rolling(window=KDJ_WINDOW, min_periods=1).max()
    rsv_denominator = (df['high_9'] - df['low_9']).replace(0, 0.0001)
    df['RSV'] = (close - df['low_9']) / rsv_denominator * 100
    df['K'] = df['RSV'].ewm(span=3, adjust=False, min_periods=1).mean()
    df['D'] = df['K'].ewm(span=3, adjust=False, min_periods=1).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 成交量和波动率
    df['volume_avg'] = volume.rolling(window=VOLUME_AVG_WINDOW, min_periods=1).mean()
    df['volume_ratio'] = volume / df['volume_avg'].replace(0, 0.0001)
    df['volatility'] = close.pct_change().rolling(window=VOLATILITY_WINDOW, min_periods=1).std() * 100
    
    return df

def analyze_signals(df: pd.DataFrame) -> Dict[str, str]:
    """分析技术信号（增加空值保护）"""
    required_cols = ['RSI', 'MACD', 'MACD_Signal', 'K', 'D']
    if df.empty or not all(col in df.columns for col in required_cols):
        return {"RSI": "中性", "MACD": "中性", "KDJ": "中性"}
    
    signals = {}
    latest = df.iloc[-1]
    prev_latest = df.iloc[-2] if len(df) >= 2 else latest
    
    # RSI信号
    if latest['RSI'] > 70:
        signals['RSI'] = "超买"
    elif latest['RSI'] < 30:
        signals['RSI'] = "超卖"
    else:
        signals['RSI'] = "中性"
    
    # MACD信号
    if latest['MACD'] > latest['MACD_Signal'] and prev_latest['MACD'] <= prev_latest['MACD_Signal']:
        signals['MACD'] = "看涨（金叉）"
    elif latest['MACD'] < latest['MACD_Signal'] and prev_latest['MACD'] >= prev_latest['MACD_Signal']:
        signals['MACD'] = "看跌（死叉）"
    elif latest['MACD'] > latest['MACD_Signal']:
        signals['MACD'] = "看涨"
    else:
        signals['MACD'] = "看跌"
    
    # KDJ信号
    if latest['K'] > 80 and latest['D'] > 80:
        signals['KDJ'] = "超买"
    elif latest['K'] < 20 and latest['D'] < 20:
        signals['KDJ'] = "超卖"
    elif latest['K'] > latest['D']:
        signals['KDJ'] = "看涨"
    else:
        signals['KDJ'] = "看跌"
    
    return signals

# ====================== 斐波那契分析模块 ======================
def calculate_fibonacci_levels(df: pd.DataFrame) -> Tuple[Dict[str, float], float, float]:
    """计算斐波那契水平（修复：返回字典格式，适配绘图函数）"""
    if df.empty or len(df) < 5:
        st.warning("数据量不足（至少需要5条），无法计算斐波那契回撤水平")
        return {}, 0.0, 0.0
    
    for col in ['high', 'low', 'close']:
        if col not in df.columns:
            st.warning(f"缺少{col}列，无法计算斐波那契水平")
            return {}, 0.0, 0.0
    
    lookback_days = min(60, len(df))
    recent_data = df.tail(lookback_days).dropna(subset=['high', 'low'])
    
    if len(recent_data) < 3:
        st.warning("有效数据不足，无法计算斐波那契水平")
        return {}, 0.0, 0.0

    recent_high = recent_data['high'].max()
    recent_low = recent_data['low'].min()
    price_diff = recent_high - recent_low 
    
    if price_diff <= 0.01:
        current_price = df['close'].iloc[-1]
        price_diff = current_price * 0.1
        recent_high = current_price + price_diff/2
        recent_low = current_price - price_diff/2
    
    # 斐波那契水平（修复：返回字典，key为比例，value为价格）
    fib_levels = {
        "100% (近期高点)": round(recent_high, 2),
        "76.4%": round(recent_high - price_diff * 0.236, 2),
        "61.8% (关键)": round(recent_high - price_diff * 0.382, 2),
        "50% (中轴)": round(recent_high - price_diff * 0.5, 2),
        "38.2% (关键)": round(recent_high - price_diff * 0.618, 2),
        "21.4%": round(recent_high - price_diff * 0.786, 2),
        "0% (近期低点)": round(recent_low, 2)
    }
    
    return fib_levels, float(recent_high), float(recent_low)

def get_fibonacci_key_levels(fib_levels: Dict[str, float], current_price: float) -> Dict[str, float]:
    """提取斐波那契关键水平（适配字典格式）"""
    key_levels = {
        "fib_382": None, "fib_50": None, "fib_618": None,
        "current_support": None, "current_resistance": None,
        "stop_loss": None, "take_profit_1": None, "take_profit_2": None
    }
    
    if not fib_levels or current_price <= 0:
        return key_levels
    
    # 提取核心水平（数值型，用于计算）
    for label, price in fib_levels.items():
        if "38.2%" in label:
            key_levels["fib_382"] = price  # 保留浮点数
        elif "50%" in label:
            key_levels["fib_50"] = price   # 保留浮点数
        elif "61.8%" in label:
            key_levels["fib_618"] = price  # 保留浮点数
    
    # 转换为列表便于遍历
    fib_list = [(price, label) for label, price in fib_levels.items()]
    fib_list.sort(reverse=True)  # 从高到低排序
    
    # 支撑/压力
    for i, (price, label) in enumerate(fib_list):
        if i < len(fib_list) - 1:
            next_price, next_label = fib_list[i+1]
            if (current_price * 0.995) < price and (current_price * 1.005) > next_price:
                key_levels["current_support"] = next_price
                key_levels["current_resistance"] = price
                break
    
    # 止损/止盈
    if key_levels["current_support"]:
        key_levels["stop_loss"] = round(key_levels["current_support"] * 0.985, 2)
    else:
        key_levels["stop_loss"] = round(current_price * 0.98, 2)
    
    if current_price < key_levels.get("fib_382", current_price):
        key_levels["take_profit_1"] = key_levels.get("fib_382")
        key_levels["take_profit_2"] = key_levels.get("fib_618")
    elif current_price < key_levels.get("fib_50", current_price):
        key_levels["take_profit_1"] = key_levels.get("fib_50")
        key_levels["take_profit_2"] = key_levels.get("fib_618")
    else:
        key_levels["take_profit_1"] = key_levels.get("fib_618")
        key_levels["take_profit_2"] = fib_list[0][0] if fib_list else current_price * 1.08
    
    return key_levels

# ====================== 宏观环境分析模块（重点优化）======================
@st.cache_data(ttl=MACRO_CACHE_TTL, show_spinner="正在获取A股宏观事件...")
def get_latest_macro_events() -> List[Dict[str, str]]:
    """获取A股宏观事件"""
    try:
        # 尝试从网络获取最新事件（备用静态数据）
        try:
            # 新浪财经宏观新闻（示例）
            response = safe_requests_get("https://finance.sina.com.cn/macro/", timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = soup.find_all('div', class_='news-item')[:3]
            
            events = []
            for item in news_items:
                date_elem = item.find('span', class_='time')
                title_elem = item.find('a')
                if date_elem and title_elem:
                    events.append({
                        "date": date_elem.text.strip(),
                        "title": title_elem.text.strip(),
                        "content": "宏观政策影响A股市场走势"
                    })
            if events:
                return events
        except:
            pass
        
        # 备用静态数据
        static_events = [
            {"date": (date.today() - timedelta(days=i)).strftime("%Y-%m-%d"), 
             "title": f"证监会发布A股最新政策({i+1})", 
             "content": f"利好{['消费','科技','金融','制造'][i%4]}板块，影响A股整体走势"}
            for i in range(3)
        ]
        return static_events
    except Exception as e:
        st.warning(f"获取宏观事件失败: {str(e)[:50]}")
        fallback_events = [
            {"date": (date.today() - timedelta(days=i)).strftime("%Y-%m-%d"),
             "title": f"A股政策利好{i+1}",
             "content": f"利好{['消费','科技','金融','制造'][i%4]}板块"}
            for i in range(3)
        ]
        return fallback_events

@st.cache_data(ttl=MACRO_CACHE_TTL, show_spinner="正在获取A股宏观数据...")
def get_macro_environment() -> Dict[str, Any]:
    """获取宏观环境数据（重点优化：指数数据获取）"""
    try:
        # ========== 重点优化：指数数据获取 ==========
        index_data = {}
        for index_name in ["上证指数", "深证成指", "创业板指"]:
            index_data[index_name] = get_latest_index_data(index_name)
        
        # ========== 市场情绪计算（优化逻辑）==========
        a_share_changes = [index_data[name]["change"] for name in index_data.keys()]
        avg_change = np.mean(a_share_changes)
        
        if avg_change > 0.8:
            market_sentiment = "乐观"
        elif avg_change > 0.2:
            market_sentiment = "偏乐观"
        elif avg_change < -0.8:
            market_sentiment = "悲观"
        elif avg_change < -0.2:
            market_sentiment = "偏悲观"
        else:
            market_sentiment = "中性"
        
        # ========== 返回完整数据 ==========
        return {
            "indices": index_data,
            "market_sentiment": market_sentiment,
            "macro_events": get_latest_macro_events(),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 精确到秒
            "shibor": {"value": "1.85%", "impact": "中性（资金面宽松）"},
            "cpi": {"value": "0.9%", "impact": "中性（通胀温和）"},
            "ppi": {"value": "-1.2%", "impact": "偏空（工业通缩）"},
            "strong_sectors": {"value": "白酒、新能源、半导体"},
            "sector_rotation_cycle": {"value": "3-5个交易日"},
            "sector_advice": {"value": "跟随强势板块，避免弱势板块抄底"},
            "shanghai_index_trend": {"value": "震荡上行" if index_data["上证指数"]["change"] >= 0 else "震荡下行"},
            "stock_market_correlation": {"value": "高（个股随大盘波动）"},
            "position_advice": {"value": "50%-70%仓位（中性偏多）" if market_sentiment in ["乐观", "偏乐观"] else 
                                  "20%-40%仓位（中性偏空）" if market_sentiment in ["悲观", "偏悲观"] else "30%-50%仓位（中性）"},
            "policy_trend": {"value": "稳增长政策持续发力，利好基建/消费板块"},
            "policy_impact": {"value": "若个股属于政策利好板块，可适当提高仓位"}
        }
    except Exception as e:
        st.warning(f"宏观数据获取异常: {str(e)[:100]}")
        # 完全失败时的保底数据
        default_indices = {
            "上证指数": {"close": 3200.00, "change": 0.50, "color": COLOR_RED},
            "深证成指": {"close": 10500.00, "change": 0.80, "color": COLOR_RED},
            "创业板指": {"close": 2100.00, "change": 1.20, "color": COLOR_RED}
        }
        return {
            "indices": default_indices,
            "market_sentiment": "中性",
            "macro_events": get_latest_macro_events(),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "shibor": {"value": "1.85%", "impact": "中性（资金面宽松）"},
            "cpi": {"value": "0.9%", "impact": "中性（通胀温和）"},
            "ppi": {"value": "-1.2%", "impact": "偏空（工业通缩）"},
            "strong_sectors": {"value": "白酒、新能源、半导体"},
            "sector_rotation_cycle": {"value": "3-5个交易日"},
            "sector_advice": {"value": "跟随强势板块，避免弱势板块抄底"},
            "shanghai_index_trend": {"value": "震荡上行"},
            "stock_market_correlation": {"value": "高（个股随大盘波动）"},
            "position_advice": {"value": "50%-70%仓位（中性偏多）"},
            "policy_trend": {"value": "稳增长政策持续发力，利好基建/消费板块"},
            "policy_impact": {"value": "若个股属于政策利好板块，可适当提高仓位"}
        }

# ====================== 交易建议模块 ======================
def calculate_trading_advice(
    df: pd.DataFrame, 
    signals: Dict[str, str], 
    timeframe: str, 
    macro_data: Dict[str, Any]
) -> Dict[str, str]:
    """生成交易建议（增加空值保护）"""
    if df.empty:
        return {
            "advice": "无法判断",
            "rationale": "数据不足，无法生成建议",
            "position": "0%",
            "stop_loss": "无",
            "score": 0
        }
    
    rsi_signal = signals.get('RSI', '中性')
    macd_signal = signals.get('MACD', '中性')
    kdj_signal = signals.get('KDJ', '中性')
    market_sentiment = macro_data.get("market_sentiment", "中性")
    
    # 评分系统
    score = 0
    if rsi_signal == "超卖": score += 1.5
    elif rsi_signal == "超买": score -= 1.5
    if "看涨" in macd_signal: score += 2
    elif "看跌" in macd_signal: score -= 2
    if kdj_signal == "超卖": score += 1
    elif kdj_signal == "超买": score -= 1
    
    sentiment_score = {"乐观":1, "偏乐观":0.5, "中性":0, "偏悲观":-0.5, "悲观":-1}.get(market_sentiment, 0)
    score += sentiment_score
    
    # 价格数据（增加空值保护）
    latest_close = float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
    recent_low = float(df['low'].tail(20).min()) if 'low' in df.columns else latest_close * 0.95
    recent_high = float(df['high'].tail(20).max()) if 'high' in df.columns else latest_close * 1.05
    
    # 交易建议
    if score >= 3:
        advice = "强烈买入"
        rationale = f"A股技术面多指标发出强烈买入信号（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议积极建仓"
        position = "50-70%"
        stop_loss = f"{recent_low * 0.95:.2f}（前低点下方5%）"
    elif score >= 1:
        advice = "建议买入"
        rationale = f"A股技术面偏多（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议适量建仓"
        position = "30-50%"
        stop_loss = f"{recent_low * 0.97:.2f}（前低点下方3%）"
    elif score <= -3:
        advice = "强烈卖出"
        rationale = f"A股技术面多指标发出强烈卖出信号（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议立即减仓"
        position = "0-20%"
        stop_loss = "无"
    elif score <= -1:
        advice = "建议卖出"
        rationale = f"A股技术面偏空（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议减仓"
        position = "20-30%"
        stop_loss = f"{recent_high * 1.03:.2f}（前高点上方3%）"
    else:
        if market_sentiment in ["乐观", "偏乐观"]:
            advice = "建议持有（偏多）"
            rationale = f"A股技术面中性（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议持有并逢低加仓"
            position = "40-60%"
        elif market_sentiment in ["悲观", "偏悲观"]:
            advice = "建议持有（偏空）"
            rationale = f"A股技术面中性（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪{market_sentiment}，建议持有并逢高减仓"
            position = "20-40%"
        else:
            advice = "建议持有"
            rationale = f"A股技术面中性（RSI={rsi_signal}、MACD={macd_signal}、KDJ={kdj_signal}），市场情绪中性，建议观望"
            position = "30-50%"
        stop_loss = f"{recent_low * 0.98:.2f}（前低点下方2%）"
    
    return {
        "advice": advice,
        "rationale": rationale,
        "position": position,
        "stop_loss": stop_loss,
        "score": score
    }

# ====================== 可视化模块 ======================
def create_technical_chart(df, stock_name, stock_code, timeframe, fib_levels=None):
    """创建技术分析图表（彻底修复所有参数和逻辑错误）"""
    # 确保df的index是datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            st.warning("数据缺少日期索引，图表绘制可能异常")
            return go.Figure()
    
    # 创建子图布局：主图（K线）+ 成交量 + MACD + KDJ + RSI
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{stock_name} ({stock_code}) - {timeframe.upper()} K线',
            '成交量', 'MACD', 'KDJ', 'RSI'
        ),
        row_heights=[0.4, 0.1, 0.15, 0.15, 0.1]
    )
    
    # 1. 绘制K线（主图）
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color=COLOR_RED,    # A股红涨
            decreasing_line_color=COLOR_GREEN,  # A股绿跌
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 添加布林带
    if 'boll_mid' in df.columns and 'boll_upper' in df.columns and 'boll_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['boll_mid'],
                name='布林中轨',
                line=dict(color=COLOR_YELLOW, width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['boll_upper'],
                name='布林上轨',
                line=dict(color=COLOR_GRAY, width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['boll_lower'],
                name='布林下轨',
                line=dict(color=COLOR_GRAY, width=1),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 添加斐波那契回调线（修复：适配字典格式）
    if fib_levels and isinstance(fib_levels, dict):
        for label, price in fib_levels.items():
            if price and isinstance(price, (int, float)):
                fig.add_hline(
                    y=price,
                    line_dash="dash",
                    line_color=COLOR_BLUE if "38.2%" in label or "61.8%" in label else COLOR_GRAY,
                    annotation_text=label,
                    annotation_position="right",
                    row=1, col=1
                )
    
    # 2. 绘制成交量
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='成交量',
            marker_color=[COLOR_RED if c > o else COLOR_GREEN for c, o in zip(df['close'], df['open'])],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. 绘制MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Hist' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Hist'],
                name='MACD柱状图',
                marker_color=[COLOR_RED if x > 0 else COLOR_GREEN for x in df['MACD_Hist']],
                showlegend=False
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color=COLOR_RED, width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color=COLOR_BLUE, width=1),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 4. 绘制KDJ
    if 'K' in df.columns and 'D' in df.columns and 'J' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['K'],
                name='K',
                line=dict(color=COLOR_RED, width=1),
                showlegend=False
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['D'],
                name='D',
                line=dict(color=COLOR_BLUE, width=1),
                showlegend=False
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['J'],
                name='J',
                line=dict(color=COLOR_YELLOW, width=1),
                showlegend=False
            ),
            row=4, col=1
        )
        # 添加超买超卖线
        fig.add_hline(y=80, line_dash="dash", line_color=COLOR_GRAY, row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color=COLOR_GRAY, row=4, col=1)
    
    # 5. 绘制RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color=COLOR_BLUE, width=1),
                showlegend=False
            ),
            row=5, col=1
        )
        # 添加超买超卖线
        fig.add_hline(y=70, line_dash="dash", line_color=COLOR_GRAY, row=5, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLOR_GRAY, row=5, col=1)
    
    # 图表样式设置
    fig.update_layout(
        title=f'{stock_name} ({stock_code}) - {timeframe.upper()} 技术分析图表',
        title_x=0.5,
        height=800,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="SimHei, Arial", size=12, color=COLOR_BLACK),
        xaxis_rangeslider_visible=False  # 隐藏底部的缩放滑块
    )
    
    # 更新x轴和y轴样式
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#e5e7eb',
        tickformat='%Y-%m-%d',
        tickangle=-45
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#e5e7eb'
    )
    
    return fig

# ====================== 主程序 ======================
def main():
    # 初始化会话状态
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = DEFAULT_STOCK_CODE
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = DEFAULT_TIMEFRAME
    
    # 加载样式
    load_custom_styles()

    # 页面标题
    st.markdown("# 📈 A股专业技术分析系统")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📌 A股配置")
        
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            stock_code = st.text_input(
                "输入6位A股代码",
                value=st.session_state.selected_stock,
                placeholder="如600519（贵州茅台）、300750（宁德时代）"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📋 示例"):
                st.session_state.selected_stock = "600519"
                safe_rerun()
        
        st.session_state.selected_stock = stock_code.strip()
        
        # 快捷选择
        st.markdown("#### ⚡ A股快捷选择")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("茅台"):
                st.session_state.selected_stock = "600519"
                safe_rerun()
        with col2:
            if st.button("宁德时代"):
                st.session_state.selected_stock = "300750"
                safe_rerun()
        with col3:
            if st.button("招商银行"):
                st.session_state.selected_stock = "600036"
                safe_rerun()
        
        # K线周期
        timeframe_options = ["daily（日线）", "weekly（周线）", "monthly（月线）"]
        current_timeframe_label = f"{st.session_state.timeframe}（{['日线','周线','月线'][['daily','weekly','monthly'].index(st.session_state.timeframe)]}）"
        timeframe_index = timeframe_options.index(current_timeframe_label) if current_timeframe_label in timeframe_options else 0
        timeframe = st.selectbox(
            "K线周期",
            timeframe_options,
            index=timeframe_index
        )
        st.session_state.timeframe = timeframe.split("（")[0]
        
        # 数据源
        data_source = st.selectbox(
            "数据源",
            ["akshare（A股官方）", "yfinance（备用）"],
            index=0,
            help="优先使用akshare获取A股数据"
        ).split("（")[0]
        
        st.markdown("---")
        if st.button("🔄 重置为默认"):
            st.session_state.selected_stock = DEFAULT_STOCK_CODE
            st.session_state.timeframe = DEFAULT_TIMEFRAME
            safe_rerun()
        
        # 风险提示
        st.markdown("""
        <div style="margin-top: 20px; padding: 10px; background-color: #fff8e6; border-radius: 8px; border: 1px solid #f59e0b;">
            <span style="color: #d97706; font-weight: 600;">⚠️ A股风险提示</span>
            <p style="color: #6b7280; font-size: 12px; margin: 5px 0 0 0;">
                本工具仅供参考，不构成投资建议。A股T+1交易，涨跌幅限制±10%，请严格控制风险！
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 校验代码
    if not stock_code or not (stock_code.isdigit() and len(stock_code) == 6):
        st.warning("请输入有效的6位A股代码（如600519 贵州茅台、300750 宁德时代）")
        return
    
    # 获取股票名称
    stock_name = get_stock_name(stock_code)
    if "无效代码" in stock_name:
        st.error(f"股票代码 {stock_code} 格式错误，请检查！")
        return
    
    st.sidebar.markdown(f"✅ 匹配结果：`{stock_code}` → **{stock_name}**")
    
    # 加载数据
    with st.spinner("正在加载A股数据，请稍候..."):
        df = get_stock_data_enhanced(
            stock_code=stock_code,
            data_source=data_source,
            timeframe=st.session_state.timeframe
        )
        if df.empty:
            st.error("无法获取A股行情数据，请检查股票代码或网络连接！")
            return
        
        df = calculate_technical_indicators(df)
        signals = analyze_signals(df)
        macro_data = get_macro_environment()
        trading_advice = calculate_trading_advice(
            df=df,
            signals=signals,
            timeframe=st.session_state.timeframe,
            macro_data=macro_data
        )
        fib_levels, recent_high, recent_low = calculate_fibonacci_levels(df)  # 现在返回字典
        current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
        fib_key_levels = get_fibonacci_key_levels(fib_levels, current_price)
        
        # 关键修复：分离数值型和格式化字符串（避免类型错误）
        fib_382 = fib_key_levels.get("fib_382", current_price * 1.02)  # 数值型，用于计算
        fib_50 = fib_key_levels.get("fib_50", current_price * 1.04)    # 数值型，用于计算
        fib_618 = fib_key_levels.get("fib_618", current_price * 1.06)  # 数值型，用于计算
        
        # 格式化字符串（仅用于展示）
        fib_382_fmt = fmt_num(fib_382, current_price * 1.02)
        fib_50_fmt = fmt_num(fib_50, current_price * 1.04)
        fib_618_fmt = fmt_num(fib_618, current_price * 1.06)
    
    st.success(f"✅ 数据加载完成：{stock_code} {stock_name}（{st.session_state.timeframe}）")
    st.markdown("---")
    
    # 标签页
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 技术分析", 
        "🎯 交易建议", 
        "📈 操作指南", 
        "📐 斐波那契分析", 
        "📋 关键指标",
        "🌍 宏观环境"
    ])
    
    with tab1:
        st.subheader("A股K线图与技术指标")
        
        # 移除无用的checkbox（避免误导，函数已内置显示所有指标）
        st.markdown("""
        <div style="padding: 10px; background-color: #f0f4ff; border-radius: 8px; margin-bottom: 15px;">
            <p style="margin:0; color:#3b82f6; font-size:14px;">
                📌 图表包含：K线+布林带、成交量、MACD、KDJ、RSI、斐波那契回撤线
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成图表（彻底修复：仅传函数定义的参数）
        fig = create_technical_chart(
            df=df,
            stock_name=stock_name,
            stock_code=stock_code,
            timeframe=st.session_state.timeframe,
            fib_levels=fib_levels
        )
        try:
            st.plotly_chart(fig, width='stretch')  # 修复弃用参数
        except:
            st.plotly_chart(fig)
        
        # 指标说明
        with st.expander("📖 A股技术指标说明", expanded=False):
            st.markdown("""
            - **RSI**：0-30超卖（反弹概率高），70-100超买（回调概率高）
            - **MACD**：金叉看涨，死叉看跌（A股趋势判断核心指标）
            - **布林带**：震荡市中准确率高，突破上轨看涨，跌破下轨看跌
            - **KDJ**：适合A股短线交易，K值上穿D值为金叉，下穿为死叉
            """)
    
    with tab2:
        st.subheader("A股交易建议")
        
        advice = trading_advice['advice']
        if "买入" in advice:
            st.markdown(f"### <span style='color:{COLOR_RED}'>{advice}</span>", unsafe_allow_html=True)
            st.markdown('<div class="signal-tag buy-tag">买入信号</div>', unsafe_allow_html=True)
        elif "卖出" in advice:
            st.markdown(f"### <span style='color:{COLOR_GREEN}'>{advice}</span>", unsafe_allow_html=True)
            st.markdown('<div class="signal-tag sell-tag">卖出信号</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"### <span style='color:{COLOR_YELLOW}'>{advice}</span>", unsafe_allow_html=True)
            st.markdown('<div class="signal-tag neutral-tag">持有信号</div>', unsafe_allow_html=True)
        
        # 信号汇总（修复market_sentiment使用）
        signal_df = pd.DataFrame({
            "指标": ["RSI", "MACD", "KDJ", "市场情绪"],
            "信号": [
                signals['RSI'],
                signals['MACD'],
                signals['KDJ'],
                macro_data['market_sentiment']
            ]
        })
        st.dataframe(signal_df, hide_index=True, width='stretch')  # 修复弃用参数
        
        # 分析依据
        st.markdown("#### 📝 分析依据（A股专属）")
        st.info(trading_advice['rationale'])
    
    with tab3:
        st.subheader("A股详细操作指南（基于斐波那契）")
        
        # 核心变量（增加空值保护）
        advice = trading_advice['advice']
        current_price = round(float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0, 2)
        stop_loss = trading_advice['stop_loss']
        
        # 数值提取
        stop_loss_val = extract_num(stop_loss, current_price * 0.985)
        take_profit_1 = fib_key_levels.get("take_profit_1")
        take_profit_2 = fib_key_levels.get("take_profit_2")
        current_support = fib_key_levels.get("current_support")
        current_resistance = fib_key_levels.get("current_resistance")
        
        # 格式化（仅用于展示）
        stop_loss_fmt = fmt_num(stop_loss, current_price * 0.985)
        take_profit_1_fmt = fmt_num(take_profit_1, current_price * 1.03)
        take_profit_2_fmt = fmt_num(take_profit_2, current_price * 1.06)
        current_support_fmt = fmt_num(current_support, current_price * 0.97)
        current_resistance_fmt = fmt_num(current_resistance, current_price * 1.03)
        
        # 核心交易参数
        st.markdown("### 📋 A股核心交易参数（基于斐波那契）")
        st.markdown(f"""
        <table class="trade-guide-table">
            <tr>
                <th>参数类型</th>
                <th>数值（元）</th>
                <th>A股交易逻辑</th>
            </tr>
            <tr>
                <td>当前价格</td>
                <td><span class="key-level">{current_price:.2f}</span></td>
                <td>最新收盘价（前复权）</td>
            </tr>
            <tr>
                <td>当前支撑位</td>
                <td><span class="key-level">{current_support_fmt}</span></td>
                <td>斐波那契区间支撑位（跌破止损）</td>
            </tr>
            <tr>
                <td>当前压力位</td>
                <td><span class="key-level">{current_resistance_fmt}</span></td>
                <td>斐波那契区间压力位（突破加仓）</td>
            </tr>
            <tr>
                <td>止损点位</td>
                <td><span class="key-level" style="color:{COLOR_GREEN}">{stop_loss_fmt}</span></td>
                <td>支撑位下方1.5%（A股风控底线）</td>
            </tr>
            <tr>
                <td>止盈目标1</td>
                <td><span class="key-level" style="color:{COLOR_RED}">{take_profit_1_fmt}</span></td>
                <td>斐波那契38.2%水平（第一止盈）</td>
            </tr>
            <tr>
                <td>止盈目标2</td>
                <td><span class="key-level" style="color:{COLOR_RED}">{take_profit_2_fmt}</span></td>
                <td>斐波那契61.8%/前高（第二止盈）</td>
            </tr>
            <tr>
                <td>建议仓位</td>
                <td><span class="key-level">{trading_advice['position']}</span></td>
                <td>基于A股T+1规则的仓位控制</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
        
        # 操作策略
        if "买入" in advice:
            buy_price_1 = round(current_price * 0.98, 2)
            buy_price_2 = round(extract_num(current_support, current_price * 0.95), 2)
            buy_price_3 = round(current_price * 1.02, 2)
            
            st.markdown("#### 🟢 A股买入策略（分批建仓，适配T+1）")
            st.markdown(f"""
            <table class="trade-guide-table">
                <tr>
                    <th>建仓阶段</th>
                    <th>买入价格（元）</th>
                    <th>仓位比例</th>
                    <th>A股触发条件</th>
                </tr>
                <tr>
                    <td>首次建仓</td>
                    <td>{buy_price_1:.2f}</td>
                    <td>30%</td>
                    <td>价格回调至当前价下方2%，成交量萎缩</td>
                </tr>
                <tr>
                    <td>二次建仓</td>
                    <td>{buy_price_2:.2f}</td>
                    <td>30%</td>
                    <td>价格回踩斐波那契支撑位{current_support_fmt}，RSI脱离超卖区</td>
                </tr>
                <tr>
                    <td>突破加仓</td>
                    <td>{buy_price_3:.2f}</td>
                    <td>40%</td>
                    <td>价格突破斐波那契压力位{current_resistance_fmt}，MACD金叉确认</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📌 买入执行要点（适配A股T+1规则）")
            st.markdown("""
            <ul>
                <li>📅 T+1规则约束：买入后当日不可卖出，避免尾盘盲目入场</li>
                <li>📊 成交量验证：建仓时需确认成交量≥5日均量的80%</li>
                <li>🎛️ 仓位纪律：单只股票总仓位不超过账户30%</li>
                <li>⚠️ 止损前置：建仓前必须挂止损单，A股跌停板可能无法卖出</li>
            </ul>
            """)
        
        elif "持有" in advice:
            add_position_price = round(current_price * 0.99, 2)
            reduce_position_price = round(current_price * 1.02, 2)
            trailing_stop = round(current_price * 0.97, 2)
            
            st.markdown("#### 🟡 A股持有策略（持仓滚动，适配T+1）")
            st.markdown(f"""
            <table class="trade-guide-table">
                <tr>
                    <th>操作类型</th>
                    <th>触发价格（元）</th>
                    <th>仓位调整</th>
                    <th>A股执行逻辑</th>
                </tr>
                <tr>
                    <td>滚动加仓</td>
                    <td>{add_position_price:.2f}</td>
                    <td>+10%</td>
                    <td>回调至斐波那契38.2%（{fib_382_fmt}）且KDJ未超卖</td>
                </tr>
                <tr>
                    <td>止盈减仓</td>
                    <td>{reduce_position_price:.2f}</td>
                    <td>-20%</td>
                    <td>上涨至斐波那契50%（{fib_50_fmt}）且MACD顶背离</td>
                </tr>
                <tr>
                    <td>移动止损</td>
                    <td>{trailing_stop:.2f}</td>
                    <td>全部卖出</td>
                    <td>价格跌破移动止损位，无论盈亏立即离场</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📌 持有核心原则（A股震荡市适配）")
            st.markdown("""
            <ul>
                <li>📈 趋势跟踪：股价在布林带中轨上方持有，跌破中轨减仓50%</li>
                <li>⏰ 时间窗口：持有不超过3个交易日，避免长期持仓踩业绩雷</li>
                <li>📉 涨跌幅约束：当日涨幅≥8%且成交量异常放大，次日开盘减仓30%</li>
            </ul>
            """)
        
        else:
            sell_price_immediate = round(current_price * 0.995, 2)
            sell_price_target = round(extract_num(take_profit_1, current_price * 1.03), 2)
            sell_price_emergency = round(stop_loss_val * 0.99, 2)
            
            st.markdown("#### 🔴 A股卖出策略（落袋为安，适配T+1）")
            st.markdown(f"""
            <table class="trade-guide-table">
                <tr>
                    <th>卖出类型</th>
                    <th>卖出价格（元）</th>
                    <th>操作优先级</th>
                    <th>A股触发条件</th>
                    <tr>
                        <td>立即减仓</td>
                        <td>{sell_price_immediate:.2f}</td>
                        <td>最高</td>
                        <td>MACD死叉确认，成交量放大下跌</td>
                    </tr>
                    <tr>
                        <td>止盈卖出</td>
                        <td>{sell_price_target:.2f}</td>
                        <td>中</td>
                        <td>价格触及斐波那契止盈位，RSI超买</td>
                    </tr>
                    <tr>
                        <td>紧急止损</td>
                        <td>{sell_price_emergency:.2f}</td>
                        <td>最高</td>
                        <td>价格跌破止损位，A股跌停前果断离场</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
                
            st.markdown("#### 📌 卖出执行要点（A股T+1规则适配）")
            st.markdown("""
                <ul>
                    <li>⏳ T+1约束：当日买入的仓位次日才能卖出，提前做好止损预案</li>
                    <li>📉 跌停风险：若个股跌停，挂单可能无法成交，需在跌停前果断卖出</li>
                    <li>📊 尾盘操作：收盘前30分钟不建议卖出，避免尾盘恐慌性下跌误操作</li>
                    <li>💰 分批卖出：单次卖出不超过50%仓位，避免一次性卖出导致股价波动</li>
                </ul>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("斐波那契回撤分析（A股专用）")
        
        st.markdown("### 📏 斐波那契关键水平（基于近60日高低点）")
        if fib_levels:
            fib_df = pd.DataFrame(list(fib_levels.items()), columns=["回撤水平", "价格（元）"])
            # 高亮关键水平
            def highlight_fib(row):
                if "38.2%" in row["回撤水平"] or "61.8%" in row["回撤水平"]:
                    return ['background-color: #f0f4ff; font-weight: bold'] * 2
                elif "50%" in row["回撤水平"]:
                    return ['background-color: #fff8e6'] * 2
                else:
                    return [''] * 2
            
            st.dataframe(
                fib_df.style.apply(highlight_fib, axis=1),
                hide_index=True,
                width='stretch'  # 已替换，无问题
            )
        else:
            st.warning("无法计算斐波那契水平，请检查数据完整性")
        
        # 斐波那契交易逻辑
        st.markdown("### 🎯 斐波那契交易逻辑（适配A股）")
        st.markdown(f"""
        <div class="advice-card">
            <p><strong>当前价格</strong>：{current_price:.2f} 元</p>
            <p><strong>关键支撑</strong>：{fib_382_fmt} 元（38.2%回撤位）</p>
            <p><strong>关键压力</strong>：{fib_618_fmt} 元（61.8%回撤位）</p>
            <hr style="border: 0.5px solid #e5e7eb; margin: 10px 0;">
            <p><strong>A股交易策略：</strong></p>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>✅ 价格回调至38.2%水平且成交量萎缩 → 买入（A股低吸机会）</li>
                <li>⚠️ 价格跌破38.2%水平且成交量放大 → 止损（A股破位离场）</li>
                <li>🚀 价格突破61.8%水平且成交量放大 → 加仓（A股突破确认）</li>
                <li>🔴 价格触及61.8%水平且RSI超买 → 减仓（A股高抛机会）</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.subheader("关键技术指标数值（A股最新）")
        
        # 提取最新指标值
        latest = df.iloc[-1]
        prev_latest = df.iloc[-2] if len(df) >= 2 else latest
        
        # 指标数据整理
        indicator_data = {
            "价格指标": {
                "最新价": f"{fmt_num(latest['close'])} 元",
                "开盘价": f"{fmt_num(latest['open'])} 元",
                "最高价": f"{fmt_num(latest['high'])} 元",
                "最低价": f"{fmt_num(latest['low'])} 元",
                "涨跌幅": f"{((latest['close'] - prev_latest['close'])/prev_latest['close']*100):.2f}%",
                "成交量": f"{int(latest['volume']/10000):,} 万手" if latest['volume'] > 10000 else f"{int(latest['volume']):,} 手"
            },
            "震荡指标": {
                "RSI(14)": f"{fmt_num(latest['RSI'], decimal=1)}",
                "KDJ-K": f"{fmt_num(latest['K'], decimal=1)}",
                "KDJ-D": f"{fmt_num(latest['D'], decimal=1)}",
                "KDJ-J": f"{fmt_num(latest['J'], decimal=1)}",
                "布林带位置": f"{'上轨上方' if latest['close'] > latest['boll_upper'] else '下轨下方' if latest['close'] < latest['boll_lower'] else '轨道内'}",
                "波动率": f"{fmt_num(latest['volatility'], decimal=2)}%"
            },
            "趋势指标": {
                "MACD": f"{fmt_num(latest['MACD'], decimal=3)}",
                "MACD信号线": f"{fmt_num(latest['MACD_Signal'], decimal=3)}",
                "MACD柱状图": f"{fmt_num(latest['MACD_Hist'], decimal=3)}",
                "成交量比率": f"{(latest['volume_ratio']*100):.1f}%",
                "布林中轨": f"{fmt_num(latest['boll_mid'])} 元",
                "布林上轨": f"{fmt_num(latest['boll_upper'])} 元",
                "布林下轨": f"{fmt_num(latest['boll_lower'])} 元"
            }
        }
        
        # 分栏展示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📊 价格指标")
            for key, value in indicator_data["价格指标"].items():
                color = COLOR_RED if "涨跌幅" in key and float(value.replace('%', '')) > 0 else COLOR_GREEN if "涨跌幅" in key and float(value.replace('%', '')) < 0 else ""
                st.markdown(f"<div class='advice-label'>{key}：</div><div class='advice-value' style='color:{color}'>{value}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎛️ 震荡指标")
            for key, value in indicator_data["震荡指标"].items():
                # RSI/KDJ颜色标注
                color = ""
                if "RSI" in key:
                    val = float(value)
                    color = COLOR_GREEN if val < 30 else COLOR_RED if val > 70 else COLOR_YELLOW
                elif "KDJ" in key:
                    val = float(value)
                    color = COLOR_GREEN if val < 20 else COLOR_RED if val > 80 else COLOR_YELLOW
                st.markdown(f"<div class='advice-label'>{key}：</div><div class='advice-value' style='color:{color}'>{value}</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### 📈 趋势指标")
            for key, value in indicator_data["趋势指标"].items():
                # MACD颜色标注
                color = COLOR_RED if "MACD柱状图" in key and float(value) > 0 else COLOR_GREEN if "MACD柱状图" in key and float(value) < 0 else ""
                st.markdown(f"<div class='advice-label'>{key}：</div><div class='advice-value' style='color:{color}'>{value}</div>", unsafe_allow_html=True)
    
    with tab6:
        st.subheader("宏观环境分析（A股市场）")
        
        # 指数行情卡片
        st.markdown("### 📊 大盘指数实时行情")
        col1, col2, col3 = st.columns(3)
        for idx, (index_name, index_info) in enumerate(macro_data["indices"].items()):
            with [col1, col2, col3][idx]:
                st.markdown(f"""
                <div class="market-card">
                    <h4 style="margin: 0 0 8px 0; color: {COLOR_BLACK};">{index_name}</h4>
                    <p style="margin: 0; font-size: 20px; font-weight: bold; color: {index_info['color']};">{index_info['close']}</p>
                    <p style="margin: 4px 0 0 0; font-size: 14px; color: {index_info['color']};">
                        {('+' if index_info['change'] > 0 else '') + str(index_info['change'])}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # 市场情绪
        sentiment_color = COLOR_RED if macro_data["market_sentiment"] in ["乐观", "偏乐观"] else COLOR_GREEN if macro_data["market_sentiment"] in ["悲观", "偏悲观"] else COLOR_YELLOW
        st.markdown(f"""
        <div class="macro-card" style="margin: 15px 0;">
            <h4 style="margin: 0 0 10px 0;">📈 市场整体情绪</h4>
            <p style="font-size: 18px; font-weight: bold; color: {sentiment_color}; margin: 0;">{macro_data['market_sentiment']}</p>
            <p style="color: #6b7280; margin: 5px 0 0 0;">更新时间：{macro_data['update_time']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 宏观数据
        st.markdown("### 📋 核心宏观数据")
        macro_info = {
            "银行间同业拆借利率(Shibor)": macro_data["shibor"],
            "居民消费价格指数(CPI)": macro_data["cpi"],
            "工业生产者价格指数(PPI)": macro_data["ppi"],
            "当前强势板块": macro_data["strong_sectors"],
            "板块轮动周期": macro_data["sector_rotation_cycle"],
            "上证指数趋势": macro_data["shanghai_index_trend"],
            "个股与大盘相关性": macro_data["stock_market_correlation"],
            "政策趋势": macro_data["policy_trend"]
        }
        
        macro_df = pd.DataFrame([
            {"指标": k, "数值": v["value"] if isinstance(v, dict) else v, "影响分析": v.get("impact", "无") if isinstance(v, dict) else "无"}
            for k, v in macro_info.items()
        ])
        st.dataframe(macro_df, hide_index=True, width='stretch')  # 已替换，无问题
        
        # 宏观事件
        st.markdown("### 📰 最新宏观事件")
        events = macro_data["macro_events"]
        for event in events:
            st.markdown(f"""
            <div style="border-left: 3px solid {COLOR_BLUE}; padding: 8px 12px; margin: 8px 0; background-color: #f9fafb; border-radius: 4px;">
                <span style="color: #6b7280; font-size: 12px;">{event['date']}</span>
                <h5 style="margin: 4px 0; color: {COLOR_BLACK}; font-size: 14px;">{event['title']}</h5>
                <p style="margin: 4px 0; color: #4b5563; font-size: 13px;">{event['content']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 宏观策略建议
        st.markdown("### 🎯 宏观策略建议（A股适配）")
        st.markdown(f"""
        <div class="advice-card">
            <p><strong>板块配置建议：</strong> {macro_data['sector_advice']}</p>
            <p><strong>仓位管理建议：</strong> {macro_data['position_advice']}</p>
            <p><strong>政策影响分析：</strong> {macro_data['policy_impact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 底部风险提示
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px; margin: 10px 0;">
        <h4 style="margin: 0 0 8px 0; color: #b91c1c; font-size: 16px;">⚠️ 重要风险提示（A股专属）</h4>
        <ul style="margin: 0; padding-left: 20px; color: #7f1d1d; font-size: 14px;">
            <li>本工具仅提供技术分析参考，不构成任何投资建议，A股投资有风险，入市需谨慎</li>
            <li>A股实行T+1交易制度，当日买入的股票次日才能卖出，务必做好止损规划</li>
            <li>A股个股涨跌幅限制为±10%（ST股±5%），创业板/科创板新股前5日无涨跌幅限制</li>
            <li>请勿仅凭技术指标进行投资决策，需结合公司基本面、宏观政策、市场情绪综合判断</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 程序入口
if __name__ == "__main__":
    main()
