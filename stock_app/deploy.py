# deploy.py
import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    try:
        import streamlit
        print("✅ Streamlit 已安装")
    except ImportError:
        print("❌ Streamlit 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    try:
        import yfinance
        print("✅ yfinance 已安装")
    except ImportError:
        print("❌ yfinance 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    
    print("✅ 所有依赖检查完成")

def deploy_local():
    """本地部署"""
    print("\n🚀 启动股票分析系统...")
    
    # 检查app.py是否存在
    if not os.path.exists("app.py"):
        print("❌ app.py 文件不存在！")
        return
    
    # 启动Streamlit
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "app.py", "--server.port", "8501", "--server.address", "localhost"
    ])
    
    print("\n" + "="*50)
    print("✅ 股票分析系统已启动！")
    print("🌐 请在浏览器中访问: http://localhost:8501")
    print("🔄 自动打开浏览器中...")
    print("="*50 + "\n")
    
    time.sleep(2)  # 等待服务器启动
    webbrowser.open("http://localhost:8501")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n👋 程序已停止")

if __name__ == "__main__":
    print("="*50)
    print("📊 股票技术分析系统部署工具")
    print("="*50)
    
    check_dependencies()
    
    deploy_local()
