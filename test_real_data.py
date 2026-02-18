#!/usr/bin/env python3
"""OpenTrade 真实数据测试"""
import asyncio
import sys
sys.path.insert(0, "/root/opentrade")

from opentrade.core.config import get_config
from opentrade.services.data_service import data_service
from opentrade.services.backtest_service import BacktestService
from datetime import datetime, timedelta

def main():
    print("=" * 60)
    print("OpenTrade 真实数据测试")
    print("=" * 60)
    
    # 1. 检查配置
    print("\n1️⃣ 检查配置...")
    config = get_config()
    print(f"   交易所: {config.exchange.name}")
    print(f"   钱包地址: {config.exchange.wallet_address[:10] if config.exchange.wallet_address else 'None'}...")
    print(f"   API Key: {config.exchange.api_key[:20] if config.exchange.api_key else 'None'}...")
    
    if not config.exchange.api_key or len(config.exchange.api_key) < 10:
        print("   ⚠️ API Key 无效，将使用模拟数据")
    
    async def test():
        # 2. 测试 API
        print("\n2️⃣ 测试 API...")
        try:
            ohlcv = await data_service.fetch_ohlcv("BTC", timeframe="1h", limit=5)
            if ohlcv:
                print(f"   ✅ 真实数据: {len(ohlcv)} 条, 最新价 ${ohlcv[-1]['close']:,.0f}")
            else:
                print("   ⚠️ 无数据")
        except Exception as e:
            print(f"   ❌ API 失败: {e}")
            print("   (这是正常的，如果 API Key 无效)")
        
        # 3. 回测测试
        print("\n3️⃣ 回测测试...")
        service = BacktestService()
        result = await service.run_backtest(
            strategy_name="trend_following",
            start_date=datetime.utcnow() - timedelta(days=5),
            end_date=datetime.utcnow(),
            symbol="BTC",
            initial_capital=10000.0,
        )
        print(f"   ✅ 交易次数: {result.total_trades}, 收益: {result.total_pnl_percent:.2%}")
        
        await data_service.close()
        return True
    
    result = asyncio.run(test())
    print("\n" + "=" * 60)
    return result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
