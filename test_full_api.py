#!/usr/bin/env python3
"""完整 API 测试 - Hyperliquid 登录和数据获取"""
import asyncio
import sys
sys.path.insert(0, "/root/opentrade")

from opentrade.core.config import get_config
from opentrade.services.data_service import data_service
from datetime import datetime, timedelta

async def full_api_test():
    print("=" * 60)
    print("OpenTrade 完整 API 测试")
    print("=" * 60)
    
    config = get_config()
    
    # 1. 检查配置
    print("\n1️⃣ 配置检查...")
    print(f"   交易所: {config.exchange.name}")
    print(f"   钱包: {config.exchange.wallet_address[:15]}...")
    print(f"   API Key: {config.exchange.api_key[:15]}...")
    
    # 2. 测试余额
    print("\n2️⃣ 测试账户余额...")
    balance = await data_service.fetch_balance()
    total = balance.get("total", {})
    print(f"   USD: ${total.get('USD', 0):,.2f}")
    print(f"   USDC: ${total.get('USDC', 0):,.2f}")
    print(f"   BTC: ${total.get('BTC', 0):,.4f}")
    print(f"   ETH: ${total.get('ETH', 0):,.4f}")
    
    # 3. 测试 K 线数据
    print("\n3️⃣ 测试 K 线数据...")
    ohlcv = await data_service.fetch_ohlcv("BTC", timeframe="1h", limit=10)
    if ohlcv:
        print(f"   ✅ 获取 {len(ohlcv)} 条 BTC/USDC K 线")
        last = ohlcv[-1]
        print(f"   最新: ${last['close']:,.2f}")
    else:
        print("   ❌ 无法获取数据")
    
    # 4. 测试 ETH
    print("\n4️⃣ 测试 ETH...")
    ohlcv_eth = await data_service.fetch_ohlcv("ETH", timeframe="1h", limit=5)
    if ohlcv_eth:
        print(f"   ✅ ETH/USDC: ${ohlcv_eth[-1]['close']:,.2f}")
    
    # 5. 测试 ticker
    print("\n5️⃣ 测试 Ticker...")
    ticker = await data_service.fetch_ticker("BTC")
    if ticker and ticker.get("last"):
        print(f"   BTC: ${ticker['last']:,.2f}")
    
    # 6. 清理
    await data_service.close()
    
    print("\n" + "=" * 60)
    print("✅ API 测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(full_api_test())
