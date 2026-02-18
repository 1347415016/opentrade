#!/usr/bin/env python3
"""使用正确的交易对格式测试 Hyperliquid"""
import asyncio
import sys
sys.path.insert(0, "/root/opentrade")

from opentrade.core.config import get_config
import ccxt.async_support as ccxt
from datetime import datetime, timedelta

async def test_correct_symbol():
    print("=" * 60)
    print("Hyperliquid 正确交易对格式测试")
    print("=" * 60)
    
    config = get_config()
    wallet = config.exchange.wallet_address
    
    # 创建 Hyperliquid 实例
    exchange = ccxt.hyperliquid({
        'apiKey': config.exchange.api_key,
        'wallet': wallet,  # Hyperliquid 需要 wallet 参数
        'enableRateLimit': True,
    })
    
    print(f"\n1️⃣ 钱包地址: {wallet}")
    print(f"   前缀: {wallet[:10]}...")
    
    # 2. 测试认证和余额
    print(f"\n2️⃣ 测试认证和余额...")
    try:
        balance = await exchange.fetch_balance()
        total_usd = balance.get('total', {}).get('USD', 0)
        print(f"   ✅ 认证成功!")
        print(f"   USD 余额: ${total_usd:,.2f}")
        print(f"   完整余额: {list(balance.get('total', {}).keys())}")
    except Exception as e:
        print(f"   ❌ 认证失败: {e}")
    
    # 3. 获取 BTC/USDC 价格
    print(f"\n3️⃣ 获取 BTC/USDC 价格...")
    try:
        ticker = await exchange.fetch_ticker('BTC/USDC')
        print(f"   ✅ BTC/USDC: ${ticker['last']:,.2f}")
        print(f"   24h 涨跌: {ticker.get('percentage', 0):.2f}%")
    except Exception as e:
        print(f"   ❌ BTC/USDC: {e}")
    
    # 4. 获取 K 线数据
    print(f"\n4️⃣ 获取 K 线数据...")
    try:
        ohlcv = await exchange.fetch_ohlcv('BTC/USDC', '1h', limit=10)
        print(f"   ✅ BTC/USDC: {len(ohlcv)} 条数据")
        if ohlcv:
            print(f"   最新: ${ohlcv[-1][4]:,.2f}")
    except Exception as e:
        print(f"   ❌ BTC/USDC: {e}")
    
    # 5. 获取账户持仓
    print(f"\n5️⃣ 获取账户持仓...")
    try:
        positions = await exchange.fetch_positions()
        print(f"   ✅ 持仓数量: {len(positions)}")
        for p in positions[:5]:
            print(f"   - {p['symbol']}: {p.get('size', 0)} @ ${p.get('entryPrice', 0):,.2f}")
    except Exception as e:
        print(f"   ❌ 持仓: {e}")
    
    # 6. 获取合约信息
    print(f"\n6️⃣ 获取合约信息...")
    try:
        market = exchange.market('BTC/USDC')
        print(f"   ✅ BTC/USDC 合约信息:")
        print(f"   - 基础货币: {market['base']}")
        print(f"   - 计价货币: {market['quote']}")
        print(f"   - 最小订单: {market['limits']['amount']['min']}")
    except Exception as e:
        print(f"   ❌ 合约信息: {e}")
    
    await exchange.close()
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_correct_symbol())
