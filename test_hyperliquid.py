#!/usr/bin/env python3
"""测试 Hyperliquid API 连接和交易对格式"""
import asyncio
import sys
sys.path.insert(0, "/root/opentrade")

from opentrade.core.config import get_config
import ccxt.async_support as ccxt

async def test_hyperliquid():
    print("=" * 60)
    print("Hyperliquid API 测试")
    print("=" * 60)
    
    config = get_config()
    
    # 创建 Hyperliquid 实例
    exchange = ccxt.hyperliquid({
        'apiKey': config.exchange.api_key,
        'enableRateLimit': True,
    })
    
    print(f"\n1️⃣ 交易所信息:")
    print(f"   ID: {exchange.id}")
    print(f"   名称: {exchange.name}")
    
    # 2. 测试 API 认证
    print(f"\n2️⃣ 测试 API 认证...")
    try:
        # 获取账户余额（这会验证 API Key）
        balance = await exchange.fetch_balance()
        print(f"   ✅ 认证成功!")
        print(f"   余额: {balance.get('total', {})}")
    except Exception as e:
        print(f"   ❌ 认证失败: {e}")
    
    # 3. 获取交易对列表
    print(f"\n3️⃣ 获取交易对列表...")
    try:
        markets = await exchange.load_markets()
        print(f"   ✅ 加载了 {len(markets)} 个交易对")
        
        # 打印 BTC 相关交易对
        btc_pairs = [m for m in markets.keys() if 'BTC' in m.upper() or 'BTC' in m]
        print(f"   BTC 相关: {btc_pairs[:20]}")
        
        # 打印所有 USDT 交易对
        usdt_pairs = [m for m in markets.keys() if 'USDT' in m]
        print(f"   USDT 交易对数量: {len(usdt_pairs)}")
        print(f"   示例: {usdt_pairs[:10]}")
        
    except Exception as e:
        print(f"   ❌ 获取交易对失败: {e}")
    
    # 4. 尝试获取 BTC 价格
    print(f"\n4️⃣ 获取 BTC 价格...")
    test_symbols = ['BTC/USDT', 'H:BTC', 'BTC-USD', 'H:BTC-PERP']
    for symbol in test_symbols:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            print(f"   ✅ {symbol}: ${ticker['last']:,.2f}")
            break
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
    
    # 5. 获取 K 线数据
    print(f"\n5️⃣ 获取 K 线数据...")
    try:
        # 尝试不同的交易对格式
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=5)
        print(f"   ✅ BTC/USDT: {len(ohlcv)} 条数据")
    except Exception as e:
        print(f"   ❌ BTC/USDT: {e}")
    
    await exchange.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(test_hyperliquid())
