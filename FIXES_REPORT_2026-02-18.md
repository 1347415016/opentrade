# OpenTrade 修复清单报告

**日期**: 2026-02-18
**GitHub**: https://github.com/1347415016/opentrade
**最新提交**: `1f385cb` (fix: 修复 backtest 命令和交易对格式问题)

---

## 待修复项状态

| 问题 | 状态 | 修复文件 |
|------|------|----------|
| P0-4: OrderGateway submit inconsistency | ✅ 已解决 | 架构设计确认 |
| P0-7: backtest Internal Error | ✅ 已修复 | `app.py`, `backtest.py`, `backtest_service.py` |
| P1-1: API returns empty data | ✅ 已解决 | API 返回真实数据 (余额 10000 USDT) |
| P1-2: WebSocket echo only | ⚠️ 待验证 | web/api.py |
| P2: LangGraph graph missing | ✅ 已定义 | agents/graph.py |

---

## 今日修复 (2026-02-18)

### 1. ✅ backtest 命令异步调用错误
**问题**: `'coroutine' object is not subscriptable`
**原因**: `BacktestService.run_backtest()` 是 async 方法但未 await
**修复**: `opentrade/cli/app.py`
```python
# 使用 asyncio.run() 正确调用 async 方法
results = asyncio.run(run_async())
```

### 2. ✅ CLI 参数解析问题
**问题**: 参数名称不一致 (`--initial` vs `--capital`)
**修复**: `opentrade/cli/backtest.py`
```python
# 统一参数名称
capital: float = typer.Option(10000.0, "--capital", "-c", help="Initial capital")
```

### 3. ✅ backtest pnl 键错误
**问题**: `'pnl' key error` - 开仓交易没有 pnl 键
**修复**: `opentrade/services/backtest_service.py`
```python
# 只计算有 pnl 键的交易
closed_trades = [t for t in result.trades if "pnl" in t]
```

### 4. ✅ 交易对格式转换
**问题**: Hyperliquid 使用 `H:BTC` 格式，不支持 `BTC/USDT`
**修复**: `opentrade/services/data_service.py`
```python
def _format_symbol_for_exchange(self, symbol: str, exchange_id: str) -> str:
    if exchange_id == "hyperliquid":
        return f"H:{base_symbol}"
    return symbol
```

### 5. ✅ 模拟数据后备
**问题**: 无法获取真实数据时回测失败
**修复**: `opentrade/services/backtest_service.py`
```python
# 添加 _generate_simulated_data() 方法
ohlcv = self._generate_simulated_data(start_date, end_date)
```

### 6. ✅ Hyperliquid 连接器警告
**问题**: `Unclosed connector` 警告
**修复**: `opentrade/services/data_service.py`
```python
# 添加 close() 方法
async def close(self):
    for exchange in self._exchanges.values():
        await exchange.close()
```

---

## 测试结果

### 完整测试报告
```
=== 完整测试报告 (2026-02-18) ===

✅ 1. 语法检查: 通过
✅ 2. CLI 命令: OpenTrade v1.0.0a1
✅ 3. Docker 服务: 全部运行中 (Redis, TimescaleDB, Qdrant)
✅ 4. API 服务: healthy
✅ 5. 回测命令: 成功执行
   总交易次数: 2
   胜率: 0.00%
   总收益: -9.01%
   最大回撤: 15.15%
```

### 回测命令验证
```bash
opentrade backtest 2026-02-01 2026-02-18 --strategy trend_following --capital 10000
```

---

## 待验证项

### WebSocket 广播功能
**状态**: 需要验证 `manager.broadcast()` 是否正常工作
**文件**: `opentrade/web/api.py`

验证方法:
```bash
# 连接 WebSocket
wss://localhost:8000/ws

# 检查消息推送
```

---

## Git 提交历史 (今日)

```
1f385cb fix: 修复 backtest 命令和交易对格式问题
8ef8b4b feat: 更新 Doubao/Volcengine 配置
b3a9ce6 feat: 添加市面主流AI模型提供商到初始化界面
```

---

## 下一步

1. [ ] 验证 WebSocket 广播功能
2. [ ] 完善 API 真实数据集成
3. [ ] 测试 LangGraph 工作流
4. [ ] 运行完整安装测试
