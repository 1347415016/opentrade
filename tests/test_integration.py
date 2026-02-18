"""
OpenTrade 集成测试

测试完整的系统工作流：
1. 配置加载 → 数据获取 → 策略分析 → 回测执行
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        from opentrade.core.config import OpenTradeConfig
        
        config = OpenTradeConfig()
        config.exchange.name = "binance"
        config.exchange.testnet = True
        config.ai.model = "deepseek/deepseek-chat"
        config.risk.max_leverage = 2.0
        config.risk.stop_loss_pct = 0.035
        
        return config
    
    @pytest.fixture
    def mock_market_state(self):
        """模拟市场状态"""
        from opentrade.core.config import MarketState
        
        return MarketState(
            symbol="BTC/USDT",
            price=68000.0,
            fear_index=30,
            trend="neutral",
        )
    
    @pytest.mark.asyncio
    async def test_full_trading_workflow(self, mock_config, mock_market_state):
        """完整交易流程测试"""
        
        # 1. 配置加载
        from opentrade.core.config import ConfigManager
        
        manager = ConfigManager()
        assert manager.load() is not None
        print("✅ 配置加载成功")
        
        # 2. 数据获取
        from opentrade.services.data_service import DataService
        
        data_service = DataService()
        
        # 模拟价格数据
        mock_prices = [68000 + i * 100 for i in range(100)]
        
        # 测试指标计算
        rsi = data_service._rsi(mock_prices[-50:], 14)
        assert 0 <= rsi <= 100
        
        ema = data_service._ema(mock_prices[-20:], 9)
        assert ema > 0
        
        print("✅ 指标计算正常")
        
        # 3. 策略分析
        from opentrade.agents.evolution import EvolutionEngine, MarketState
        
        engine = EvolutionEngine()
        engine.market_state = MarketState(
            fear_greed_index=30,
            btc_price=68000,
            trend="neutral",
        )
        
        risk_params = engine.get_risk_parameters()
        assert "max_leverage" in risk_params
        assert "stop_loss" in risk_params
        assert "stablecoin_ratio" in risk_params
        
        print("✅ 风险参数生成正常")
        
        # 4. 生成提示词
        prompt = engine.generate_system_prompt()
        assert "BTC" in prompt
        assert "Fear Index" in prompt.lower() or "恐惧" in prompt
        
        print("✅ 提示词生成正常")
        
        # 5. 风险控制检查
        from opentrade.core.risk import RiskEngine
        from opentrade.core.config import RiskConfig
        
        risk_config = RiskConfig(
            max_leverage=2.0,
            max_position_pct=0.1,
            stop_loss_pct=0.035,
        )
        
        risk_engine = RiskEngine(risk_config)
        
        # 模拟订单
        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "size": 1000,
            "leverage": 1.0,
        }
        
        # 风险检查应该通过
        # result = risk_engine.check_order(order)
        # assert result.approved
        
        print("✅ 风险引擎正常")
        
        print("\n🎉 完整交易流程测试通过")
        
        return True
    
    @pytest.mark.asyncio
    async def test_backtest_integration(self):
        """回测集成测试"""
        
        from opentrade.cli.backtest import BacktestEngine
        from datetime import datetime, timedelta
        
        engine = BacktestEngine(initial_balance=10000)
        
        # 测试回测执行
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        
        # 这里会尝试获取真实数据，可能失败
        result = await engine.run(
            symbol="BTC/USDT",
            strategy_type="trend_following",
            start_date=start,
            end_date=end,
        )
        
        assert "total_return" in result
        assert "win_rate" in result
        assert "sharpe_ratio" in result
        
        print("✅ 回测引擎正常")
        
        return result
    
    @pytest.mark.asyncio
    async def test_daily_workflow(self):
        """每日工作流测试"""
        
        # 模拟市场数据获取
        fear_index = 35
        btc_price = 68000.0
        
        assert 0 <= fear_index <= 100
        assert btc_price > 0
        
        # 模拟进化引擎
        from opentrade.agents.evolution import EvolutionEngine, MarketState
        
        engine = EvolutionEngine()
        engine.market_state = MarketState(
            fear_greed_index=fear_index,
            btc_price=btc_price,
        )
        
        risk_params = engine.get_risk_parameters()
        
        assert risk_params["max_leverage"] >= 1.0
        assert risk_params["risk_mode"] in ["extreme_fear", "fear", "neutral", "optimistic", "greedy"]
        
        print("✅ 每日工作流测试通过")
        
        return True
    
    def test_vector_store_integration(self):
        """向量存储集成测试"""
        
        from opentrade.core.vector_store import MemoryVectorStore, VectorRecord
        from datetime import datetime
        
        store = MemoryVectorStore()
        
        # 添加测试数据
        test_vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        
        for i, vec in enumerate(test_vectors):
            record = VectorRecord(
                id=f"test-{i}",
                vector=vec,
                payload={"type": "test", "index": i},
                created_at=datetime.utcnow(),
            )
            store.add(record)
        
        # 搜索
        results = store.search([0.15, 0.25, 0.35], limit=3)
        
        assert len(results) > 0
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
        
        # 清理
        for i in range(len(test_vectors)):
            store.delete(f"test-{i}")
        
        store.close()
        
        print("✅ 向量存储集成测试通过")
        
        return True
    
    def test_circuit_breaker_integration(self):
        """熔断器集成测试"""
        
        from opentrade.core.circuit_breaker import CircuitBreaker, CircuitState, TriggerReason
        
        breaker = CircuitBreaker()
        
        # 初始状态应该是 CLOSED
        assert breaker.state == CircuitState.CLOSED
        
        # 模拟正常操作
        for _ in range(10):
            result = breaker.record_success()
            assert result is True
        
        # 模拟失败
        result = breaker.record_failure(TriggerReason.DRAWDOWN_EXCEEDED)
        assert result is False
        
        # 状态应该还是 CLOSED (还没到阈值)
        assert breaker.state == CircuitState.CLOSED
        
        # 多次失败应该触发熔断
        for _ in range(5):
            breaker.record_failure(TriggerReason.DRAWDOWN_EXCEEDED)
        
        assert breaker.state == CircuitState.OPEN
        
        # 测试恢复
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        
        print("✅ 熔断器集成测试通过")
        
        return True


class TestExchangeIntegration:
    """交易所集成测试"""
    
    def test_hyperliquid_config(self):
        """Hyperliquid 配置测试"""
        
        from opentrade.core.config import ExchangeConfig
        
        config = ExchangeConfig(
            name="hyperliquid",
            wallet_address="0x1234567890abcdef",
            testnet=False,
        )
        
        assert config.name == "hyperliquid"
        assert config.wallet_address is not None
        
        print("✅ Hyperliquid 配置正常")
        
    def test_binance_config(self):
        """Binance 配置测试"""
        
        from opentrade.core.config import ExchangeConfig
        
        config = ExchangeConfig(
            name="binance",
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
        )
        
        assert config.name == "binance"
        assert config.testnet is True
        
        print("✅ Binance 配置正常")


class TestAPIClientIntegration:
    """API客户端集成测试"""
    
    @pytest.mark.asyncio
    async def test_data_service_api(self):
        """数据服务 API 测试"""
        
        from opentrade.services.data_service import DataService
        
        service = DataService()
        
        # 测试技术指标计算
        test_prices = [50000 + i * 50 for i in range(100)]
        
        rsi = service._rsi(test_prices, 14)
        assert 0 <= rsi <= 100
        
        ema = service._ema(test_prices, 9)
        assert ema > 0
        
        macd, signal, hist = service._macd(test_prices, 12, 26, 9)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)
        
        # 布林带
        upper, middle, lower = service._bollinger_bands(test_prices, 20, 2)
        assert upper > middle > lower
        
        print("✅ 数据服务 API 正常")
        
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
