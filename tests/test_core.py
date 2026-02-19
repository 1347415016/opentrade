"""
OpenTrade 测试包

测试核心模块的功能
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """配置测试"""
    
    def test_exchange_config(self):
        """交易所配置"""
        from opentrade.core.config import ExchangeConfig
        
        config = ExchangeConfig(
            name="binance",
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
        )
        
        assert config.name == "binance"
        assert config.testnet is True
    
    def test_ai_config(self):
        """AI配置"""
        from opentrade.core.config import AIConfig
        
        config = AIConfig(
            model="deepseek/deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
        )
        
        assert config.model == "deepseek/deepseek-chat"
        assert config.temperature == 0.7
    
    def test_gateway_config(self):
        """网关配置测试"""
        from opentrade.core.config import GatewayConfig
        
        config = GatewayConfig(
            host="127.0.0.1",
            port=18790,
            web_port=3000,
        )
        
        assert config.host == "127.0.0.1"
        assert config.port == 18790
    
    def test_notification_config(self):
        """通知配置测试"""
        from opentrade.core.config import NotificationConfig
        
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test-token",
        )
        
        assert config.telegram_enabled is True
    
    def test_storage_config(self):
        """存储配置测试"""
        from opentrade.core.config import StorageConfig
        
        config = StorageConfig(
            database_url="postgresql://localhost/db",
        )
        
        assert "postgresql" in config.database_url
    
    def test_web_config(self):
        """Web配置测试"""
        from opentrade.core.config import WebConfig
        
        config = WebConfig(
            enabled=True,
            title="OpenTrade",
        )
        
        assert config.enabled is True
    
    def test_risk_config(self):
        """风险配置测试"""
        from opentrade.core.config import RiskConfig
        
        config = RiskConfig(
            max_leverage=2.0,
            max_position_pct=0.1,
            stop_loss_pct=0.035,
        )
        
        assert config.max_leverage == 2.0
        assert config.max_position_pct == 0.1


class TestCircuitBreaker:
    """熔断机制测试"""
    
    def test_circuit_state_enum(self):
        """熔断状态枚举测试"""
        from opentrade.core.circuit_breaker import CircuitState
        
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"
    
    def test_circuit_breaker_config(self):
        """熔断器配置测试"""
        from opentrade.core.circuit_breaker import CircuitBreakerConfig
        
        config = CircuitBreakerConfig()
        assert config.max_consecutive_failures == 5


class TestEncryption:
    """加密模块测试"""
    
    def test_key_file_creation(self):
        """测试密钥文件创建"""
        from opentrade.core.encryption import get_or_create_key
        import os
        import tempfile
        
        # 在临时目录测试
        with tempfile.TemporaryDirectory() as tmpdir:
            from opentrade.core.encryption import KEY_FILE
            original_key_file = str(KEY_FILE)
            
            # 临时修改路径
            import opentrade.core.encryption as enc
            enc.KEY_FILE = tmpdir + "/test.key"
            
            try:
                key = get_or_create_key()
                assert len(key) > 0
            finally:
                enc.KEY_FILE = original_key_file


class TestNetwork:
    """网络模块测试"""
    
    def test_network_error_type(self):
        """网络错误类型测试"""
        from opentrade.core.network import NetworkErrorType
        
        assert NetworkErrorType.TIMEOUT.value == "timeout"
        assert NetworkErrorType.CONNECTION.value == "connection"
        assert NetworkErrorType.RATE_LIMIT.value == "rate_limit"
    
    def test_network_config(self):
        """网络配置测试"""
        from opentrade.core.network import NetworkConfig
        
        config = NetworkConfig(
            default_timeout_seconds=30.0,
            max_retries=3,
        )
        
        assert config.default_timeout_seconds == 30.0
        assert config.max_retries == 3


class TestOrder:
    """订单模块测试"""
    
    def test_order_idempotency_config(self):
        """订单幂等性配置测试"""
        from opentrade.core.order import OrderIdempotencyConfig
        
        config = OrderIdempotencyConfig()
        assert config.ttl_seconds > 0
    
    def test_order_deduplication_manager(self):
        """订单去重管理器测试"""
        from opentrade.core.order import OrderIdempotencyManager
        import uuid
        
        manager = OrderIdempotencyManager()
        
        # 生成订单ID
        order_id = str(uuid.uuid4())
        
        # 标记为已处理
        manager.record_order_id(order_id)
        
        # 检查（应该已存在）
        assert manager.is_duplicate(order_id) is True
        
        # 新的订单ID
        new_order_id = str(uuid.uuid4())
        assert manager.is_duplicate(new_order_id) is False


class TestVectorStore:
    """向量存储测试"""
    
    def test_memory_vector_store(self):
        """内存向量存储测试"""
        from opentrade.core.vector_store import MemoryVectorStore, VectorRecord
        from datetime import datetime
        
        store = MemoryVectorStore()
        
        record = VectorRecord(
            id="test-1",
            vector=[0.1, 0.2, 0.3],
            payload={"type": "test"},
            created_at=datetime.utcnow(),
        )
        
        # 添加
        result = store.add(record)
        assert result == "test-1"
        
        # 搜索
        results = store.search([0.1, 0.2, 0.3], limit=5)
        assert len(results) >= 1
        
        # 删除
        assert store.delete("test-1") is True
        
        # 关闭
        store.close()
    
    def test_strategy_experience_store(self):
        """策略经验存储测试"""
        from opentrade.core.vector_store import StrategyExperienceStore
        
        store = StrategyExperienceStore()
        
        # 存储经验
        record_id = store.store_experience(
            strategy_name="trend_following",
            market_condition={"fear_index": 30, "volatility": 0.02},
            action="buy",
            result="success",
            pnl=0.05,
            vector=[0.3, 0.2, 0.1],
        )
        
        assert record_id is not None
        
        store.close()


class TestEvolutionEngine:
    """策略进化引擎测试"""
    
    def test_risk_parameters_extreme_fear(self):
        """极端恐惧模式风险参数"""
        from opentrade.agents.evolution import EvolutionEngine, MarketState
        
        engine = EvolutionEngine()
        engine.market_state = MarketState(fear_greed_index=10)
        
        params = engine.get_risk_parameters()
        
        assert params["max_leverage"] == 1.0
        assert params["stop_loss"] == 0.02
    
    def test_risk_parameters_greedy(self):
        """贪婪模式风险参数"""
        from opentrade.agents.evolution import EvolutionEngine, MarketState
        
        engine = EvolutionEngine()
        engine.market_state = MarketState(fear_greed_index=85)
        
        params = engine.get_risk_parameters()
        
        assert params["max_leverage"] == 2.0
        assert params["risk_mode"] == "greedy"
    
    def test_generate_prompt(self):
        """生成提示词测试"""
        from opentrade.agents.evolution import EvolutionEngine, MarketState
        
        engine = EvolutionEngine()
        engine.market_state = MarketState(
            fear_greed_index=50,
            btc_price=68000.0,
        )
        
        prompt = engine.generate_system_prompt()
        
        assert "OpenTrade" in prompt
        assert isinstance(prompt, str)
    
    def test_evolution_report(self):
        """进化报告测试"""
        from opentrade.agents.evolution import EvolutionEngine
        
        engine = EvolutionEngine()
        report = engine.get_evolution_report()
        
        assert "strategy_weights" in report
        assert "market_state" in report


class TestBacktestEngine:
    """回测引擎测试"""
    
    def test_backtest_engine_creation(self):
        """回测引擎创建测试"""
        from opentrade.cli.backtest import BacktestEngine
        from datetime import datetime
        
        engine = BacktestEngine(
            initial_balance=10000,
            fee_rate=0.001,
        )
        
        assert engine.initial_balance == 10000
        assert engine.fee_rate == 0.001
        assert engine.balance == 10000
    
    def test_simulated_prices(self):
        """模拟价格生成测试"""
        from opentrade.cli.backtest import BacktestEngine
        from datetime import datetime
        
        engine = BacktestEngine(initial_balance=10000)
        
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 10)
        
        prices = engine._generate_simulated_prices(start, end)
        
        assert len(prices) > 0
        assert all(isinstance(p, float) for p in prices)
        assert all(p > 0 for p in prices)


class TestInitCommand:
    """初始化命令测试"""
    
    def test_ai_providers_exist(self):
        """AI提供商配置测试"""
        from opentrade.cli.init import AI_PROVIDERS
        
        assert "deepseek" in AI_PROVIDERS
        assert "openai" in AI_PROVIDERS
        assert "anthropic" in AI_PROVIDERS
        
        # 检查必需字段
        for provider_key, provider in AI_PROVIDERS.items():
            assert "name" in provider
            assert "models" in provider
            assert "default_model" in provider


class TestDailyWorkflow:
    """每日工作流测试"""
    
    def test_sentiment_label_extreme_fear(self):
        """极端恐惧标签测试"""
        from scripts.daily_workflow import get_sentiment_label
        
        label = get_sentiment_label(10)
        assert "Fear" in label or "fear" in label
    
    def test_sentiment_label_greedy(self):
        """贪婪标签测试"""
        from scripts.daily_workflow import get_sentiment_label
        
        label = get_sentiment_label(80)
        assert "Greed" in label or "greed" in label


class TestBackup:
    """备份模块测试"""
    
    def test_timestamp_format(self):
        """时间戳格式测试"""
        from scripts.backup import get_timestamp
        import re
        
        timestamp = get_timestamp()
        # 格式: YYYYMMDD_HHMMSS
        assert re.match(r"\d{8}_\d{6}", timestamp)


class TestDataService:
    """数据服务测试"""
    
    def test_indicator_calculation(self):
        """测试技术指标计算"""
        from opentrade.services.data_service import DataService
        
        service = DataService()
        
        # 模拟价格数据
        closes = [50000 + i * 100 for i in range(50)]
        
        # 测试 EMA 计算
        ema = service._ema(closes, 9)
        assert ema > 0
        
        # 测试 RSI 计算
        rsi = service._rsi(closes, 14)
        assert 0 <= rsi <= 100


class TestCoordinator:
    """协调器测试"""
    
    def test_agent_type(self):
        """AgentType 测试"""
        from opentrade.agents.coordinator import AgentType
        
        assert AgentType.MARKET.value == "market"
        assert AgentType.STRATEGY.value == "strategy"
        assert AgentType.RISK.value == "risk"
        assert AgentType.ONCHAIN.value == "onchain"
        assert AgentType.SENTIMENT.value == "sentiment"
        assert AgentType.MACRO.value == "macro"
    
    def test_market_direction(self):
        """MarketDirection 测试"""
        from opentrade.agents.coordinator import MarketDirection
        
        assert MarketDirection.BULLISH.value == "bullish"
        assert MarketDirection.BEARISH.value == "bearish"
        assert MarketDirection.NEUTRAL.value == "neutral"
    
    def test_agent_vote(self):
        """AgentVote 测试"""
        from opentrade.agents.coordinator import AgentVote, AgentType
        
        vote = AgentVote(
            agent_type=AgentType.MARKET,
            agent_name="Market",
            direction="bullish",
            confidence=0.8,
            score=0.75,
        )
        
        assert vote.agent_type == AgentType.MARKET
        assert vote.direction == "bullish"
    
    def test_agent_input(self):
        """AgentInput 测试"""
        from opentrade.agents.coordinator import AgentInput
        
        input_data = AgentInput(
            symbol="BTC/USDT",
            price=68000.0,
        )
        
        assert input_data.symbol == "BTC/USDT"
        assert input_data.price == 68000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
