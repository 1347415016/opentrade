"""
OpenTrade LangGraph Agents Test Suite

测试多智能体协作系统
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAgentsGraph:
    """Agents Graph 测试"""
    
    def test_graph_creation(self):
        """测试图创建"""
        try:
            from opentrade.agents.graph import create_trading_graph, TradingState
            
            # 创建图
            graph = create_trading_graph()
            
            assert graph is not None
            
        except ImportError as e:
            pytest.skip(f"LangGraph 不可用: {e}")
    
    def test_trading_state(self):
        """交易状态测试"""
        try:
            from opentrade.agents.graph import TradingState
            from datetime import datetime
            
            state = TradingState(
                symbol="BTC/USDT",
                current_price=68000.0,
                timestamp=datetime.utcnow(),
            )
            
            assert state.symbol == "BTC/USDT"
            assert state.current_price == 68000.0
            
        except ImportError:
            pytest.skip("TradingState 不可用")
    
    def test_agent_type_enum(self):
        """AgentType 枚举测试"""
        from opentrade.agents.coordinator import AgentType
        
        assert AgentType.MARKET.value == "market"
        assert AgentType.STRATEGY.value == "strategy"
        assert AgentType.RISK.value == "risk"
    
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
        assert vote.confidence == 0.8
    
    def test_agent_output(self):
        """AgentOutput 测试"""
        from opentrade.agents.coordinator import AgentOutput, AgentType
        
        output = AgentOutput(
            agent_type=AgentType.MARKET,
            agent_name="Market",
            direction="bullish",
            confidence=0.8,
            score=0.75,
            reasoning=["test"],
            details={},
        )
        
        assert output.agent_type == AgentType.MARKET
        assert output.reasoning == ["test"]


class TestMarketAgent:
    """市场代理测试"""
    
    def test_market_agent_exists(self):
        """市场代理存在测试"""
        from opentrade.agents.market import MarketAgent
        
        agent = MarketAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")
    
    def test_market_analyze_returns_dict(self):
        """市场分析返回字典测试"""
        from opentrade.agents.market import MarketAgent
        from datetime import datetime
        
        agent = MarketAgent()
        
        # 简单的输入测试
        result = asyncio.get_event_loop().run_until_complete(
            agent.analyze({"symbol": "BTC/USDT", "price": 68000.0})
        )
        
        assert isinstance(result, dict)


class TestStrategyAgent:
    """策略代理测试"""
    
    def test_strategy_agent_exists(self):
        """策略代理存在测试"""
        from opentrade.agents.strategy import StrategyAgent
        
        agent = StrategyAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")
    
    def test_strategy_analyze_returns_dict(self):
        """策略分析返回字典测试"""
        from opentrade.agents.strategy import StrategyAgent
        
        agent = StrategyAgent()
        
        result = asyncio.get_event_loop().run_until_complete(
            agent.analyze({"symbol": "BTC/USDT", "price": 68000.0})
        )
        
        assert isinstance(result, dict)


class TestRiskAgent:
    """风险代理测试"""
    
    def test_risk_agent_exists(self):
        """风险代理存在测试"""
        from opentrade.agents.risk import RiskAgent
        
        agent = RiskAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")
    
    def test_risk_level_enum(self):
        """风险等级枚举测试"""
        from opentrade.core.risk import RiskLevel
        
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"


class TestOnchainAgent:
    """链上代理测试"""
    
    def test_onchain_agent_exists(self):
        """链上代理存在测试"""
        from opentrade.agents.onchain import OnchainAgent
        
        agent = OnchainAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")


class TestSentimentAgent:
    """情绪代理测试"""
    
    def test_sentiment_agent_exists(self):
        """情绪代理存在测试"""
        from opentrade.agents.sentiment import SentimentAgent
        
        agent = SentimentAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")


class TestMacroAgent:
    """宏观代理测试"""
    
    def test_macro_agent_exists(self):
        """宏观代理存在测试"""
        from opentrade.agents.macro import MacroAgent
        
        agent = MacroAgent()
        assert hasattr(agent, "name")
        assert hasattr(agent, "analyze")


class TestCoordinator:
    """协调器测试"""
    
    def test_agent_coordinator_exists(self):
        """AgentCoordinator 存在测试"""
        from opentrade.agents.coordinator import AgentCoordinator
        
        coordinator = AgentCoordinator()
        assert hasattr(coordinator, "run_analysis")
    
    def test_agent_input(self):
        """AgentInput 测试"""
        from opentrade.agents.coordinator import AgentInput
        
        input_data = AgentInput(
            symbol="BTC/USDT",
            price=68000.0,
            ohlcv={},
            risk_level="medium",
            max_leverage=2.0,
        )
        
        assert input_data.symbol == "BTC/USDT"
        assert input_data.price == 68000.0


class TestGraphNodes:
    """Graph 节点测试"""
    
    def test_market_analysis_node(self):
        """市场分析节点测试"""
        try:
            from opentrade.agents.graph import market_analysis_node
            
            state = {
                "symbol": "BTC/USDT",
                "price": 68000.0,
                "ohlcv": {},
                "risk_level": "medium",
                "max_leverage": 2.0,
                "trace_id": "test-123",
                "agent_outputs": {},
            }
            
            result = asyncio.get_event_loop().run_until_complete(
                market_analysis_node(state)
            )
            
            assert "agent_outputs" in result
            
        except ImportError:
            pytest.skip("LangGraph 不可用")


class TestAgentTypes:
    """Agent 类型测试"""
    
    def test_market_direction(self):
        """MarketDirection 枚举测试"""
        from opentrade.agents.coordinator import MarketDirection
        
        assert MarketDirection.BULLISH.value == "bullish"
        assert MarketDirection.BEARISH.value == "bearish"
        assert MarketDirection.NEUTRAL.value == "neutral"


class TestFinalDecision:
    """最终决策测试"""
    
    def test_final_decision_type(self):
        """FinalDecision 类型测试"""
        from opentrade.agents.coordinator import FinalDecision
        
        assert hasattr(FinalDecision, "__members__")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
