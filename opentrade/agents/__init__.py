"""
OpenTrade Agents Package

多智能体交易系统，包含：
- 基础 Agent 类
- 6 大专业 Agent
- 协调器 (Coordinator)
- 身份系统 (Identity)
- 进化引擎 (Evolution)
- 大脑系统 (Brain)
"""

from opentrade.agents.base import (
    BaseAgent,
    MarketState,
    SignalConfidence,
    SignalType,
    TradeDecision,
)
from opentrade.agents.coordinator import AgentCoordinator
from opentrade.agents.graph import create_trading_graph, get_trading_graph, run_graph
from opentrade.agents.macro import MacroAgent
from opentrade.agents.market import MarketAgent
from opentrade.agents.onchain import OnchainAgent
from opentrade.agents.risk import RiskAgent
from opentrade.agents.sentiment import SentimentAgent
from opentrade.agents.strategy import StrategyAgent

# 身份系统
from opentrade.agents.identity import (
    get_agent_team,
    AgentTeam,
    AgentRole,
    AGENT_IDENTITIES,
)

# 进化引擎
from opentrade.agents.evolution import (
    get_evolution_engine,
    EvolutionEngine,
)

# 大脑系统
from opentrade.agents.brain import (
    get_brain,
    OpenTradeBrain,
    TradingDecision,
    analyze_market,
    record_result,
    run_evolution,
)

__all__ = [
    # Base
    "BaseAgent",
    "MarketState",
    "SignalType",
    "SignalConfidence",
    "TradeDecision",
    # Agents
    "AgentCoordinator",
    "MarketAgent",
    "StrategyAgent",
    "RiskAgent",
    "OnchainAgent",
    "SentimentAgent",
    "MacroAgent",
    # Graph
    "create_trading_graph",
    "get_trading_graph",
    "run_graph",
    # Identity
    "get_agent_team",
    "AgentTeam",
    "AgentRole",
    "AGENT_IDENTITIES",
    # Evolution
    "get_evolution_engine",
    "EvolutionEngine",
    # Brain
    "get_brain",
    "OpenTradeBrain",
    "TradingDecision",
    "analyze_market",
    "record_result",
    "run_evolution",
]
