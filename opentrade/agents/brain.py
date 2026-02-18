"""
OpenTrade 自主进化系统

这是整个系统的核心大脑，负责：
1. 协调所有 Agent
2. 管理历史数据
3. 自动进化策略
4. 生成智能决策
"""

import json
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from opentrade.core.config import get_config
from opentrade.agents.evolution import get_evolution_engine, EvolutionEngine
from opentrade.agents.identity import get_agent_team, AgentTeam, AgentRole
from opentrade.data.history_manager import get_history_manager, HistoryDataManager


@dataclass
class TradingDecision:
    """交易决策"""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    symbol: str = "BTC/USDT"
    action: str = "hold"
    leverage: float = 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    risk_level: str = "medium"
    reasoning: str = ""
    agent_votes: Dict[str, dict] = field(default_factory=dict)
    final_decision: str = "hold_by_default"


class OpenTradeBrain:
    """
    OpenTrade 大脑 - 自主进化交易系统
    
    设计理念：
    1. 多 Agent 协作决策
    2. 从历史中学习
    3. 自动进化策略
    4. 持续优化性能
    """
    
    _instance: Optional["OpenTradeBrain"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        
        # 初始化子系统
        self.evolution = get_evolution_engine()
        self.agents = get_agent_team()
        self.history = get_history_manager()
        
        # 决策历史
        self.decision_history: List[TradingDecision] = []
        
        # 系统状态
        self.status = {
            "initialized": True,
            "last_decision": None,
            "total_decisions": 0,
            "correct_predictions": 0,
            "system_health": "healthy",
        }
        
        self._initialized = True
        
        # 打印系统状态
        print("=" * 60)
        print("🧠 OpenTrade 大脑已启动")
        print("=" * 60)
        print(f"  Agent 团队: {len(self.agents.agents)} 个成员")
        print(f"  历史数据: {len(self.history.market_events)} 条事件")
        print(f"  进化引擎: 已就绪")
        print("=" * 60)
    
    def analyze_market(self, market_data: dict) -> TradingDecision:
        """
        分析市场并生成决策
        
        流程：
        1. 更新市场状态
        2. 收集各 Agent 意见
        3. 综合决策
        4. 执行风险检查
        """
        symbol = market_data.get("symbol", "BTC/USDT")
        price = market_data.get("price", 68000)
        
        print(f"\n🧠 分析市场: {symbol} @ ${price:,.2f}")
        
        # 1. 更新市场状态到进化引擎
        fear_index = market_data.get("fear_greed_index", 50)
        self.evolution.update_market_state(
            fear_greed_index=fear_index,
            btc_price=price,
            volatility=market_data.get("volatility", 0.02),
            trend=market_data.get("trend", "neutral"),
        )
        
        # 2. 获取各 Agent 意见（模拟）
        agent_votes = self._collect_agent_opinions(market_data)
        
        # 3. 综合决策
        decision = self._synthesize_decision(symbol, price, agent_votes)
        decision.agent_votes = agent_votes
        
        # 4. 应用风险控制
        risk_params = self.evolution.get_risk_parameters()
        decision = self._apply_risk_control(decision, price, risk_params)
        
        # 5. 记录决策
        self.decision_history.append(decision)
        self.status["last_decision"] = decision.timestamp
        self.status["total_decisions"] += 1
        
        print(f"   决策: {decision.action} {decision.symbol}")
        print(f"   杠杆: {decision.leverage}x")
        print(f"   置信度: {decision.confidence:.2f}")
        print(f"   风险: {decision.risk_level}")
        
        return decision
    
    def _collect_agent_opinions(self, market_data: dict) -> Dict[str, dict]:
        """收集各 Agent 意见"""
        votes = {}
        symbol = market_data.get("symbol", "BTC/USDT")
        price = market_data.get("price", 68000)
        fear = market_data.get("fear_greed_index", 50)
        
        # 遍历所有 Agent
        for role in AgentRole:
            identity = self.agents.agents[role]["identity"]
            
            # 根据角色生成意见
            opinion = self._generate_agent_opinion(role, market_data)
            votes[role.value] = {
                "name": identity.name,
                "action": opinion["action"],
                "confidence": opinion["confidence"],
                "reasoning": opinion["reasoning"],
                "risk_level": opinion.get("risk_level", "medium"),
            }
        
        return votes
    
    def _generate_agent_opinion(self, role: AgentRole, market_data: dict) -> dict:
        """生成 Agent 意见"""
        price = market_data.get("price", 68000)
        fear = market_data.get("fear_greed_index", 50)
        trend = market_data.get("trend", "neutral")
        
        identity = self.agents.agents[role]["identity"]
        
        if role == AgentRole.MARKET_ANALYST:
            # 技术分析 Agent
            if trend == "bullish":
                return {"action": "buy", "confidence": 0.75, "reasoning": "技术指标显示上涨趋势", "risk_level": "medium"}
            elif trend == "bearish":
                return {"action": "sell", "confidence": 0.7, "reasoning": "技术指标显示下跌趋势", "risk_level": "high"}
            return {"action": "hold", "confidence": 0.6, "reasoning": "技术指标不明确", "risk_level": "low"}
        
        elif role == AgentRole.RISK_MANAGER:
            # 风控 Agent - 总是偏保守
            if fear < 20:
                return {"action": "hold", "confidence": 0.8, "reasoning": "极度恐惧市场，建议观望", "risk_level": "low"}
            elif fear > 80:
                return {"action": "sell", "confidence": 0.75, "reasoning": "过度贪婪，风险增加", "risk_level": "high"}
            return {"action": "hold", "confidence": 0.7, "reasoning": "风险可控，观望为主", "risk_level": "medium"}
        
        elif role == AgentRole.SENTIMENT_ANALYST:
            # 情绪 Agent - 逆向思维
            if fear < 15:
                return {"action": "buy", "confidence": 0.7, "reasoning": "极度恐惧可能是买入机会", "risk_level": "medium"}
            elif fear > 75:
                return {"action": "sell", "confidence": 0.65, "reasoning": "过度贪婪，考虑减仓", "risk_level": "high"}
            return {"action": "hold", "confidence": 0.6, "reasoning": "情绪中性，等待信号", "risk_level": "low"}
        
        elif role == AgentRole.STRATEGIST:
            # 策略 Agent - 趋势跟踪
            if trend == "bullish":
                return {"action": "buy", "confidence": 0.8, "reasoning": "趋势向上，顺势而为", "risk_level": "medium"}
            elif trend == "bearish":
                return {"action": "sell", "confidence": 0.75, "reasoning": "趋势向下，顺势做空", "risk_level": "high"}
            return {"action": "hold", "confidence": 0.5, "reasoning": "趋势不明，等待突破", "risk_level": "low"}
        
        elif role == AgentRole.ONCHAIN_ANALYST:
            # 链上 Agent
            return {"action": "hold", "confidence": 0.6, "reasoning": "等待链上数据确认", "risk_level": "low"}
        
        elif role == AgentRole.MACRO_ANALYST:
            # 宏观 Agent
            return {"action": "hold", "confidence": 0.65, "reasoning": "关注宏观政策变化", "risk_level": "medium"}
        
        else:
            # 默认
            return {"action": "hold", "confidence": 0.5, "reasoning": "收集信息中", "risk_level": "medium"}
    
    def _synthesize_decision(self, symbol: str, price: float, agent_votes: dict) -> TradingDecision:
        """综合所有 Agent 意见生成最终决策"""
        
        # 统计投票
        votes = {"buy": 0, "sell": 0, "hold": 0}
        total_confidence = 0
        weighted_confidence = 0
        
        for role, vote in agent_votes.items():
            action = vote["action"]
            confidence = vote["confidence"]
            
            votes[action] += 1
            total_confidence += confidence
            weighted_confidence += confidence * (1 if action != "hold" else 0.5)
        
        # 决定最终动作
        if votes["buy"] > votes["sell"] and votes["buy"] >= votes["hold"]:
            final_action = "buy"
        elif votes["sell"] > votes["buy"] and votes["sell"] >= votes["hold"]:
            final_action = "sell"
        else:
            final_action = "hold"
        
        # 计算置信度
        avg_confidence = total_confidence / len(agent_votes) if agent_votes else 0.5
        
        # 生成推理
        reasoning_parts = []
        for role, vote in agent_votes.items():
            if vote["action"] == final_action:
                reasoning_parts.append(f"{vote['name']}: {vote['reasoning']}")
        
        reasoning = " | ".join(reasoning_parts[:3]) if reasoning_parts else "等待更明确信号"
        
        return TradingDecision(
            symbol=symbol,
            action=final_action,
            leverage=1.0,
            entry_price=price if final_action != "hold" else None,
            confidence=avg_confidence,
            reasoning=reasoning,
            final_decision=f"{final_action}_by_consensus",
        )
    
    def _apply_risk_control(self, decision: TradingDecision, price: float, risk_params: dict) -> TradingDecision:
        """应用风险控制"""
        
        # 如果是 hold，不做处理
        if decision.action == "hold":
            decision.stop_loss = None
            decision.take_profit = None
            decision.risk_level = "low"
            return decision
        
        # 设置杠杆上限
        decision.leverage = min(decision.leverage, risk_params.get("max_leverage", 2.0))
        
        # 设置止损
        stop_loss_pct = risk_params.get("stop_loss", 0.035)
        if decision.action == "buy":
            decision.stop_loss = price * (1 - stop_loss_pct)
        else:  # sell
            decision.stop_loss = price * (1 + stop_loss_pct)
        
        # 设置止盈
        take_profit_pct = risk_params.get("take_profit_pct", 0.07)
        if decision.action == "buy":
            decision.take_profit = price * (1 + take_profit_pct)
        else:
            decision.take_profit = price * (1 - take_profit_pct)
        
        # 评估风险等级
        if stop_loss_pct <= 0.025:
            decision.risk_level = "low"
        elif stop_loss_pct <= 0.04:
            decision.risk_level = "medium"
        else:
            decision.risk_level = "high"
        
        return decision
    
    def record_result(self, decision: TradingDecision, actual_pnl_pct: float, correct: bool):
        """记录决策结果，用于学习"""
        # 记录到历史
        self.history.add_trading_signal({
            "date": decision.timestamp,
            "symbol": decision.symbol,
            "signal_type": decision.action,
            "strategy": "multi_agent",
            "entry_price": decision.entry_price,
            "exit_price": decision.entry_price * (1 + actual_pnl_pct/100) if decision.entry_price else None,
            "pnl_pct": actual_pnl_pct,
            "confidence": decision.confidence,
            "reason": decision.reasoning,
            "executed": True,
        })
        
        # 更新 Agent 表现
        for role, vote in decision.agent_votes.items():
            if vote["action"] == decision.action:
                self.agents.record_prediction(AgentRole(role), correct, vote["confidence"])
        
        # 如果正确，增加计数
        if correct:
            self.status["correct_predictions"] += 1
    
    def evolve(self):
        """执行进化"""
        print("\n🔄 执行系统进化...")
        
        # 1. 进化策略权重
        new_weights = self.evolution.evolve()
        print(f"   新策略权重: {json.dumps(new_weights, indent=2)}")
        
        # 2. 进化 Agent 提示词
        self.agents.evolve_prompts()
        print("   Agent 提示词已更新")
        
        # 3. 导出更新后的配置
        self.agents.export_identities()
        print("   身份配置已保存")
        
        print("   ✅ 进化完成")
        
        return new_weights
    
    def get_system_report(self) -> dict:
        """获取系统状态报告"""
        return {
            "status": self.status,
            "evolution": self.evolution.get_evolution_report(),
            "agents": self.agents.get_team_report(),
            "history_summary": {
                "events": len(self.history.market_events),
                "patterns": len(self.history.price_patterns),
                "signals": len(self.history.trading_signals),
            },
            "recent_decisions": len(self.decision_history[-10:]),
        }
    
    def generate_ai_context(self) -> str:
        """生成 AI 上下文 - 精选数据喂给 AI"""
        
        # 获取精选历史摘要
        summary = self.history.get精选_summary()
        
        # 获取当前系统状态
        risk_params = self.evolution.get_risk_parameters()
        weights = self.evolution.strategy_weights
        
        # 生成上下文
        context = f"""# OpenTrade AI 上下文

## 当前市场状态
- BTC 价格: ${self.evolution.market_state.btc_price:,.0f}
- 恐惧指数: {self.evolution.market_state.fear_greed_index}/100
- 市场趋势: {self.evolution.market_state.trend}

## 风险控制参数
- 最大杠杆: {risk_params['max_leverage']}x
- 单仓止损: {risk_params['stop_loss']*100:.1f}%
- 总敞口限制: {risk_params['max_exposure']*100:.0f}%
- 稳定币配置: {risk_params['stablecoin_ratio']*100:.0f}%
- 风险模式: {risk_params['risk_mode']}

## 策略权重
{json.dumps(weights, indent=2)}

## 历史经验
### 成功形态
"""
        
        for pattern in summary["successful_patterns"][:5]:
            context += f"- {pattern['pattern_type']}: {pattern['profit_pct']:.2f}% 盈利\n"
        
        context += f"""
### 关键教训
"""
        for lesson in summary["key_lessons"][:5]:
            context += f"- {lesson}\n"
        
        context += f"""
### 交易表现
- 总交易: {summary['performance']['total_trades']}
- 胜率: {summary['performance']['win_rate']:.2%}
- 平均盈亏: {summary['performance']['average_pnl_pct']:.2f}%

## Agent 团队意见
"""
        
        for role, perf in self.agents.get_team_report().items():
            if isinstance(perf, dict) and "accuracy" in perf:
                context += f"- {perf.get('name', role)}: 准确率 {perf['accuracy']:.1%}\n"
        
        return context
    
    def print_status(self):
        """打印系统状态"""
        report = self.get_system_report()
        
        print("\n" + "=" * 60)
        print("🧠 OpenTrade 大脑状态")
        print("=" * 60)
        print(f"  系统健康: {report['status']['system_health']}")
        print(f"  总决策数: {report['status']['total_decisions']}")
        print(f"  正确预测: {report['status']['correct_predictions']}")
        if report['status']['total_decisions'] > 0:
            accuracy = report['status']['correct_predictions'] / report['status']['total_decisions']
            print(f"  准确率: {accuracy:.1%}")
        print(f"  历史事件: {report['history_summary']['events']}")
        print(f"  Agent 数量: {len(report['agents']['agents'])}")
        print("=" * 60)


# 单例访问
def get_brain() -> OpenTradeBrain:
    """获取大脑单例"""
    return OpenTradeBrain()


# 便捷函数
def analyze_market(market_data: dict) -> TradingDecision:
    """快速分析市场"""
    brain = get_brain()
    return brain.analyze_market(market_data)


def record_result(decision: TradingDecision, pnl: float, correct: bool):
    """记录结果"""
    brain = get_brain()
    brain.record_result(decision, pnl, correct)


def run_evolution():
    """执行进化"""
    brain = get_brain()
    return brain.evolve()
