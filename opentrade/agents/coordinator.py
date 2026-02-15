"""
OpenTrade 多 Agent 协调器 - LangGraph 工作流

基于 LangGraph 的事件驱动工作流，实现六大 Agent 并行分析、辩论共识。

工作流:
    ┌─────────────────────────────────────────────────────────────┐
    │                    MarketState 输入                          │
    └─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Market  │     │Strategy │     │  Risk   │
        │  Agent  │     │  Agent  │     │  Agent  │
        └────┬────┘     └────┬────┘     └────┬────┘
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                    ┌────────────────┐
                    │  辩论/共识引擎   │
                    │ (Bull vs Bear)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Onchain │     │Sentiment│     │  Macro  │
        │  Agent  │     │  Agent  │     │  Agent  │
        └─────────┘     └─────────┘     └─────────┘
                             │
                    ┌────────┴────────┐
                    │   最终决策合成    │
                    └────────┬────────┘
                             ▼
                    ┌────────────────┐
                    │  风控最终校验    │
                    └────────┬────────┘
                             ▼
                    ┌────────────────┐
                    │  TradeDecision  │
                    │    (输出)       │
                    └────────────────┘
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from rich import print


# ============ 统一 Schema ============

class AgentType(str, Enum):
    """Agent 类型"""
    MARKET = "market"
    STRATEGY = "strategy"
    RISK = "risk"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    MACRO = "macro"


class MarketDirection(str, Enum):
    """市场方向"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class AgentVote(BaseModel):
    """Agent 投票"""
    agent_type: AgentType
    agent_name: str
    direction: MarketDirection
    confidence: float = Field(..., ge=0, le=1, description="置信度 0-1")
    score: float = Field(..., description="得分 -1 到 1", ge=-1, le=1)
    reasons: list[str] = Field(default_factory=list, description="投票理由")
    evidence: dict[str, Any] = Field(default_factory=dict, description="支持证据")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentOutput(BaseModel):
    """Agent 输出 - 统一格式"""
    agent_type: AgentType
    agent_name: str
    direction: MarketDirection = MarketDirection.UNKNOWN
    score: float = 0.0  # -1 (强烈看跌) 到 1 (强烈看涨)
    confidence: float = 0.5  # 0-1

    # 分析结果
    analysis: dict[str, Any] = Field(default_factory=dict)

    # 投票详情
    vote: AgentVote = Field(default_factory=AgentVote)

    # 置信度分数分解
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)

    # 关键发现
    key_findings: list[str] = Field(default_factory=list)

    # 风险提示
    risks: list[str] = Field(default_factory=list)

    # 时间戳
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # 追溯 ID
    trace_id: Optional[str] = None


class ConsensusResult(BaseModel):
    """共识结果"""
    direction: MarketDirection
    overall_score: float
    confidence: float

    # 投票统计
    votes: list[AgentVote] = Field(default_factory=list)
    vote_counts: dict[str, int] = Field(default_factory=dict)

    # 权重计算
    weighted_score: float = 0.0

    # 共识强度
    consensus_strength: float = Field(..., description="0-1, 越接近1共识越强")

    # 冲突解释
    conflicts: list[dict[str, Any]] = Field(default_factory=list)

    # 最终理由
    final_reasons: list[str] = Field(default_factory=list)


class AgentInput(BaseModel):
    """Agent 输入 - 统一格式"""
    symbol: str = Field(..., description="交易对")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="追溯ID")

    # 市场数据
    price: float = Field(..., description="当前价格")
    ohlcv: dict[str, Any] = Field(default_factory=dict, description="K线数据")

    # 配置
    risk_level: str = Field(default="medium", description="风险等级")
    max_leverage: float = Field(default=2.0, description="最大杠杆")

    # 时间
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class FinalDecision(BaseModel):
    """最终决策"""
    # 基本信息
    trace_id: str
    symbol: str
    timestamp: str

    # 决策结果
    action: str = Field(..., description="BUY/SELL/HOLD/CLOSE")
    direction: MarketDirection

    # 交易参数
    size: float = Field(..., description="仓位大小 0-1")
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # 置信度
    confidence: float = 0.5
    risk_score: float = 0.5

    # 分析来源
    agent_votes: list[AgentVote] = Field(default_factory=list)
    consensus: ConsensusResult = Field(default_factory=ConsensusResult)

    # 原因
    reasons: list[str] = Field(default_factory=list)
    summary: str = ""

    # 风控
    risk_check_passed: bool = False
    risk_message: str = ""

    # 元数据
    processing_time_ms: float = 0.0


# ============ 辩论引擎 ============

class DebateEngine:
    """
    辩论/共识引擎

    实现多空对抗、权重聚合、冲突解释
    """

    # Agent 权重配置
    AGENT_WEIGHTS: dict[AgentType, float] = {
        AgentType.MARKET: 0.25,      # 技术分析权重最高
        AgentType.STRATEGY: 0.20,    # 策略次之
        AgentType.RISK: 0.20,        # 风控重要
        AgentType.ONCHAIN: 0.12,     # 链上数据
        AgentType.SENTIMENT: 0.12,   # 情绪
        AgentType.MACRO: 0.11,       # 宏观权重最低
    }

    # 方向分数映射
    DIRECTION_SCORES: dict[MarketDirection, float] = {
        MarketDirection.BULLISH: 1.0,
        MarketDirection.NEUTRAL: 0.0,
        MarketDirection.BEARISH: -1.0,
        MarketDirection.UNKNOWN: 0.0,
    }

    def calculate_consensus(self, outputs: list[AgentOutput]) -> ConsensusResult:
        """
        计算共识

        Args:
            outputs: 各 Agent 的输出

        Returns:
            ConsensusResult: 共识结果
        """
        if not outputs:
            return ConsensusResult(
                direction=MarketDirection.UNKNOWN,
                overall_score=0.0,
                confidence=0.0,
                consensus_strength=0.0,
            )

        votes = []
        vote_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
        weighted_score_sum = 0.0
        weight_total = 0.0

        for output in outputs:
            # 创建投票
            vote = AgentVote(
                agent_type=output.agent_type,
                agent_name=output.agent_name,
                direction=output.direction,
                confidence=output.confidence,
                score=output.score,
                reasons=output.key_findings,
                evidence=output.analysis,
            )
            votes.append(vote)

            # 统计投票
            vote_counts[output.direction.value] += 1

            # 加权分数
            weight = self.AGENT_WEIGHTS.get(output.agent_type, 0.1)
            weighted_score_sum += output.score * weight
            weight_total += weight

        # 计算最终分数
        overall_score = weighted_score_sum / weight_total if weight_total > 0 else 0.0

        # 确定方向
        if overall_score > 0.1:
            direction = MarketDirection.BULLISH
        elif overall_score < -0.1:
            direction = MarketDirection.BEARISH
        else:
            direction = MarketDirection.NEUTRAL

        # 计算共识强度 (0-1)
        max_count = max(vote_counts.values())
        total = len(outputs)
        consensus_strength = (max_count / total) if total > 0 else 0.0

        # 检测冲突
        conflicts = self._detect_conflicts(outputs)

        # 计算置信度
        avg_confidence = sum(o.confidence for o in outputs) / len(outputs)
        confidence = avg_confidence * consensus_strength

        # 生成最终理由
        final_reasons = self._generate_final_reasons(outputs, direction)

        return ConsensusResult(
            direction=direction,
            overall_score=overall_score,
            confidence=confidence,
            votes=votes,
            vote_counts=vote_counts,
            weighted_score=overall_score,
            consensus_strength=consensus_strength,
            conflicts=conflicts,
            final_reasons=final_reasons,
        )

    def _detect_conflicts(self, outputs: list[AgentOutput]) -> list[dict[str, Any]]:
        """检测冲突"""
        conflicts = []

        # 找出方向不一致的 Agent 对
        bullish = [o for o in outputs if o.direction == MarketDirection.BULLISH]
        bearish = [o for o in outputs if o.direction == MarketDirection.BEARISH]

        if bullish and bearish:
            conflicts.append({
                "type": "direction_conflict",
                "bullish_agents": [o.agent_name for o in bullish],
                "bearish_agents": [o.agent_name for o in bearish],
                "description": f"{len(bullish)} 个 Agent 看涨，{len(bearish)} 个 Agent 看跌",
            })

        # 检测高置信度冲突
        for o1 in outputs:
            for o2 in outputs:
                if o1.agent_type != o2.agent_type:
                    # 方向相反且都高置信度
                    if (o1.direction != o2.direction and
                        o1.confidence > 0.7 and o2.confidence > 0.7):
                        conflicts.append({
                            "type": "high_confidence_conflict",
                            "agents": [o1.agent_name, o2.agent_name],
                            "description": f"高置信度冲突: {o1.agent_name} ({o1.direction.value}) vs {o2.agent_name} ({o2.direction.value})",
                        })

        return conflicts

    def _generate_final_reasons(
        self,
        outputs: list[AgentOutput],
        direction: MarketDirection,
    ) -> list[str]:
        """生成最终理由"""
        reasons = []

        # 按方向收集关键发现
        target_direction = direction
        relevant_outputs = [o for o in outputs if o.direction == target_direction]

        # 收集最有力的理由
        if relevant_outputs:
            # 按置信度排序
            sorted_outputs = sorted(
                relevant_outputs,
                key=lambda x: x.confidence * abs(x.score),
                reverse=True,
            )

            for output in sorted_outputs[:3]:
                for finding in output.key_findings[:2]:
                    if finding not in reasons:
                        reasons.append(f"[{output.agent_name}] {finding}")

        return reasons


# ============ 审计追溯 ============

class AuditLogger:
    """审计日志"""

    def __init__(self):
        self.traces: dict[str, dict] = {}

    def start_trace(self, trace_id: str, input_data: AgentInput) -> dict:
        """开始追溯"""
        trace = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data.model_dump(),
            "agent_outputs": [],
            "consensus": None,
            "final_decision": None,
            "processing_time_ms": 0.0,
        }
        self.traces[trace_id] = trace
        trace["start_time"] = datetime.utcnow()
        return trace

    def log_agent_output(self, trace_id: str, output: AgentOutput):
        """记录 Agent 输出"""
        if trace_id in self.traces:
            self.traces[trace_id]["agent_outputs"].append({
                "agent_type": output.agent_type.value,
                "agent_name": output.agent_name,
                "direction": output.direction.value,
                "score": output.score,
                "confidence": output.confidence,
                "timestamp": output.timestamp,
            })

    def log_consensus(self, trace_id: str, consensus: ConsensusResult):
        """记录共识结果"""
        if trace_id in self.traces:
            self.traces[trace_id]["consensus"] = consensus.model_dump()

    def log_decision(self, trace_id: str, decision: FinalDecision):
        """记录最终决策"""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace["final_decision"] = decision.model_dump()
            trace["end_time"] = datetime.utcnow()
            trace["processing_time_ms"] = (
                (trace["end_time"] - trace["start_time"]).total_seconds() * 1000
            )

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取追溯记录"""
        return self.traces.get(trace_id)

    def get_recent_traces(self, limit: int = 10) -> list[dict]:
        """获取最近追溯"""
        sorted_traces = sorted(
            self.traces.items(),
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True,
        )
        return [t[1] for t in sorted_traces[:limit]]

    def print_trace(self, trace_id: str):
        """打印追溯信息"""
        trace = self.get_trace(trace_id)
        if not trace:
            print(f"[Audit] 未找到追溯: {trace_id}")
            return

        print(f"\n[bold]=== 追溯: {trace_id} ===[/bold]")
        print(f"时间: {trace.get('timestamp', 'N/A')}")
        print(f"处理时间: {trace.get('processing_time_ms', 0):.1f}ms")

        print("\n[bold]Agent 输出:[/bold]")
        for output in trace.get("agent_outputs", []):
            print(f"  • {output['agent_name']}: {output['direction']} ({output['confidence']:.0%})")

        consensus = trace.get("consensus")
        if consensus:
            print(f"\n[bold]共识:[/bold]")
            print(f"  方向: {consensus['direction']}")
            print(f"  分数: {consensus['overall_score']:.3f}")
            print(f"  强度: {consensus['consensus_strength']:.0%}")

        decision = trace.get("final_decision")
        if decision:
            print(f"\n[bold]决策:[/bold]")
            print(f"  操作: {decision['action']}")
            print(f"  置信度: {decision['confidence']:.0%}")


# 全局审计日志
audit_logger = AuditLogger()


# ============ Agent 协调器 ============

class AgentCoordinator:
    """
    Agent 协调器 - LangGraph 工作流入口

    负责:
    1. 协调各 Agent 并行分析
    2. 辩论共识
    3. 生成最终决策
    4. 审计追溯
    """

    def __init__(self):
        self.debate_engine = DebateEngine()
        self.audit_logger = audit_logger

    async def analyze(
        self,
        symbol: str,
        price: float,
        market_data: dict,
        risk_level: str = "medium",
        max_leverage: float = 2.0,
    ) -> FinalDecision:
        """
        综合分析入口

        Args:
            symbol: 交易对
            price: 当前价格
            market_data: 市场数据
            risk_level: 风险等级
            max_leverage: 最大杠杆

        Returns:
            FinalDecision: 最终决策
        """
        import time
        start_time = time.time()

        # 生成追溯 ID
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # 创建输入
        input_data = AgentInput(
            symbol=symbol,
            trace_id=trace_id,
            price=price,
            ohlcv=market_data,
            risk_level=risk_level,
            max_leverage=max_leverage,
        )

        # 开始追溯
        self.audit_logger.start_trace(trace_id, input_data)

        # 并行执行各 Agent
        agent_outputs = await self._run_agents_parallel(input_data)

        # 记录 Agent 输出
        for output in agent_outputs:
            self.audit_logger.log_agent_output(trace_id, output)

        # 辩论共识
        consensus = self.debate_engine.calculate_consensus(agent_outputs)
        self.audit_logger.log_consensus(trace_id, consensus)

        # 生成最终决策
        decision = self._synthesize_decision(
            trace_id=trace_id,
            symbol=symbol,
            price=price,
            consensus=consensus,
            agent_outputs=agent_outputs,
            risk_level=risk_level,
            max_leverage=max_leverage,
        )

        # 记录决策
        self.audit_logger.log_decision(trace_id, decision)

        # 计算处理时间
        processing_time_ms = (time.time() - start_time) * 1000
        decision.processing_time_ms = processing_time_ms

        return decision

    async def _run_agents_parallel(self, input_data: AgentInput) -> list[AgentOutput]:
        """并行执行所有 Agent"""
        import asyncio

        # 获取各 Agent 实例
        agents = self._get_agents()

        # 并发执行
        tasks = [
            agent.analyze(input_data)
            for agent in agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[Agent] {agents[i].name} 分析失败: {result}")
                continue
            outputs.append(result)

        return outputs

    def _get_agents(self) -> list:
        """获取所有 Agent"""
        from opentrade.agents.market import MarketAgent
        from opentrade.agents.strategy import StrategyAgent
        from opentrade.agents.risk import RiskAgent
        from opentrade.agents.onchain import OnchainAgent
        from opentrade.agents.sentiment import SentimentAgent
        from opentrade.agents.macro import MacroAgent

        return [
            MarketAgent(),
            StrategyAgent(),
            RiskAgent(),
            OnchainAgent(),
            SentimentAgent(),
            MacroAgent(),
        ]

    def _synthesize_decision(
        self,
        trace_id: str,
        symbol: str,
        price: float,
        consensus: ConsensusResult,
        agent_outputs: list[AgentOutput],
        risk_level: str,
        max_leverage: float,
    ) -> FinalDecision:
        """合成最终决策"""

        # 确定操作
        if consensus.direction.value == "neutral" or consensus.confidence < 0.4:
            action = "HOLD"
        elif consensus.direction.value == "bullish":
            action = "BUY"
        else:
            action = "SELL"

        # 计算仓位大小
        if action == "HOLD":
            size = 0.0
        else:
            # 置信度决定仓位
            base_size = {
                "low": 0.05,
                "medium": 0.10,
                "high": 0.20,
            }.get(risk_level, 0.10)

            size = base_size * consensus.confidence * (1 + abs(consensus.overall_score))

        # 计算杠杆
        leverage = min(max_leverage, 1 + int(abs(consensus.overall_score) * (max_leverage - 1)))

        # 计算止损止盈
        stop_loss_pct = 0.03 if risk_level == "low" else (0.05 if risk_level == "medium" else 0.08)
        take_profit_pct = stop_loss_pct * 2

        stop_loss = price * (1 - stop_loss_pct) if action == "BUY" else price * (1 + stop_loss_pct)
        take_profit = price * (1 + take_profit_pct) if action == "BUY" else price * (1 - take_profit_pct)

        # 生成摘要
        vote_summary = ", ".join(
            f"{v.agent_name[:8]}({v.direction.value[0]})"
            for v in consensus.votes
        )

        return FinalDecision(
            trace_id=trace_id,
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            direction=consensus.direction,
            size=min(size, 0.5),  # 限制最大仓位
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=consensus.confidence,
            risk_score=1 - consensus.confidence,
            agent_votes=consensus.votes,
            consensus=consensus,
            reasons=consensus.final_reasons,
            summary=f"共识{consensus.direction.value} {consensus.confidence:.0%} | {vote_summary}",
        )


# ============ 便捷函数 ============

async def quick_analyze(
    symbol: str,
    price: float,
    market_data: dict,
) -> FinalDecision:
    """快速分析"""
    coordinator = AgentCoordinator()
    return await coordinator.analyze(symbol, price, market_data)


def get_trace(trace_id: str) -> Optional[dict]:
    """获取追溯"""
    return audit_logger.get_trace(trace_id)


def print_trace(trace_id: str):
    """打印追溯"""
    audit_logger.print_trace(trace_id)
