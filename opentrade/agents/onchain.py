"""
OpenTrade Onchain Agent - 链上数据分析

分析区块链数据、交易所流入流出、巨鲸行为等
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class OnchainAgent(BaseAgent):
    """链上数据分析 Agent"""

    @property
    def name(self) -> str:
        return "OnchainAgent"

    @property
    def description(self) -> str:
        return "链上数据分析师，专注于交易所余额、巨鲸行为、稳定币流动"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析链上数据"""
        import asyncio

        if isinstance(market_state, dict):
            from dataclasses import asdict
            market_state = MarketState(**market_state)

        await asyncio.sleep(0.01)

        score = 0.0
        direction = MarketDirection.NEUTRAL
        confidence = 0.5
        key_findings = []
        risks = []
        analysis = {}

        # 1. 交易所净流入分析
        flow_score = self._analyze_exchange_flow(market_state)
        analysis["exchange_flow"] = flow_score
        score += flow_score * 0.3

        if flow_score > 0:
            key_findings.append(f"净流入 {market_state.exchange_net_flow:+.0f}")
        else:
            risks.append(f"净流出 {market_state.exchange_net_flow:.0f}")

        # 2. 巨鲸行为分析
        whale_score = self._analyze_whale_activity(market_state)
        analysis["whale_activity"] = whale_score
        score += whale_score * 0.25

        if whale_score > 0:
            key_findings.append(f"巨鲸活跃度: {market_state.whale_transactions}")
        elif whale_score < 0:
            risks.append("巨鲸活动减少")

        # 3. 稳定币流动分析
        stable_score = self._analyze_stablecoin_flow(market_state)
        analysis["stablecoin_flow"] = stable_score
        score += stable_score * 0.25

        if stable_score > 0:
            key_findings.append("稳定币净流入，场外资金充足")
        elif stable_score < 0:
            risks.append("稳定币净流出")

        # 4. 链上指标综合
        onchain_score = self._analyze_onchain_metrics(market_state)
        analysis["onchain_metrics"] = onchain_score
        score += onchain_score * 0.2

        # 综合方向
        if score > 0.1:
            direction = MarketDirection.BULLISH
        elif score < -0.1:
            direction = MarketDirection.BEARISH

        # 置信度
        confidence = min(0.85, 0.5 + abs(score) * 0.35)

        return AgentOutput(
            agent_type=AgentType.ONCHAIN,
            agent_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:4],
            risks=risks[:3],
        )

    def _analyze_exchange_flow(self, state: MarketState) -> float:
        """交易所流入流出分析"""
        flow = state.exchange_net_flow

        if not flow:
            return 0.0

        # 净流入为正 = 币从交易所流出 = 投资者持有意愿强 = 看涨
        # 净流入为负 = 币流入交易所 = 投资者准备卖出 = 看跌

        # 假设总流通市值约 1万亿，流入流出按比例
        flow_normalized = flow / 1000000  # 归一化

        if flow_normalized > 10:  # 大幅净流出
            return 0.4
        elif flow_normalized > 5:
            return 0.25
        elif flow_normalized > 1:
            return 0.1
        elif flow_normalized < -10:  # 大幅净流入
            return -0.4
        elif flow_normalized < -5:
            return -0.25
        elif flow_normalized < -1:
            return -0.1

        return 0.0

    def _analyze_whale_activity(self, state: MarketState) -> float:
        """巨鲸行为分析"""
        whales = state.whale_transactions

        if whales > 20:
            return 0.2  # 巨鲸高度活跃
        elif whales > 10:
            return 0.1
        elif whales > 5:
            return 0.05
        elif whales < 2:
            return -0.1  # 巨鲸不活跃
        else:
            return 0.0

    def _analyze_stablecoin_flow(self, state: MarketState) -> float:
        """稳定币流动分析"""
        mint = state.stablecoin_mint

        if mint > 100000000:  # 1亿以上
            return 0.3  # 大量稳定币铸造，可能入场
        elif mint > 50000000:
            return 0.2
        elif mint > 10000000:
            return 0.1
        elif mint < -10000000:
            return -0.1  # 赎回增加
        else:
            return 0.0

    def _analyze_onchain_metrics(self, state: MarketState) -> float:
        """其他链上指标"""
        # 综合评分
        score = 0.0

        # 这里可以添加更多指标，如 NVT、MVRV、交易所持仓比例等
        # 暂时返回 0

        return score
