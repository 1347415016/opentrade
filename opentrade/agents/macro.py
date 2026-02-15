"""
OpenTrade Macro Agent - 宏观数据分析

分析宏观经济因素对加密市场的影响
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class MacroAgent(BaseAgent):
    """宏观数据分析 Agent"""

    @property
    def name(self) -> str:
        return "MacroAgent"

    @property
    def description(self) -> str:
        return "宏观分析师，专注于美元、美股、黄金、国债等宏观因素"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析宏观环境"""
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

        # 1. 美元指数分析
        dxy_score = self._analyze_dxy(market_state)
        analysis["dxy"] = dxy_score
        score += dxy_score * 0.25

        if dxy_score > 0:
            key_findings.append(f"美元指数偏弱 ({market_state.dxy_index:.2f})")
        elif dxy_score < 0:
            risks.append(f"美元指数偏强 ({market_state.dxy_index:.2f})")

        # 2. 美股分析
        sp500_score = self._analyze_sp500(market_state)
        analysis["sp500"] = sp500_score
        score += sp500_score * 0.25

        if sp500_score > 0:
            key_findings.append(f"美股偏强 ({market_state.sp500_change:+.2f}%)")
        elif sp500_score < 0:
            risks.append(f"美股偏弱 ({market_state.sp500_change:+.2f}%)")

        # 3. 黄金分析
        gold_score = self._analyze_gold(market_state)
        analysis["gold"] = gold_score
        score += gold_score * 0.2

        # 4. 美债收益率分析
        bond_score = self._analyze_bonds(market_state)
        analysis["bonds"] = bond_score
        score += bond_score * 0.15

        if bond_score < -0.2:
            risks.append(f"美债收益率高企 ({market_state.bond_yield_10y:.2f}%)")

        # 5. VIX 恐慌指数分析
        vix_score = self._analyze_vix(market_state)
        analysis["vix"] = vix_score
        score += vix_score * 0.15

        if vix_score < -0.2:
            risks.append(f"市场恐慌指数升高 (VIX: {market_state.vix_index:.2f})")

        # 综合方向
        if score > 0.1:
            direction = MarketDirection.BULLISH
        elif score < -0.1:
            direction = MarketDirection.BEARISH

        # 置信度
        confidence = min(0.75, 0.5 + abs(score) * 0.25)

        return AgentOutput(
            agent_type=AgentType.MACRO,
            agent_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:4],
            risks=risks[:3],
        )

    def _analyze_dxy(self, state: MarketState) -> float:
        """美元指数分析"""
        dxy = state.dxy_index

        if dxy > 105:
            return -0.3  # 美元强势，利空风险资产
        elif dxy > 103:
            return -0.15
        elif dxy < 100:
            return 0.2  # 美元弱势，利多风险资产
        elif dxy < 102:
            return 0.1
        else:
            return 0.0

    def _analyze_sp500(self, state: MarketState) -> float:
        """美股分析"""
        change = state.sp500_change

        if change > 2:
            return 0.3  # 大涨
        elif change > 1:
            return 0.2
        elif change > 0:
            return 0.1
        elif change < -2:
            return -0.3  # 大跌
        elif change < -1:
            return -0.2
        elif change < 0:
            return -0.1
        else:
            return 0.0

    def _analyze_gold(self, state: MarketState) -> float:
        """黄金分析"""
        # 黄金与比特币有一定的正相关性（都是抗通胀资产）
        change = (state.gold_price - 2000) / 2000 if state.gold_price > 0 else 0

        if change > 0.05:
            return 0.15
        elif change < -0.05:
            return -0.1
        else:
            return 0.0

    def _analyze_bonds(self, state: MarketState) -> float:
        """美债收益率分析"""
        yield_10y = state.bond_yield_10y

        # 高收益率 = 资金回流债市 = 利空风险资产
        if yield_10y > 5.0:
            return -0.3
        elif yield_10y > 4.5:
            return -0.2
        elif yield_10y > 4.0:
            return -0.1
        elif yield_10y < 3.5:
            return 0.1
        else:
            return 0.0

    def _analyze_vix(self, state: MarketState) -> float:
        """VIX 分析"""
        vix = state.vix_index

        # VIX 低 = 市场平静，VIX 高 = 市场恐慌
        if vix > 30:
            return -0.3  # 市场恐慌
        elif vix > 25:
            return -0.15
        elif vix < 15:
            return 0.1  # 市场平静
        else:
            return 0.0
