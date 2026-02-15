"""
OpenTrade Risk Agent - 风险评估

评估交易风险和账户风险
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class RiskAgent(BaseAgent):
    """风险分析 Agent"""

    @property
    def name(self) -> str:
        return "RiskAgent"

    @property
    def description(self) -> str:
        return "风险控制专家，专注于风险评估和仓位管理"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析市场风险，输出风险评分"""
        import asyncio

        if isinstance(market_state, dict):
            from dataclasses import asdict
            market_state = MarketState(**market_state)

        await asyncio.sleep(0.01)

        score = 0.0  # 正数表示风险低，负数表示风险高
        direction = MarketDirection.NEUTRAL
        confidence = 0.5
        key_findings = []
        risks = []
        analysis = {}

        # 1. 价格波动风险
        volatility_risk = self._analyze_volatility_risk(market_state)
        analysis["volatility_risk"] = volatility_risk
        score += volatility_risk * 0.2

        if volatility_risk < -0.3:
            risks.append(f"波动率风险高 (ATR: {market_state.atr:.2f})")

        # 2. 资金费率风险
        funding_risk = self._analyze_funding_risk(market_state)
        analysis["funding_risk"] = funding_risk
        score += funding_risk * 0.15

        if funding_risk < -0.2:
            risks.append("资金费率过高，做空成本大")

        # 3. 持仓集中度风险
        oi_risk = self._analyze_oi_risk(market_state)
        analysis["oi_risk"] = oi_risk
        score += oi_risk * 0.15

        if oi_risk < -0.2:
            risks.append("持仓量变化异常，可能多空博弈激烈")

        # 4. 市场结构风险
        structure_risk = self._analyze_structure_risk(market_state)
        analysis["structure_risk"] = structure_risk
        score += structure_risk * 0.2

        if structure_risk < -0.2:
            risks.append("市场结构偏弱")

        # 5. 宏观风险
        macro_risk = self._analyze_macro_risk(market_state)
        analysis["macro_risk"] = macro_risk
        score += macro_risk * 0.15

        if macro_risk < -0.2:
            risks.append("宏观环境不确定性高")

        # 6. 链上风险
        onchain_risk = self._analyze_onchain_risk(market_state)
        analysis["onchain_risk"] = onchain_risk
        score += onchain_risk * 0.15

        if onchain_risk < -0.2:
            risks.append("链上数据偏空")

        # 综合方向 (Risk Agent 返回的是风险，所以取反)
        if score > 0.1:
            direction = MarketDirection.BULLISH  # 风险低，可以做多
        elif score < -0.1:
            direction = MarketDirection.BEARISH  # 风险高，应该谨慎
        else:
            direction = MarketDirection.NEUTRAL

        # 置信度
        confidence = min(0.9, 0.5 + abs(score) * 0.4)

        return AgentOutput(
            agent_type=AgentType.RISK,
            agent_name=self.name,
            direction=direction,
            score=score,  # 正数=低风险，负数=高风险
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:4],
            risks=risks[:4],
        )

    def _analyze_volatility_risk(self, state: MarketState) -> float:
        """波动率风险"""
        atr_pct = state.atr / state.price if state.price > 0 else 0

        # 波动率风险: 过高或过低都有风险
        if atr_pct > 0.08:  # 日波动 > 8%
            return -0.4
        elif atr_pct > 0.05:
            return -0.2
        elif atr_pct < 0.01:
            return -0.1  # 波动太低可能是暴风雨前的宁静
        else:
            return 0.2  # 正常波动率

    def _analyze_funding_risk(self, state: MarketState) -> float:
        """资金费率风险"""
        rate = state.funding_rate

        if rate > 0.1:  # 资金费率 > 0.1%
            return -0.3
        elif rate > 0.05:
            return -0.15
        elif rate < 0:  # 负费率
            return 0.1
        else:
            return 0.1

    def _analyze_oi_risk(self, state: MarketState) -> float:
        """持仓量变化风险"""
        change = state.open_interest_change

        if change > 0.1:  # 持仓大增
            return 0.0  # 中性
        elif change > 0.05:
            return 0.1
        elif change < -0.1:  # 持仓大降
            return -0.3
        elif change < -0.05:
            return -0.15
        else:
            return 0.1

    def _analyze_structure_risk(self, state: MarketState) -> float:
        """市场结构风险"""
        price = state.price
        ema_fast = state.ema_fast
        ema_slow = state.ema_slow

        if ema_fast == 0 or ema_slow == 0:
            return 0.0

        # 检查是否在关键位置
        bb_middle = state.bollinger_middle
        if bb_middle == 0:
            return 0.0

        # 价格相对于中轨的位置
        price_position = (price - bb_middle) / (state.bollinger_upper - bb_middle + 0.001)

        if price_position > 0.8:  # 接近上轨
            return -0.2
        elif price_position < 0.2:  # 接近下轨
            return 0.2

        return 0.0

    def _analyze_macro_risk(self, state: MarketState) -> float:
        """宏观风险"""
        # 简化的宏观风险评估
        risk_score = 0.0

        # VIX 指数
        if state.vix_index > 25:
            risk_score -= 0.2  # 市场恐慌
        elif state.vix_index > 20:
            risk_score -= 0.1

        # 美债收益率
        if state.bond_yield_10y > 5.0:
            risk_score -= 0.2  # 高收益率利空风险资产
        elif state.bond_yield_10y > 4.0:
            risk_score -= 0.1

        # S&P 500 下跌
        if state.sp500_change < -2:
            risk_score -= 0.2
        elif state.sp500_change < -1:
            risk_score -= 0.1

        return risk_score

    def _analyze_onchain_risk(self, state: MarketState) -> float:
        """链上风险"""
        risk_score = 0.0

        # 交易所净流入
        if state.exchange_net_flow > 1000:
            risk_score -= 0.2  # 大量币流向交易所，可能要卖
        elif state.exchange_net_flow < -1000:
            risk_score += 0.1  # 币流出交易所，可能要买

        # 巨鲸交易
        if state.whale_transactions > 10:
            risk_score -= 0.1

        return risk_score
