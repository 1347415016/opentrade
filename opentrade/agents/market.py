"""
OpenTrade Market Agent - 技术分析

基于技术指标的买卖信号分析
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class MarketAgent(BaseAgent):
    """市场技术分析 Agent"""

    @property
    def name(self) -> str:
        return "MarketAgent"

    @property
    def description(self) -> str:
        return "技术分析专家，专注于价格行为、趋势和技术指标"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析市场状态，生成技术分析结果"""
        import asyncio

        # 如果传入的是 dict，转换为 MarketState
        if isinstance(market_state, dict):
            from dataclasses import asdict
            market_state = MarketState(**market_state)

        # 异步模拟延迟
        await asyncio.sleep(0.01)

        score = 0.0
        direction = MarketDirection.NEUTRAL
        confidence = 0.5
        key_findings = []
        risks = []
        analysis = {}

        # 1. 趋势分析 (EMA)
        ema_score = self._analyze_ema(market_state)
        analysis["ema"] = ema_score
        score += ema_score * 0.2

        if ema_score > 0:
            key_findings.append(f"价格站上 EMA，均线多头排列")
        elif ema_score < 0:
            risks.append(f"价格跌破 EMA，均线空头排列")

        # 2. 动量分析 (RSI)
        rsi_score = self._analyze_rsi(market_state)
        analysis["rsi"] = rsi_score
        score += rsi_score * 0.15

        if rsi_score > 0:
            key_findings.append(f"RSI 处于强势区间 ({market_state.rsi:.1f})")
        elif rsi_score < 0:
            risks.append(f"RSI 处于超卖区域 ({market_state.rsi:.1f})")

        # 3. MACD 分析
        macd_score = self._analyze_macd(market_state)
        analysis["macd"] = macd_score
        score += macd_score * 0.15

        if macd_score > 0:
            key_findings.append("MACD 金叉向上，动能增强")
        elif macd_score < 0:
            risks.append("MACD 死叉向下，动能减弱")

        # 4. 布林带分析
        bb_score = self._analyze_bollinger(market_state)
        analysis["bollinger"] = bb_score
        score += bb_score * 0.15

        # 5. 成交量分析
        volume_score = self._analyze_volume(market_state)
        analysis["volume"] = volume_score
        score += volume_score * 0.1

        if volume_score > 0:
            key_findings.append(f"成交量放大 ({market_state.volume_ratio:.2f}x)")
        elif volume_score < 0:
            risks.append("成交量萎缩，需谨慎")

        # 6. 资金费率分析
        funding_score = self._analyze_funding(market_state)
        analysis["funding"] = funding_score
        score += funding_score * 0.1

        if funding_score > 0:
            key_findings.append("资金费率健康，多头占优")
        elif funding_score < 0:
            risks.append("资金费率过高，可能有回调风险")

        # 7. 持仓量分析
        oi_score = self._analyze_open_interest(market_state)
        analysis["open_interest"] = oi_score
        score += oi_score * 0.1

        # 8. 综合方向
        if score > 0.1:
            direction = MarketDirection.BULLISH
        elif score < -0.1:
            direction = MarketDirection.BEARISH

        # 9. 置信度 (基于信号一致性)
        positive_signals = sum(1 for s in [ema_score, rsi_score, macd_score, bb_score, volume_score, funding_score, oi_score] if s > 0)
        negative_signals = sum(1 for s in [ema_score, rsi_score, macd_score, bb_score, volume_score, funding_score, oi_score] if s < 0)
        confidence = max(0.3, min(0.95, 0.5 + abs(score) * 0.3 + (positive_signals - negative_signals) * 0.05))

        return AgentOutput(
            agent_type=AgentType.MARKET,
            agent_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:5],
            risks=risks[:3],
        )

    def _analyze_ema(self, state: MarketState) -> float:
        """EMA 分析"""
        if state.price > state.ema_fast > state.ema_slow:
            return 0.5
        elif state.price < state.ema_fast < state.ema_slow:
            return -0.5
        elif state.price > state.ema_fast:
            return 0.2
        elif state.price < state.ema_fast:
            return -0.2
        return 0.0

    def _analyze_rsi(self, state: MarketState) -> float:
        """RSI 分析"""
        rsi = state.rsi
        if rsi > 70:
            return -0.3  # 超买
        elif rsi < 30:
            return 0.3  # 超卖
        elif rsi > 60:
            return 0.2
        elif rsi < 40:
            return -0.2
        return 0.0

    def _analyze_macd(self, state: MarketState) -> float:
        """MACD 分析"""
        macd = state.macd
        signal = state.macd_signal

        if macd > signal > 0:
            return 0.3
        elif macd > signal:
            return 0.1
        elif macd < signal < 0:
            return -0.3
        elif macd < signal:
            return -0.1
        return 0.0

    def _analyze_bollinger(self, state: MarketState) -> float:
        """布林带分析"""
        price = state.price
        upper = state.bollinger_upper
        middle = state.bollinger_middle
        lower = state.bollinger_lower

        if upper == 0:
            return 0.0

        # 价格位置
        if price > upper:
            return -0.2  # 价格突破上轨，可能回调
        elif price < lower:
            return 0.3  # 价格跌破下轨，可能反弹
        elif price > middle:
            return 0.1
        else:
            return -0.1

    def _analyze_volume(self, state: MarketState) -> float:
        """成交量分析"""
        ratio = state.volume_ratio

        if ratio > 2.0:
            return 0.3  # 放量
        elif ratio > 1.5:
            return 0.15
        elif ratio < 0.5:
            return -0.1  # 缩量
        elif ratio < 0.8:
            return -0.05
        return 0.0

    def _analyze_funding(self, state: MarketState) -> float:
        """资金费率分析"""
        rate = state.funding_rate

        if rate > 0.1:  # 超过 0.1% 过高
            return -0.2
        elif rate > 0.05:
            return -0.1
        elif rate < 0:
            return 0.1  # 负费率，看跌方付钱
        return 0.0

    def _analyze_open_interest(self, state: MarketState) -> float:
        """持仓量分析"""
        change = state.open_interest_change

        if change > 0.05:
            return 0.2  # 持仓增加
        elif change > 0.02:
            return 0.1
        elif change < -0.05:
            return -0.2  # 持仓大降
        elif change < -0.02:
            return -0.1
        return 0.0
