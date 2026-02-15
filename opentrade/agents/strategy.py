"""
OpenTrade Strategy Agent - 策略分析

基于价格模式和历史规律的策略信号
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class StrategyAgent(BaseAgent):
    """策略分析 Agent"""

    @property
    def name(self) -> str:
        return "StrategyAgent"

    @property
    def description(self) -> str:
        return "量化策略分析师，专注于价格模式和策略信号"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析市场状态，生成策略信号"""
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

        # 1. 趋势持续性分析
        trend_score = self._analyze_trend_continuation(market_state)
        analysis["trend_continuation"] = trend_score
        score += trend_score * 0.25

        if trend_score > 0:
            key_findings.append("趋势持续性良好")

        # 2. 支撑阻力分析
        sr_score = self._analyze_support_resistance(market_state)
        analysis["support_resistance"] = sr_score
        score += sr_score * 0.2

        # 3. 形态识别
        pattern_score = self._analyze_patterns(market_state)
        analysis["patterns"] = pattern_score
        score += pattern_score * 0.2

        if pattern_score > 0.2:
            key_findings.append("检测到看涨形态")
        elif pattern_score < -0.2:
            risks.append("检测到看跌形态")

        # 4. 波动率分析
        volatility_score = self._analyze_volatility(market_state)
        analysis["volatility"] = volatility_score
        score += volatility_score * 0.15

        if volatility_score < -0.2:
            risks.append("波动率异常放大")

        # 5. 动量确认
        momentum_score = self._analyze_momentum(market_state)
        analysis["momentum"] = momentum_score
        score += momentum_score * 0.2

        if momentum_score > 0:
            key_findings.append("动量指标确认趋势")

        # 综合方向
        if score > 0.1:
            direction = MarketDirection.BULLISH
        elif score < -0.1:
            direction = MarketDirection.BEARISH

        # 置信度
        confidence = min(0.9, 0.5 + abs(score) * 0.4)

        return AgentOutput(
            agent_type=AgentType.STRATEGY,
            agent_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:4],
            risks=risks[:3],
        )

    def _analyze_trend_continuation(self, state: MarketState) -> float:
        """趋势持续性"""
        # 简单判断：价格与均线的相对位置
        price = state.price
        ema_fast = state.ema_fast
        ema_slow = state.ema_slow

        if ema_fast == 0 or ema_slow == 0:
            return 0.0

        # 检查趋势
        if price > ema_fast > ema_slow:
            # 上涨趋势
            # 检查是否在回调
            if price > ema_fast * 0.98:
                return 0.3  # 趋势持续
            else:
                return 0.0  # 回调中
        elif price < ema_fast < ema_slow:
            # 下跌趋势
            if price < ema_fast * 1.02:
                return -0.3
            else:
                return 0.0

        return 0.0

    def _analyze_support_resistance(self, state: MarketState) -> float:
        """支撑阻力分析"""
        price = state.price
        bb_upper = state.bollinger_upper
        bb_lower = state.bollinger_lower

        if bb_upper == 0 or bb_lower == 0:
            return 0.0

        # 价格接近下轨有支撑
        if price < bb_lower * 1.02:
            return 0.2
        # 价格接近上轨有压力
        elif price > bb_upper * 0.98:
            return -0.2

        return 0.0

    def _analyze_patterns(self, state: MarketState) -> float:
        """价格形态识别 (简化版)"""
        # 实际项目中应该用更复杂的算法
        # 这里用简单的价格变化模式

        ohlcv = state.ohlcv_1h if state.ohlcv_1h else {}
        if not ohlcv:
            return 0.0

        # 检查最近 5 根 K 线
        closes = []
        for candle in list(ohlcv.get("closes", []))[-5:]:
            if isinstance(candle, (int, float)):
                closes.append(candle)

        if len(closes) < 3:
            return 0.0

        # 检查是否在学习上涨
        if len(closes) >= 3:
            if closes[-1] > closes[-2] > closes[-3]:
                return 0.2  # 连续上涨
            elif closes[-1] < closes[-2] < closes[-3]:
                return -0.2  # 连续下跌

        return 0.0

    def _analyze_volatility(self, state: MarketState) -> float:
        """波动率分析"""
        atr = state.atr
        price = state.price

        if price == 0:
            return 0.0

        atr_pct = atr / price

        # 正常波动率范围 1-5%
        if atr_pct > 0.08:
            return -0.3  # 波动过大
        elif atr_pct > 0.05:
            return -0.1  # 波动偏高
        elif atr_pct < 0.01:
            return 0.1  # 波动极低，可能突破
        else:
            return 0.1  # 正常波动

    def _analyze_momentum(self, state: MarketState) -> float:
        """动量确认"""
        rsi = state.rsi
        macd = state.macd - state.macd_signal  # Histogram

        # RSI 确认
        if 45 < rsi < 55:
            rsi_confirm = 0
        elif rsi >= 55:
            rsi_confirm = 0.1
        else:
            rsi_confirm = -0.1

        # MACD 确认
        if macd > 0:
            macd_confirm = 0.1
        elif macd < 0:
            macd_confirm = -0.1
        else:
            macd_confirm = 0

        return rsi_confirm + macd_confirm
