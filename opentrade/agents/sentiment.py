"""
OpenTrade Sentiment Agent - 市场情绪分析

分析恐惧贪婪指数、社交媒体情绪、新闻情绪
"""

from typing import Any

from opentrade.agents.base import BaseAgent
from opentrade.agents.coordinator import (
    AgentOutput,
    AgentType,
    MarketDirection,
)
from opentrade.agents.base import MarketState


class SentimentAgent(BaseAgent):
    """市场情绪分析 Agent"""

    @property
    def name(self) -> str:
        return "SentimentAgent"

    @property
    def description(self) -> str:
        return "情绪分析师，专注于恐惧贪婪指数和社交媒体情绪"

    async def analyze(
        self,
        market_state: MarketState | dict,
    ) -> AgentOutput:
        """分析市场情绪"""
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

        # 1. 恐惧贪婪指数分析
        fg_score = self._analyze_fear_greed(market_state)
        analysis["fear_greed"] = fg_score
        score += fg_score * 0.4

        if fg_score > 0:
            key_findings.append(f"情绪偏乐观 ({market_state.fear_greed_index}/100)")
        elif fg_score < 0:
            risks.append(f"情绪偏恐慌 ({market_state.fear_greed_index}/100)")

        # 2. 社交情绪分析
        social_score = self._analyze_social_sentiment(market_state)
        analysis["social_sentiment"] = social_score
        score += social_score * 0.3

        if social_score > 0:
            key_findings.append("社交媒体情绪积极")
        elif social_score < 0:
            risks.append("社交媒体情绪消极")

        # 3. 社交媒体活跃度
        activity_score = self._analyze_social_activity(market_state)
        analysis["social_activity"] = activity_score
        score += activity_score * 0.2

        if activity_score > 0:
            key_findings.append(f"社交媒体讨论量: {market_state.twitter_volume}")

        # 4. 极端情绪检测
        extreme_score = self._analyze_extreme_sentiment(market_state)
        analysis["extreme_sentiment"] = extreme_score
        score += extreme_score * 0.1

        # 综合方向
        if score > 0.1:
            direction = MarketDirection.BULLISH
        elif score < -0.1:
            direction = MarketDirection.BEARISH

        # 置信度
        confidence = min(0.8, 0.5 + abs(score) * 0.3)

        return AgentOutput(
            agent_type=AgentType.SENTIMENT,
            agent_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            analysis=analysis,
            key_findings=key_findings[:4],
            risks=risks[:3],
        )

    def _analyze_fear_greed(self, state: MarketState) -> float:
        """恐惧贪婪指数分析"""
        index = state.fear_greed_index

        # 0-100, 0=极端恐惧, 100=极端贪婪
        if index <= 20:
            return 0.4  # 极端恐惧，往往是买入机会
        elif index <= 40:
            return 0.2  # 恐惧
        elif index <= 60:
            return 0.0  # 中性
        elif index <= 80:
            return -0.2  # 贪婪
        else:
            return -0.3  # 极端贪婪，风险累积

    def _analyze_social_sentiment(self, state: MarketState) -> float:
        """社交媒体情绪分析"""
        sentiment = state.social_sentiment

        # -1 到 1
        if sentiment > 0.5:
            return 0.2
        elif sentiment > 0.2:
            return 0.1
        elif sentiment < -0.5:
            return -0.2
        elif sentiment < -0.2:
            return -0.1
        else:
            return 0.0

    def _analyze_social_activity(self, state: MarketState) -> float:
        """社交媒体活跃度分析"""
        volume = state.twitter_volume

        # 活跃度不是直接的信号，但高活跃度通常伴随高波动
        if volume > 10000:
            return 0.0  # 高活跃，但方向不确定
        elif volume > 5000:
            return 0.05
        elif volume < 100:
            return -0.1  # 活跃度过低
        else:
            return 0.0

    def _analyze_extreme_sentiment(self, state: MarketState) -> float:
        """极端情绪检测"""
        index = state.fear_greed_index

        # 极端情绪往往预示反转
        if index <= 10:  # 极度恐惧
            return 0.15  # 可能反转
        elif index >= 90:  # 极度贪婪
            return -0.15  # 风险累积
        else:
            return 0.0
