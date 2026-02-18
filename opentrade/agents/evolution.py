"""
OpenTrade 策略进化引擎

核心功能：
1. 从回测结果中学习，调整策略权重
2. 根据市场状态动态调整风险参数
3. 实现 AI 模型的持续进化
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
import os

from opentrade.core.config import get_config


@dataclass
class StrategyPerformance:
    """策略表现记录"""
    strategy_name: str
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    trade_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketState:
    """市场状态"""
    fear_greed_index: int = 50
    btc_price: float = 68000.0
    btc_volatility: float = 0.02
    trend: str = "neutral"  # bullish, bearish, neutral


class EvolutionEngine:
    """策略进化引擎
    
    负责：
    1. 记录策略表现
    2. 分析回测结果
    3. 调整策略权重
    4. 生成动态提示词
    """
    
    _instance: "EvolutionEngine" = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        self.performance_history: list[StrategyPerformance] = []
        self.strategy_weights = {
            "trend_following": 0.25,
            "mean_reversion": 0.25,
            "rsi_strategy": 0.25,
            "momentum": 0.25,
        }
        self.market_state = MarketState()
        self.learning_rate = 0.1  # 学习率
        self.evolution_log = []
        
        # 进化历史文件
        self.history_file = "/root/opentrade/data/evolution_history.json"
        self._load_history()
        
        self._initialized = True
    
    def _load_history(self):
        """加载历史数据"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.performance_history = [
                        StrategyPerformance(**p) for p in data.get("history", [])
                    ]
                    self.strategy_weights = data.get("weights", self.strategy_weights)
            except Exception:
                pass
    
    def _save_history(self):
        """保存历史数据"""
        data = {
            "history": [
                {
                    "strategy_name": p.strategy_name,
                    "total_return": p.total_return,
                    "win_rate": p.win_rate,
                    "max_drawdown": p.max_drawdown,
                    "sharpe_ratio": p.sharpe_ratio,
                    "trade_count": p.trade_count,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in self.performance_history[-100:]  # 只保留最近100条
            ],
            "weights": self.strategy_weights,
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def update_market_state(
        self,
        fear_greed_index: int,
        btc_price: float,
        volatility: float = 0.02,
        trend: str = "neutral",
    ):
        """更新市场状态"""
        self.market_state = MarketState(
            fear_greed_index=fear_greed_index,
            btc_price=btc_price,
            btc_volatility=volatility,
            trend=trend,
        )
    
    def record_performance(self, performance: StrategyPerformance):
        """记录策略表现"""
        self.performance_history.append(performance)
        self._log_evolution(f"记录 {performance.strategy_name}: 收益={performance.total_return:.2%}, 胜率={performance.win_rate:.2%}")
        self._save_history()
    
    def evolve(self) -> dict:
        """执行进化：根据历史表现调整策略权重"""
        if len(self.performance_history) < 3:
            return self.strategy_weights
        
        # 分析最近的表现
        recent = self.performance_history[-10:]
        
        # 计算每个策略的平均表现
        strategy_scores = {}
        for perf in recent:
            if perf.strategy_name not in strategy_scores:
                strategy_scores[perf.strategy_name] = []
            score = self._calculate_strategy_score(perf)
            strategy_scores[perf.strategy_name].append(score)
        
        # 计算加权分数
        avg_scores = {}
        for name, scores in strategy_scores.items():
            avg_scores[name] = sum(scores) / len(scores) if scores else 0.5
        
        # 归一化权重
        total_score = sum(avg_scores.values()) or 1
        new_weights = {
            name: score / total_score
            for name, score in avg_scores.items()
        }
        
        # 应用学习率限制变化幅度
        for name in self.strategy_weights:
            old_weight = self.strategy_weights[name]
            new_weight = new_weights.get(name, old_weight)
            # 限制每次变化不超过 10%
            max_change = 0.1
            if new_weight > old_weight + max_change:
                new_weight = old_weight + max_change
            elif new_weight < old_weight - max_change:
                new_weight = old_weight - max_change
            self.strategy_weights[name] = new_weight
        
        self._log_evolution(f"进化完成: {json.dumps(self.strategy_weights, indent=2)}")
        self._save_history()
        
        return self.strategy_weights
    
    def _calculate_strategy_score(self, perf: StrategyPerformance) -> float:
        """计算策略综合得分"""
        # 综合评分：收益、胜率、回撤、夏普
        return (
            max(0, perf.total_return) * 0.3 +
            perf.win_rate * 0.2 +
            (1 - min(1, perf.max_drawdown)) * 0.2 +
            max(0, min(1, perf.sharpe_ratio / 10)) * 0.3
        )
    
    def get_risk_parameters(self) -> dict:
        """根据市场状态获取风险参数"""
        fear = self.market_state.fear_greed_index
        
        if fear <= 10:
            return {
                "max_leverage": 1.0,
                "stop_loss": 0.02,
                "max_exposure": 0.10,
                "stablecoin_ratio": 0.80,
                "risk_mode": "extreme_fear",
            }
        elif fear <= 25:
            return {
                "max_leverage": 1.2,
                "stop_loss": 0.025,
                "max_exposure": 0.15,
                "stablecoin_ratio": 0.70,
                "risk_mode": "high_fear",
            }
        elif fear <= 40:
            return {
                "max_leverage": 1.5,
                "stop_loss": 0.03,
                "max_exposure": 0.20,
                "stablecoin_ratio": 0.60,
                "risk_mode": "fear",
            }
        elif fear <= 60:
            return {
                "max_leverage": 2.0,
                "stop_loss": 0.035,
                "max_exposure": 0.30,
                "stablecoin_ratio": 0.50,
                "risk_mode": "neutral",
            }
        elif fear <= 75:
            return {
                "max_leverage": 2.5,
                "stop_loss": 0.04,
                "max_exposure": 0.40,
                "stablecoin_ratio": 0.40,
                "risk_mode": "optimistic",
            }
        else:
            return {
                "max_leverage": 2.0,
                "stop_loss": 0.035,
                "max_exposure": 0.30,
                "stablecoin_ratio": 0.50,
                "risk_mode": "greedy",
            }
    
    def generate_system_prompt(self) -> str:
        """生成动态系统提示词"""
        risk_params = self.get_risk_parameters()
        weights = self.strategy_weights
        
        # 计算权重百分比
        total = sum(weights.values()) or 1
        weight_str = "\n".join([
            f"  - {name}: {w/total*100:.1f}%"
            for name, w in weights.items()
        ])
        
        # 市场状态描述
        fear = self.market_state.fear_greed_index
        if fear <= 25:
            sentiment = "极度恐惧"
        elif fear <= 40:
            sentiment = "恐惧"
        elif fear <= 60:
            sentiment = "中性"
        elif fear <= 75:
            sentiment = "乐观"
        else:
            sentiment = "贪婪"
        
        prompt = f"""# OpenTrade 加密货币交易 AI

## 角色
你是一位专业的加密货币交易分析师，擅长技术分析、情绪分析和风险管理。

## 当前市场环境
- **恐惧贪婪指数**: {fear}/100 ({sentiment})
- **BTC 价格**: ${self.market_state.btc_price:,.0f}
- **波动率**: {self.market_state.btc_volatility*100:.1f}%
- **市场趋势**: {self.market_state.trend}

## 风险控制参数 (自动进化)
- **最大杠杆**: {risk_params['max_leverage']}x
- **单仓止损**: {risk_params['stop_loss']*100:.1f}%
- **总敞口限制**: {risk_params['max_exposure']*100:.0f}%
- **稳定币配置**: {risk_params['stablecoin_ratio']*100:.0f}%
- **当前模式**: {risk_params['risk_mode']}

## 策略权重 (自动进化)
{weight_str}

## 进化历史
- 已分析 {len(self.performance_history)} 条策略表现记录
- 上次进化: {self._get_last_evolution_time()}

## 决策规则
1. 分析市场多时间框架
2. 结合技术指标和情绪数据
3. 根据策略权重综合决策
4. 应用风险控制规则
5. 输出交易信号

## 输出格式
```json
{{"action": "buy|sell|hold", "symbol": "BTC/USDT", "leverage": 1.0, "stop_loss": 68000, "confidence": 0.85, "reasoning": "..."}}
```

## 重要规则
- 只在置信度 >70% 时开仓
- 连续亏损后自动降低仓位
- 极度恐惧时可考虑左侧抄底
- 极度贪婪时减仓避险
"""
        return prompt
    
    def _log_evolution(self, message: str):
        """记录进化日志"""
        self.evolution_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
        })
        print(f"[进化] {message}")
    
    def _get_last_evolution_time(self) -> str:
        """获取上次进化时间"""
        if self.evolution_log:
            return self.evolution_log[-1]["timestamp"]
        return "从未进化"
    
    def get_evolution_report(self) -> dict:
        """获取进化报告"""
        return {
            "strategy_weights": self.strategy_weights,
            "market_state": {
                "fear_greed_index": self.market_state.fear_greed_index,
                "btc_price": self.market_state.btc_price,
                "trend": self.market_state.trend,
            },
            "risk_parameters": self.get_risk_parameters(),
            "performance_count": len(self.performance_history),
            "evolution_count": len(self.evolution_log),
            "last_evolution": self._get_last_evolution_time(),
        }


# 单例访问函数
def get_evolution_engine() -> EvolutionEngine:
    """获取进化引擎单例"""
    return EvolutionEngine()
