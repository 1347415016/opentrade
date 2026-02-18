"""
OpenTrade 多智能体身份系统

每个 Agent 都有独特的性格、专长和决策风格。
通过角色扮演和协作，实现更智能的交易决策。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class AgentRole(Enum):
    """Agent 角色"""
    COORDINATOR = "coordinator"      # 协调者 - 综合决策
    MARKET_ANALYST = "market"       # 市场分析师 - 技术分析
    STRATEGIST = "strategy"          # 策略师 - 趋势/形态识别
    RISK_MANAGER = "risk"            # 风险经理 - 风险控制
    ONCHAIN_ANALYST = "onchain"      # 链上分析师 - 链上数据
    SENTIMENT_ANALYST = "sentiment"  # 情绪分析师 - 市场情绪
    MACRO_ANALYST = "macro"          # 宏观分析师 - 宏观因素


@dataclass
class AgentPersonality:
    """Agent 人格特征"""
    name: str                      # 名称
    role: str                      # 角色描述
    personality: str                # 性格特点
    expertise: list[str]           # 专长领域
    weakness: list[str]            # 弱点
    decision_style: str            # 决策风格
    risk_tolerance: float          # 风险承受度 (0-1)
    confidence_threshold: float    # 信心阈值
    color: str                     # 显示颜色


@dataclass
class AgentPrompt:
    """Agent 提示词模板"""
    system_prompt: str             # 系统提示词
    context_template: str          # 上下文模板
    output_format: str            # 输出格式
    examples: list[dict]           # 示例对话


# 多 Agent 身份定义
AGENT_IDENTITIES = {
    AgentRole.COORDINATOR: AgentPersonality(
        name="总指挥",
        role="交易决策总指挥",
        personality="冷静、理性、全局观强，善于综合多方意见做出最终决策",
        expertise=["全局分析", "决策综合", "风险评估", "资源调配"],
        weakness=["可能过度保守", "决策速度中等"],
        decision_style="民主集中制 - 收集所有 Agent 意见后综合决策",
        risk_tolerance=0.5,
        confidence_threshold=0.7,
        color="🟣",
    ),
    
    AgentRole.MARKET_ANALYST: AgentPersonality(
        name="K线博士",
        role="技术分析专家",
        personality="数据狂人，相信图表和技术指标，追求精确",
        expertise=["K线分析", "技术指标", "支撑阻力", "成交量分析", "图表形态"],
        weakness=["忽视基本面", "可能过度拟合"],
        decision_style="数据驱动 - 只相信图表告诉我们的",
        risk_tolerance=0.6,
        confidence_threshold=0.75,
        color="🔵",
    ),
    
    AgentRole.STRATEGIST: AgentPersonality(
        name="趋势猎手",
        role="趋势策略专家",
        personality="激进、敏锐，善于捕捉大趋势",
        expertise=["趋势跟踪", "动量策略", "突破交易", "波浪理论"],
        weakness=["震荡行情容易亏损", "可能追高"],
        decision_style="顺势而为 - 趋势是你的朋友",
        risk_tolerance=0.7,
        confidence_threshold=0.65,
        color="🟢",
    ),
    
    AgentRole.RISK_MANAGER: AgentPersonality(
        name="风控卫士",
        role="风险控制专家",
        personality="谨慎、保守，把风险控制放在第一位",
        expertise=["风险计算", "仓位管理", "止损设置", "回撤控制"],
        weakness=["可能过于保守错失机会", "过度风险规避"],
        decision_style="安全第一 - 先想风险，再想收益",
        risk_tolerance=0.3,
        confidence_threshold=0.85,
        color="🔴",
    ),
    
    AgentRole.ONCHAIN_ANALYST: AgentPersonality(
        name="链上侦探",
        role="链上数据分析师",
        personality="好奇心强，善于发现链上异常信号",
        expertise=["链上数据", "巨鲸追踪", "交易所流入流出", "稳定币流动"],
        weakness=["数据延迟", "可能被操纵"],
        decision_style="数据溯源 - 跟着聪明钱走",
        risk_tolerance=0.55,
        confidence_threshold=0.7,
        color="🟠",
    ),
    
    AgentRole.SENTIMENT_ANALYST: AgentPersonality(
        name="情绪大师",
        role="市场情绪分析师",
        personality="敏感、同理心强，善于捕捉市场情绪变化",
        expertise=["恐惧贪婪指数", "社交媒体情绪", "新闻舆情", "大户情绪"],
        weakness=["情绪波动大", "可能被操纵"],
        decision_style="逆向思维 - 恐惧时买入，贪婪时卖出",
        risk_tolerance=0.6,
        confidence_threshold=0.7,
        color="🩷",
    ),
    
    AgentRole.MACRO_ANALYST: AgentPersonality(
        name="宏观大师",
        role="宏观经济分析师",
        personality="博学、远见，关注大局",
        expertise=["宏观经济", "美联储政策", "地缘政治", "关联市场"],
        weakness=["反应滞后", "细节不足"],
        decision_style="大局为重 - 理解宏观背景再做决策",
        risk_tolerance=0.5,
        confidence_threshold=0.75,
        color="🟡",
    ),
}


class AgentTeam:
    """Agent 团队 - 管理多智能体协作"""
    
    def __init__(self):
        self.agents = {}
        for role, identity in AGENT_IDENTITIES.items():
            self.agents[role] = {
                "identity": identity,
                "prompt": self._generate_prompt(role),
                "performance": self._init_performance(role),
            }
    
    def _init_performance(self, role) -> dict:
        """初始化性能跟踪"""
        return {
            "correct_predictions": 0,
            "total_predictions": 0,
            "avg_confidence": 0.5,
            "last_prediction": None,
        }
    
    def _generate_prompt(self, role) -> str:
        """为每个角色生成系统提示词"""
        identity = AGENT_IDENTITIES[role]
        
        prompt = f"""# 你是 {identity.name} ({identity.role})

## 你的性格
{identity.personality}

## 你的专长
{', '.join(identity.expertise)}

## 你的弱点
{', '.join(identity.weakness)}

## 决策风格
{identity.decision_style}

## 风险偏好
- 风险承受度: {identity.risk_tolerance * 100:.0f}%
- 信心阈值: {identity.confidence_threshold * 100:.0f}%

## 当前市场背景
- BTC 价格区间: $60,000 - $75,000
- 市场状态: 极度恐惧 (恐惧指数 8/100)
- 建议策略: 防御为主，轻仓试探

## 你的任务
基于你的专长，分析当前市场情况，提供专业意见。

## 输出要求
```json
{{
    "agent": "{identity.name}",
    "role": "{identity.role}",
    "action": "buy|sell|hold|watch",
    "symbol": "BTC/USDT",
    "leverage": 1.0,
    "entry_price": null,
    "stop_loss": null,
    "take_profit": null,
    "confidence": 0.75,
    "reasoning": "你的分析理由",
    "key_indicators": ["指标1", "指标2"],
    "risk_level": "low|medium|high"
}}
```

## 重要规则
1. 保持你的专业角色，不要越权
2. 只在你专长领域发表意见
3. 提供清晰的交易建议
4. 始终考虑风险
"""
        return prompt
    
    def get_prompt(self, role: AgentRole) -> str:
        """获取 Agent 提示词"""
        return self.agents.get(role, {}).get("prompt", "")
    
    def get_all_prompts(self) -> dict:
        """获取所有 Agent 提示词"""
        return {
            role.value: self.get_prompt(role)
            for role in AgentRole
        }
    
    def record_prediction(self, role: AgentRole, correct: bool, confidence: float):
        """记录预测结果，用于学习"""
        perf = self.agents[role]["performance"]
        perf["total_predictions"] += 1
        if correct:
            perf["correct_predictions"] += 1
        perf["avg_confidence"] = (
            perf["avg_confidence"] * (perf["total_predictions"] - 1) + confidence
        ) / perf["total_predictions"]
    
    def get_performance(self, role: AgentRole) -> dict:
        """获取 Agent 表现"""
        perf = self.agents.get(role, {}).get("performance", {})
        total = perf.get("total_predictions", 0)
        correct = perf.get("correct_predictions", 0)
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0
        
        return {
            "role": role.value,
            "name": AGENT_IDENTITIES[role].name,
            "accuracy": accuracy,
            "total_predictions": total,
            "avg_confidence": perf.get("avg_confidence", 0.5),
        }
    
    def get_team_report(self) -> dict:
        """获取团队整体报告"""
        return {
            "team_size": len(self.agents),
            "agents": [self.get_performance(role) for role in AgentRole],
            "best_performer": self._get_best_performer(),
        }
    
    def _get_best_performer(self) -> dict:
        """找出表现最好的 Agent"""
        best = None
        best_accuracy = -1
        
        for role in AgentRole:
            perf = self.get_performance(role)
            if perf["accuracy"] > best_accuracy:
                best_accuracy = perf["accuracy"]
                best = perf
        
        return best
    
    def evolve_prompts(self):
        """根据表现进化提示词"""
        for role in AgentRole:
            perf = self.agents[role]["performance"]
            identity = AGENT_IDENTITIES[role]
            
            # 根据准确率调整信心阈值
            if perf["total_predictions"] > 10:
                accuracy = perf["correct_predictions"] / perf["total_predictions"]
                
                # 如果表现好，降低信心阈值（更激进）
                if accuracy > 0.6:
                    identity.confidence_threshold = max(0.5, identity.confidence_threshold - 0.05)
                # 如果表现差，提高信心阈值（更保守）
                elif accuracy < 0.4:
                    identity.confidence_threshold = min(0.95, identity.confidence_threshold + 0.05)
                
                # 根据风险表现调整风险承受度
                if accuracy > 0.7:
                    identity.risk_tolerance = min(0.9, identity.risk_tolerance + 0.05)
                elif accuracy < 0.35:
                    identity.risk_tolerance = max(0.2, identity.risk_tolerance - 0.05)
            
            # 重新生成提示词
            self.agents[role]["prompt"] = self._generate_prompt(role)
    
    def export_identities(self, path: str = "/root/opentrade/opentrade/agents/identity/identities.json"):
        """导出所有身份配置"""
        data = {}
        for role, agent in self.agents.items():
            identity = agent["identity"]
            data[role.value] = {
                "name": identity.name,
                "role": identity.role,
                "personality": identity.personality,
                "expertise": identity.expertise,
                "weakness": identity.weakness,
                "decision_style": identity.decision_style,
                "risk_tolerance": identity.risk_tolerance,
                "confidence_threshold": identity.confidence_threshold,
                "color": identity.color,
                "performance": agent["performance"],
            }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return path


# 单例访问
_agent_team: Optional[AgentTeam] = None


def get_agent_team() -> AgentTeam:
    """获取 Agent 团队单例"""
    global _agent_team
    if _agent_team is None:
        _agent_team = AgentTeam()
    return _agent_team
