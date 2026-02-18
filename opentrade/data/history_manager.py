"""
OpenTrade 精选历史数据系统

设计原则：
1. 精选关键事件和数据，不喂全量
2. 分类存储，便于检索
3. 自动更新和清理
4. 支持多维度分析
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import hashlib


# 数据目录
DATA_DIR = "/root/opentrade/opentrade/data"
HISTORY_DIR = f"{DATA_DIR}/history"


@dataclass
class MarketEvent:
    """市场重大事件"""
    date: str                    # 日期
    event_type: str              # 事件类型
    title: str                   # 标题
    description: str             # 描述
    impact: str                  # 影响程度 (high/medium/low)
    price_before: float          # 事件前价格
    price_after: float           # 事件后价格
    change_pct: float            # 变化百分比
    source: str                  # 数据来源
    tags: list[str]              # 标签
    lessons: list[str]           # 经验教训


@dataclass  
class PricePattern:
    """价格形态"""
    date: str                    # 识别日期
    pattern_type: str           # 形态类型
    symbol: str                 # 交易对
    entry: float                 # 入场价
    exit: float                  # 出场价
    duration_hours: int          # 持续时间
    profit_pct: float            # 盈利百分比
    success: bool                # 是否成功
    timeframe: str              # 时间框架
    indicators: dict            # 相关指标


@dataclass
class TradingSignal:
    """交易信号记录"""
    date: str                    # 信号日期
    symbol: str                  # 交易对
    signal_type: str            # 信号类型 (buy/sell)
    strategy: str                # 策略名称
    entry_price: float           # 入场价
    exit_price: Optional[float]  # 出场价
    pnl_pct: float              # 盈亏百分比
    confidence: float           # 置信度
    reason: str                 # 信号原因
    executed: bool               # 是否执行


# 精选历史数据类型
HISTORY_TYPES = {
    "market_events": f"{HISTORY_DIR}/market_events.json",
    "price_patterns": f"{HISTORY_DIR}/price_patterns.json",
    "trading_signals": f"{HISTORY_DIR}/trading_signals.json",
    "performance_records": f"{HISTORY_DIR}/performance_records.json",
}


class HistoryDataManager:
    """历史数据管理器"""
    
    def __init__(self):
        self._ensure_directories()
        self._load_all_data()
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(HISTORY_DIR, exist_ok=True)
        os.makedirs(f"{DATA_DIR}/evolution", exist_ok=True)
    
    def _load_all_data(self):
        """加载所有数据"""
        self.market_events = self._load_json("market_events", [])
        self.price_patterns = self._load_json("price_patterns", [])
        self.trading_signals = self._load_json("trading_signals", [])
        self.performance_records = self._load_json("performance_records", [])
    
    def _load_json(self, name: str, default):
        """加载 JSON 文件"""
        path = HISTORY_TYPES.get(name, f"{HISTORY_DIR}/{name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass
        return default
    
    def _save_json(self, name: str, data):
        """保存 JSON 文件"""
        path = HISTORY_TYPES.get(name, f"{HISTORY_DIR}/{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ========== 市场事件管理 ==========
    
    def add_market_event(self, event: MarketEvent):
        """添加市场事件"""
        self.market_events.append({
            "date": event.date,
            "event_type": event.event_type,
            "title": event.title,
            "description": event.description,
            "impact": event.impact,
            "price_before": event.price_before,
            "price_after": event.price_after,
            "change_pct": event.change_pct,
            "source": event.source,
            "tags": event.tags,
            "lessons": event.lessons,
        })
        # 按日期排序
        self.market_events.sort(key=lambda x: x["date"], reverse=True)
        self._save_json("market_events", self.market_events)
    
    def get_recent_events(self, days: int = 30) -> list:
        """获取最近事件"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.market_events if e["date"] >= cutoff]
    
    def get_events_by_type(self, event_type: str) -> list:
        """按类型获取事件"""
        return [e for e in self.market_events if e["event_type"] == event_type]
    
    def get_high_impact_events(self) -> list:
        """获取高影响力事件"""
        return [e for e in self.market_events if e["impact"] == "high"]
    
    # ========== 价格形态管理 ==========
    
    def add_price_pattern(self, pattern: PricePattern):
        """添加价格形态"""
        self.price_patterns.append({
            "date": pattern.date,
            "pattern_type": pattern.pattern_type,
            "symbol": pattern.symbol,
            "entry": pattern.entry,
            "exit": pattern.exit,
            "duration_hours": pattern.duration_hours,
            "profit_pct": pattern.profit_pct,
            "success": pattern.success,
            "timeframe": pattern.timeframe,
            "indicators": pattern.indicators,
        })
        self._save_json("price_patterns", self.price_patterns)
    
    def get_successful_patterns(self) -> list:
        """获取成功形态"""
        return [p for p in self.price_patterns if p["success"]]
    
    def get_patterns_by_type(self, pattern_type: str) -> list:
        """按类型获取形态"""
        return [p for p in self.price_patterns if p["pattern_type"] == pattern_type]
    
    # ========== 交易信号管理 ==========
    
    def add_trading_signal(self, signal: TradingSignal):
        """添加交易信号"""
        self.trading_signals.append({
            "date": signal.date,
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "strategy": signal.strategy,
            "entry_price": signal.entry_price,
            "exit_price": signal.exit_price,
            "pnl_pct": signal.pnl_pct,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "executed": signal.executed,
        })
        self.trading_signals.sort(key=lambda x: x["date"], reverse=True)
        self._save_json("trading_signals", self.trading_signals)
    
    def get_executed_signals(self) -> list:
        """获取已执行信号"""
        return [s for s in self.trading_signals if s["executed"]]
    
    # ========== 数据精选摘要 ==========
    
    def get精选_summary(self, max_items: int = 50) -> dict:
        """生成精选摘要 - 用于喂给 AI"""
        
        # 精选高影响力事件
        high_events = self.get_high_impact_events()[:10]
        
        # 精选成功形态
        successful_patterns = self.get_successful_patterns()[:15]
        
        # 精选已执行信号
        executed = self.get_executed_signals()[:20]
        
        # 计算成功率
        if executed:
            win_rate = sum(1 for s in executed if s.get("pnl_pct", 0) > 0) / len(executed)
            avg_pnl = sum(s.get("pnl_pct", 0) for s in executed) / len(executed)
        else:
            win_rate = 0
            avg_pnl = 0
        
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "data_summary": {
                "total_events": len(self.market_events),
                "total_patterns": len(self.price_patterns),
                "total_signals": len(self.trading_signals),
            },
            "high_impact_events": high_events,
            "successful_patterns": successful_patterns,
            "executed_signals": executed,
            "performance": {
                "total_trades": len(executed),
                "win_rate": win_rate,
                "average_pnl_pct": avg_pnl,
            },
            "key_lessons": self._extract_key_lessons(),
        }
        
        return summary
    
    def _extract_key_lessons(self) -> list:
        """提取关键经验教训"""
        lessons = set()
        
        for event in self.get_high_impact_events():
            for lesson in event.get("lessons", []):
                lessons.add(lesson)
        
        for pattern in self.get_successful_patterns():
            if pattern.get("success"):
                pattern_type = pattern.get("pattern_type", "")
                if pattern_type:
                    lessons.add(f"{pattern_type} 在趋势行情中成功率较高")
        
        return list(lessons)[:10]
    
    def generate_ai_training_data(self) -> str:
        """生成 AI 训练数据 - 精简版"""
        summary = self.get精选_summary()
        
        # 生成结构化训练数据
        training_data = f"""# OpenTrade AI 训练数据
生成时间: {summary['generated_at']}

## 数据概览
- 市场事件数: {summary['data_summary']['total_events']}
- 价格形态数: {summary['data_summary']['total_patterns']}
- 交易信号数: {summary['data_summary']['total_signals']}

## 核心经验

### 1. 市场事件影响
"""
        
        for event in summary["high_impact_events"]:
            training_data += f"""
**{event['title']}** ({event['date']})
- 类型: {event['event_type']}
- 影响: {event['impact']}
- 价格变化: {event['change_pct']:.2f}%
- 经验: {', '.join(event.get('lessons', [])[:3])}
"""
        
        training_data += """
### 2. 成功的价格形态
"""
        for pattern in summary["successful_patterns"][:5]:
            training_data += f"""
- {pattern['pattern_type']} on {pattern['symbol']}
  盈利: {pattern['profit_pct']:.2f}%
  时间: {pattern['duration_hours']}小时
"""
        
        training_data += f"""
### 3. 交易表现
- 总交易数: {summary['performance']['total_trades']}
- 胜率: {summary['performance']['win_rate']:.2%}
- 平均盈亏: {summary['performance']['average_pnl_pct']:.2f}%

### 4. 关键教训
"""
        for lesson in summary["key_lessons"][:5]:
            training_data += f"- {lesson}\n"
        
        return training_data
    
    def export_for_ai(self, path: str = f"{DATA_DIR}/evolution/ai_training_data.md"):
        """导出 AI 训练数据"""
        data = self.generate_ai_training_data()
        with open(path, "w") as f:
            f.write(data)
        return path


# 初始化
_history_manager: Optional[HistoryDataManager] = None


def get_history_manager() -> HistoryDataManager:
    """获取历史数据管理器单例"""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryDataManager()
    return _history_manager


def init_sample_data():
    """初始化示例数据"""
    manager = get_history_manager()
    
    # 添加一些示例市场事件
    manager.add_market_event(MarketEvent(
        date="2026-02-05",
        event_type="market_crash",
        title="加密货币黑色周末",
        description="BTC单日暴跌14%，从$73K跌至$60K",
        impact="high",
        price_before=73000,
        price_after=60000,
        change_pct=-17.8,
        source="market_data",
        tags=["暴跌", "恐慌", "清算"],
        lessons=[
            "极端行情下杠杆风险极高",
            "设置止损是必要的",
            "恐惧指数低于10可能是买入信号"
        ]
    ))
    
    manager.add_market_event(MarketEvent(
        date="2026-02-13",
        event_type="price_consolidation",
        title="BTC在$66K附近震荡",
        description="BTC失守$66K支撑，连续第4周收跌",
        impact="medium",
        price_before=68000,
        price_after=66000,
        change_pct=-2.9,
        source="market_data",
        tags=["震荡", "支撑", "筑底"],
        lessons=[
            "震荡行情适合高抛低吸",
            "突破区间后顺势操作"
        ]
    ))
    
    print("✅ 示例历史数据已初始化")
    return manager
