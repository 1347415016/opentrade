"""
OpenTrade 策略生命周期管理

管理策略从 Draft → Paper → Canary → Production → Retired 完整生命周期

生命周期:
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  Draft  │ ──▶ │  Paper  │ ──▶ │ Canary  │ ──▶ │Production│
    │ 草稿    │     │ 模拟盘  │     │ 小资金  │     │  实盘   │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
        │                                              │
        │                                              ▼
        │                                       ┌─────────┐
        │                                       │Retired  │
        │                                       │ 已停用  │
        │                                       └─────────┘

状态门禁:
- Draft → Paper: 基本回测验证
- Paper → Canary: 30天稳定性
- Canary → Production: 3个月合规运行
- Production → Retired: 手动标记或自动触发
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel


# ============ 生命周期状态 ============

class LifecycleStage(str, Enum):
    """策略生命周期阶段"""
    DRAFT = "draft"           # 草稿
    PAPER = "paper"           # 模拟盘
    CANARY = "canary"         # 小资金实盘
    PRODUCTION = "production" # 全量实盘
    RETIRED = "retired"       # 已停用


class StageTransition(str, Enum):
    """阶段转换原因"""
    BACKTEST_PASSED = "backtest_passed"       # 回测通过
    PAPER_STABLE = "paper_stable"             # 模拟盘稳定
    CANARY_PROVEN = "canary_proven"           # 小资金验证
    MANUAL_UPGRADE = "manual_upgrade"         # 手动升级
    MANUAL_RETIRE = "manual_retire"           # 手动停用
    RISK_TRIGGERED = "risk_triggered"         # 风险触发
    PERFORMANCE_DEGRADED = "performance_degraded"  # 性能下降
    EXPIRED = "expired"                       # 过期


# ============ 门禁要求 ============

@dataclass
class GateRequirements:
    """阶段门禁要求"""
    min_days: int = 0           # 最少天数
    min_trades: int = 0         # 最少交易次数
    min_win_rate: float = 0.0   # 最低胜率
    max_drawdown: float = 100.0  # 最大回撤限制
    min_sharpe: float = 0.0     # 最低夏普
    min_profit: float = 0.0    # 最低收益
    max_loss_streak: int = 10   # 最大连亏次数

    def check(self, stats: dict) -> tuple[bool, str]:
        """检查是否满足要求"""
        if stats.get("days", 0) < self.min_days:
            return False, f"运行天数不足: {stats['days']}/{self.min_days}"

        if stats.get("trades", 0) < self.min_trades:
            return False, f"交易次数不足: {stats['trades']}/{self.min_trades}"

        if stats.get("win_rate", 1) < self.min_win_rate:
            return False, f"胜率不足: {stats['win_rate']:.1%}/{self.min_win_rate:.1%}"

        if stats.get("max_drawdown", 0) > self.max_drawdown:
            return False, f"回撤过大: {stats['max_drawdown']:.1%}/{self.max_drawdown:.1%}"

        if stats.get("sharpe_ratio", 0) < self.min_sharpe:
            return False, f"夏普不足: {stats['sharpe_ratio']:.2f}/{self.min_sharpe:.2f}"

        if stats.get("total_return", 0) < self.min_profit:
            return False, f"收益不足: {stats['total_return']:.1%}/{self.min_profit:.1%}"

        if stats.get("loss_streak", 0) > self.max_loss_streak:
            return False, f"连亏过多: {stats['loss_streak']}/{self.max_loss_streak}"

        return True, "通过"


# 各阶段门禁配置
GATE_CONFIGS: dict[LifecycleStage, GateRequirements] = {
    LifecycleStage.DRAFT: GateRequirements(
        min_days=0,
        min_trades=0,
    ),
    LifecycleStage.PAPER: GateRequirements(
        min_days=7,
        min_trades=10,
        min_win_rate=0.4,
        max_drawdown=20.0,
        min_sharpe=0.5,
    ),
    LifecycleStage.CANARY: GateRequirements(
        min_days=30,
        min_trades=50,
        min_win_rate=0.45,
        max_drawdown=15.0,
        min_sharpe=0.8,
        min_profit=5.0,
    ),
    LifecycleStage.PRODUCTION: GateRequirements(
        min_days=90,
        min_trades=200,
        min_win_rate=0.5,
        max_drawdown=10.0,
        min_sharpe=1.0,
        min_profit=15.0,
    ),
    LifecycleStage.RETIRED: GateRequirements(
        min_days=0,
        min_trades=0,
    ),
}


# ============ 策略元数据 ============

@dataclass
class StrategyMetadata:
    """策略元数据"""
    strategy_id: str
    name: str
    version: str
    created_at: str
    updated_at: str

    # 阶段
    stage: LifecycleStage = LifecycleStage.DRAFT

    # 资源配置
    capital_allocation: float = 0.0  # 分配资金
    max_position_pct: float = 0.1    # 最大仓位

    # 所有者
    owner: str = "system"
    description: str = ""

    # 标签
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage": self.stage.value,
            "capital_allocation": self.capital_allocation,
            "max_position_pct": self.max_position_pct,
            "owner": self.owner,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyMetadata":
        return cls(
            strategy_id=data.get("strategy_id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Unnamed Strategy"),
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            stage=LifecycleStage(data.get("stage", "draft")),
            capital_allocation=data.get("capital_allocation", 0.0),
            max_position_pct=data.get("max_position_pct", 0.1),
            owner=data.get("owner", "system"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


# ============ 策略性能统计 ============

@dataclass
class StrategyStats:
    """策略性能统计"""
    strategy_id: str

    # 时间
    days: int = 0
    last_trade_at: str | None = None

    # 交易
    trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    loss_streak: int = 0
    max_loss_streak: int = 0

    # 收益
    total_return: float = 0.0
    avg_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # 风险
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    var_daily: float = 0.0

    # 资金
    final_capital: float = 0.0
    initial_capital: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "days": self.days,
            "last_trade_at": self.last_trade_at,
            "trades": self.trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": self.win_rate,
            "loss_streak": self.loss_streak,
            "max_loss_streak": self.max_loss_streak,
            "total_return": self.total_return,
            "avg_return": self.avg_return,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "var_daily": self.var_daily,
            "final_capital": self.final_capital,
            "initial_capital": self.initial_capital,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyStats":
        return cls(
            strategy_id=data.get("strategy_id", ""),
            **data,
        )


# ============ 阶段转换记录 ============

@dataclass
class StageTransitionRecord:
    """阶段转换记录"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy_id: str = ""
    from_stage: LifecycleStage = LifecycleStage.DRAFT
    to_stage: LifecycleStage = LifecycleStage.DRAFT
    reason: StageTransition = StageTransition.MANUAL_UPGRADE

    # 验证
    gate_passed: bool = True
    gate_details: dict = field(default_factory=dict)

    # 统计快照
    stats_snapshot: dict = field(default_factory=dict)

    # 时间
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    performed_by: str = "system"

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "strategy_id": self.strategy_id,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "reason": self.reason.value,
            "gate_passed": self.gate_passed,
            "gate_details": self.gate_details,
            "stats_snapshot": self.stats_snapshot,
            "timestamp": self.timestamp,
            "performed_by": self.performed_by,
        }


# ============ 生命周期管理器 ============

class LifecycleManager:
    """
    策略生命周期管理器

    功能:
    1. 策略注册和阶段管理
    2. 阶段门禁检查
    3. 自动升级/降级
    4. 历史追溯
    """

    def __init__(self, storage_path: str = "./data/strategies"):
        self.storage_path = storage_path
        self.strategies: dict[str, StrategyMetadata] = {}
        self.stats: dict[str, StrategyStats] = {}
        self.transitions: list[StageTransitionRecord] = []

        # 加载已有策略
        self._load_all()

    # ============ 策略管理 ============

    def register_strategy(
        self,
        name: str,
        version: str = "1.0.0",
        owner: str = "system",
        description: str = "",
        tags: list[str] = None,
    ) -> StrategyMetadata:
        """注册新策略"""
        strategy_id = str(uuid.uuid4())[:8]

        metadata = StrategyMetadata(
            strategy_id=strategy_id,
            name=name,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            stage=LifecycleStage.DRAFT,
            owner=owner,
            description=description,
            tags=tags or [],
        )

        self.strategies[strategy_id] = metadata
        self.stats[strategy_id] = StrategyStats(strategy_id=strategy_id)

        self._save_strategy(strategy_id)

        return metadata

    def get_strategy(self, strategy_id: str) -> StrategyMetadata | None:
        """获取策略"""
        return self.strategies.get(strategy_id)

    def get_strategies_by_stage(self, stage: LifecycleStage) -> list[StrategyMetadata]:
        """按阶段查询策略"""
        return [s for s in self.strategies.values() if s.stage == stage]

    def get_all_strategies(self) -> list[StrategyMetadata]:
        """获取所有策略"""
        return list(self.strategies.values())

    # ============ 阶段转换 ============

    def transition_to(
        self,
        strategy_id: str,
        target_stage: LifecycleStage,
        reason: StageTransition = StageTransition.MANUAL_UPGRADE,
        performed_by: str = "system",
    ) -> tuple[bool, str]:
        """
        阶段转换 (带门禁检查)

        Returns:
            (success, message)
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return False, f"策略不存在: {strategy_id}"

        current_stage = strategy.stage
        stats = self.stats.get(strategy_id, StrategyStats(strategy_id=strategy_id))

        # 目标阶段检查
        if target_stage == current_stage:
            return False, f"已经在 {target_stage.value} 阶段"

        # 顺序检查 (不能跳过阶段)
        stage_order = [
            LifecycleStage.DRAFT,
            LifecycleStage.PAPER,
            LifecycleStage.CANARY,
            LifecycleStage.PRODUCTION,
            LifecycleStage.RETIRED,
        ]

        try:
            current_idx = stage_order.index(current_stage)
            target_idx = stage_order.index(target_stage)

            if target_idx > current_idx + 1:
                return False, f"不能跳过阶段，请按顺序升级"

        except ValueError:
            return False, f"无效阶段: {target_stage}"

        # 门禁检查
        if target_stage != LifecycleStage.RETIRED:
            requirements = GATE_CONFIGS.get(target_stage, GateRequirements())
            stats_dict = stats.to_dict()
            passed, message = requirements.check(stats_dict)

            if not passed:
                return False, f"门禁未通过: {message}"

        # 执行转换
        record = StageTransitionRecord(
            strategy_id=strategy_id,
            from_stage=current_stage,
            to_stage=target_stage,
            reason=reason,
            gate_passed=target_stage == LifecycleStage.RETIRED,
            gate_details=self.stats.get(strategy_id, StrategyStats(strategy_id=strategy_id)).to_dict(),
            stats_snapshot=stats.to_dict(),
            performed_by=performed_by,
        )
        self.transitions.append(record)

        # 更新策略
        strategy.stage = target_stage
        strategy.updated_at = datetime.utcnow().isoformat()

        # 更新资金配置
        if target_stage == LifecycleStage.PAPER:
            strategy.capital_allocation = 10000  # 模拟盘 1万
        elif target_stage == LifecycleStage.CANARY:
            strategy.capital_allocation = 100000  # 小资金 10万
        elif target_stage == LifecycleStage.PRODUCTION:
            strategy.capital_allocation = 500000  # 全量 50万

        self._save_strategy(strategy_id)

        return True, f"{current_stage.value} → {target_stage.value}"

    def check_upgrade(self, strategy_id: str) -> tuple[bool, str]:
        """检查是否可以升级"""
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return False, f"策略不存在: {strategy_id}"

        current_stage = strategy.stage

        # 确定下一个阶段
        next_stage_map = {
            LifecycleStage.DRAFT: LifecycleStage.PAPER,
            LifecycleStage.PAPER: LifecycleStage.CANARY,
            LifecycleStage.CANARY: LifecycleStage.PRODUCTION,
        }

        next_stage = next_stage_map.get(current_stage)
        if not next_stage:
            return False, f"无法从 {current_stage.value} 升级"

        # 门禁检查
        return self.transition_to(
            strategy_id,
            next_stage,
            reason=StageTransition.BACKTEST_PASSED,
            performed_by="system",
        )

    def retire_strategy(self, strategy_id: str, reason: str = "manual") -> tuple[bool, str]:
        """停用策略"""
        return self.transition_to(
            strategy_id,
            LifecycleStage.RETIRED,
            reason=StageTransition.MANUAL_RETIRE,
            performed_by=reason,
        )

    # ============ 统计更新 ============

    def update_stats(self, strategy_id: str, stats: dict):
        """更新策略统计"""
        current = self.stats.get(strategy_id, StrategyStats(strategy_id=strategy_id))

        for key, value in stats.items():
            if hasattr(current, key):
                setattr(current, key, value)

        self.stats[strategy_id] = current
        self._save_stats(strategy_id)

    def record_trade(self, strategy_id: str, trade: dict):
        """记录交易并更新统计"""
        # 简化的交易记录
        pass

    # ============ 历史和审计 ============

    def get_transitions(self, strategy_id: str) -> list[StageTransitionRecord]:
        """获取策略转换历史"""
        return [t for t in self.transitions if t.strategy_id == strategy_id]

    def get_all_transitions(self) -> list[StageTransitionRecord]:
        """获取所有转换历史"""
        return self.transitions

    def get_strategy_history(self, strategy_id: str) -> dict:
        """获取策略完整历史"""
        strategy = self.strategies.get(strategy_id)
        stats = self.stats.get(strategy_id, StrategyStats(strategy_id=strategy_id))
        transitions = self.get_transitions(strategy_id)

        return {
            "metadata": strategy.to_dict() if strategy else None,
            "stats": stats.to_dict(),
            "transitions": [t.to_dict() for t in transitions],
        }

    # ============ 持久化 ============

    def _save_strategy(self, strategy_id: str):
        """保存策略"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)

        strategy = self.strategies.get(strategy_id)
        if strategy:
            path = f"{self.storage_path}/{strategy_id}.json"
            with open(path, "w") as f:
                json.dump(strategy.to_dict(), f, indent=2)

    def _save_stats(self, strategy_id: str):
        """保存统计"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)

        stats = self.stats.get(strategy_id)
        if stats:
            path = f"{self.storage_path}/{strategy_id}_stats.json"
            with open(path, "w") as f:
                json.dump(stats.to_dict(), f, indent=2)

    def _load_all(self):
        """加载所有策略"""
        import os

        if not os.path.exists(self.storage_path):
            return

        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json") and not filename.endswith("_stats.json"):
                strategy_id = filename.replace(".json", "")
                path = f"{self.storage_path}/{filename}"

                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.strategies[strategy_id] = StrategyMetadata.from_dict(data)
                except Exception:
                    pass

            elif filename.endswith("_stats.json"):
                strategy_id = filename.replace("_stats.json", "")
                path = f"{self.storage_path}/{filename}"

                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.stats[strategy_id] = StrategyStats.from_dict(data)
                except Exception:
                    pass

    # ============ 报告 ============

    def get_lifecycle_report(self) -> dict:
        """获取生命周期报告"""
        report = {
            "total_strategies": len(self.strategies),
            "by_stage": {},
            "recent_transitions": [],
            "upgrade_candidates": [],
        }

        # 按阶段统计
        for stage in LifecycleStage:
            strategies = self.get_strategies_by_stage(stage)
            report["by_stage"][stage.value] = {
                "count": len(strategies),
                "strategies": [s.name for s in strategies],
            }

        # 最近转换
        sorted_transitions = sorted(
            self.transitions,
            key=lambda t: t.timestamp,
            reverse=True,
        )
        report["recent_transitions"] = [t.to_dict() for t in sorted_transitions[:10]]

        # 升级候选 (检查下一个阶段)
        for strategy_id, strategy in self.strategies.items():
            if strategy.stage in [LifecycleStage.DRAFT, LifecycleStage.PAPER, LifecycleStage.CANARY]:
                can_upgrade, _ = self.check_upgrade(strategy_id)
                if can_upgrade:
                    report["upgrade_candidates"].append({
                        "strategy_id": strategy_id,
                        "name": strategy.name,
                        "current_stage": strategy.stage.value,
                        "next_stage": {
                            LifecycleStage.DRAFT: LifecycleStage.PAPER,
                            LifecycleStage.PAPER: LifecycleStage.CANARY,
                            LifecycleStage.CANARY: LifecycleStage.PRODUCTION,
                        }[strategy.stage].value,
                    })

        return report


# ============ 便捷函数 ============

def create_lifecycle_manager(storage_path: str = "./data/strategies") -> LifecycleManager:
    """创建生命周期管理器"""
    return LifecycleManager(storage_path=storage_path)
