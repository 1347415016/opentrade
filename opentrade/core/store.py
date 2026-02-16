"""
OpenTrade 内存存储

提供简单的内存存储，用于 API 数据查询。
生产环境应使用 Redis/PostgreSQL。
"""

from datetime import datetime
from typing import Any
from uuid import uuid4


class MemoryStore:
    """简单的内存存储"""

    def __init__(self):
        self._orders: dict[str, Any] = {}
        self._positions: dict[str, Any] = {}
        self._balance: dict[str, float] = {
            "total_equity": 10000.0,
            "available": 10000.0,
            "positions_value": 0.0,
        }
        self._strategies: dict[str, Any] = {}
        self._events: list[Any] = []

    # ============ 订单 ============

    def create_order(self, order_data: dict) -> dict:
        """创建订单"""
        order_id = str(uuid4())[:8]
        order = {
            "id": order_id,
            "symbol": order_data.get("symbol"),
            "side": order_data.get("side"),
            "status": "open",
            "size": order_data.get("size", 0.0),
            "filled_size": 0.0,
            "average_price": None,
            "created_at": datetime.utcnow().isoformat(),
            **order_data,
        }
        self._orders[order_id] = order
        self._emit("order_created", order)
        return order

    def get_orders(self, symbol: str | None = None) -> list[dict]:
        """获取订单列表"""
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]
        return orders

    def get_order(self, order_id: str) -> dict | None:
        """获取单个订单"""
        return self._orders.get(order_id)

    def fill_order(self, order_id: str, fill_data: dict):
        """订单成交"""
        if order_id in self._orders:
            self._orders[order_id].update({
                "status": "filled",
                "filled_size": fill_data.get("size", 0.0),
                "average_price": fill_data.get("price"),
            })
            self._emit("order_filled", self._orders[order_id])

    def cancel_order(self, order_id: str):
        """取消订单"""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            self._emit("order_cancelled", self._orders[order_id])

    # ============ 持仓 ============

    def update_position(self, symbol: str, position_data: dict):
        """更新持仓"""
        self._positions[symbol] = {
            "symbol": symbol,
            "side": position_data.get("side", "long"),
            "size": position_data.get("size", 0.0),
            "entry_price": position_data.get("entry_price", 0.0),
            "mark_price": position_data.get("mark_price", 0.0),
            "pnl": position_data.get("pnl", 0.0),
            "pnl_pct": position_data.get("pnl_pct", 0.0),
            "updated_at": datetime.utcnow().isoformat(),
        }

    def get_positions(self) -> list[dict]:
        """获取所有持仓"""
        return list(self._positions.values())

    def close_position(self, symbol: str):
        """平仓"""
        if symbol in self._positions:
            del self._positions[symbol]
            self._emit("position_closed", {"symbol": symbol})

    # ============ 余额 ============

    def get_balance(self) -> dict:
        """获取余额"""
        return self._balance.copy()

    def update_balance(self, **kwargs):
        """更新余额"""
        self._balance.update(kwargs)

    # ============ 策略 ============

    def get_strategies(self) -> list[dict]:
        """获取策略列表"""
        return list(self._strategies.values())

    def register_strategy(self, strategy_id: str, strategy_data: dict):
        """注册策略"""
        self._strategies[strategy_id] = {
            "id": strategy_id,
            "name": strategy_data.get("name", strategy_id),
            "status": "active",
            "enabled": True,
            "config": strategy_data,
            "created_at": datetime.utcnow().isoformat(),
        }

    def set_strategy_status(self, strategy_id: str, status: str):
        """设置策略状态"""
        if strategy_id in self._strategies:
            self._strategies[strategy_id]["status"] = status
            self._emit("strategy_updated", {"id": strategy_id, "status": status})

    # ============ 事件 ============

    def _emit(self, event_type: str, data: dict):
        """内部事件触发"""
        self._events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_events(self, event_type: str | None = None) -> list[dict]:
        """获取事件"""
        events = self._events
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-100:]  # 只返回最近100条


# 全局单例
store = MemoryStore()
