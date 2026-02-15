"""
OpenTrade Execution Engine

统一执行接口，支持 Simulated/CCXT adapter
"""

from opentrade.engine.executor import (
    AccountBalance,
    ExecutionAdapter,
    Fill,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Ticker,
    TradeExecutor,
    create_simulated_executor,
    create_ccxt_executor,
)

__all__ = [
    # 数据模型
    "OrderRequest",
    "Order",
    "Fill",
    "Position",
    "Ticker",
    "AccountBalance",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSide",
    # Adapter
    "ExecutionAdapter",
    # 执行器
    "TradeExecutor",
    # 工厂
    "create_simulated_executor",
    "create_ccxt_executor",
]
