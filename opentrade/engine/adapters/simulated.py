"""
OpenTrade Simulated Adapter - 模拟交易

用于回测和 Paper Trading，模拟真实交易所行为

特性:
1. 精确的订单模拟 (部分成交、滑点)
2. 资金费率模拟
3. 完整的成交记录
4. 支持多品种
5. 统计报告生成
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any

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
)


class SimulatedAdapter(ExecutionAdapter):
    """模拟交易 Adapter"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fees: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        funding_rate: float = 0.0001,  # 0.01%
        auto_liquidate: bool = True,
    ):
        self.name = "simulated"
        self.is_simulated = True

        # 账户状态
        self._balances: dict[str, float] = {"USDT": initial_balance, "BTC": 0, "ETH": 0}
        self._total_balance = initial_balance
        self._available_balance = initial_balance

        # 订单和持仓
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._trades: list[dict] = []

        # 市场数据
        self._tickers: dict[str, Ticker] = {}
        self._price_history: dict[str, list[dict]] = {}

        # 配置
        self._fees = fees
        self._slippage = slippage
        self._funding_rate = funding_rate
        self._auto_liquidate = auto_liquidate

        # 事件
        self._on_order_update: list[callable] = []
        self._on_position_update: list[callable] = []
        self._on_balance_update: list[callable] = []

        # 运行时
        self._running = False
        self._task: asyncio.Task | None = None

    # ============ 连接管理 ============

    async def connect(self) -> bool:
        """连接模拟交易所"""
        self._running = True
        return True

    async def disconnect(self):
        """断开连接"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ============ 订单操作 ============

    async def create_order(self, request: OrderRequest) -> Order:
        """创建订单"""
        import random

        # 生成订单 ID
        order_id = f"sim_{uuid.uuid4().hex[:16]}"

        # 创建订单
        now = datetime.utcnow().isoformat()
        order = Order(
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.PENDING,
            quantity=request.quantity,
            filled_quantity=0.0,
            remaining_quantity=request.quantity,
            price=request.price,
            avg_fill_price=0.0,
            leverage=request.leverage,
            take_profit=request.take_profit,
            stop_loss=request.stop_loss,
            client_order_id=request.client_order_id,
            strategy_id=request.strategy_id,
            strategy_name=request.strategy_name,
            trace_id=request.trace_id,
            created_at=now,
            updated_at=now,
        )

        # 立即执行市价单
        if request.order_type == OrderType.MARKET:
            order = await self._execute_market_order(order, request)
        elif request.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]:
            order.status = OrderStatus.PENDING
            self._orders[order_id] = order

        return order

    async def _execute_market_order(self, order: Order, request: OrderRequest) -> Order:
        """执行市价单"""
        # 获取当前价格
        ticker = await self.get_ticker(request.symbol)
        if not ticker:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "No ticker data"
            return order

        # 计算滑点后的成交价
        base_price = ticker.price
        if request.side == OrderSide.BUY:
            fill_price = base_price * (1 + self._slippage)
        else:
            fill_price = base_price * (1 - self._slippage)

        # 计算数量和手续费
        quantity = request.quantity
        fee = quantity * fill_price * self._fees

        # 检查余额
        cost = quantity * fill_price + fee
        balance_key = "USDT" if "USDT" in request.symbol else request.symbol.split("/")[0]

        if self._balances.get(balance_key, 0) < cost and request.side == OrderSide.BUY:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Insufficient balance"
            return order

        # 更新余额
        if balance_key in self._balances:
            self._balances[balance_key] -= cost
            self._available_balance -= cost

        # 创建成交记录
        fill = Fill(
            fill_id=f"fill_{uuid.uuid4().hex[:12]}",
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.utcnow().isoformat(),
        )

        order.fills = [fill]
        order.filled_quantity = quantity
        order.remaining_quantity = 0
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.filled_at = fill.timestamp

        # 更新持仓
        await self._update_position_from_fill(order, fill)

        # 记录交易
        self._trades.append({
            "order_id": order.order_id,
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "timestamp": fill.timestamp,
        })

        # 触发事件
        self._notify_order_update(order)

        return order

    async def _update_position_from_fill(self, order: Order, fill: Fill):
        """根据成交更新持仓"""
        symbol = fill.symbol
        existing = self._positions.get(symbol)

        if order.side == OrderSide.BUY:
            # 买入 - 增加多头或开多
            if existing and existing.side == PositionSide.LONG:
                # 增持
                total_qty = existing.quantity + fill.quantity
                new_entry = (existing.entry_price * existing.quantity + fill.price * fill.quantity) / total_qty
                existing.quantity = total_qty
                existing.entry_price = new_entry
            else:
                # 新建多头
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=fill.quantity,
                    entry_price=fill.price,
                    mark_price=fill.price,
                    pnl=0.0,
                    pnl_pct=0.0,
                    leverage=order.leverage,
                    opened_at=fill.timestamp,
                    updated_at=fill.timestamp,
                )
        else:
            # 卖出 - 平多或开空
            if existing and existing.side == PositionSide.LONG:
                if existing.quantity <= fill.quantity:
                    # 全部平仓
                    del self._positions[symbol]
                else:
                    # 部分平仓
                    existing.quantity -= fill.quantity
                    existing.updated_at = fill.timestamp

        # 更新余额 (收到卖出所得)
        self._balances["USDT"] += fill.quantity * fill.price - fill.fee
        self._available_balance = self._balances["USDT"]

        # 触发事件
        if symbol in self._positions:
            self._notify_position_update(self._positions[symbol])

    # ============ 订单管理 ============

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow().isoformat()
                self._notify_order_update(order)
                return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        """查询订单"""
        return self._orders.get(order_id)

    async def get_orders(self, symbol: str | None = None) -> list[Order]:
        """查询订单列表"""
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # ============ 持仓和余额 ============

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """查询持仓"""
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    async def get_balance(self) -> AccountBalance:
        """查询余额"""
        total = self._total_balance
        return AccountBalance(
            total_balance=total,
            available_balance=self._available_balance,
            margin_balance=self._available_balance,
            unrealized_pnl=0.0,
            balances=self._balances.copy(),
        )

    # ============ 行情 ============

    async def get_ticker(self, symbol: str) -> Ticker | None:
        """查询行情"""
        if symbol in self._tickers:
            return self._tickers[symbol]
        return None

    async def get_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        """批量查询行情"""
        return {s: self._tickers[s] for s in symbols if s in self._tickers}

    def set_price(self, symbol: str, price: float, volume: float = 0.0):
        """设置当前价格 (用于回测)"""
        now = datetime.utcnow().isoformat()
        ticker = Ticker(
            symbol=symbol,
            price=price,
            bid=price * 0.9998,
            ask=price * 1.0002,
            volume=volume,
            timestamp=now,
        )
        self._tickers[symbol] = ticker

        # 记录价格历史
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append({
            "price": price,
            "volume": volume,
            "timestamp": now,
        })

    def set_prices(self, prices: dict[str, float], volume: float = 0.0):
        """批量设置价格"""
        for symbol, price in prices.items():
            self.set_price(symbol, price, volume)

    # ============ 统计数据 ============

    def get_stats(self) -> dict[str, Any]:
        """获取统计报告"""
        trades = self._trades

        if not trades:
            return {"total_trades": 0}

        # 计算盈亏
        pnl_by_symbol: dict[str, list] = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in pnl_by_symbol:
                pnl_by_symbol[symbol] = []
            pnl_by_symbol[symbol].append({
                "side": trade["side"],
                "price": trade["price"],
                "quantity": trade["quantity"],
                "fee": trade["fee"],
            })

        total_fees = sum(t["fee"] for t in trades)

        return {
            "total_trades": len(trades),
            "total_fees": total_fees,
            "symbols": list(pnl_by_symbol.keys()),
            "positions": len(self._positions),
            "open_orders": len([o for o in self._orders.values() if o.status == OrderStatus.PENDING]),
            "current_balance": self._balances,
        }

    def get_trades(self) -> list[dict]:
        """获取所有交易记录"""
        return self._trades.copy()

    def get_price_history(self, symbol: str) -> list[dict]:
        """获取价格历史"""
        return self._price_history.get(symbol, [])

    def reset(self, balance: float | None = None):
        """重置模拟状态"""
        if balance:
            self._balances = {"USDT": balance, "BTC": 0, "ETH": 0}
            self._total_balance = balance
            self._available_balance = balance

        self._orders = {}
        self._positions = {}
        self._trades = []
        self._tickers = {}
        self._price_history = {}

    # ============ 事件 ============

    def on_order_update(self, callback: callable):
        """订阅订单更新事件"""
        self._on_order_update.append(callback)

    def on_position_update(self, callback: callable):
        """订阅持仓更新事件"""
        self._on_position_update.append(callback)

    def on_balance_update(self, callback: callable):
        """订阅余额更新事件"""
        self._on_balance_update.append(callback)

    def _notify_order_update(self, order: Order):
        """通知订单更新"""
        for cb in self._on_order_update:
            try:
                cb(order)
            except Exception:
                pass

    def _notify_position_update(self, position: Position):
        """通知持仓更新"""
        for cb in self._on_position_update:
            try:
                cb(position)
            except Exception:
                pass

    def _notify_balance_update(self):
        """通知余额更新"""
        for cb in self._on_balance_update:
            try:
                cb(self._balances)
            except Exception:
                pass
