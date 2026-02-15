"""
OpenTrade CCXT Adapter - 交易所连接

使用 CCXT 库连接 100+ 交易所

支持:
- 现货交易
- 期货交易 (需扩展)
- 杠杆交易
- 订单管理
- 余额查询
- 行情获取
"""

import asyncio
from datetime import datetime
from typing import Any

from ccxt.async_support import ccxt

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


class CCXTAdapter(ExecutionAdapter):
    """CCXT 交易所 Adapter"""

    def __init__(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        password: str | None = None,
        testnet: bool = True,
        timeout: int = 10000,
    ):
        self.name = f"ccxt_{exchange}"
        self.is_simulated = False

        # 初始化 CCXT
        exchange_class = getattr(ccxt, exchange, None)
        if not exchange_class:
            raise ValueError(f"Unknown exchange: {exchange}")

        self._exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "password": password,
            "enableRateLimit": True,
            "timeout": timeout,
            "options": {
                "defaultType": "spot",
                "testnet": testnet,
            },
        })

        # 缓存
        self._tickers_cache: dict[str, Ticker] = {}
        self._cache_time = 0
        self._cache_ttl = 1.0  # 1秒缓存

        # 运行时
        self._connected = False

    # ============ 连接管理 ============

    @property
    def exchange(self):
        """获取 CCXT 交易所实例"""
        return self._exchange

    async def connect(self) -> bool:
        """连接交易所"""
        try:
            # 测试连接
            await self._exchange.fetch_balance()
            self._connected = True
            return True
        except Exception as e:
            print(f"[CCXT] Connection failed: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        if self._connected:
            await self._exchange.close()
            self._connected = False

    # ============ 辅助方法 ============

    def _to_ccxt_side(self, side: OrderSide) -> str:
        """转换为 CCXT 订单方向"""
        return "buy" if side == OrderSide.BUY else "sell"

    def _from_ccxt_side(self, side: str) -> OrderSide:
        """从 CCXT 转换订单方向"""
        return OrderSide.BUY if side == "buy" else OrderSide.SELL

    def _to_ccxt_order_type(self, order_type: OrderType) -> str:
        """转换为 CCXT 订单类型"""
        type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }
        return type_map.get(order_type, "limit")

    def _from_ccxt_status(self, status: str) -> OrderStatus:
        """从 CCXT 转换订单状态"""
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(status, OrderStatus.PENDING)

    def _parse_symbol(self, symbol: str) -> str:
        """解析交易对格式"""
        # CCXT 通常使用 BTC/USDT 格式
        return symbol

    # ============ 订单操作 ============

    async def create_order(self, request: OrderRequest) -> Order:
        """创建订单"""
        try:
            ccxt_type = self._to_ccxt_order_type(request.order_type)
            side = self._to_ccxt_side(request.side)

            params = {}
            if request.stop_loss:
                params["stopLossPrice"] = request.stop_loss
            if request.take_profit:
                params["takeProfitPrice"] = request.take_profit

            # 创建订单
            ccxt_order = await self._exchange.create_order(
                symbol=request.symbol,
                type=ccxt_type,
                side=side,
                amount=request.quantity,
                price=request.price,
                params=params,
            )

            # 转换为统一格式
            return self._parse_ccxt_order(ccxt_order, request)

        except Exception as e:
            return Order(
                order_id=f"err_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                status=OrderStatus.REJECTED,
                quantity=request.quantity,
                rejection_reason=str(e),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
            )

    def _parse_ccxt_order(self, ccxt_order: dict, request: OrderRequest = None) -> Order:
        """解析 CCXT 订单"""
        return Order(
            order_id=str(ccxt_order.get("id", "")),
            symbol=ccxt_order.get("symbol", request.symbol if request else ""),
            side=self._from_ccxt_side(ccxt_order.get("side", "")),
            order_type=self._ccxt_type_to_order_type(ccxt_order.get("type", "limit")),
            status=self._from_ccxt_status(ccxt_order.get("status", "open")),
            quantity=ccxt_order.get("amount", 0),
            filled_quantity=ccxt_order.get("filled", 0),
            remaining_quantity=ccxt_order.get("remaining", 0),
            price=ccxt_order.get("price"),
            avg_fill_price=ccxt_order.get("average"),
            leverage=request.leverage if request else 1.0,
            take_profit=request.take_profit if request else None,
            stop_loss=request.stop_loss if request else None,
            client_order_id=ccxt_order.get("clientOrderId"),
            strategy_id=request.strategy_id if request else None,
            trace_id=request.trace_id if request else None,
            created_at=datetime.fromtimestamp(ccxt_order.get("timestamp", 0) / 1000).isoformat() if ccxt_order.get("timestamp") else datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            filled_at=datetime.fromtimestamp(ccxt_order.get("lastTradeTimestamp", 0) / 1000).isoformat() if ccxt_order.get("lastTradeTimestamp") else None,
        )

    def _ccxt_type_to_order_type(self, ccxt_type: str) -> OrderType:
        """CCXT 订单类型转统一类型"""
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT,
        }
        return type_map.get(ccxt_type, OrderType.LIMIT)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            await self._exchange.cancel_order(order_id, symbol=symbol)
            return True
        except Exception:
            return False

    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        """查询订单"""
        try:
            ccxt_order = await self._exchange.fetch_order(order_id, symbol)
            return self._parse_ccxt_order(ccxt_order)
        except Exception:
            return None

    async def get_orders(self, symbol: str | None = None) -> list[Order]:
        """查询订单列表"""
        try:
            orders = await self._exchange.fetch_orders(symbol=symbol)
            return [self._parse_ccxt_order(o) for o in orders]
        except Exception:
            return []

    # ============ 持仓和余额 ============

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """查询持仓 (期货)"""
        # 期货需要额外处理
        try:
            if symbol:
                position = await self._exchange.fetch_position(symbol)
                return [self._parse_ccxt_position(position)]
            else:
                positions = await self._exchange.fetch_positions()
                return [self._parse_ccxt_position(p) for p in positions]
        except Exception:
            return []

    def _parse_ccxt_position(self, ccxt_position: dict) -> Position:
        """解析 CCXT 持仓"""
        return Position(
            symbol=ccxt_position.get("symbol", ""),
            side=PositionSide.LONG if ccxt_position.get("side") == "long" else PositionSide.SHORT,
            quantity=abs(ccxt_position.get("contracts", 0)),
            entry_price=ccxt_position.get("entryPrice", 0),
            mark_price=ccxt_position.get("markPrice"),
            pnl=ccxt_position.get("unrealizedPnl", 0),
            pnl_pct=ccxt_position.get("percentage", 0),
            leverage=ccxt_position.get("leverage", 1),
            liquidation_price=ccxt_position.get("liquidationPrice"),
            opened_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )

    async def get_balance(self) -> AccountBalance:
        """查询余额"""
        try:
            balance = await self._exchange.fetch_balance()

            total = 0
            available = 0
            balances = {}

            for currency, data in balance.get("total", {}).items():
                if data and data > 0:
                    balances[currency] = data
                    # 估算总价值 (简化处理)
                    if currency in ["USDT", "BTC", "ETH"]:
                        total += data

            return AccountBalance(
                total_balance=total,
                available_balance=balance.get("free", {}).get("USDT", 0),
                margin_balance=total,
                unrealized_pnl=balance.get("total", {}).get("USDT", 0) - total,
                balances=balances,
            )
        except Exception:
            return AccountBalance(total_balance=0, available_balance=0, margin_balance=0)

    # ============ 行情 ============

    async def get_ticker(self, symbol: str) -> Ticker | None:
        """查询行情"""
        try:
            now = datetime.utcnow().timestamp()

            # 检查缓存
            if symbol in self._tickers_cache:
                cached = self._tickers_cache[symbol]
                if now - self._cache_time < self._cache_ttl:
                    return cached

            ticker = await self._exchange.fetch_ticker(symbol)

            result = Ticker(
                symbol=symbol,
                price=ticker.get("last", 0),
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                volume=ticker.get("baseVolume", 0),
                timestamp=datetime.fromtimestamp(ticker.get("timestamp", 0) / 1000).isoformat() if ticker.get("timestamp") else datetime.utcnow().isoformat(),
            )

            self._tickers_cache[symbol] = result
            self._cache_time = now

            return result

        except Exception:
            return None

    async def get_tickers(self, symbols: list[str]) -> dict[str, Ticker]:
        """批量查询行情"""
        tickers = {}
        for symbol in symbols:
            ticker = await self.get_ticker(symbol)
            if ticker:
                tickers[symbol] = ticker
        return tickers

    # ============ 市场数据 ============

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list]:
        """获取 K 线数据"""
        try:
            ohlcv = await self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            return ohlcv
        except Exception:
            return []

    async def fetch_order_book(self, symbol: str, limit: int = 10) -> dict:
        """获取订单簿"""
        try:
            orderbook = await self._exchange.fetch_order_book(symbol, limit)
            return {
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", []),
            }
        except Exception:
            return {"bids": [], "asks": []}


# ============ 常用交易所配置 ============

EXCHANGE_CONFIGS: dict[str, dict] = {
    "hyperliquid": {
        "features": ["spot", "perp"],
        "testnet": True,
        "class": "hyperliquid",
    },
    "binance": {
        "features": ["spot", "perp", "margin"],
        "testnet": True,
        "class": "binance",
    },
    "bybit": {
        "features": ["spot", "perp"],
        "testnet": True,
        "class": "bybit",
    },
}


def create_exchange_adapter(
    exchange: str,
    api_key: str,
    api_secret: str,
    password: str | None = None,
    testnet: bool = True,
) -> CCXTAdapter:
    """创建交易所 Adapter"""
    return CCXTAdapter(
        exchange=exchange,
        api_key=api_key,
        api_secret=api_secret,
        password=password,
        testnet=testnet,
    )
