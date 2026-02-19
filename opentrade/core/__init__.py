"""
OpenTrade 核心模块
"""
from opentrade.core.config import (
    OpenTradeConfig,
    ConfigManager,
    ExchangeConfig,
    AIConfig,
    RiskConfig,
    get_config,
    settings,
)

from opentrade.core.encryption import (
    encrypt_api_key,
    decrypt_api_key,
    SecureConfig,
    get_or_create_key,
)

from opentrade.core.gateway import (
    RiskEngine,
    RiskConfig as GatewayRiskConfig,
    RiskCheckResult,
    get_risk_engine,
)

from opentrade.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    get_circuit_breaker,
)

from opentrade.core.order import (
    OrderIdempotencyManager,
    OrderIdempotencyConfig,
    OrderDeduplicator,
    get_idempotency_manager,
)

from opentrade.core.network import (
    NetworkHandler,
    NetworkConfig,
    NetworkException,
    get_network_handler,
)

__all__ = [
    # Config
    "OpenTradeConfig",
    "ConfigManager",
    "ExchangeConfig",
    "AIConfig",
    "RiskConfig",
    "get_config",
    "settings",
    # Encryption
    "encrypt_api_key",
    "decrypt_api_key",
    "SecureConfig",
    "get_or_create_key",
    # Gateway
    "RiskEngine",
    "RiskCheckResult",
    "get_risk_engine",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "get_circuit_breaker",
    # Order
    "OrderIdempotencyManager",
    "OrderIdempotencyConfig",
    "OrderDeduplicator",
    "get_idempotency_manager",
    # Network
    "NetworkHandler",
    "NetworkConfig",
    "NetworkException",
    "get_network_handler",
]
