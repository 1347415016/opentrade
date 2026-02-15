"""
OpenTrade 强化学习模块 - RL

实现:
1. PPO/A2C 策略网络
2. 经验回放
3. 环境模拟器
4. 离线训练 pipeline

注意:
- 实际部署需要 PyTorch + Stable-Baselines3
- 这里提供简化版实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class ActionType(str, Enum):
    """动作类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class State:
    """状态"""
    prices: list[float] = field(default_factory=list)  # 最近 N 个价格
    portfolio_value: float = 10000.0
    position: float = 0.0  # 持仓数量
    cash: float = 10000.0

    # 技术指标
    rsi: float = 50.0
    macd: float = 0.0
    bollinger_position: float = 0.5

    def to_array(self) -> np.ndarray:
        """转换为数组"""
        prices_array = self.prices[-20:] if len(self.prices) >= 20 else [0] * 20
        return np.array([
            *prices_array,
            self.portfolio_value / 10000.0,  # 归一化
            self.position,
            self.rsi / 100.0,
            self.macd,
            self.bollinger_position,
        ])


@dataclass
class Action:
    """动作"""
    action_type: ActionType
    size: float = 1.0  # 0-1, 仓位比例


@dataclass
class Reward:
    """奖励"""
    reward: float
    done: bool = False
    info: dict = field(default_factory=dict)


# ============ 简化版 RL 实现 ============

class RLPolic(ABC):
    """RL 策略基类"""

    @abstractmethod
    def predict(self, state: State) -> Action:
        """预测动作"""
        pass

    @abstractmethod
    def update(self, state: State, action: Action, reward: float, next_state: State):
        """更新策略"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass


class RandomPolicy(RLPolic):
    """随机策略 (基线)"""

    def predict(self, state: State) -> Action:
        return Action(
            action_type=ActionType.HOLD,
            size=1.0,
        )

    def update(self, state: State, action: Action, reward: float, next_state: State):
        pass

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"type": "random"}, f)

    def load(self, path: str):
        pass


class MomentumPolicy(RLPolic):
    """动量策略 (简单策略)"""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def predict(self, state: State) -> Action:
        if len(state.prices) < 2:
            return Action(ActionType.HOLD)

        # 动量
        momentum = (state.prices[-1] - state.prices[-2]) / state.prices[-2]

        if momentum > self.threshold:
            return Action(ActionType.BUY, size=min(abs(momentum) * 10, 1.0))
        elif momentum < -self.threshold:
            return Action(ActionType.SELL, size=min(abs(momentum) * 10, 1.0))

        return Action(ActionType.HOLD)

    def update(self, state: State, action: Action, reward: float, next_state: State):
        pass

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"type": "momentum", "threshold": self.threshold}, f)

    def load(self, path: str):
        pass


class EpsilonGreedyPolicy(RLPolic):
    """ε-贪婪策略 (简化 Q-Learning)"""

    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q 表 (简化: 离散化状态)
        self.q_table: dict[str, list[float]] = {}

    def _get_state_key(self, state: State) -> str:
        """离散化状态"""
        # 简化: 用动量方向 + RSI 区间
        if len(state.prices) < 2:
            return "neutral_50"

        momentum = state.prices[-1] - state.prices[-2]
        momentum_dir = "up" if momentum > 0 else "down" if momentum < 0 else "neutral"

        rsi_range = int(state.rsi / 20) * 20  # 0, 20, 40, 60, 80, 100

        return f"{momentum_dir}_{rsi_range}"

    def _get_action_index(self, action_type: ActionType) -> int:
        """动作转索引"""
        mapping = {
            ActionType.HOLD: 0,
            ActionType.BUY: 1,
            ActionType.SELL: 2,
            ActionType.CLOSE: 3,
        }
        return mapping.get(action_type, 0)

    def _get_action_from_index(self, index: int) -> ActionType:
        """索引转动作"""
        mapping = ["HOLD", "BUY", "SELL", "CLOSE"]
        return ActionType(mapping[index] if index < len(mapping) else "HOLD")

    def predict(self, state: State) -> Action:
        state_key = self._get_state_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions

        # ε-贪婪
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = np.argmax(self.q_table[state_key])

        return Action(
            action_type=self._get_action_from_index(action_idx),
            size=1.0,
        )

    def update(self, state: State, action: Action, reward: float, next_state: State):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.n_actions

        action_idx = self._get_action_index(action.action_type)

        # Q-Learning 更新
        current_q = self.q_table[state_key][action_idx]
        max_next_q = max(self.q_table[next_state_key])

        self.q_table[state_key][action_idx] = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({
                "type": "epsilon_greedy",
                "q_table": self.q_table,
                "epsilon": self.epsilon,
            }, f)

    def load(self, path: str):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = data.get("q_table", {})
        self.epsilon = data.get("epsilon", 0.1)


# ============ 环境模拟器 ============

class TradingEnv:
    """交易环境 (Gymnasium-like)"""

    def __init__(
        self,
        prices: list[float],
        initial_balance: float = 10000.0,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
    ):
        self.prices = prices
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost

        self.reset()

    def reset(self) -> State:
        """重置环境"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.trades: list[dict] = []
        self.portfolio_values = [self.initial_balance]

        return self._get_state()

    def step(self, action: Action) -> tuple[State, Reward, bool]:
        """执行一步"""
        price = self.prices[self.current_step]

        # 执行动作
        if action.action_type == ActionType.BUY:
            # 买入
            cost = self.balance * action.size * (1 - self.transaction_cost)
            quantity = cost / price
            self.position += quantity
            self.balance -= cost
            self.trades.append({
                "type": "BUY",
                "price": price,
                "quantity": quantity,
                "step": self.current_step,
            })

        elif action.action_type == ActionType.SELL:
            # 卖出 (减仓)
            quantity = self.position * action.size
            revenue = quantity * price * (1 - self.transaction_cost)
            self.position -= quantity
            self.balance += revenue
            self.trades.append({
                "type": "SELL",
                "price": price,
                "quantity": quantity,
                "step": self.current_step,
            })

        elif action.action_type == ActionType.CLOSE:
            # 全平
            if self.position > 0:
                revenue = self.position * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.trades.append({
                    "type": "CLOSE",
                    "price": price,
                    "quantity": self.position,
                    "step": self.current_step,
                })
                self.position = 0

        # 推进
        self.current_step += 1

        # 计算奖励和状态
        new_state = self._get_state()

        # 组合价值
        portfolio_value = self.balance + self.position * price
        portfolio_values = self.portfolio_values + [portfolio_value]
        self.portfolio_values.append(portfolio_value)

        # 奖励: 收益率变化
        if len(portfolio_values) > 1:
            reward = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
        else:
            reward = 0

        # 奖励: 对数收益
        reward = np.log(portfolio_value / self.portfolio_values[-2]) if len(self.portfolio_values) > 1 else 0

        # 结束条件
        done = self.current_step >= len(self.prices) - 1

        info = {
            "portfolio_value": portfolio_value,
            "position": self.position,
            "balance": self.balance,
            "trades": len(self.trades),
        }

        return new_state, Reward(reward=reward, done=done, info=info), done

    def _get_state(self) -> State:
        """获取当前状态"""
        price = self.prices[self.current_step]

        # 最近价格
        window = min(20, self.current_step + 1)
        recent_prices = self.prices[self.current_step - window + 1:self.current_step + 1]

        # 计算 RSI
        rsi = self._calculate_rsi(recent_prices)

        # 计算 MACD
        ema_fast = np.mean(recent_prices[-12:]) if len(recent_prices) >= 12 else recent_prices[-1]
        ema_slow = np.mean(recent_prices[-26:]) if len(recent_prices) >= 26 else recent_prices[-1]
        macd = ema_fast - ema_slow

        # 布林带位置
        bb_middle = np.mean(recent_prices)
        bb_std = np.std(recent_prices)
        bb_position = (price - bb_middle) / (bb_std * 2 + 0.001)

        return State(
            prices=recent_prices,
            portfolio_value=self.balance + self.position * price,
            position=self.position,
            cash=self.balance,
            rsi=rsi,
            macd=macd,
            bollinger_position=bb_position,
        )

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> float:
        """计算 RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


# ============ 训练 pipeline ============

class RLTrainer:
    """RL 训练器"""

    def __init__(
        self,
        policy: RLPolic,
        env: TradingEnv,
        eval_env: TradingEnv | None = None,
    ):
        self.policy = policy
        self.env = env
        self.eval_env = eval_env or env

        self.episode_rewards: list[float] = []
        self.best_reward = float("-inf")

    def train(
        self,
        n_episodes: int = 100,
        eval_interval: int = 10,
        verbose: bool = True,
    ) -> dict:
        """训练"""
        results = {
            "episode_rewards": [],
            "eval_rewards": [],
            "best_model_path": None,
        }

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.policy.predict(state)
                next_state, reward, done = self.env.step(action)

                self.policy.update(state, action, reward, next_state)

                total_reward += reward.reward
                state = next_state

                if done.done:
                    break

            self.episode_rewards.append(total_reward)

            # 评估
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate()
                results["eval_rewards"].append(eval_reward)

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = f"model_best_{episode}.json"
                    self.policy.save(model_path)
                    results["best_model_path"] = model_path

                if verbose:
                    print(f"Episode {episode + 1}/{n_episodes} | "
                          f"Train: {total_reward:.4f} | Eval: {eval_reward:.4f}")

            results["episode_rewards"].append(total_reward)

        return results

    def evaluate(self, n_episodes: int = 5) -> float:
        """评估"""
        total_rewards = []

        for _ in range(n_episodes):
            state = self.eval_env.reset()
            total_reward = 0.0

            while True:
                action = self.policy.predict(state)
                next_state, reward, done = self.eval_env.step(action)

                total_reward += reward.reward
                state = next_state

                if done.done:
                    break

            total_rewards.append(total_reward)

        return np.mean(total_rewards)

    def get_stats(self) -> dict:
        """获取统计"""
        return {
            "mean_train_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            "best_reward": self.best_reward,
            "total_episodes": len(self.episode_rewards),
        }


# ============ 便捷函数 ============

def create_momentum_env(
    prices: list[float],
    initial_balance: float = 10000.0,
) -> TradingEnv:
    """创建动量交易环境"""
    return TradingEnv(prices=prices, initial_balance=initial_balance)


def quick_train_rl(
    prices: list[float],
    n_episodes: int = 100,
    policy_type: str = "epsilon_greedy",
) -> tuple[RLPolic, dict]:
    """快速训练 RL 策略"""
    import numpy as np

    # 准备数据
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    eval_prices = prices[train_size:]

    # 创建环境和策略
    env = create_momentum_env(train_prices)
    eval_env = create_momentum_env(eval_prices)

    if policy_type == "epsilon_greedy":
        policy = EpsilonGreedyPolicy()
    elif policy_type == "momentum":
        policy = MomentumPolicy()
    else:
        policy = RandomPolicy()

    # 训练
    trainer = RLTrainer(policy, env, eval_env)
    results = trainer.train(n_episodes=n_episodes)

    return policy, {
        **results,
        **trainer.get_stats(),
    }
