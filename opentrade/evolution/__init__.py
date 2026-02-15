"""
OpenTrade Evolution Module

GA + RL Strategy Evolution:
- ga.py: Genetic Algorithm for parameter optimization
- rl.py: Reinforcement Learning for policy training
"""

from opentrade.evolution.ga import (
    GeneticAlgorithm,
    StrategyGenome,
    Gene,
    GeneType,
    FitnessEvaluator,
    FitnessResult,
    StrategyReplay,
    create_ga_optimizer,
    quick_optimize,
)

from opentrade.evolution.rl import (
    RLPolic,
    RandomPolicy,
    MomentumPolicy,
    EpsilonGreedyPolicy,
    TradingEnv,
    State,
    Action,
    ActionType,
    RLTrainer,
    quick_train_rl,
)

__all__ = [
    # GA
    "GeneticAlgorithm",
    "StrategyGenome",
    "Gene",
    "GeneType",
    "FitnessEvaluator",
    "FitnessResult",
    "StrategyReplay",
    "create_ga_optimizer",
    "quick_optimize",
    # RL
    "RLPolic",
    "RandomPolicy",
    "MomentumPolicy",
    "EpsilonGreedyPolicy",
    "TradingEnv",
    "State",
    "Action",
    "ActionType",
    "RLTrainer",
    "quick_train_rl",
]
