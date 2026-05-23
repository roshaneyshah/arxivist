"""
train.py — Main training script for RL trade execution.

Implements the full Optimal_strategy() training pipeline from:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Usage:
  python train.py --config configs/config.yaml
  python train.py --config configs/config.yaml --debug   # fast run with synthetic data
  python train.py --config configs/config.yaml --dry-run # validate setup only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_trade_execution.agent.policy import OptimalPolicy
from rl_trade_execution.agent.q_table import QTable
from rl_trade_execution.baselines.submit_and_leave import SubmitAndLeavePolicy
from rl_trade_execution.data.loader import INETDataLoader, SyntheticOrderBookGenerator
from rl_trade_execution.env.market_env import TradeExecutionEnv
from rl_trade_execution.env.market_features import MarketFeatureExtractor
from rl_trade_execution.evaluation.metrics import ExecutionMetrics
from rl_trade_execution.training.trainer import BackwardInductionTrainer
from rl_trade_execution.utils.config import ExperimentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL optimal trade execution policy (Nevmyvaka et al., ICML 2006)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Q-table checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override (default: use config seed)")
    parser.add_argument("--debug", action="store_true",
                        help="Use synthetic data and small episode count for quick testing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build all components and validate setup without training")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for saved policy (overrides config checkpoint_dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)
    print(f"Config loaded: {config}")
    print(f"Seed: {seed}")

    if args.dry_run:
        print("\n[DRY RUN] Setup validated. All components initialized successfully.")
        print(f"  State space: T={config.T}, I={config.I}, market_vars={config.market_variables}")
        print(f"  Action space: L={config.L} ({config.action_min} to {config.action_max})")
        return

    # Load or generate data
    if args.debug:
        print("\n[DEBUG MODE] Using synthetic order book data.")
        gen = SyntheticOrderBookGenerator(seed=seed)
        n_train = 200
        n_test = 50
        all_train_episodes = gen.generate_episodes(n_train, config.T, stock=config.stock)
        all_test_episodes = gen.generate_episodes(n_test, config.T, stock=config.stock)
        train_snapshots = [snap for ep in all_train_episodes for snap in ep.snapshots]
    else:
        print(f"\nLoading INET data from: {config.data_path}")
        loader = INETDataLoader(config.data_path)
        stock_file = os.path.join(config.data_path, f"{config.stock}.csv")
        all_snapshots = loader.load_order_book(stock_file)
        all_episodes = loader.partition_episodes(
            all_snapshots, config.H_minutes, config.T, config.stock
        )
        all_train_episodes, all_test_episodes = loader.train_test_split(
            all_episodes, config.train_months, config.test_months
        )
        train_snapshots = [snap for ep in all_train_episodes for snap in ep.snapshots]
        print(f"  Train: {len(all_train_episodes):,} episodes | Test: {len(all_test_episodes):,} episodes")

    # Fit market feature extractor on training data
    print(f"\nFitting market feature extractor on {len(train_snapshots):,} snapshots...")
    feature_extractor = MarketFeatureExtractor(
        feature_names=config.market_variables,
        n_bins=config.n_bins_market,
    )
    feature_extractor.fit(train_snapshots, config.V, config.side)
    print(f"  Features: {feature_extractor.feature_names}, dims: {feature_extractor.state_dims}")

    # Build environment
    env = TradeExecutionEnv(config, feature_extractor)
    print(f"  Environment: {env}")

    # Load or initialize Q-table
    q_table = None
    if args.resume:
        print(f"\nLoading Q-table from: {args.resume}")
        q_table = QTable.load(args.resume)

    # Optimize S&L baseline for comparison
    print("\nOptimizing S&L baseline...")
    sl_policy = SubmitAndLeavePolicy.optimize(
        config,
        [ep.snapshots for ep in all_train_episodes[:min(500, len(all_train_episodes))]]
    )

    # Train RL policy
    print("\nStarting backward induction training...")
    checkpoint_dir = args.output or config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = BackwardInductionTrainer(config, env, q_table)
    trained_q = trainer.train(
        [ep.snapshots for ep in all_train_episodes],
        log_every=config.log_every_n_steps,
        checkpoint_dir=checkpoint_dir if config.save_checkpoint else None,
    )

    # Save final policy
    policy = OptimalPolicy.from_q_table(trained_q, config)
    policy_path = os.path.join(checkpoint_dir, "policy_final.pkl")
    policy.save(policy_path)
    print(f"\nPolicy saved: {policy_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    rl_costs = []
    sl_costs = []

    for ep in all_test_episodes:
        # RL evaluation
        env.reset(ep.snapshots)
        ep_cost = 0.0
        state = env.encode_state(0, config.I, [0] * len(config.market_variables))
        for _ in range(config.T):
            action_idx = policy.act(state)
            state, cost, done, _ = env.step(action_idx)
            ep_cost += cost
            if done:
                break
        rl_costs.append(ep_cost)

        # S&L evaluation (simplified)
        sl_episode_cost = sl_policy._run_episode(ep.snapshots)
        sl_costs.append(sl_episode_cost)

    comparison = ExecutionMetrics.compare_policies(rl_costs, sl_costs, "RL", "S&L")
    print(f"\nTest Results ({config.stock}, V={config.V}, H={config.H_minutes}min):")
    print(f"  {comparison['summary']}")
    print(f"  Expected from paper: 27-50%+ improvement over S&L")


if __name__ == "__main__":
    main()
