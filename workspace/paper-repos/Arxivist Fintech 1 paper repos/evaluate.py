"""
evaluate.py — Evaluate a saved policy on test data and produce results.

Usage:
  python evaluate.py --policy models/policy_final.pkl --config configs/config.yaml
  python evaluate.py --policy models/policy_final.pkl --debug
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from rl_trade_execution.agent.policy import OptimalPolicy
from rl_trade_execution.baselines.market_order import MarketOrderPolicy
from rl_trade_execution.baselines.submit_and_leave import SubmitAndLeavePolicy
from rl_trade_execution.data.loader import INETDataLoader, SyntheticOrderBookGenerator
from rl_trade_execution.env.market_env import TradeExecutionEnv
from rl_trade_execution.env.market_features import MarketFeatureExtractor
from rl_trade_execution.evaluation.metrics import ExecutionMetrics
from rl_trade_execution.utils.config import ExperimentConfig, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RL trade execution policy")
    parser.add_argument("--policy", type=str, required=True, help="Path to saved policy .pkl")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--debug", action="store_true", help="Use synthetic data")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.seed)

    print(f"Loading policy: {args.policy}")
    policy = OptimalPolicy.load(args.policy)

    if args.debug:
        gen = SyntheticOrderBookGenerator(seed=config.seed)
        test_episodes = gen.generate_episodes(100, config.T, config.stock)
        train_snapshots = gen.generate_episodes(200, config.T, config.stock)
        train_snapshots = [s for ep in train_snapshots for s in ep.snapshots]
    else:
        loader = INETDataLoader(config.data_path)
        stock_file = os.path.join(config.data_path, f"{config.stock}.csv")
        all_snapshots = loader.load_order_book(stock_file)
        all_episodes = loader.partition_episodes(all_snapshots, config.H_minutes, config.T, config.stock)
        _, test_episodes = loader.train_test_split(all_episodes, config.train_months, config.test_months)
        train_snapshots = [s for ep, _ in [(e, None) for e in all_episodes[:int(len(all_episodes)*0.67)]] for s in ep.snapshots]

    feature_extractor = MarketFeatureExtractor(config.market_variables, config.n_bins_market)
    feature_extractor.fit(train_snapshots, config.V, config.side)
    env = TradeExecutionEnv(config, feature_extractor)

    sl_policy = SubmitAndLeavePolicy.optimize(config, [ep.snapshots for ep in test_episodes[:100]])
    mo_policy = MarketOrderPolicy(config)

    rl_costs, sl_costs, mo_costs = [], [], []

    for ep in test_episodes:
        # RL
        env.reset(ep.snapshots)
        cost = 0.0
        state = env.encode_state(0, config.I, [0] * len(config.market_variables))
        for _ in range(config.T):
            action_idx = policy.act(state)
            state, c, done, _ = env.step(action_idx)
            cost += c
            if done:
                break
        rl_costs.append(cost)
        sl_costs.append(sl_policy._run_episode(ep.snapshots))
        mo_costs.append(ep.snapshots[0].market_order_cost_bps(config.V, config.side))

    rl_vs_sl = ExecutionMetrics.compare_policies(rl_costs, sl_costs, "RL", "S&L")
    rl_vs_mo = ExecutionMetrics.compare_policies(rl_costs, mo_costs, "RL", "MarketOrder")

    print(f"\nResults ({config.stock}, V={config.V}, H={config.H_minutes}min):")
    print(f"  {rl_vs_sl['summary']}")
    print(f"  {rl_vs_mo['summary']}")

    results = {
        "config": str(config),
        "n_test_episodes": len(test_episodes),
        "rl_vs_sl": rl_vs_sl,
        "rl_vs_market_order": rl_vs_mo,
        "rl_stats": ExecutionMetrics.aggregate_episodes(rl_costs),
        "sl_stats": ExecutionMetrics.aggregate_episodes(sl_costs),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
