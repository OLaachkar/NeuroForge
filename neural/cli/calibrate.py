"""
Calibration harness to tune tonic input toward target firing rates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np

from neural.config.brain_config import load_brain_config, create_brain_from_config


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default_brain.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate firing rates via tonic input.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to brain config JSON.",
    )
    parser.add_argument("--duration-ms", type=float, default=1000.0, help="Duration per iteration.")
    parser.add_argument("--iterations", type=int, default=5, help="Calibration iterations.")
    parser.add_argument("--gain", type=float, default=0.02, help="Tonic current gain per Hz error.")
    parser.add_argument("--tonic-min", type=float, default=0.0, help="Minimum tonic current.")
    parser.add_argument("--tonic-max", type=float, default=1.0, help="Maximum tonic current.")
    return parser.parse_args()


def _evaluate(config: Dict[str, Any], duration_ms: float) -> Dict[str, float]:
    brain = create_brain_from_config(config)
    counts = {name: 0 for name in brain.regions}
    steps = int(duration_ms / brain.simulation_dt)

    for _ in range(steps):
        brain.step()
        for name, region in brain.regions.items():
            counts[name] += sum(1 for n in region.neurons if n.state.just_spiked)

    duration_s = duration_ms / 1000.0
    rates = {}
    for name, region in brain.regions.items():
        rates[name] = counts[name] / duration_s / max(1, len(region.neurons))
    return rates


def calibrate(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    simulation = config.setdefault("simulation", {})
    regions = config.setdefault("regions", {})

    for iteration in range(args.iterations):
        rates = _evaluate(config, args.duration_ms)
        print(f"\nIteration {iteration + 1}/{args.iterations}")

        for name, rate in rates.items():
            region_cfg = regions.setdefault(name, {})
            target = region_cfg.get(
                "homeostasis_target_hz",
                simulation.get("homeostasis_target_hz", 5.0),
            )
            current = region_cfg.get("tonic_current", simulation.get("tonic_current", 0.2))
            error = target - rate
            updated = float(np.clip(current + args.gain * error, args.tonic_min, args.tonic_max))
            region_cfg["tonic_current"] = updated
            print(
                f"  {name}: rate={rate:.2f} Hz target={target:.2f} Hz "
                f"tonic={current:.3f} -> {updated:.3f}"
            )

    return config


def main() -> None:
    args = parse_args()
    config = load_brain_config(args.config)
    calibrate(config, args)


if __name__ == "__main__":
    main()
