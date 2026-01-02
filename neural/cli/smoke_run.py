"""
Smoke-run script for the artificial brain simulation.
Runs a short simulation and prints quick diagnostics for sanity checks.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np

from neural.engines.artificial_brain import ArtificialBrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke run for the artificial brain simulation.")
    parser.add_argument("--duration-ms", type=float, default=5000.0, help="Simulation time in ms.")
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep in ms.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--cortex", type=int, default=200, help="Number of cortical neurons.")
    parser.add_argument("--hippocampus", type=int, default=100, help="Number of hippocampal neurons.")
    parser.add_argument("--thalamus", type=int, default=50, help="Number of thalamic neurons.")
    parser.add_argument("--sample-synapses", type=int, default=200, help="Number of synapses to sample.")
    parser.add_argument("--tonic-current", type=float, default=0.2, help="Baseline current per neuron.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Noise std-dev for background current.")
    parser.add_argument("--no-homeostasis", action="store_true", help="Disable homeostatic threshold control.")
    parser.add_argument("--target-hz", type=float, default=5.0, help="Homeostatic target firing rate.")
    return parser.parse_args()


def run_smoke(args: argparse.Namespace) -> None:
    if args.duration_ms <= 0 or args.dt <= 0:
        raise ValueError("duration-ms and dt must be positive.")

    brain = ArtificialBrain(
        num_cortical_neurons=args.cortex,
        num_hippocampal_neurons=args.hippocampus,
        num_thalamic_neurons=args.thalamus,
        simulation_dt=args.dt,
        seed=args.seed,
        tonic_current=args.tonic_current,
        noise_std=args.noise_std,
        enable_homeostasis=not args.no_homeostasis,
        homeostasis_target_hz=args.target_hz,
    )

    region_spikes: Dict[str, int] = {name: 0 for name in brain.regions}
    min_v = math.inf
    max_v = -math.inf

    num_steps = int(args.duration_ms / args.dt)
    for _ in range(num_steps):
        brain.step()

        for name, region in brain.regions.items():
            region_spikes[name] += sum(1 for n in region.neurons if n.state.just_spiked)
            for neuron in region.neurons:
                v = neuron.state.membrane_potential
                if v < min_v:
                    min_v = v
                if v > max_v:
                    max_v = v

    duration_s = args.duration_ms / 1000.0
    print("\n=== Smoke Run Summary ===")
    print(f"Simulated time: {args.duration_ms:.1f} ms (dt={args.dt:.3f} ms)")
    print(f"Total neurons: {brain.get_total_neurons()}")
    print(f"Total synapses: {brain.get_total_synapses()}")
    print(f"Membrane potential range: {min_v:.2f} mV to {max_v:.2f} mV")

    print("\nSpikes per region per second:")
    for name, count in region_spikes.items():
        spikes_per_s = count / duration_s
        mean_rate = spikes_per_s / len(brain.regions[name].neurons)
        print(f"  {name}: {spikes_per_s:.2f} spikes/s (mean {mean_rate:.2f} Hz)")

    synapses = list(brain.iter_synapses())
    sample = synapses[: max(1, min(args.sample_synapses, len(synapses)))]
    if sample:
        weights = np.array([s.get_weight() for s in sample])
        print("\nSampled synapse weights:")
        print(
            f"  n={len(sample)} min={weights.min():.4f} "
            f"max={weights.max():.4f} mean={weights.mean():.4f}"
        )


if __name__ == "__main__":
    run_smoke(parse_args())
