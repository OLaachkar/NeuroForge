"""
Unified runner for object-model or scaled brain backends.
"""

from __future__ import annotations

import argparse

from neural.engines.artificial_brain import ArtificialBrain
from neural.config.brain_config import create_brain_from_config
from neural.engines.scaled_brain import ScaledArtificialBrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an artificial brain backend.")
    parser.add_argument("--backend", choices=["object", "scaled"], default="object")
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--config", default=None, help="Config JSON (object backend only).")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for scaled backend.")
    parser.add_argument("--cortex", type=int, default=1000)
    parser.add_argument("--hippocampus", type=int, default=500)
    parser.add_argument("--thalamus", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == "object":
        if args.config:
            brain = create_brain_from_config(args.config)
        else:
            brain = ArtificialBrain(
                num_cortical_neurons=args.cortex,
                num_hippocampal_neurons=args.hippocampus,
                num_thalamic_neurons=args.thalamus,
            )
        brain.run(args.duration_ms)
        brain.print_state()
        return

    brain = ScaledArtificialBrain(
        num_cortical_neurons=args.cortex,
        num_hippocampal_neurons=args.hippocampus,
        num_thalamic_neurons=args.thalamus,
        use_gpu=args.use_gpu,
    )
    brain.run(args.duration_ms)
    brain.print_state()


if __name__ == "__main__":
    main()
