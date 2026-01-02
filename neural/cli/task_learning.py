"""
Simple task-driven learning loop (sensory -> decision -> reward).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from neural.config.brain_config import create_brain_from_config, load_brain_config


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default_brain.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-driven learning demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Brain config JSON.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes.")
    parser.add_argument("--stim-ms", type=float, default=20.0, help="Stimulus duration per episode.")
    parser.add_argument("--decision-ms", type=float, default=20.0, help="Decision window per episode.")
    parser.add_argument("--rest-ms", type=float, default=10.0, help="Rest period between episodes.")
    parser.add_argument("--reward-correct", type=float, default=0.5, help="Reward for correct choice.")
    parser.add_argument("--reward-incorrect", type=float, default=-0.2, help="Reward for incorrect choice.")
    parser.add_argument("--pattern-strength", type=float, default=0.6, help="Stimulus strength.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    return parser.parse_args()


def make_stimulus(num_inputs: int, label: int, strength: float) -> np.ndarray:
    half = max(1, num_inputs // 2)
    stimulus = np.zeros(num_inputs, dtype=float)
    if label == 0:
        stimulus[:half] = strength
    else:
        stimulus[half:] = strength
    return stimulus


def run_episode(
    brain,
    label: int,
    stim_steps: int,
    decision_steps: int,
    rest_steps: int,
    strength: float,
    reward_correct: float,
    reward_incorrect: float,
) -> bool:
    thalamus_size = len(brain.regions["thalamus"].neurons)
    stimulus = make_stimulus(thalamus_size, label, strength)
    brain.add_sensory_input("task", stimulus)

    for _ in range(stim_steps):
        brain.step()

    motor_size = len(brain.motor_neurons)
    mid = max(1, motor_size // 2)
    motor_counts = np.zeros(motor_size, dtype=float)
    for _ in range(decision_steps):
        brain.step()
        motor_counts += brain.get_motor_output()

    left_score = float(np.sum(motor_counts[:mid]))
    right_score = float(np.sum(motor_counts[mid:]))
    action = 0 if left_score >= right_score else 1
    correct = action == label

    reward = reward_correct if correct else reward_incorrect
    brain.add_sensory_input("task", np.zeros(thalamus_size, dtype=float))
    brain.step(external_reward=reward)

    for _ in range(rest_steps):
        brain.step()

    return correct


def main() -> None:
    args = parse_args()
    config = load_brain_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    brain = create_brain_from_config(config)

    stim_steps = int(args.stim_ms / brain.simulation_dt)
    decision_steps = int(args.decision_ms / brain.simulation_dt)
    rest_steps = int(args.rest_ms / brain.simulation_dt)

    correct = 0
    for episode in range(1, args.episodes + 1):
        label = np.random.randint(0, 2)
        if run_episode(
            brain,
            label,
            stim_steps,
            decision_steps,
            rest_steps,
            args.pattern_strength,
            args.reward_correct,
            args.reward_incorrect,
        ):
            correct += 1

        if episode % 10 == 0 or episode == 1:
            accuracy = correct / episode
            print(f"Episode {episode:3d} | accuracy={accuracy:.2%}")


if __name__ == "__main__":
    main()
