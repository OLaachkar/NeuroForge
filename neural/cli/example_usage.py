"""
Quick example showing how to use the artificial brain.
"""

import argparse
from typing import Optional

import numpy as np
from neural.engines.artificial_brain import ArtificialBrain


def main(seed: Optional[int] = None):
    print("Creating a small artificial brain...")
    
    brain = ArtificialBrain(
        num_cortical_neurons=500,
        num_hippocampal_neurons=200,
        num_thalamic_neurons=100,
        simulation_dt=0.1,
        seed=seed,
    )
    
    print("\n1. Running brain with no input (spontaneous activity)...")
    brain.run(50.0)  # Run for 50ms
    brain.print_state()
    
    print("\n2. Adding sensory input and running...")
    visual_pattern = np.random.rand(50) * 0.5
    brain.add_sensory_input('vision', visual_pattern)
    brain.run(100.0)
    brain.print_state()
    
    print("\n3. Learning a pattern...")
    pattern = np.zeros(brain.regions['hippocampus'].num_neurons)
    pattern[:20] = 1.0  # First 20 neurons active
    pattern_id = brain.learn_pattern(pattern)
    print(f"Pattern encoded with ID: {pattern_id}")
    brain.run(50.0)
    
    print("\n4. Recalling the pattern...")
    partial = np.zeros(brain.regions['cortex'].num_neurons)
    partial[:10] = 1.0
    recalled = brain.recall_pattern(partial)
    
    if recalled is not None:
        print("Pattern recalled successfully!")
        original_expanded = np.zeros(brain.regions['cortex'].num_neurons)
        original_expanded[:20] = 1.0
        similarity = np.dot(recalled, original_expanded) / (
            np.linalg.norm(recalled) * np.linalg.norm(original_expanded) + 1e-10
        )
        print(f"Similarity to original: {similarity:.2%}")
    else:
        print("Pattern not recalled (may need more learning time)")
    
    print("\n5. Reward-based learning...")
    for i in range(5):
        brain.step(external_reward=0.5 if i % 2 == 0 else 0.0)
        print(f"  Step {i}: Dopamine = {brain.neurotransmitters.levels['dopamine']:.3f}")
    
    print("\n6. Final brain state:")
    brain.print_state()
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nThe brain demonstrates:")
    print("  - Spontaneous neural activity")
    print("  - Sensory input processing")
    print("  - Pattern learning and memory")
    print("  - Pattern recall from partial cues")
    print("  - Reward-based learning (dopamine)")
    print("\nTo scale this further, see README.md (Scaling section).")


def parse_args():
    parser = argparse.ArgumentParser(description="Example artificial brain usage")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(seed=args.seed)

