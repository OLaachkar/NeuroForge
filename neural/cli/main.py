"""
Main script to run the artificial brain simulation.
Demonstrates the brain's capabilities: learning, memory, pattern recognition.
"""

import argparse
from typing import Optional

import numpy as np
from neural.engines.artificial_brain import ArtificialBrain


def demonstrate_learning(seed: Optional[int] = None):
    """Demonstrate the brain's learning capabilities"""
    print("=" * 60)
    print("ARTIFICIAL BIOLOGICAL BRAIN SIMULATION")
    print("=" * 60)
    
    brain = ArtificialBrain(
        num_cortical_neurons=500,
        num_hippocampal_neurons=200,
        num_thalamic_neurons=100,
        simulation_dt=0.1,
        seed=seed,
    )
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION 1: Basic Brain Activity")
    print("=" * 60)
    
    print("\nRunning brain for 100ms (spontaneous activity)...")
    brain.run(100.0)
    brain.print_state()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: Sensory Input Processing")
    print("=" * 60)
    
    visual_input = np.random.rand(50) * 0.5  # Random visual pattern
    brain.add_sensory_input('vision', visual_input)
    print("\nAdded visual input (50 neurons)")
    
    print("Running brain for 200ms with sensory input...")
    brain.run(200.0)
    brain.print_state()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: Pattern Learning and Memory")
    print("=" * 60)
    
    patterns = [
        np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1] + [0] * 40),  # Pattern A
        np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 0] + [0] * 40),  # Pattern B
        np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 0] + [0] * 40),  # Pattern C
    ]
    
    print("\nLearning 3 patterns...")
    pattern_ids = []
    for i, pattern in enumerate(patterns):
        full_pattern = np.zeros(brain.regions['hippocampus'].num_neurons)
        pattern_size = min(len(pattern), len(full_pattern))
        full_pattern[:pattern_size] = pattern[:pattern_size]
        
        pattern_id = brain.learn_pattern(full_pattern)
        pattern_ids.append(pattern_id)
        print(f"  Pattern {i+1} encoded (ID: {pattern_id})")
        
        brain.run(50.0)
    
    brain.print_state()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION 4: Pattern Recall")
    print("=" * 60)
    
    print("\nTesting pattern recall from partial cues...")
    for i, pattern in enumerate(patterns):
        partial = pattern.copy()
        mask = np.random.random(len(partial)) > 0.5
        partial[mask] = 0
        
        full_partial = np.zeros(brain.regions['cortex'].num_neurons)
        pattern_size = min(len(partial), len(full_partial))
        full_partial[:pattern_size] = partial[:pattern_size]
        
        recalled = brain.recall_pattern(full_partial)
        
        if recalled is not None:
            original_expanded = np.zeros(brain.regions['cortex'].num_neurons)
            original_expanded[:pattern_size] = pattern[:pattern_size]
            similarity = np.dot(recalled, original_expanded) / (
                np.linalg.norm(recalled) * np.linalg.norm(original_expanded) + 1e-10
            )
            print(f"  Pattern {i+1}: Recalled with {similarity:.2%} similarity")
        else:
            print(f"  Pattern {i+1}: Could not recall")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION 5: Reward-Based Learning")
    print("=" * 60)
    
    print("\nSimulating reward-based learning...")
    for step in range(10):
        brain.add_sensory_input('vision', np.random.rand(50) * 0.3)
        
        reward = 0.5 if step % 3 == 0 else 0.0
        brain.step(external_reward=reward)
        
        if reward > 0:
            print(f"  Step {step}: Reward given (dopamine: {brain.neurotransmitters.levels['dopamine']:.3f})")
    
    brain.print_state()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nThis artificial brain demonstrates:")
    print("  - Biological neuron dynamics (spiking, refractory periods)")
    print("  - Synaptic plasticity (STDP, Hebbian learning)")
    print("  - Brain regions with specialized functions")
    print("  - Memory formation and recall")
    print("  - Neurotransmitter systems (dopamine, etc.)")
    print("  - Sensory processing and motor output")
    print("\nNote: This is a simplified model. A full human brain simulation")
    print("would require billions of neurons and trillions of synapses.")


def interactive_mode(seed: Optional[int] = None):
    """Interactive mode for exploring the brain"""
    print("=" * 60)
    print("INTERACTIVE BRAIN EXPLORATION")
    print("=" * 60)
    
    brain = ArtificialBrain(
        num_cortical_neurons=200,
        num_hippocampal_neurons=100,
        num_thalamic_neurons=50,
        seed=seed,
    )
    
    print("\nCommands:")
    print("  'run <ms>' - Run simulation for X milliseconds")
    print("  'input <values>' - Add sensory input (comma-separated)")
    print("  'learn <pattern>' - Learn a pattern")
    print("  'recall <partial>' - Recall from partial pattern")
    print("  'reward <value>' - Give reward signal")
    print("  'state' - Show brain state")
    print("  'quit' - Exit")
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'run' and len(cmd) > 1:
                duration = float(cmd[1])
                brain.run(duration)
                print(f"Ran for {duration}ms")
            elif cmd[0] == 'state':
                brain.print_state()
            elif cmd[0] == 'reward' and len(cmd) > 1:
                reward = float(cmd[1])
                brain.step(external_reward=reward)
                print(f"Reward: {reward}, Dopamine: {brain.neurotransmitters.levels['dopamine']:.3f}")
            else:
                print("Unknown command or missing arguments")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Artificial brain simulation")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["demo", "interactive"],
        default="demo",
        help="Run demo (default) or interactive mode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "interactive":
        interactive_mode(seed=args.seed)
    else:
        demonstrate_learning(seed=args.seed)

