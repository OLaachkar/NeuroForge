import numpy as np

from neural.regions.brain_region import BrainRegion
from neural.systems.memory_system import MemorySystem


def test_memory_replay_strengthens_trace():
    hippocampus = BrainRegion(
        region_name="hippocampus",
        num_neurons=10,
        connectivity_density=0.0,
    )
    cortex = BrainRegion(
        region_name="cortex",
        num_neurons=10,
        connectivity_density=0.0,
    )
    memory = MemorySystem(
        hippocampus,
        cortex,
        config={
            "replay_interval_ms": 1.0,
            "replay_batch_size": 1,
            "replay_strength": 0.2,
            "replay_dropout": 0.0,
            "replay_min_strength": 0.1,
        },
        seed=1,
    )

    pattern = np.zeros(10)
    pattern[:3] = 1.0
    memory.encode(pattern, hippocampus)
    trace = memory.working_memory[0]
    trace.strength = 0.5

    memory.update(1.0)

    assert memory.replay_count >= 1
    assert trace.strength > 0.5
