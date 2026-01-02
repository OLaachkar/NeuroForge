"""
Memory System
Implements biological memory mechanisms:
- Working memory (short-term)
- Long-term memory (LTP-based)
- Memory consolidation
- Pattern completion and recall
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from ..regions.brain_region import BrainRegion
from ..core.synapse import Synapse


class MemoryTrace:
    """Represents a memory trace in the system"""
    
    def __init__(self, pattern: np.ndarray, strength: float = 1.0):
        self.pattern = pattern
        self.strength = strength
        self.age = 0.0
        self.access_count = 0
    
    def decay(self, dt: float, decay_rate: float = 0.001):
        """Decay memory strength over time"""
        self.strength *= np.exp(-decay_rate * dt)
        self.age += dt
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen memory through rehearsal"""
        self.strength = min(1.0, self.strength + amount)
        self.access_count += 1


class MemorySystem:
    """
    Biological memory system with working and long-term memory.
    Uses Hebbian plasticity, pattern completion, and replay cycles.
    """
    
    def __init__(
        self,
        hippocampus: BrainRegion,
        cortex: BrainRegion,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.hippocampus = hippocampus  # Fast learning, temporary storage
        self.cortex = cortex  # Slow learning, permanent storage
        self.rng = np.random.default_rng(seed)
        config = config or {}
        
        self.working_memory: List[MemoryTrace] = []  # Short-term (seconds to minutes)
        self.long_term_memory: List[MemoryTrace] = []  # Long-term (hours to lifetime)
        
        self.consolidation_threshold = float(config.get("consolidation_threshold", 0.7))
        self.working_memory_capacity = int(config.get("working_memory_capacity", 7))

        self.replay_enabled = bool(config.get("replay_enabled", True))
        self.replay_interval_ms = float(config.get("replay_interval_ms", 500.0))
        self.replay_batch_size = int(config.get("replay_batch_size", 2))
        self.replay_strength = float(config.get("replay_strength", 0.1))
        self.replay_dropout = float(config.get("replay_dropout", 0.3))
        self.replay_min_strength = float(config.get("replay_min_strength", 0.2))
        self.replay_dropout = float(np.clip(self.replay_dropout, 0.0, 1.0))
        self.replay_timer_ms = 0.0
        self.replay_count = 0
        self.last_replay_ms = 0.0
        self.time_ms = 0.0

        self.memory_patterns: Dict[int, np.ndarray] = {}
        self.next_pattern_id = 0
    
    def encode(self, pattern: np.ndarray, region: BrainRegion = None) -> int:
        """
        Encode a pattern into memory.
        Returns pattern ID.
        """
        if region is None:
            region = self.hippocampus
        
        pattern_id = self.next_pattern_id
        self.next_pattern_id += 1
        
        trace = MemoryTrace(pattern.copy(), strength=1.0)
        self.working_memory.append(trace)
        
        if len(self.working_memory) > self.working_memory_capacity:
            self.working_memory.sort(key=lambda x: (x.strength, -x.age))
            removed = self.working_memory.pop(0)
            if removed.strength >= self.consolidation_threshold:
                self._consolidate(removed)
        
        self.memory_patterns[pattern_id] = pattern
        
        self._strengthen_pattern_synapses(region, pattern)
        
        return pattern_id
    
    def _strengthen_pattern_synapses(self, region: BrainRegion, pattern: np.ndarray):
        """Strengthen synapses corresponding to active pattern"""
        active_neurons = np.where(pattern > 0.5)[0]
        
        for neuron_idx in active_neurons:
            if neuron_idx < len(region.neurons):
                neuron = region.neurons[neuron_idx]
                for synapse in neuron.output_synapses:
                    if synapse.post_neuron in region.neurons:
                        synapse.set_weight(synapse.get_weight() * 1.1)
    
    def recall(self, partial_pattern: np.ndarray, 
              region: BrainRegion = None) -> Optional[np.ndarray]:
        """
        Recall a complete pattern from partial cue.
        Uses pattern completion.
        """
        if region is None:
            region = self.cortex
        
        best_match = None
        best_similarity = 0.0
        
        for trace in self.long_term_memory:
            if len(trace.pattern) == len(partial_pattern):
                similarity = np.dot(trace.pattern, partial_pattern) / (
                    np.linalg.norm(trace.pattern) * np.linalg.norm(partial_pattern) + 1e-10
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = trace.pattern
        
        for trace in self.working_memory:
            if len(trace.pattern) == len(partial_pattern):
                similarity = np.dot(trace.pattern, partial_pattern) / (
                    np.linalg.norm(trace.pattern) * np.linalg.norm(partial_pattern) + 1e-10
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = trace.pattern
                    trace.strengthen()  # Rehearsal strengthens memory
        
        if best_similarity > 0.3:  # Threshold for recall
            return best_match
        return None
    
    def _consolidate(self, trace: MemoryTrace):
        """Consolidate memory from hippocampus to cortex"""
        self.long_term_memory.append(MemoryTrace(trace.pattern.copy(), trace.strength))
        
        self._strengthen_pattern_synapses(self.cortex, trace.pattern)

    def _replay_once(self) -> None:
        """Replay a batch of memories to strengthen cortical traces."""
        candidates = [
            trace
            for trace in (self.working_memory + self.long_term_memory)
            if trace.strength >= self.replay_min_strength
        ]
        if not candidates:
            return

        weights = np.array(
            [trace.strength / (1.0 + trace.age) for trace in candidates],
            dtype=float,
        )
        total = float(np.sum(weights))
        if total > 0.0:
            weights = weights / total
        else:
            weights = None

        count = min(self.replay_batch_size, len(candidates))
        indices = self.rng.choice(len(candidates), size=count, replace=False, p=weights)
        for idx in indices:
            trace = candidates[int(idx)]
            pattern = trace.pattern.copy()
            if self.replay_dropout > 0.0:
                mask = self.rng.random(pattern.shape) > self.replay_dropout
                pattern = pattern * mask
            self._strengthen_pattern_synapses(self.cortex, pattern)
            trace.strengthen(self.replay_strength)

        self.replay_count += 1
        self.last_replay_ms = self.time_ms
    
    def update(self, dt: float):
        """Update memory system (decay, consolidation)"""
        self.time_ms += dt

        for trace in self.working_memory:
            trace.decay(dt, decay_rate=0.01)  # Faster decay
        
        for trace in self.long_term_memory:
            trace.decay(dt, decay_rate=0.0001)  # Very slow decay
        
        self.working_memory = [t for t in self.working_memory if t.strength > 0.1]
        self.long_term_memory = [t for t in self.long_term_memory if t.strength > 0.01]
        
        for trace in list(self.working_memory):
            if trace.strength >= self.consolidation_threshold and trace.age > 1.0:
                self._consolidate(trace)
                self.working_memory.remove(trace)

        if self.replay_enabled and self.replay_interval_ms > 0.0:
            self.replay_timer_ms += dt
            while self.replay_timer_ms >= self.replay_interval_ms:
                self.replay_timer_ms -= self.replay_interval_ms
                self._replay_once()
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about memory system"""
        return {
            'working_memory_count': len(self.working_memory),
            'long_term_memory_count': len(self.long_term_memory),
            'total_patterns': len(self.memory_patterns),
            'avg_working_strength': np.mean([t.strength for t in self.working_memory]) if self.working_memory else 0.0,
            'avg_longterm_strength': np.mean([t.strength for t in self.long_term_memory]) if self.long_term_memory else 0.0,
            'replay_count': self.replay_count,
            'last_replay_ms': self.last_replay_ms,
        }

