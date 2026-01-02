"""
Cortical column model with layered neuron populations and motifs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

from .brain_region import BrainRegion
from ..core.neuron import BiologicalNeuron, NeuronType
from ..core.synapse import Synapse


DEFAULT_LAYER_SPECS = [
    {"name": "L2/3", "ratio": 0.35, "excitatory_ratio": 0.8},
    {"name": "L4", "ratio": 0.2, "excitatory_ratio": 0.85},
    {"name": "L5", "ratio": 0.3, "excitatory_ratio": 0.75},
    {"name": "L6", "ratio": 0.15, "excitatory_ratio": 0.7},
]

DEFAULT_LAYER_CONNECTIVITY: Dict[Tuple[str, str], float] = {
    ("L4", "L2/3"): 0.10,
    ("L2/3", "L5"): 0.08,
    ("L5", "L6"): 0.05,
    ("L6", "L4"): 0.05,
    ("L2/3", "L2/3"): 0.05,
    ("L4", "L4"): 0.04,
    ("L5", "L5"): 0.04,
    ("L6", "L6"): 0.03,
}


class CorticalColumn(BrainRegion):
    """Cortical column with layered structure and layer-specific connectivity."""

    def __init__(
        self,
        region_name: str,
        num_neurons: int,
        excitatory_ratio: float = 0.8,
        connectivity_density: float = 0.1,
        region_type: str = "cortical_column",
        layer_specs: Optional[List[Dict[str, float]]] = None,
        layer_connectivity: Optional[Dict[Tuple[str, str], float]] = None,
        layer_connectivity_scale: float = 1.0,
        **kwargs,
    ):
        self.layer_specs = layer_specs or list(DEFAULT_LAYER_SPECS)
        self.layer_connectivity = layer_connectivity or dict(DEFAULT_LAYER_CONNECTIVITY)
        self.layer_connectivity_scale = layer_connectivity_scale
        self.layer_indices: Dict[str, List[int]] = {}
        super().__init__(
            region_name=region_name,
            num_neurons=num_neurons,
            excitatory_ratio=excitatory_ratio,
            connectivity_density=connectivity_density,
            region_type=region_type,
            **kwargs,
        )

    def _create_neurons(self):
        """Create neurons with layer assignments."""
        self.neurons = []
        self.layer_indices = {}

        allocated = 0
        for i, layer in enumerate(self.layer_specs):
            ratio = float(layer.get("ratio", 0.0))
            layer_count = int(round(self.num_neurons * ratio))
            if i == len(self.layer_specs) - 1:
                layer_count = self.num_neurons - allocated
            allocated += layer_count

            layer_name = str(layer.get("name", f"layer_{i}"))
            excit_ratio = float(layer.get("excitatory_ratio", self.excitatory_ratio))
            num_excitatory = int(layer_count * excit_ratio)

            start_idx = len(self.neurons)
            for j in range(layer_count):
                neuron_type = NeuronType.EXCITATORY if j < num_excitatory else NeuronType.INHIBITORY
                neuron = BiologicalNeuron(
                    neuron_id=len(self.neurons),
                    neuron_type=neuron_type,
                    neuron_model=self.neuron_model,
                    izhikevich_params=self.izhikevich_params,
                )
                neuron.layer = layer_name
                self.neurons.append(neuron)

            end_idx = len(self.neurons)
            self.layer_indices[layer_name] = list(range(start_idx, end_idx))

    def _create_connections(self, density: float):
        """Create connections using layer motifs."""
        self.synapses = []

        for (pre_layer, post_layer), base_density in self.layer_connectivity.items():
            pre_indices = self.layer_indices.get(pre_layer, [])
            post_indices = self.layer_indices.get(post_layer, [])
            if not pre_indices or not post_indices:
                continue

            effective_density = base_density * self.layer_connectivity_scale
            num_connections = int(len(pre_indices) * len(post_indices) * effective_density)
            if num_connections <= 0:
                continue

            for _ in range(num_connections):
                pre_idx = int(np.random.choice(pre_indices))
                post_idx = int(np.random.choice(post_indices))
                if pre_idx == post_idx:
                    continue

                pre_neuron = self.neurons[pre_idx]
                post_neuron = self.neurons[post_idx]

                if pre_neuron.neuron_type == NeuronType.EXCITATORY:
                    initial_weight = float(np.random.uniform(0.3, 0.7))
                else:
                    initial_weight = float(np.random.uniform(0.5, 1.0))

                synapse = Synapse(pre_neuron, post_neuron, initial_weight=initial_weight)
                self.synapses.append(synapse)
                pre_neuron.connect_to(post_neuron, synapse)
