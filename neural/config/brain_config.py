"""
Utilities for loading brain configuration files and constructing a brain.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from ..engines.artificial_brain import ArtificialBrain


def load_brain_config(path: str) -> Dict[str, Any]:
    """Load brain configuration from JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_brain_from_config(config_or_path: Union[str, Dict[str, Any]]) -> ArtificialBrain:
    """Create an ArtificialBrain from a config dict or path."""
    config = load_brain_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    simulation = config.get("simulation", {})
    regions = config.get("regions", {})
    neuromodulation = config.get("neuromodulation", {})
    basal_ganglia = config.get("basal_ganglia", {})
    memory_cfg = config.get("memory", {})

    cortex_cfg = dict(regions.get("cortex", {}))
    hippocampus_cfg = dict(regions.get("hippocampus", {}))
    thalamus_cfg = dict(regions.get("thalamus", {}))

    num_cortical = int(cortex_cfg.pop("num_neurons", 1000))
    num_hippocampal = int(hippocampus_cfg.pop("num_neurons", 500))
    num_thalamic = int(thalamus_cfg.pop("num_neurons", 300))

    region_params = {
        "cortex": cortex_cfg,
        "hippocampus": hippocampus_cfg,
        "thalamus": thalamus_cfg,
    }

    return ArtificialBrain(
        num_cortical_neurons=num_cortical,
        num_hippocampal_neurons=num_hippocampal,
        num_thalamic_neurons=num_thalamic,
        simulation_dt=simulation.get("dt", 0.1),
        seed=config.get("seed"),
        tonic_current=simulation.get("tonic_current", 0.2),
        noise_std=simulation.get("noise_std", 0.02),
        enable_homeostasis=simulation.get("enable_homeostasis", True),
        homeostasis_target_hz=simulation.get("homeostasis_target_hz", 5.0),
        homeostasis_tau_ms=simulation.get("homeostasis_tau_ms", 1000.0),
        homeostasis_gain=simulation.get("homeostasis_gain", 0.001),
        homeostasis_offset_limit=simulation.get("homeostasis_offset_limit", 15.0),
        region_params=region_params,
        neuromodulation_config=neuromodulation,
        basal_ganglia_config=basal_ganglia,
        memory_config=memory_cfg,
    )
