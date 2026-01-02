from neural.engines.artificial_brain import ArtificialBrain


def test_region_params_override():
    brain = ArtificialBrain(
        num_cortical_neurons=10,
        num_hippocampal_neurons=6,
        num_thalamic_neurons=4,
        simulation_dt=1.0,
        seed=0,
        region_params={
            "cortex": {"tonic_current": 0.42, "noise_std": 0.0},
            "hippocampus": {"excitatory_ratio": 0.5},
        },
    )

    assert brain.regions["cortex"].tonic_current == 0.42
    assert brain.regions["cortex"].noise_std == 0.0
    assert brain.regions["hippocampus"].excitatory_ratio == 0.5
