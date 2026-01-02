from neural.engines.artificial_brain import ArtificialBrain


def test_neuromodulation_keeps_base_rates():
    brain = ArtificialBrain(
        num_cortical_neurons=5,
        num_hippocampal_neurons=3,
        num_thalamic_neurons=2,
        simulation_dt=1.0,
        seed=2,
        noise_std=0.0,
        tonic_current=0.0,
        enable_homeostasis=False,
    )

    synapse = next(brain.iter_synapses())
    base_ltp = synapse.a_ltp_base
    base_ltd = synapse.a_ltd_base

    for _ in range(5):
        brain.neurotransmitters.release("dopamine", 0.2)
        brain.neurotransmitters.modulate_region(brain.regions["cortex"])

    assert synapse.a_ltp_base == base_ltp
    assert synapse.a_ltd_base == base_ltd
