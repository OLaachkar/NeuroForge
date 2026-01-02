from neural.engines.artificial_brain import ArtificialBrain


def test_synapse_updated_once_per_step():
    brain = ArtificialBrain(
        num_cortical_neurons=6,
        num_hippocampal_neurons=4,
        num_thalamic_neurons=3,
        simulation_dt=1.0,
        seed=1,
        noise_std=0.0,
        tonic_current=0.0,
        enable_homeostasis=False,
    )

    synapses = list(brain.iter_synapses())
    assert synapses

    counts = {id(synapse): 0 for synapse in synapses}
    for synapse in synapses:
        original = synapse.update

        def make_wrapper(syn, orig):
            def _wrapped(dt, t_ms):
                counts[id(syn)] += 1
                return orig(dt, t_ms)

            return _wrapped

        synapse.update = make_wrapper(synapse, original)

    brain.step()

    assert all(count == 1 for count in counts.values())
