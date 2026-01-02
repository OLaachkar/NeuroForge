from neural.core.neuron import BiologicalNeuron


def test_refractory_ends():
    neuron = BiologicalNeuron(neuron_id=0, refractory_period=2.0)
    dt = 1.0

    fired = neuron.update(dt, external_current=5.0, t_ms=0.0)
    assert fired
    assert neuron.state.refractory_remaining_ms > 0.0

    fired = neuron.update(dt, external_current=5.0, t_ms=1.0)
    assert not fired

    fired = neuron.update(dt, external_current=5.0, t_ms=2.0)
    assert fired


def test_just_spiked_flag():
    neuron = BiologicalNeuron(neuron_id=1, refractory_period=2.0)
    dt = 1.0

    neuron.update(dt, external_current=0.0, t_ms=0.0)
    assert not neuron.state.just_spiked

    neuron.update(dt, external_current=5.0, t_ms=1.0)
    assert neuron.state.just_spiked

    neuron.update(dt, external_current=0.0, t_ms=2.0)
    assert not neuron.state.just_spiked
