from neural.core.neuron import BiologicalNeuron


def test_izhikevich_spikes_with_current():
    neuron = BiologicalNeuron(
        neuron_id=0,
        neuron_model="izhikevich",
        izhikevich_params={"input_scale": 80.0},
    )
    dt = 1.0
    fired = False
    for step in range(20):
        fired = neuron.update(dt, external_current=0.5, t_ms=float(step))
        if fired:
            break
    assert fired
