import numpy as np

from neural.core.neuron import BiologicalNeuron
from neural.core.synapse import Synapse


def test_synapse_releases_only_on_spike():
    np.random.seed(0)
    pre = BiologicalNeuron(neuron_id=0)
    post = BiologicalNeuron(neuron_id=1)
    synapse = Synapse(pre, post, use_probability=1.0, initial_weight=1.0)

    synapse.state.neurotransmitter_level = 0.0
    pre.state.just_spiked = False
    pre.state.membrane_potential = pre.threshold + 5.0
    synapse.update(dt=1.0, t_ms=0.0)
    assert synapse.state.neurotransmitter_level == 0.0

    pre.state.just_spiked = True
    synapse.update(dt=1.0, t_ms=1.0)
    assert synapse.state.neurotransmitter_level > 0.0
