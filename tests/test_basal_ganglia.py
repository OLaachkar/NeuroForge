import numpy as np

from neural.systems.basal_ganglia import BasalGangliaSystem


def test_basal_ganglia_selects_action_and_gates():
    bg = BasalGangliaSystem(num_actions=3, motor_output_size=6, seed=1)
    cortex = np.ones(10, dtype=float)
    action, probs = bg.select_action(cortex, {"dopamine": 0.5, "norepinephrine": 0.5, "acetylcholine": 0.5})

    assert 0 <= action < 3
    assert probs.shape == (3,)

    motor = np.ones(6, dtype=float)
    gated = bg.apply_gate(motor, {"dopamine": 0.8})
    assert gated.shape == motor.shape
    assert np.max(gated) == 1.0
    assert np.min(gated) <= 1.0
