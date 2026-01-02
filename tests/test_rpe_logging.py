from neural.engines.artificial_brain import ArtificialBrain


def test_rpe_logging_updates_with_reward():
    brain = ArtificialBrain(
        num_cortical_neurons=5,
        num_hippocampal_neurons=3,
        num_thalamic_neurons=2,
        simulation_dt=1.0,
        seed=1,
        noise_std=0.0,
        tonic_current=0.0,
        enable_homeostasis=False,
    )

    brain.step(experience={"reward": 0.5})
    state = brain.get_brain_state()

    assert state["last_rpe"] > 0.0
    assert state["rpe_history"]
