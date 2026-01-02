from neural.engines.artificial_brain import ArtificialBrain


def test_self_model_report_contains_label():
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

    brain.step(experience={"reward": 1.0})
    report = brain.get_phenomenology_report()

    assert "Internally generated state report (simulation)" in report
    state = brain.get_brain_state()
    assert "self_model_state" in state
