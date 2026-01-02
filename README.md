# Artificial Biological Brain (Neural)

This repository is a biologically inspired brain simulation written in Python.
It models spiking neurons, synaptic plasticity, neuromodulation, memory replay,
and action selection, and can optionally modulate an LLM through brain state.

This is a research simulation. It does not claim verified consciousness.

## Quick Start

```bash
pip install -r requirements.txt
```

Smoke run (sanity check):

```bash
python -m neural.cli.smoke_run --duration-ms 5000 --seed 1
```

Run the basic simulation:

```bash
python -m neural.cli.main
```

## Chat Interfaces

LLM + brain state using llama.cpp (GPU):

```bash
python -m neural.cli.chat_llama_cpp --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Optional internal state report after each response:

```bash
python -m neural.cli.chat_llama_cpp --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf --phenomenology
```

Ollama-based chat:

```bash
python -m neural.cli.chat --model llama3 --config configs/default_brain.json
```

Chat commands:
- `/help` `/state` `/phenomenology` `/reward <value>` `/remember <text>` `/memory` `/reset` `/quit`

Conversation memory is stored in `data/chat_memory.json`.

## GPU + llama.cpp Setup (Windows)

Install the CUDA wheel for llama-cpp-python (RTX 3060 friendly):

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

If you want richer affect inference (optional):

```bash
pip install sentence-transformers
set NEURAL_USE_SENTENCE_TRANSFORMERS=1
```

If `llama.dll` fails to load, ensure CUDA 12.1 is installed and on PATH.

## Architecture (Code Layout)

```
neural/
  core/                neuron + synapse models
  regions/             cortex/hippocampus/thalamus + cortical column
  systems/             memory, neuromodulators, basal ganglia, self-model
  engines/             artificial_brain + scaled_brain
  llm/                 llama.cpp + affect inference + logits modulation
  cli/                 runnable entry points
configs/               JSON configs
models/                GGUF models
data/                  chat memory
tests/                 pytest tests
```

## Core Features

- Neurons: LIF by default, optional Izhikevich model per region
- Synapses: STDP, short-term plasticity, event-based updates
- Regions: cortex, hippocampus, thalamus, cortical column support
- Neuromodulators: dopamine/serotonin/acetylcholine/norepinephrine/GABA/glutamate
- Memory: working + long-term storage with replay-based consolidation
- Basal ganglia loop: action selection + reward prediction error logging
- Self-model: internal state tracking + labeled phenomenology report (simulation)
- LLM coupling: logits modulation and affect-driven sampling

## Configuration

Configs live in `configs/`. You can tune:
- `regions`: size, connectivity, neuron model
- `neuromodulation`: gains and time constants
- `basal_ganglia`: action selection behavior
- `memory`: replay cadence and strength

Example: enable Izhikevich in cortex:

```json
{
  "regions": {
    "cortex": {
      "neuron_model": "izhikevich",
      "izhikevich_params": {
        "a": 0.02,
        "b": 0.2,
        "c": -65.0,
        "d": 8.0,
        "input_scale": 100.0
      }
    }
  }
}
```

## Scaling

Use the vectorized backend for large runs:

```bash
python -m neural.cli.scale_demo
python -m neural.cli.scaling_benchmark
```

## Tests

```bash
pytest -q
```

## On Consciousness Claims

Consciousness cannot currently be verified or measured in software systems.
This project does not claim subjective experience as fact.
Any self-reports are derived from internal state variables and labeled as simulation.
This is an experiment exploring whether consciousness-like behavior can emerge.

## What This System Is Good For

- Brain-inspired experimentation (reward, neuromodulation, memory replay)
- Behavioral prototyping with a perception-action loop
- Research scaffolding for more detailed neural models
- LLM modulation from biologically grounded state variables

If you want a specific roadmap item added, open an issue or ask for the next feature.
