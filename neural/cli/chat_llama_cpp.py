"""
Chat interface using llama.cpp with brain-driven logits modulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from neural.config.brain_config import load_brain_config, create_brain_from_config
from neural.llm.llama_cpp_chat import BrainLlamaChat
from neural.llm.text_stimulus import text_to_stimulus, text_to_affect
from neural.systems.memory_store import MemoryStore


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default_brain.json"
DEFAULT_MEMORY = ROOT_DIR / "data" / "chat_memory.json"
DEFAULT_MODEL = ROOT_DIR / "models" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with brain-modulated llama.cpp.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Brain config JSON.")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY), help="Memory store path.")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory retrieval.")
    parser.add_argument("--brain-ms", type=float, default=50.0, help="Simulated ms per user turn.")
    parser.add_argument("--stimulus-strength", type=float, default=0.3, help="Text stimulus strength.")
    parser.add_argument("--max-history", type=int, default=12, help="Max history turns.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty.")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context length.")
    parser.add_argument("--n-gpu-layers", type=int, default=35, help="Number of GPU layers.")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size for llama.cpp.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for llama.cpp.")
    parser.add_argument(
        "--phenomenology",
        action="store_true",
        help="Print simulated internal state report after each response.",
    )
    return parser.parse_args()


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help                 Show this help\n"
        "  /state                Show brain state summary\n"
        "  /phenomenology        Show simulated internal state report\n"
        "  /reward <value>       Apply reward (dopamine signal)\n"
        "  /remember <text>      Store a memory snippet\n"
        "  /memory               Show recent stored memories\n"
        "  /reset                Clear conversation history\n"
        "  /quit                 Exit\n"
    )


def _format_brain_state(state: Dict) -> str:
    neuro = state.get("neurotransmitter_levels", {})
    affect = state.get("affect_state", {})
    bg = state.get("basal_ganglia_state") or {}
    rates = state.get("spike_rates", {})
    lines = [
        "Brain state summary:",
        f"- neurotransmitters: dopamine={neuro.get('dopamine', 0.0):.2f}, "
        f"serotonin={neuro.get('serotonin', 0.0):.2f}, "
        f"acetylcholine={neuro.get('acetylcholine', 0.0):.2f}, "
        f"gaba={neuro.get('gaba', 0.0):.2f}, "
        f"glutamate={neuro.get('glutamate', 0.0):.2f}, "
        f"norepinephrine={neuro.get('norepinephrine', 0.0):.2f}",
        f"- affect: valence={affect.get('valence', 0.0):.2f}, "
        f"arousal={affect.get('arousal', 0.0):.2f}, "
        f"tension={affect.get('tension', 0.0):.2f}",
    ]
    if bg.get("last_action") is not None:
        lines.append(f"- basal ganglia: last_action={bg.get('last_action')}")
    if state.get("last_rpe") is not None:
        lines.append(f"- reward prediction error: {state.get('last_rpe', 0.0):.3f}")
    if rates:
        rate_text = ", ".join(f"{name}={rate:.2f} Hz" for name, rate in rates.items())
        lines.append(f"- spike rates: {rate_text}")
    return "\n".join(lines)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _adjust_sampling(
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    affect: Dict[str, float],
) -> Tuple[float, float, float]:
    valence = float(affect.get("valence", 0.0))
    arousal = float(affect.get("arousal", 0.0))
    tension = float(affect.get("tension", 0.0))

    temp = temperature + 0.35 * (arousal - 0.3) + 0.2 * valence - 0.2 * tension
    top_p_adj = top_p + 0.15 * (arousal - 0.3) + 0.1 * valence - 0.15 * tension
    repeat_adj = repeat_penalty + 0.15 * tension - 0.1 * valence

    temp = _clamp(temp, 0.2, 1.5)
    top_p_adj = _clamp(top_p_adj, 0.5, 0.98)
    repeat_adj = _clamp(repeat_adj, 1.0, 1.4)
    return temp, top_p_adj, repeat_adj


def _memory_context(memory: MemoryStore, query: str, affect: Dict[str, float] | None, limit: int = 3) -> str:
    results = memory.search(query, limit=limit, affect=affect)
    if not results:
        return ""
    lines = ["Relevant memories:"]
    for entry in results:
        lines.append(f"- {entry.get('text', '')}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_brain_config(args.config)

    brain = create_brain_from_config(config)
    memory = MemoryStore(args.memory)

    if not Path(args.model).exists():
        print(f"Model file not found: {args.model}")
        print("Download a GGUF model and pass --model <path>.")
        sys.exit(1)

    chat = BrainLlamaChat(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        seed=args.seed,
    )

    history: List[Tuple[str, str]] = []
    recent_user_texts: List[str] = []
    print("\nBrain-modulated llama.cpp chat is ready. Type /help for commands.")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if command in ("/quit", "/exit"):
                break
            if command == "/help":
                _print_help()
                continue
            if command == "/state":
                print(_format_brain_state(brain.get_brain_state()))
                continue
            if command == "/phenomenology":
                print(brain.get_phenomenology_report())
                continue
            if command == "/reward":
                try:
                    value = float(arg)
                except ValueError:
                    print("Usage: /reward <value>")
                    continue
                brain.step(external_reward=value)
                levels = brain.neurotransmitters.get_state()
                print(f"Dopamine now {levels.get('dopamine', 0.0):.3f}")
                continue
            if command == "/remember":
                if not arg:
                    print("Usage: /remember <text>")
                    continue
                memory.add(arg, tags=["manual"], affect=brain.get_brain_state().get("affect_state"))
                memory.save()
                print("Memory saved.")
                continue
            if command == "/memory":
                recent = memory.entries[-5:]
                if not recent:
                    print("No memories stored yet.")
                else:
                    for entry in recent:
                        print(f"- {entry.get('text', '')}")
                continue
            if command == "/reset":
                history = []
                print("History cleared.")
                continue

            print("Unknown command. Type /help for help.")
            continue

        experience = text_to_affect(user_input, recent_user_texts[-5:])
        recent_user_texts.append(user_input)
        if args.brain_ms > 0:
            thalamus_size = len(brain.regions["thalamus"].neurons)
            stimulus = text_to_stimulus(user_input, thalamus_size, args.stimulus_strength)
            brain.add_sensory_input("text", stimulus)
            brain.run(args.brain_ms, experience=experience)

        state = brain.get_brain_state()
        context = ""
        if not args.no_memory:
            context = _memory_context(memory, user_input, state.get("affect_state"))
        phenomenology = brain.get_phenomenology_report() if args.phenomenology else ""
        if phenomenology:
            context = f"{context}\n{phenomenology}".strip()

        if args.max_history > 0:
            history = history[-args.max_history :]

        prompt = chat.build_prompt(history, user_input, memory=context)
        temp, top_p, repeat_penalty = _adjust_sampling(
            args.temperature,
            args.top_p,
            args.repeat_penalty,
            state.get("affect_state", {}),
        )
        response = chat.generate(
            prompt=prompt,
            brain_state=state,
            temperature=temp,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=args.max_tokens,
        )

        print(f"Neural> {response}")
        if args.phenomenology:
            print(phenomenology)

        history.append(("user", user_input))
        history.append(("assistant", response))
        memory.add(
            f"User: {user_input}\nAssistant: {response}",
            tags=["dialogue"],
            affect=state.get("affect_state"),
        )
        memory.save()


if __name__ == "__main__":
    main()
