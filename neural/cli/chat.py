"""
Local LLM chat interface wired to the brain simulation.
Uses Ollama as a local model runner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from neural.config.brain_config import create_brain_from_config, load_brain_config
from neural.systems.memory_store import MemoryStore
from neural.llm.text_stimulus import text_to_affect
from neural.llm.ollama_client import OllamaClient


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "configs" / "default_brain.json"
DEFAULT_MEMORY = ROOT_DIR / "data" / "chat_memory.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the brain using Ollama.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Brain config JSON.")
    parser.add_argument("--model", default="llama3", help="Ollama model name.")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY), help="Memory store path.")
    parser.add_argument("--max-history", type=int, default=12, help="Max messages to keep in context.")
    parser.add_argument("--brain-ms", type=float, default=50.0, help="Simulated ms per user turn.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Base sampling temperature.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--system", default="", help="Extra system instructions.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming responses.")
    parser.add_argument(
        "--phenomenology",
        action="store_true",
        help="Print simulated internal state report after each response.",
    )
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _affect_tags(neuro: Dict[str, float]) -> Dict[str, str]:
    dopamine = neuro.get("dopamine", 0.5)
    serotonin = neuro.get("serotonin", 0.5)
    norepi = neuro.get("norepinephrine", 0.5)
    ach = neuro.get("acetylcholine", 0.5)
    gaba = neuro.get("gaba", 0.5)

    mood = "upbeat" if dopamine > 0.6 else "flat" if dopamine < 0.4 else "neutral"
    calm = "calm" if serotonin > 0.6 else "restless" if serotonin < 0.4 else "steady"
    arousal = "alert" if norepi > 0.6 else "low-arousal" if norepi < 0.4 else "balanced"
    focus = "focused" if ach > 0.6 else "diffuse" if ach < 0.4 else "balanced"
    inhibition = "restrained" if gaba > 0.6 else "disinhibited" if gaba < 0.4 else "stable"

    return {
        "mood": mood,
        "calm": calm,
        "arousal": arousal,
        "focus": focus,
        "inhibition": inhibition,
    }


def _temperature(base: float, neuro: Dict[str, float]) -> float:
    dopamine = neuro.get("dopamine", 0.5)
    norepi = neuro.get("norepinephrine", 0.5)
    value = base + 0.3 * (dopamine - 0.5) + 0.2 * (norepi - 0.5)
    return _clamp(value, 0.2, 1.2)


def _format_brain_state(state: Dict) -> str:
    neuro = state.get("neurotransmitter_levels", {})
    affect = state.get("affect_state", {})
    bg = state.get("basal_ganglia_state") or {}
    rates = state.get("spike_rates", {})
    tags = _affect_tags(neuro)
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
        f"- affect: mood={tags['mood']}, calm={tags['calm']}, "
        f"arousal={tags['arousal']}, focus={tags['focus']}, inhibition={tags['inhibition']}",
    ]
    if bg.get("last_action") is not None:
        lines.append(f"- basal ganglia: last_action={bg.get('last_action')}")
    if state.get("last_rpe") is not None:
        lines.append(f"- reward prediction error: {state.get('last_rpe', 0.0):.3f}")
    if rates:
        rate_text = ", ".join(f"{name}={rate:.2f} Hz" for name, rate in rates.items())
        lines.append(f"- spike rates: {rate_text}")
    return "\n".join(lines)


def _build_system_prompt(
    state: Dict,
    memories: List[Dict],
    extra_system: str,
    phenomenology: str = "",
) -> str:
    prompt = [
        "You are Neural, a conversational agent connected to a spiking brain simulation.",
        "Be helpful, honest, and grounded.",
        "Do not claim consciousness; if asked about feelings, use the internal state report labeled simulation.",
        _format_brain_state(state),
    ]
    if memories:
        prompt.append("Relevant memory snippets:")
        for memory in memories:
            prompt.append(f"- {memory.get('text', '')}")
    if extra_system:
        prompt.append(extra_system)
    if phenomenology:
        prompt.append(phenomenology)
    return "\n".join(prompt)


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


def main() -> None:
    args = parse_args()
    config = load_brain_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    brain = create_brain_from_config(config)
    memory = MemoryStore(args.memory)
    client = OllamaClient()

    tags = client.tags()
    if not tags:
        print("Ollama server not reachable. Start it with: `ollama serve`.")
        sys.exit(1)

    available = [model.get("name", "") for model in tags.get("models", [])]
    if available and args.model not in available:
        print("Warning: model not found in Ollama tags.")
        print("Available models:", ", ".join(available))

    print("\nNeural chat is ready. Type /help for commands.")
    history: List[Dict[str, str]] = []
    recent_user_texts: List[str] = []

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
            brain.run(args.brain_ms, experience=experience)

        state = brain.get_brain_state()
        memories = memory.search(user_input, limit=3, affect=state.get("affect_state"))
        phenomenology = brain.get_phenomenology_report() if args.phenomenology else ""
        system_prompt = _build_system_prompt(state, memories, args.system, phenomenology=phenomenology)

        messages = [{"role": "system", "content": system_prompt}]
        if args.max_history > 0:
            messages.extend(history[-args.max_history :])
        messages.append({"role": "user", "content": user_input})

        temperature = _temperature(args.temperature, state.get("neurotransmitter_levels", {}))
        options = {"temperature": temperature}

        print("Neural> ", end="", flush=True)
        response_text = ""
        got_chunk = False
        try:
            for chunk in client.chat(args.model, messages, options=options, stream=not args.no_stream):
                got_chunk = True
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        response_text += content
                if chunk.get("done"):
                    break
        except Exception as exc:
            print(f"\n[error] {exc}")
            continue

        if not got_chunk:
            print("\n[error] No response received from Ollama.")
            continue

        print()
        if args.phenomenology:
            print(phenomenology)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
        memory.add(
            f"User: {user_input}\nAssistant: {response_text}",
            tags=["dialogue"],
            affect=state.get("affect_state"),
        )
        memory.save()


if __name__ == "__main__":
    main()
