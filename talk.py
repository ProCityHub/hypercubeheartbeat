#!/usr/bin/env python3
"""
talk.py — a conversation with the instrument.
You type; the brain scores what you said and reports its internal state.
It does not understand you. It will never pretend to.
Runs in Termux: python3 talk.py   (requires lattice_bridge.py + core modules)
"""

from lattice_bridge import LatticeBrain


def main():
    brain = LatticeBrain()
    print("LATTICE TALK — type text, get state. 'quit' to exit.")
    print("(This is an instrument reading, not a reply.)\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        r = brain.perceive(text)
        print(f"brain> score={r['score']:.4f}  O={r['O']} A={r['A']} B={r['B']}  [{r['confidence']}]")
        print(f"       SIGHT   {r['faculties']['SIGHT']}")
        print(f"       GENOME  {r['faculties']['GENOME']}   VOICE {r['faculties']['VOICE']}")
        print(f"       {r['faculties']['RESEARCH']}\n")
    print("session ended — node potentials reset on next run.")


if __name__ == "__main__":
    main()
