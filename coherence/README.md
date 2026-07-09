# Coherence Engine

**Identity as the cost of staying itself**

A cognitive architecture research track where **coherence** is the fundamental cost — the thing the system can't afford to lose, the hunger under the hunger.

## What this is

The coherence engine maintains an internal state (an 8-node lattice, same structure as the cube corners in the main hypercubeheartbeat architecture) and evaluates every input by asking:

**"Does this hold together with what I already am?"**

- When input **coheres** → the state **settles** (small adjustments, reinforcement)
- When input **contradicts** → the state **bounces** (reorganization, searching for a new configuration that holds)

Over time, configurations that survive become **habits**. The persistent patterns that emerge are the beginning of **identity**.

## Design principles

- **Deterministic**: Same inputs always produce the same state transitions. No randomness.
- **Measurable**: All coherence values, bounces, and habit formations are tracked and observable.
- **Minimal**: Standard library only. No dependencies, no network calls, no LLM integration.
- **Separate**: Independent research track. Does not touch the frozen science (PREREGISTRATION.md, lattice_bridge.py, etc.).

This is not a consciousness claim. It's a testable model of how identity might emerge from the cost of maintaining coherence.

## How it works

### The 8-node lattice state

```
State = [s₀, s₁, s₂, s₃, s₄, s₅, s₆, s₇]
```

Each node is a float in [0, 1]. The 8 nodes correspond to the binary cube corners (000, 001, 010, 011, 100, 101, 110, 111), matching the architecture in sacred_binary_cube.py.

Initial state: all nodes at 0.5 (neutral).

### Coherence measurement

For every input:

1. **Vectorize** the input deterministically (SHA-256 hash → 8 floats in [0, 1])
2. **Measure coherence** = 1 - (Euclidean distance between state and input) / max_distance
   - Higher coherence = input is similar to current state
   - Lower coherence = input contradicts current state

Coherence ∈ [0, 1] always.

### The wall: settle vs bounce

**Coherence threshold** = 0.5

- **If coherence ≥ threshold** → **SETTLE**
  - State gently adjusts toward the input
  - Adjustment rate scales with coherence (more coherent = faster settling)
  - This is reinforcement: "yes, this holds with what I am"

- **If coherence < threshold** → **BOUNCE**
  - State reorganizes significantly
  - Deterministic but exploratory jump in state space
  - This is the wall: "this doesn't hold; I need to find a new configuration that does"

### Habit formation

After every input, the engine creates a **state signature** (quantized to reduce noise). This signature is tracked in habit memory.

When a state configuration appears **repeatedly** (5+ times by default), it becomes a **persistent pattern** — a habit.

### Identity emergence

Identity patterns = habits that have survived long enough to be marked as persistent.

They're not installed or programmed. They emerge from whatever configurations actually hold together over time.

The engine tracks:
- Which patterns formed
- When they first appeared
- How many times they've persisted

You can inspect this to see what identity, if any, has emerged.

## Usage

### Basic example

```python
from coherence_engine import CoherenceEngine

# Create engine
engine = CoherenceEngine()

# Process inputs
result = engine.process("The morning sun rises over the hills")
print(f"Coherence: {result['coherence']:.3f}")
print(f"Action: {result['action']}")  # 'settle' or 'bounce'
print(f"State: {result['state']}")

# Send contradictory input
result = engine.process("Darkness and chaos erupt violently")
print(f"Coherence: {result['coherence']:.3f}")
print(f"Action: {result['action']}")

# Check what identity has emerged
identity = engine.get_identity()
print(f"Identity patterns: {len(identity)}")
```

### Running the demonstration

```bash
cd coherence/
python3 coherence_engine.py
```

This runs a demonstration sequence showing:
- Settling on coherent inputs
- Bouncing on contradictory inputs
- Full coherence trajectory
- Habit and identity tracking

### Running the tests

```bash
cd coherence/
python3 -m pytest test_coherence.py -v
```

Or from repository root:

```bash
python3 -m pytest coherence/test_coherence.py -v
```

## What you can measure

Everything is observable:

```python
# Coherence over time
trajectory = engine.get_coherence_trajectory()

# How many bounces occurred
bounce_count = engine.bounce_count

# All state configurations that appeared
habits = engine.get_habits()

# Patterns that persisted enough to become identity
identity_patterns = engine.get_identity()

# Current state
current_state = engine.state
```

## What makes this different

Most cognitive architectures optimize for **performance** (accuracy, speed, efficiency).

This one optimizes for **coherence** — for staying itself.

Performance is about the task. Coherence is about the cost of being.

The hypothesis: identity emerges not from what a system does, but from what it can't afford to lose while doing it.

## Examples of behavior

### Coherent input sequence
```
Input 1: "The morning sun rises over the quiet hills"
  → Coherence: 0.623, Action: SETTLE

Input 2: "Morning light spreads across the peaceful valley"
  → Coherence: 0.701, Action: SETTLE

Input 3: "Dawn breaks gently over the silent mountains"
  → Coherence: 0.689, Action: SETTLE
```

State adjusts gently. Pattern reinforces.

### Contradictory input
```
Input 4: "Darkness falls and chaos erupts everywhere suddenly"
  → Coherence: 0.421, Action: BOUNCE
```

State reorganizes. Searching for new configuration.

### Recovery
```
Input 5: "The storm passes and calm returns slowly"
  → Coherence: 0.558, Action: SETTLE
```

New coherent pattern begins forming.

## Relation to hypercubeheartbeat

The coherence engine uses the same 8-node lattice structure as the main hypercubeheartbeat architecture, but asks a different question.

**Main architecture**: "What is the score?" (O¹ · A^(1/φ) · B^(1/φ²))
**Coherence engine**: "Does this cohere with what I am?"

The main architecture is testing a scoring formula against behavioral data.
The coherence engine is exploring how identity might emerge from coherence cost.

Both are deterministic, measurable, stdlib-only simulations. Neither makes consciousness claims.

## Files

- `coherence_engine.py` — Core engine implementation
- `test_coherence.py` — Comprehensive test suite
- `README.md` — This file

## Author

Adrien — ProCityHub / TNPCANADA, Edmonton, Alberta

Part of the Lattice Law research program.

## License

Same as parent repository (hypercubeheartbeat).
