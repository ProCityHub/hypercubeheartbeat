#!/usr/bin/env python3
"""
COHERENCE ENGINE — Identity as the cost of staying itself
==========================================================

A cognitive architecture where coherence is the fundamental cost.
The system maintains an 8-node lattice state and asks of every input:
"Does this hold together with what I already am?"

When input coheres → state settles (reinforcement)
When input contradicts → state bounces (reorganization)

Over time, configurations that survive become habits.
The persistent pattern that emerges is the beginning of identity.

DESIGN PRINCIPLES:
- Deterministic: same inputs → same state transitions
- Measurable: all coherence values, bounces, and habits are tracked
- Minimal: stdlib only, no dependencies, no randomness
- Separate: independent research track, doesn't touch frozen science

ARCHITECTURE:
- 8-node lattice (binary cube corners 000-111)
- Each node holds activation value in [0, 1]
- Coherence = how well current state aligns with new input
- Threshold-based bounce when coherence drops too low
- Habit memory tracks configurations that persist
"""

import hashlib
import math
from typing import Dict, List, Tuple, Optional


class CoherenceEngine:
    """
    A coherence-driven cognitive architecture.
    
    The engine maintains an internal 8-node lattice state and evaluates
    every input against the current state. Coherent inputs settle the state
    (small adjustments). Contradictory inputs trigger bounces (reorganization).
    """
    
    # Thresholds
    COHERENCE_THRESHOLD = 0.5  # Below this triggers a bounce
    SETTLE_RATE = 0.1          # How fast state adjusts to coherent input
    BOUNCE_MAGNITUDE = 0.7     # How much state reorganizes on contradiction
    HABIT_PERSISTENCE = 5      # Times a pattern must appear to become a habit
    
    def __init__(self):
        """Initialize the engine with neutral state."""
        # 8-node lattice (one per cube corner 000-111)
        self.state = [0.5] * 8
        
        # History tracking
        self.coherence_history: List[float] = []
        self.bounce_count = 0
        self.input_count = 0
        
        # Habit formation: tracks state configurations that persist
        # Format: {state_signature: occurrence_count}
        self.habit_memory: Dict[str, int] = {}
        
        # Identity emergence: persistent patterns that survive
        self.identity_patterns: List[Dict] = []
    
    def _vectorize_input(self, text: str) -> List[float]:
        """
        Convert input text to 8-dimensional vector deterministically.
        Uses SHA-256 hashing for reproducibility.
        
        Args:
            text: Input string to vectorize
            
        Returns:
            8-element list of floats in [0, 1]
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Use first 8 bytes, normalize to [0, 1]
        return [h[i] / 255.0 for i in range(8)]
    
    def _measure_coherence(self, input_vector: List[float]) -> float:
        """
        Measure how well the input coheres with current state.
        
        Coherence = inverse of Euclidean distance between state and input.
        Higher values = more coherent (similar).
        
        Args:
            input_vector: 8D vector representing new input
            
        Returns:
            Coherence score in [0, 1]
        """
        # Euclidean distance
        distance = math.sqrt(sum((s - i)**2 for s, i in zip(self.state, input_vector)))
        
        # Normalize to [0, 1], invert so higher = more coherent
        # Max distance in 8D unit cube is sqrt(8) ≈ 2.83
        max_distance = math.sqrt(8.0)
        coherence = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, coherence))
    
    def _settle_state(self, input_vector: List[float], coherence: float):
        """
        Gently adjust state toward coherent input.
        The state "settles" — small movements toward alignment.
        
        Args:
            input_vector: The coherent input vector
            coherence: How coherent it is (used to scale adjustment)
        """
        # Adjust each node toward the input, scaled by settle rate and coherence
        adjustment_rate = self.SETTLE_RATE * coherence
        for i in range(8):
            delta = input_vector[i] - self.state[i]
            self.state[i] += delta * adjustment_rate
            # Keep in bounds
            self.state[i] = max(0.0, min(1.0, self.state[i]))
    
    def _bounce_state(self, input_vector: List[float]):
        """
        Reorganize state when input contradicts.
        The state "bounces" — searching for a new configuration that might hold.
        
        Uses the input vector as a forcing function but adds rotation to explore
        the state space rather than just jumping to the input.
        
        Args:
            input_vector: The contradictory input vector
        """
        self.bounce_count += 1
        
        # Reorganize: blend current state with input, then perturb
        # This creates a deterministic but exploratory jump
        for i in range(8):
            # Mix input with perpendicular component from adjacent nodes
            next_i = (i + 1) % 8
            prev_i = (i - 1) % 8
            
            blend = (
                input_vector[i] * self.BOUNCE_MAGNITUDE +
                self.state[next_i] * (1 - self.BOUNCE_MAGNITUDE) * 0.5 +
                self.state[prev_i] * (1 - self.BOUNCE_MAGNITUDE) * 0.5
            )
            
            self.state[i] = max(0.0, min(1.0, blend))
    
    def _state_signature(self) -> str:
        """
        Create a signature for the current state for habit tracking.
        Quantizes state to reduce noise while preserving structure.
        
        Returns:
            String signature of current state
        """
        # Quantize to 10 levels (0.0, 0.1, 0.2, ..., 0.9, 1.0)
        quantized = [round(s * 10) / 10 for s in self.state]
        return ",".join(f"{v:.1f}" for v in quantized)
    
    def _update_habits(self):
        """
        Track state configurations that persist.
        When a pattern appears repeatedly, it becomes a habit.
        """
        sig = self._state_signature()
        
        # Increment occurrence count
        if sig not in self.habit_memory:
            self.habit_memory[sig] = 0
        self.habit_memory[sig] += 1
        
        # If this configuration has persisted enough, mark as identity pattern
        if self.habit_memory[sig] == self.HABIT_PERSISTENCE:
            self.identity_patterns.append({
                "signature": sig,
                "first_seen": self.input_count,
                "persistence": self.HABIT_PERSISTENCE
            })
    
    def process(self, text: str) -> Dict:
        """
        Process an input and update internal state.
        
        This is the core loop:
        1. Vectorize input
        2. Measure coherence with current state
        3. If coherent: settle (small adjustment)
        4. If contradictory: bounce (reorganize)
        5. Track habits and identity emergence
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing:
                - coherence: measured coherence score
                - action: "settle" or "bounce"
                - state: current 8-node state
                - bounce_count: total bounces so far
                - habits: current habit count
                - identity_patterns: emerged identity patterns
        """
        self.input_count += 1
        
        # 1. Vectorize
        input_vector = self._vectorize_input(text)
        
        # 2. Measure coherence
        coherence = self._measure_coherence(input_vector)
        self.coherence_history.append(coherence)
        
        # 3. Settle or bounce
        if coherence >= self.COHERENCE_THRESHOLD:
            self._settle_state(input_vector, coherence)
            action = "settle"
        else:
            self._bounce_state(input_vector)
            action = "bounce"
        
        # 4. Update habits
        self._update_habits()
        
        # 5. Return full state report
        return {
            "coherence": coherence,
            "action": action,
            "state": self.state.copy(),
            "bounce_count": self.bounce_count,
            "habits": len(self.habit_memory),
            "identity_patterns": self.identity_patterns.copy(),
            "input_count": self.input_count
        }
    
    def get_coherence_trajectory(self) -> List[float]:
        """Return the full history of coherence measurements."""
        return self.coherence_history.copy()
    
    def get_habits(self) -> Dict[str, int]:
        """Return all tracked habits and their occurrence counts."""
        return self.habit_memory.copy()
    
    def get_identity(self) -> List[Dict]:
        """Return emerged identity patterns (habits that persisted)."""
        return self.identity_patterns.copy()
    
    def reset(self):
        """Reset the engine to initial neutral state."""
        self.state = [0.5] * 8
        self.coherence_history = []
        self.bounce_count = 0
        self.input_count = 0
        self.habit_memory = {}
        self.identity_patterns = []


def demonstrate():
    """
    Demonstrate the coherence engine in action.
    Shows settling on coherent input and bouncing on contradictory input.
    """
    engine = CoherenceEngine()
    
    print("=" * 70)
    print("COHERENCE ENGINE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Test sequence: coherent inputs followed by contradiction
    test_inputs = [
        "The morning sun rises over the quiet hills",
        "Morning light spreads across the peaceful valley",
        "Dawn breaks gently over the silent mountains",
        "Darkness falls and chaos erupts everywhere suddenly",  # Contradiction!
        "Dark storm clouds gather with violent intensity",
        "The storm passes and calm returns slowly",
    ]
    
    for i, text in enumerate(test_inputs, 1):
        result = engine.process(text)
        
        print(f"Input {i}: {text[:50]}...")
        print(f"  Coherence: {result['coherence']:.3f}")
        print(f"  Action:    {result['action'].upper()}")
        print(f"  Bounces:   {result['bounce_count']}")
        print(f"  Habits:    {result['habits']}")
        print(f"  State:     [{', '.join(f'{s:.2f}' for s in result['state'])}]")
        if result['identity_patterns']:
            print(f"  Identity:  {len(result['identity_patterns'])} pattern(s) emerged")
        print()
    
    print("=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    print(f"Total bounces: {engine.bounce_count}")
    print(f"Habit patterns tracked: {len(engine.get_habits())}")
    print(f"Identity patterns emerged: {len(engine.get_identity())}")
    print()
    print("Coherence trajectory:")
    print("  " + " → ".join(f"{c:.2f}" for c in engine.get_coherence_trajectory()))
    print()


if __name__ == "__main__":
    demonstrate()
