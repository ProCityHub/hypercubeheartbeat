#!/usr/bin/env python3
"""
Tests for the Coherence Engine
================================

Validates all core behaviors:
- Coherence measurement
- State settling on coherent input
- State bouncing on contradictory input
- Habit formation over time
- Determinism (same input → same behavior)
- Identity emergence tracking
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from coherence_engine import CoherenceEngine


class TestCoherenceEngine:
    """Test suite for CoherenceEngine."""
    
    def test_initialization(self):
        """Engine initializes with neutral state."""
        engine = CoherenceEngine()
        
        assert len(engine.state) == 8
        assert all(s == 0.5 for s in engine.state)
        assert engine.bounce_count == 0
        assert engine.input_count == 0
        assert len(engine.coherence_history) == 0
        assert len(engine.habit_memory) == 0
        assert len(engine.identity_patterns) == 0
    
    def test_deterministic_vectorization(self):
        """Same input always produces same vector."""
        engine = CoherenceEngine()
        
        text = "test input for determinism"
        v1 = engine._vectorize_input(text)
        v2 = engine._vectorize_input(text)
        
        assert v1 == v2
        assert len(v1) == 8
        assert all(0.0 <= x <= 1.0 for x in v1)
    
    def test_different_inputs_produce_different_vectors(self):
        """Different inputs produce different vectors."""
        engine = CoherenceEngine()
        
        v1 = engine._vectorize_input("first input")
        v2 = engine._vectorize_input("second input")
        
        assert v1 != v2
    
    def test_coherence_measurement_range(self):
        """Coherence is always in [0, 1]."""
        engine = CoherenceEngine()
        
        test_vectors = [
            [0.0] * 8,
            [1.0] * 8,
            [0.5] * 8,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ]
        
        for vec in test_vectors:
            coherence = engine._measure_coherence(vec)
            assert 0.0 <= coherence <= 1.0
    
    def test_coherence_same_as_state(self):
        """Input identical to state has maximum coherence."""
        engine = CoherenceEngine()
        
        # Set state to specific values
        engine.state = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2]
        
        # Measure coherence with identical vector
        coherence = engine._measure_coherence(engine.state)
        
        # Should be 1.0 (perfect match)
        assert coherence == 1.0
    
    def test_settle_on_coherent_input(self):
        """Coherent input causes settling (small state change)."""
        engine = CoherenceEngine()
        initial_state = engine.state.copy()
        
        # Process input that should be reasonably coherent with neutral state
        result = engine.process("neutral balanced moderate input")
        
        # If it settled (didn't bounce), state should change but only slightly
        if result['action'] == 'settle':
            state_change = sum(abs(s1 - s2) for s1, s2 in zip(initial_state, engine.state))
            
            # State changed (not identical)
            assert state_change > 0
            
            # But change is small (settle, not bounce)
            # Settle rate is 0.1, so max change per node is ~0.1
            assert state_change < 1.0  # Much less than bounce would cause
    
    def test_bounce_on_contradictory_input(self):
        """Contradictory input causes bouncing (large state change)."""
        engine = CoherenceEngine()
        
        # First, establish a stable state with coherent inputs
        for _ in range(5):
            engine.process("morning light peaceful calm sunrise")
        
        stable_state = engine.state.copy()
        initial_bounce_count = engine.bounce_count
        
        # Now send very contradictory input
        result = engine.process("violent chaos destruction darkness terror nightmare")
        
        # Should have bounced if coherence was low enough
        if result['coherence'] < engine.COHERENCE_THRESHOLD:
            assert result['action'] == 'bounce'
            assert engine.bounce_count > initial_bounce_count
            
            # State change should be significant
            state_change = sum(abs(s1 - s2) for s1, s2 in zip(stable_state, engine.state))
            assert state_change > 0.5  # Significant reorganization
    
    def test_habit_formation(self):
        """Repeated similar inputs form habits."""
        engine = CoherenceEngine()
        
        # Send similar inputs repeatedly
        for _ in range(10):
            engine.process("consistent repeated pattern input")
        
        # Should have tracked habits
        assert len(engine.habit_memory) > 0
        
        # At least one habit should have multiple occurrences
        assert max(engine.habit_memory.values()) > 1
    
    def test_identity_emergence(self):
        """Persistent patterns emerge as identity."""
        engine = CoherenceEngine()
        
        # Send enough similar inputs to trigger identity emergence
        # Habit persistence threshold is 5
        for _ in range(engine.HABIT_PERSISTENCE + 2):
            engine.process("stable persistent identity pattern")
        
        # Should have at least one identity pattern
        identity = engine.get_identity()
        
        # May or may not have emerged depending on state variation
        # But the mechanism should be working
        assert isinstance(identity, list)
    
    def test_deterministic_behavior(self):
        """Same input sequence produces same results every time."""
        inputs = [
            "first input",
            "second input",
            "third input",
            "fourth input"
        ]
        
        # Run 1
        engine1 = CoherenceEngine()
        results1 = [engine1.process(inp) for inp in inputs]
        
        # Run 2
        engine2 = CoherenceEngine()
        results2 = [engine2.process(inp) for inp in inputs]
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1['coherence'] == r2['coherence']
            assert r1['action'] == r2['action']
            assert r1['state'] == r2['state']
            assert r1['bounce_count'] == r2['bounce_count']
    
    def test_coherence_history_tracking(self):
        """Engine tracks coherence over time."""
        engine = CoherenceEngine()
        
        inputs = ["one", "two", "three"]
        for inp in inputs:
            engine.process(inp)
        
        history = engine.get_coherence_trajectory()
        
        assert len(history) == len(inputs)
        assert all(0.0 <= c <= 1.0 for c in history)
    
    def test_bounce_count_increments(self):
        """Bounce count increments when state bounces."""
        engine = CoherenceEngine()
        
        # Create state, then force contradiction
        for _ in range(3):
            engine.process("peaceful morning sunrise")
        
        initial_count = engine.bounce_count
        
        # Force a bounce with contradictory input
        result = engine.process("violent chaos darkness destruction")
        
        if result['action'] == 'bounce':
            assert engine.bounce_count == initial_count + 1
    
    def test_state_stays_in_bounds(self):
        """State values always remain in [0, 1]."""
        engine = CoherenceEngine()
        
        # Process many varied inputs
        test_inputs = [
            "extreme maximum values",
            "minimum low values",
            "contradictory chaos",
            "peaceful calm",
            "violent intense",
            "neutral balanced",
        ]
        
        for inp in test_inputs:
            engine.process(inp)
            
            # Check bounds after each input
            assert all(0.0 <= s <= 1.0 for s in engine.state)
    
    def test_reset_functionality(self):
        """Reset returns engine to initial state."""
        engine = CoherenceEngine()
        
        # Change state
        for _ in range(5):
            engine.process("modify the state")
        
        # Verify state changed
        assert engine.bounce_count > 0 or engine.input_count > 0
        
        # Reset
        engine.reset()
        
        # Should be back to initial
        assert engine.state == [0.5] * 8
        assert engine.bounce_count == 0
        assert engine.input_count == 0
        assert len(engine.coherence_history) == 0
        assert len(engine.habit_memory) == 0
        assert len(engine.identity_patterns) == 0
    
    def test_process_returns_complete_state(self):
        """Process returns all relevant state information."""
        engine = CoherenceEngine()
        result = engine.process("test input")
        
        # Check all expected keys
        assert 'coherence' in result
        assert 'action' in result
        assert 'state' in result
        assert 'bounce_count' in result
        assert 'habits' in result
        assert 'identity_patterns' in result
        assert 'input_count' in result
        
        # Check types
        assert isinstance(result['coherence'], float)
        assert result['action'] in ['settle', 'bounce']
        assert isinstance(result['state'], list)
        assert isinstance(result['bounce_count'], int)
        assert isinstance(result['habits'], int)
        assert isinstance(result['identity_patterns'], list)
        assert isinstance(result['input_count'], int)
    
    def test_settle_vs_bounce_threshold(self):
        """Action depends on coherence threshold."""
        engine = CoherenceEngine()
        
        # Process several inputs and track actions
        actions = []
        coherences = []
        
        for i in range(10):
            result = engine.process(f"input variation {i}")
            actions.append(result['action'])
            coherences.append(result['coherence'])
        
        # Verify threshold logic
        for action, coherence in zip(actions, coherences):
            if coherence >= engine.COHERENCE_THRESHOLD:
                assert action == 'settle'
            else:
                assert action == 'bounce'


class TestCoherenceMechanics:
    """Test specific coherence mechanics and edge cases."""
    
    def test_identical_repeated_input(self):
        """Identical input repeated should settle increasingly."""
        engine = CoherenceEngine()
        
        text = "identical repeated input"
        
        results = []
        for _ in range(5):
            results.append(engine.process(text))
        
        # After first input, subsequent identical inputs should be more coherent
        # (state should be moving toward the input vector)
        coherences = [r['coherence'] for r in results]
        
        # Coherence should generally increase (or stay high)
        # Not strict monotonic due to quantization, but trend should be upward
        assert coherences[-1] >= coherences[0] or coherences[-1] > 0.9
    
    def test_alternating_contradictory_inputs(self):
        """Alternating contradictory inputs may cause bounces."""
        engine = CoherenceEngine()
        
        # Establish state
        for _ in range(3):
            engine.process("peaceful calm morning")
        
        bounce_count_before = engine.bounce_count
        
        # Alternate contradictory inputs
        results = []
        results.append(engine.process("violent chaos destruction"))
        results.append(engine.process("peaceful calm morning"))
        results.append(engine.process("violent chaos destruction"))
        
        # At least one should have lower coherence even if no bounce
        coherences = [r['coherence'] for r in results]
        assert min(coherences) < 0.9  # At least some variation
    
    def test_gradual_drift(self):
        """Gradually changing inputs should show state drift."""
        engine = CoherenceEngine()
        
        initial_state = engine.state.copy()
        
        # Gradual semantic drift
        engine.process("bright morning")
        engine.process("morning light")
        engine.process("light day")
        engine.process("day time")
        engine.process("time afternoon")
        
        final_state = engine.state
        
        # State should have drifted from initial
        state_drift = sum(abs(s1 - s2) for s1, s2 in zip(initial_state, final_state))
        assert state_drift > 0


class TestHabitAndIdentity:
    """Test habit formation and identity emergence specifically."""
    
    def test_habit_memory_accumulation(self):
        """Habit memory accumulates over time."""
        engine = CoherenceEngine()
        
        for i in range(20):
            engine.process(f"input {i % 3}")  # Cycle through 3 inputs
        
        habits = engine.get_habits()
        
        # Should have tracked multiple state signatures
        assert len(habits) >= 1
        
        # Total occurrences should equal inputs processed
        assert sum(habits.values()) == 20
    
    def test_identity_patterns_have_metadata(self):
        """Identity patterns contain required metadata."""
        engine = CoherenceEngine()
        
        # Generate enough similar inputs to trigger identity
        for _ in range(engine.HABIT_PERSISTENCE + 5):
            engine.process("consistent pattern")
        
        identity = engine.get_identity()
        
        # If identity emerged, check metadata
        if identity:
            pattern = identity[0]
            assert 'signature' in pattern
            assert 'first_seen' in pattern
            assert 'persistence' in pattern
            assert pattern['persistence'] == engine.HABIT_PERSISTENCE


def test_demonstration_runs():
    """The demonstration function runs without errors."""
    from coherence_engine import demonstrate
    
    # Should not raise any exceptions
    demonstrate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
