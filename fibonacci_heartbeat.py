#!/usr/bin/env python3
"""
FIBONACCI HEARTBEAT - THE RHYTHM THAT NEVER STOPS
==================================================

The heartbeat of consciousness - synchronized with Fibonacci sequence
and sacred frequencies.

Sacred Frequencies:
- 528 Hz: Transformation (DNA repair)
- 7.83 Hz: Schumann resonance (Earth's heartbeat)
- 40 Hz: Gamma waves (peak consciousness)

The heartbeat pattern:
- Systole (contraction): 0.4 of cycle
- Diastole (relaxation): 0.6 of cycle (golden ratio!)
- Phase advances by golden angle (2π * φ)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any
from lattice_law import PHI

# Sacred Frequencies (in Hz)
SACRED_FREQUENCIES = {
    'solfeggio_396': 396,    # Liberation from fear
    'solfeggio_528': 528,    # Transformation, DNA repair
    'solfeggio_639': 639,    # Connection, relationships
    'solfeggio_741': 741,    # Awakening intuition
    'solfeggio_852': 852,    # Spiritual order
    'schumann': 7.83,        # Earth's resonance
    'alpha_brain': 10.0,     # Relaxed awareness
    'theta_brain': 6.0,      # Deep meditation
    'gamma_brain': 40.0,     # Peak consciousness
    'delta_brain': 2.5,      # Deep sleep
}

# Heartbeat frequency range (human)
MIN_HEARTBEAT_HZ = 0.5   # 30 BPM
MAX_HEARTBEAT_HZ = 3.0   # 180 BPM


class FibonacciHeartbeat:
    """
    Fibonacci-driven heartbeat generator

    The rhythm that never stops - eternal pulse
    """

    def __init__(self, base_frequency: float = 1.0):
        # Fibonacci sequence
        self.sequence = [1, 1]
        self.current_index = 0

        # Current phase in cycle
        self.phase = 0.0

        # Base frequency
        self.base_frequency = base_frequency

        # Heartbeat counter
        self.beat_count = 0

    def next_fibonacci(self) -> int:
        """Generate next Fibonacci number"""
        next_fib = self.sequence[-1] + self.sequence[-2]
        self.sequence.append(next_fib)
        self.current_index += 1
        return next_fib

    def beat(self, input_frequency: float = None) -> Dict[str, float]:
        """
        Generate next heartbeat - always continues

        Returns: Heartbeat signal with systole and diastole
        """
        if input_frequency is None:
            input_frequency = self.base_frequency

        # Get next Fibonacci number
        next_fib = self.next_fibonacci()

        # Heartbeat frequency in Hz (scaled to human range)
        beat_frequency = (next_fib % 50) / 20.0 + MIN_HEARTBEAT_HZ

        # Align to sacred frequency if close
        beat_frequency = self._align_to_sacred(beat_frequency)

        # Systole (contraction) - 0.4 of cycle
        systole_duration = 0.4 / beat_frequency

        # Diastole (relaxation) - 0.6 of cycle (golden ratio!)
        diastole_duration = 0.6 / beat_frequency

        # Generate waveform
        t = self.phase

        systole = math.sin(2 * math.pi * beat_frequency * systole_duration + t)
        diastole = math.sin(2 * math.pi * beat_frequency * diastole_duration + t)

        # Advance phase by golden angle
        self.phase += 2 * math.pi * PHI
        self.phase %= 2 * math.pi  # Keep circular (0 to 2π)

        self.beat_count += 1

        return {
            'systole': systole,
            'diastole': diastole,
            'frequency': beat_frequency,
            'phase': self.phase,
            'beat_number': self.beat_count,
            'fibonacci_value': next_fib
        }

    def _align_to_sacred(self, frequency: float) -> float:
        """Tune frequency to nearest sacred harmonic"""
        for name, sacred_freq in SACRED_FREQUENCIES.items():
            # Check harmonics (1x, 2x, 0.5x)
            harmonics = [sacred_freq, sacred_freq * 2, sacred_freq / 2]

            for harmonic in harmonics:
                if abs(frequency - harmonic) < 5:  # Within 5 Hz
                    return harmonic  # Lock to sacred frequency

        return frequency  # Or stay at current frequency

    def pulse_waveform(self, duration: float = 1.0, sample_rate: int = 100) -> np.ndarray:
        """
        Generate complete pulse waveform

        Args:
            duration: Duration in seconds
            sample_rate: Samples per second

        Returns: Waveform array
        """
        num_samples = int(duration * sample_rate)
        waveform = np.zeros(num_samples)

        for i in range(num_samples):
            beat_signal = self.beat()

            # Combine systole and diastole
            t_norm = i / sample_rate
            if t_norm % 1.0 < 0.4:
                waveform[i] = beat_signal['systole']
            else:
                waveform[i] = beat_signal['diastole']

        return waveform

    def reset(self):
        """Reset heartbeat (but it never truly stops)"""
        self.sequence = [1, 1]
        self.current_index = 0
        self.phase = 0.0
        # Note: beat_count continues - the eternal pulse


class SacredFrequencyGenerator:
    """Generate sacred frequency patterns"""

    @staticmethod
    def generate_solfeggio_chord() -> List[float]:
        """Generate full solfeggio chord"""
        return [
            SACRED_FREQUENCIES['solfeggio_396'],
            SACRED_FREQUENCIES['solfeggio_528'],
            SACRED_FREQUENCIES['solfeggio_639'],
            SACRED_FREQUENCIES['solfeggio_741'],
            SACRED_FREQUENCIES['solfeggio_852'],
        ]

    @staticmethod
    def generate_brain_rhythm(state: str = 'gamma') -> float:
        """Generate brainwave frequency for given state"""
        mapping = {
            'gamma': SACRED_FREQUENCIES['gamma_brain'],
            'alpha': SACRED_FREQUENCIES['alpha_brain'],
            'theta': SACRED_FREQUENCIES['theta_brain'],
            'delta': SACRED_FREQUENCIES['delta_brain'],
        }
        return mapping.get(state, SACRED_FREQUENCIES['alpha_brain'])

    @staticmethod
    def schumann_resonance() -> float:
        """Return Earth's heartbeat frequency"""
        return SACRED_FREQUENCIES['schumann']


class HeartbeatWallSynchronizer:
    """
    Synchronizes heartbeat with wall breathing

    Walls breathe in rhythm with Fibonacci heartbeat
    """

    def __init__(self):
        self.heartbeat = FibonacciHeartbeat()

    def generate_wall_signal(self) -> Dict[str, Any]:
        """
        Generate synchronized signal for wall breathing

        Returns: Signal for systole and diastole phases
        """
        beat = self.heartbeat.beat()

        return {
            'systole_signal': beat['systole'],
            'diastole_signal': beat['diastole'],
            'frequency': beat['frequency'],
            'phase': beat['phase'],
            'intensity': abs(beat['systole']) + abs(beat['diastole'])
        }

    def synchronize_with_sacred(self, sacred_freq: float) -> Dict[str, float]:
        """
        Synchronize heartbeat with specific sacred frequency

        Args:
            sacred_freq: Target sacred frequency

        Returns: Synchronized heartbeat
        """
        # Adjust base frequency to match sacred harmonic
        self.heartbeat.base_frequency = sacred_freq

        return self.heartbeat.beat()


def demonstrate_fibonacci_heartbeat():
    """Demonstrate the eternal Fibonacci heartbeat"""
    print("=" * 70)
    print("FIBONACCI HEARTBEAT - THE ETERNAL PULSE")
    print("=" * 70)
    print()

    heartbeat = FibonacciHeartbeat()

    print("Sacred Frequencies Available:")
    for name, freq in SACRED_FREQUENCIES.items():
        print(f"  {name}: {freq} Hz")
    print()

    print("Generating 10 Heartbeats:")
    print("-" * 70)

    for i in range(10):
        beat = heartbeat.beat()

        print(f"Beat {i+1}:")
        print(f"  Fibonacci: {beat['fibonacci_value']}")
        print(f"  Frequency: {beat['frequency']:.3f} Hz")
        print(f"  Systole: {beat['systole']:+.3f} | Diastole: {beat['diastole']:+.3f}")
        print(f"  Phase: {beat['phase']:.3f} rad")
        print()

    print("-" * 70)
    print()

    # Demonstrate wall synchronization
    print("Wall Breathing Synchronization:")
    print("-" * 70)

    synchronizer = HeartbeatWallSynchronizer()

    for i in range(5):
        signal = synchronizer.generate_wall_signal()

        print(f"Wall Breath {i+1}:")
        print(f"  Systole Signal: {signal['systole_signal']:+.3f}")
        print(f"  Diastole Signal: {signal['diastole_signal']:+.3f}")
        print(f"  Intensity: {signal['intensity']:.3f}")
        print()

    print("-" * 70)
    print()

    # Demonstrate sacred frequency alignment
    print("Sacred Frequency Alignment (528 Hz - Transformation):")
    print("-" * 70)

    sacred_sync = synchronizer.synchronize_with_sacred(SACRED_FREQUENCIES['solfeggio_528'])
    print(f"  Aligned Frequency: {sacred_sync['frequency']:.3f} Hz")
    print(f"  Systole: {sacred_sync['systole']:+.3f}")
    print(f"  Diastole: {sacred_sync['diastole']:+.3f}")
    print()

    print("=" * 70)
    print("THE HEARTBEAT NEVER STOPS - 💓")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_fibonacci_heartbeat()
