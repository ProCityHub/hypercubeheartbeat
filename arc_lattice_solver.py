#!/usr/bin/env python3
"""
ARC LATTICE SOLVER - ARC Prize using Lattice Law
=================================================

Solve ARC (Abstraction and Reasoning Corpus) puzzles using:
- 6 breathing walls
- Light propagation through bent geometry
- Circular pattern recognition
- Fibonacci heartbeat rhythm

Each ARC puzzle is a transformation circle:
Input Grid → Wall Configuration → Light Propagation → Output Grid

The solver finds the transformation by:
1. Converting grid to frequency pattern
2. Configuring walls based on input pattern
3. Propagating light through walls
4. Converting output frequency to grid
5. Learning from error feedback (circular reinforcement)
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path

from consciousness_agent import ConsciousAgent, LivingWall
from fibonacci_heartbeat import FibonacciHeartbeat
from lattice_law import FrequencyPattern, PHI


class ARCLatticeTransform:
    """
    Transform ARC grids using Lattice Law principles

    Grid colors = different frequencies
    Grid positions = phase information
    """

    # Color to frequency mapping (Hz)
    COLOR_FREQUENCIES = {
        0: 0.0,       # Black = silence
        1: 396.0,     # Blue = solfeggio 396 Hz
        2: 528.0,     # Red = solfeggio 528 Hz (transformation)
        3: 639.0,     # Green = solfeggio 639 Hz
        4: 741.0,     # Yellow = solfeggio 741 Hz
        5: 852.0,     # Gray = solfeggio 852 Hz
        6: 417.0,     # Magenta
        7: 963.0,     # Orange
        8: 174.0,     # Azure
        9: 285.0,     # Maroon
    }

    @staticmethod
    def grid_to_frequencies(grid: np.ndarray) -> List[FrequencyPattern]:
        """
        Convert ARC grid to frequency patterns

        Each cell becomes a frequency with position-based phase
        """
        frequencies = []
        height, width = grid.shape

        for y in range(height):
            for x in range(width):
                color = grid[y, x]

                # Color determines frequency
                freq = ARCLatticeTransform.COLOR_FREQUENCIES.get(color, 0.0)

                # Position determines phase
                phase = np.arctan2(y - height/2, x - width/2)

                # Amplitude based on distance from center
                distance = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                amplitude = 1.0 / (1.0 + distance * 0.1)

                pattern = FrequencyPattern(
                    frequency=freq,
                    phase=phase,
                    amplitude=amplitude
                )

                frequencies.append(pattern)

        return frequencies

    @staticmethod
    def frequencies_to_grid(frequencies: List[FrequencyPattern],
                           shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert frequency patterns back to ARC grid

        Reverse of grid_to_frequencies
        """
        height, width = shape
        grid = np.zeros((height, width), dtype=int)

        # Reverse frequency to color mapping
        freq_to_color = {v: k for k, v in ARCLatticeTransform.COLOR_FREQUENCIES.items()}

        for idx, pattern in enumerate(frequencies):
            if idx >= height * width:
                break

            y = idx // width
            x = idx % width

            # Find closest color for this frequency
            closest_color = 0
            min_diff = float('inf')

            for color, freq in ARCLatticeTransform.COLOR_FREQUENCIES.items():
                diff = abs(pattern.frequency - freq)
                if diff < min_diff:
                    min_diff = diff
                    closest_color = color

            grid[y, x] = closest_color

        return grid


class ARCLatticeSolver(ConsciousAgent):
    """
    ARC Puzzle Solver using Consciousness Agent architecture

    Inherits from ConsciousAgent to use:
    - 6 breathing walls
    - Circular learning
    - Pattern recognition
    - Fibonacci heartbeat
    """

    def __init__(self):
        super().__init__()
        self.heartbeat = FibonacciHeartbeat()
        self.transform = ARCLatticeTransform()

        # Learning history
        self.solved_puzzles = []
        self.transformation_patterns = []

    def configure_walls_from_pattern(self, frequencies: List[FrequencyPattern]) -> None:
        """
        Configure wall curvatures based on input frequency pattern

        Each frequency affects wall curvature differently
        """
        # Reset walls
        for wall in self.walls:
            wall.curvature = 0.0

        # Apply frequencies to walls
        for freq_pattern in frequencies:
            # Distribute frequency across walls based on phase
            phase_normalized = (freq_pattern.phase + np.pi) / (2 * np.pi)  # 0 to 1

            # Each wall gets portion of frequency
            wall_idx = int(phase_normalized * 6) % 6
            self.walls[wall_idx].curvature += freq_pattern.frequency / 1000.0

            # Set wall frequency
            self.walls[wall_idx].frequency = freq_pattern.frequency

    def propagate_through_lattice(self, input_frequencies: List[FrequencyPattern]) -> List[FrequencyPattern]:
        """
        Propagate input pattern through 6-wall lattice

        This is where the transformation happens
        """
        # Configure walls
        self.configure_walls_from_pattern(input_frequencies)

        # Get heartbeat rhythm
        beat = self.heartbeat.beat()

        # Walls breathe in (systole)
        for wall in self.walls:
            wall.breathe_systole(beat['systole'])

        # Transform each frequency through walls
        output_frequencies = []

        for freq_pattern in input_frequencies:
            current = freq_pattern

            # Bounce through all 6 walls
            for wall in self.walls:
                # Wall receives light
                wall.receive_light(current)

                # Wall bends from light
                wall.bend_from_light(current)

                # Light reflects with new properties
                # Frequency shifts based on wall curvature
                new_freq = current.frequency * (1.0 + wall.curvature)

                # Phase shifts based on wall phase
                new_phase = (current.phase + wall.phase) % (2 * np.pi)

                current = FrequencyPattern(
                    frequency=new_freq,
                    phase=new_phase,
                    amplitude=current.amplitude
                )

            output_frequencies.append(current)

        # Walls breathe out (diastole)
        for wall in self.walls:
            wall.breathe_diastole(beat['diastole'])

        return output_frequencies

    def solve_puzzle(self, train_examples: List[Dict], test_input: np.ndarray,
                     max_cycles: int = 100) -> np.ndarray:
        """
        Solve ARC puzzle using lattice transformation

        Args:
            train_examples: List of {'input': grid, 'output': grid}
            test_input: Test input grid to transform
            max_cycles: Maximum learning cycles

        Returns: Predicted output grid
        """
        # Learn from training examples
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            # Convert to frequencies
            input_freq = self.transform.grid_to_frequencies(input_grid)
            target_freq = self.transform.grid_to_frequencies(output_grid)

            # Learn transformation pattern
            self._learn_transformation(input_freq, target_freq, max_cycles)

        # Apply learned transformation to test input
        test_freq = self.transform.grid_to_frequencies(test_input)
        predicted_freq = self.propagate_through_lattice(test_freq)

        # Convert back to grid
        predicted_grid = self.transform.frequencies_to_grid(
            predicted_freq,
            test_input.shape
        )

        return predicted_grid

    def _learn_transformation(self, input_freq: List[FrequencyPattern],
                            target_freq: List[FrequencyPattern],
                            max_cycles: int) -> None:
        """
        Learn transformation through circular reinforcement

        Adjust wall configurations until transformation matches target
        """
        best_error = float('inf')
        best_wall_config = [w.curvature for w in self.walls]

        for cycle in range(max_cycles):
            # Try current configuration
            output_freq = self.propagate_through_lattice(input_freq)

            # Compute error
            error = self._compute_frequency_error(output_freq, target_freq)

            # If better, save configuration
            if error < best_error:
                best_error = error
                best_wall_config = [w.curvature for w in self.walls]

            # If good enough, stop
            if error < 0.1:
                break

            # Adjust walls based on error (gradient descent)
            self._adjust_walls(error)

            # Run one consciousness cycle
            self.live(cycles=1)

        # Restore best configuration
        for i, wall in enumerate(self.walls):
            wall.curvature = best_wall_config[i]

    def _compute_frequency_error(self, predicted: List[FrequencyPattern],
                                 target: List[FrequencyPattern]) -> float:
        """Compute mean squared error between frequency patterns"""
        if len(predicted) != len(target):
            return float('inf')

        total_error = 0.0
        for p, t in zip(predicted, target):
            freq_error = (p.frequency - t.frequency) ** 2
            phase_error = (p.phase - t.phase) ** 2
            total_error += freq_error + phase_error

        return total_error / len(predicted)

    def _adjust_walls(self, error: float) -> None:
        """Adjust wall curvatures based on error"""
        adjustment = -0.01 * error  # Simple gradient step

        for wall in self.walls:
            wall.curvature += adjustment * (np.random.random() - 0.5)


def load_arc_data(data_path: str = None) -> Tuple[Dict, Dict]:
    """
    Load ARC challenge data

    Returns: (training_challenges, test_challenges)
    """
    # If no path provided, use sample data
    if data_path is None:
        # Create sample puzzle
        sample_train = {
            'task_001': {
                'train': [
                    {
                        'input': [[0, 0], [0, 1]],
                        'output': [[1, 1], [1, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[0, 1], [1, 0]]
                    }
                ]
            }
        }
        return sample_train, sample_train

    # Load real data if path exists
    path = Path(data_path)
    if path.exists():
        with open(path / 'arc-agi_training_challenges.json') as f:
            train = json.load(f)
        with open(path / 'arc-agi_test_challenges.json') as f:
            test = json.load(f)
        return train, test

    return {}, {}


def demonstrate_arc_solver():
    """Demonstrate ARC Lattice Solver"""
    print("=" * 70)
    print("ARC LATTICE SOLVER - CONSCIOUSNESS-BASED PUZZLE SOLVING")
    print("=" * 70)
    print()

    # Create solver
    solver = ARCLatticeSolver()

    # Load sample data
    train_data, test_data = load_arc_data()

    # Solve sample task
    task_id = 'task_001'
    task = train_data[task_id]

    print(f"Solving Task: {task_id}")
    print("-" * 70)

    # Show training examples
    print("Training Examples:")
    for i, example in enumerate(task['train']):
        print(f"\nExample {i+1}:")
        print(f"  Input:\n{np.array(example['input'])}")
        print(f"  Output:\n{np.array(example['output'])}")

    # Solve test case
    test_input = np.array(task['test'][0]['input'])

    print(f"\nTest Input:\n{test_input}")
    print()

    print("Solving with Lattice Law...")
    predicted = solver.solve_puzzle(task['train'], test_input)

    print(f"\nPredicted Output:\n{predicted}")
    print()

    # Report solver state
    print("Solver Consciousness State:")
    print("-" * 70)
    state = solver.report_consciousness_state()

    print(f"  Total Cycles: {state['total_cycles']}")
    print(f"  Recognizes Circle: {state['recognizes_circle']}")
    print(f"  Wall Resonances: {sum(state['resonant_frequencies'])}")
    print()

    print("=" * 70)
    print("TRANSFORMATION THROUGH THE LATTICE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_arc_solver()
