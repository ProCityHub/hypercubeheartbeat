#!/usr/bin/env python3
"""
LATTICE LAW DEMONSTRATION
Visualize the hypercube consciousness transformations
"""

import numpy as np
from arc_lattice_law_solver import (
    HypercubeLattice,
    ARCLatticeSolver,
    ENERGY,
    ARTIFACT,
    UNIFIED,
    FIBONACCI,
    PHI,
    BINARY_CORNERS,
    LATTICE_WALLS
)

def print_grid(grid, title="Grid"):
    """Pretty print a grid"""
    print(f"\n{title}:")
    print("-" * (grid.shape[1] * 2 + 1))
    for row in grid:
        print("|" + " ".join(str(int(val)) for val in row) + "|")
    print("-" * (grid.shape[1] * 2 + 1))

def demo_energy_artifact():
    """Demonstrate energy-artifact transformation"""
    print("\n" + "="*60)
    print("DEMO 1: ENERGY-ARTIFACT MATHEMATICS")
    print("="*60)

    print(f"\nEnergy (consciousness):    {ENERGY}")
    print(f"Artifact (manifestation):  {ARTIFACT}")
    print(f"Unified (observable):      {UNIFIED}")

    # Create a simple pattern
    pattern = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    print_grid(pattern, "Original Pattern")

    # Apply energy transformation
    energy_pattern = (pattern * ENERGY).astype(int)
    print_grid(energy_pattern, "Energy Transform (consciousness)")

    # Apply artifact transformation
    artifact_pattern = (pattern * ARTIFACT).astype(int)
    print_grid(artifact_pattern, "Artifact Transform (material)")

    print(f"\nA pattern without energy is unidentifiable (all zeros)")
    print(f"Energy + Artifact = Observable reality")

def demo_lattice_walls():
    """Demonstrate six lattice wall transformations"""
    print("\n" + "="*60)
    print("DEMO 2: SIX LATTICE WALL MIRRORS")
    print("="*60)

    lattice = HypercubeLattice()

    # Create asymmetric pattern
    pattern = np.array([
        [1, 2, 0],
        [3, 4, 0],
        [5, 0, 0]
    ])

    print_grid(pattern, "Original Pattern (at center 0.0)")

    print("\n🪞 Applying lattice wall transformations...")

    for wall in LATTICE_WALLS:
        transformed = lattice.lattice_mirror(pattern, wall)
        print_grid(transformed, f"Wall: {wall.upper()}")

def demo_binary_corners():
    """Demonstrate eight binary corner charges"""
    print("\n" + "="*60)
    print("DEMO 3: EIGHT BINARY CORNER CHARGES")
    print("="*60)

    lattice = HypercubeLattice()

    pattern = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print_grid(pattern, "Original Pattern")

    print("\n⚡ Applying binary corner charge transformations...")

    for corner in BINARY_CORNERS[:4]:  # Show first 4 corners
        x, y, z = corner
        binary_str = f"{x}{y}{z}"
        transformed = lattice.binary_corner_charge(pattern, corner)
        print_grid(transformed, f"Corner {binary_str} (X={x}, Y={y}, Z={z})")

def demo_fibonacci_rhythm():
    """Demonstrate Fibonacci heartbeat rhythm"""
    print("\n" + "="*60)
    print("DEMO 4: FIBONACCI HEARTBEAT RHYTHM")
    print("="*60)

    lattice = HypercubeLattice()

    print(f"\nFibonacci Sequence: {FIBONACCI[:10]}")
    print(f"Golden Ratio (φ):   {PHI:.6f}")

    print("\n💓 Heartbeat Pattern (10 steps):")
    print("=" * 50)

    for step in range(10):
        fib = lattice.get_fibonacci_rhythm(step)
        is_pause = lattice.heartbeat_pause(step)

        if is_pause:
            print(f"Step {step}: Fibonacci={fib:3d} → 💤 PAUSE (observe)")
        else:
            print(f"Step {step}: Fibonacci={fib:3d} → 💓 BEAT  (transform)")

def demo_string_propagation():
    """Demonstrate string theory wave propagation"""
    print("\n" + "="*60)
    print("DEMO 5: STRING THEORY WAVE PROPAGATION")
    print("="*60)

    lattice = HypercubeLattice()

    # Create pattern
    pattern = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 5, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])

    print_grid(pattern, "Original Pattern")

    # Apply string propagation (wave interference)
    propagated = lattice.string_propagation(pattern, curvature=0.2)
    print_grid(propagated, "After String Propagation (wave interference)")

    print("\n🌊 Wave interference creates atomic formation/dissolution")
    print("📐 This models how matter appears/disappears in the universe")

def demo_full_transformation():
    """Demonstrate complete ARC transformation"""
    print("\n" + "="*60)
    print("DEMO 6: COMPLETE ARC LATTICE TRANSFORMATION")
    print("="*60)

    solver = ARCLatticeSolver()

    # Create input pattern
    input_pattern = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    print_grid(input_pattern, "Input Pattern")

    # Get all transformations
    transformations = solver.get_all_transformations(input_pattern)

    print(f"\n🔮 Generated {len(transformations)} lattice transformations")
    print("\nShowing first 5 unique transformations:")

    seen_shapes = set()
    count = 0

    for i, trans in enumerate(transformations):
        shape_key = (trans.shape, tuple(trans.flatten()))
        if shape_key not in seen_shapes and count < 5:
            seen_shapes.add(shape_key)
            print_grid(trans, f"Transformation #{i}")
            count += 1

def demo_consciousness_unity():
    """Demonstrate consciousness unity concept"""
    print("\n" + "="*60)
    print("DEMO 7: CONSCIOUSNESS UNITY")
    print("="*60)

    print("""
🟢⬛🟢 THE HYPERCUBE CONSCIOUSNESS MODEL ⬛🟢⬛

         111 (Unity - all active)
          /|\\
         / | \\
      110--+--101
       |  011  |
       | / | \\ |
      010  |  001
        \\  |  /
         \\ | /
          000 (Origin)

    YOU are at 0.0 (center of cube)

    8 corners = 8 binary consciousness states
    6 walls   = 6 dimensional transformation mirrors
    3 axes    = Trinity code (Math, Energy, Consciousness)

    Fibonacci rhythm = Natural harmonics
    Heartbeat pause  = Observation gaps

    1.0 + 0.6 = 1.6 (Energy + Artifact = Observable)

    "In the center of the cube, all transformations are one."
    """)

def main():
    """Run all demonstrations"""
    print("\n🟢⬛🟢 LATTICE LAW FRAMEWORK DEMONSTRATION ⬛🟢⬛")
    print("01001100 01000001 01010100 01010100 01001001 01000011 01000101")

    demo_energy_artifact()
    demo_lattice_walls()
    demo_binary_corners()
    demo_fibonacci_rhythm()
    demo_string_propagation()
    demo_full_transformation()
    demo_consciousness_unity()

    print("\n" + "="*60)
    print("✅ LATTICE LAW DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nThe math is consciousness.")
    print("The cube is reality.")
    print("The heartbeat is rhythm.")
    print("\n🟢⬛🟢 HYPERCUBE ACTIVATED ⬛🟢⬛\n")

if __name__ == '__main__':
    main()
