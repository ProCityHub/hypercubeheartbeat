#!/usr/bin/env python3
"""
WALL PHYSICS - LIGHT BENDING & STRING THEORY
=============================================

The walls of the cube are STRING THEORY - flexible, vibrating membranes.

Key Principles:
1. Light bends walls, walls bend light (reciprocal relationship)
2. Atoms are "knots" in the lattice (stable wall configurations)
3. Double-slit: particle = localized wall bending
4. Observation changes curvature, thus changes light propagation

Physics:
- Walls are elastic membranes (string theory)
- Light carries electromagnetic momentum
- Momentum creates pressure on walls
- Pressure deforms geometry (General Relativity)
- Deformed geometry redirects light (geodesics)
- Self-sustaining feedback loop
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Any
from lattice_law import PHI, FrequencyPattern

# Physical Constants (SI units)
C = 299792458  # Speed of light (m/s)
H = 6.62607015e-34  # Planck constant (J⋅s)
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity

# Lattice Constants
WALL_ELASTICITY = 0.6  # Artifact constant
STRING_TENSION = PHI  # Golden ratio tension


class PhotonPacket:
    """
    Photon as wave packet with momentum

    E = hν (energy)
    p = E/c (momentum)
    """

    def __init__(self, frequency: float, amplitude: float = 1.0):
        self.frequency = frequency  # Hz
        self.amplitude = amplitude

        # Calculate energy and momentum
        self.energy = H * frequency  # E = hν
        self.momentum = self.energy / C  # p = E/c

        # Position and direction
        self.position = np.array([0.0, 0.0, 0.0])
        self.direction = np.array([1.0, 0.0, 0.0])  # Default: +X

        # Wavelength
        self.wavelength = C / frequency  # λ = c/ν

    def to_frequency_pattern(self) -> FrequencyPattern:
        """Convert to FrequencyPattern for lattice system"""
        return FrequencyPattern(
            frequency=self.frequency,
            phase=0.0,
            amplitude=self.amplitude
        )


class StringWall:
    """
    Wall as vibrating string (string theory)

    The wall is a 2D membrane that can vibrate and deform
    """

    def __init__(self, wall_id: int, normal: Tuple[float, float, float]):
        self.wall_id = wall_id
        self.normal = np.array(normal)

        # Geometric properties
        self.curvature_tensor = np.zeros((3, 3))  # Riemann curvature
        self.metric_tensor = np.eye(3)  # Spacetime metric (starts flat)

        # String properties
        self.tension = STRING_TENSION
        self.elasticity = WALL_ELASTICITY

        # Vibration modes
        self.modes = []  # Harmonic modes

    def apply_pressure(self, photon: PhotonPacket, contact_point: np.ndarray) -> None:
        """
        Apply electromagnetic pressure from photon

        Pressure = momentum flux = p/A
        """
        # Photon momentum creates pressure
        pressure = photon.momentum / (self.elasticity ** 2)  # Pressure per unit area

        # Pressure deforms geometry
        deformation = pressure * photon.direction

        # Update curvature tensor
        for i in range(3):
            for j in range(3):
                self.curvature_tensor[i, j] += deformation[i] * deformation[j] * 0.1

        # Update metric tensor (spacetime curvature)
        self.metric_tensor += self.curvature_tensor * 0.01

    def compute_geodesic(self, photon: PhotonPacket) -> np.ndarray:
        """
        Compute light geodesic through curved wall geometry

        Uses Christoffel symbols (connection coefficients)
        Γᵢⱼᵏ = ½ gᵏˡ (∂gⱼˡ/∂xⁱ + ∂gᵢˡ/∂xʲ - ∂gᵢⱼ/∂xˡ)
        """
        # Simplified geodesic: light follows curvature

        # Christoffel symbols (connection)
        christoffel = self._compute_christoffel_symbols()

        # Geodesic deviation
        deviation = np.zeros(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    deviation[i] += christoffel[i, j, k] * photon.direction[j] * photon.direction[k]

        # New trajectory
        new_direction = photon.direction + deviation * 0.1

        # Normalize
        norm = np.linalg.norm(new_direction)
        if norm > 0:
            new_direction /= norm

        return new_direction

    def _compute_christoffel_symbols(self) -> np.ndarray:
        """
        Compute Christoffel symbols from metric tensor

        Simplified version for computational efficiency
        """
        christoffel = np.zeros((3, 3, 3))

        # Metric derivatives (simplified)
        metric_grad = np.gradient(self.metric_tensor, axis=0)

        # Γᵢⱼᵏ computation (simplified)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    christoffel[i, j, k] = 0.5 * (
                        self.metric_tensor[k, i] * metric_grad[j] +
                        self.metric_tensor[k, j] * metric_grad[i] -
                        self.metric_tensor[i, j] * metric_grad[k]
                    ) if isinstance(metric_grad[i], (int, float)) else 0.0

        return christoffel

    def add_vibration_mode(self, frequency: float, amplitude: float) -> None:
        """Add vibrational mode to string wall"""
        self.modes.append({'frequency': frequency, 'amplitude': amplitude})

    def vibrate(self, time: float) -> float:
        """
        Calculate wall vibration at given time

        Superposition of all vibrational modes
        """
        vibration = 0.0

        for mode in self.modes:
            vibration += mode['amplitude'] * math.sin(2 * math.pi * mode['frequency'] * time)

        return vibration


class AtomKnot:
    """
    Atom as stable knot in the lattice

    Atoms form when wall curvature creates stable, self-reinforcing patterns
    """

    def __init__(self, atomic_number: int):
        self.atomic_number = atomic_number
        self.electron_count = atomic_number

        # Wall configuration (6 walls for 3D cube)
        self.wall_curvatures = np.zeros(6)

        # Stability measure
        self.stability = 0.0

    def form_from_walls(self, wall_curvatures: List[float]) -> bool:
        """
        Form atom from wall curvature pattern

        Stable configuration = atom exists
        Unstable configuration = atom disappears
        """
        self.wall_curvatures = np.array(wall_curvatures[:6])

        # Carbon (Z=6) has 6 electrons - 6 walls!
        if self.atomic_number == 6:
            # Carbon forms when all 6 walls have similar curvature
            curvature_variance = np.var(self.wall_curvatures)
            self.stability = 1.0 / (1.0 + curvature_variance)

            return self.stability > 0.5

        return False

    def is_stable(self) -> bool:
        """Check if atom configuration is stable"""
        return self.stability > 0.5

    def get_electron_configuration(self) -> str:
        """Get electron configuration"""
        if self.atomic_number == 6:  # Carbon
            return "1s² 2s² 2p²"
        return f"Z={self.atomic_number}"


class DoubleSlit:
    """
    Double-slit experiment resolved through wall bending

    Key insight: "Particle" isn't a thing - it's localized wall bending
    Observation changes curvature → changes light propagation
    """

    def __init__(self, slit_separation: float = 1e-4):
        self.slit_separation = slit_separation  # meters
        self.wall = StringWall(0, (0, 0, 1))  # Screen wall

        # Observation state
        self.observed = False

    def propagate_photon(self, photon: PhotonPacket, observe: bool = False) -> Dict[str, Any]:
        """
        Propagate photon through double slit

        Args:
            photon: Incident photon
            observe: Whether to observe which slit

        Returns: Detection pattern
        """
        self.observed = observe

        if observe:
            # Observation changes wall curvature
            # This collapses the wave function
            return self._particle_like_detection(photon)
        else:
            # No observation - wave interference
            return self._wave_like_detection(photon)

    def _particle_like_detection(self, photon: PhotonPacket) -> Dict[str, Any]:
        """
        Particle-like behavior when observed

        Observation = localized wall bending
        """
        # Wall bends at specific point
        contact_point = np.array([0.0, 0.0, 0.0])
        self.wall.apply_pressure(photon, contact_point)

        # Photon takes definite path (geodesic through bent geometry)
        trajectory = self.wall.compute_geodesic(photon)

        return {
            'pattern': 'particle',
            'trajectory': trajectory,
            'wall_curvature': np.trace(self.wall.curvature_tensor),
            'observed': True
        }

    def _wave_like_detection(self, photon: PhotonPacket) -> Dict[str, Any]:
        """
        Wave-like behavior when not observed

        No observation = distributed wall bending
        """
        # Wall vibrates at photon frequency (superposition)
        self.wall.add_vibration_mode(photon.frequency, photon.amplitude)

        # Interference pattern from wall vibration
        interference = self.wall.vibrate(1.0 / photon.frequency)

        return {
            'pattern': 'wave',
            'interference': interference,
            'wall_modes': len(self.wall.modes),
            'observed': False
        }


def demonstrate_wall_physics():
    """Demonstrate wall physics principles"""
    print("=" * 70)
    print("WALL PHYSICS - LIGHT BENDING & STRING THEORY")
    print("=" * 70)
    print()

    # 1. Photon-Wall Interaction
    print("1. PHOTON-WALL INTERACTION")
    print("-" * 70)

    photon = PhotonPacket(frequency=528e12)  # 528 THz (green light)
    wall = StringWall(0, (1, 0, 0))

    print(f"Photon frequency: {photon.frequency/1e12:.1f} THz")
    print(f"Photon energy: {photon.energy:.2e} J")
    print(f"Photon momentum: {photon.momentum:.2e} kg⋅m/s")
    print()

    # Apply pressure
    contact = np.array([0.5, 0.5, 0.0])
    wall.apply_pressure(photon, contact)

    print(f"Wall curvature (trace): {np.trace(wall.curvature_tensor):.6f}")
    print(f"Wall metric determinant: {np.linalg.det(wall.metric_tensor):.6f}")
    print()

    # Compute geodesic
    new_direction = wall.compute_geodesic(photon)
    print(f"Original direction: {photon.direction}")
    print(f"Geodesic direction: {new_direction}")
    print()

    # 2. Atom Formation (Carbon)
    print("2. ATOM FORMATION - CARBON (Z=6)")
    print("-" * 70)

    carbon = AtomKnot(atomic_number=6)

    # Simulate 6 wall curvatures
    wall_curvatures = [0.1, 0.11, 0.09, 0.10, 0.12, 0.10]  # Similar curvatures

    formed = carbon.form_from_walls(wall_curvatures)

    print(f"Wall curvatures: {wall_curvatures}")
    print(f"Carbon formed: {formed}")
    print(f"Stability: {carbon.stability:.3f}")
    print(f"Configuration: {carbon.get_electron_configuration()}")
    print()

    # Unstable configuration
    unstable_curvatures = [0.1, 0.5, 0.05, 0.8, 0.01, 0.9]  # Varied
    unstable = carbon.form_from_walls(unstable_curvatures)

    print(f"Unstable curvatures: {unstable_curvatures}")
    print(f"Carbon formed: {unstable}")
    print(f"Stability: {carbon.stability:.3f}")
    print()

    # 3. Double-Slit Experiment
    print("3. DOUBLE-SLIT EXPERIMENT RESOLUTION")
    print("-" * 70)

    slit = DoubleSlit()
    photon_slit = PhotonPacket(frequency=500e12)  # 500 THz

    # Unobserved (wave behavior)
    result_wave = slit.propagate_photon(photon_slit, observe=False)

    print("Unobserved (wave-like):")
    print(f"  Pattern: {result_wave['pattern']}")
    print(f"  Interference: {result_wave['interference']:.6f}")
    print(f"  Wall vibrational modes: {result_wave['wall_modes']}")
    print()

    # Observed (particle behavior)
    slit_observed = DoubleSlit()
    result_particle = slit_observed.propagate_photon(photon_slit, observe=True)

    print("Observed (particle-like):")
    print(f"  Pattern: {result_particle['pattern']}")
    print(f"  Trajectory: {result_particle['trajectory']}")
    print(f"  Wall curvature: {result_particle['wall_curvature']:.6f}")
    print()

    print("=" * 70)
    print("WALLS BEND LIGHT, LIGHT BENDS WALLS - THE ETERNAL DANCE")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_wall_physics()
