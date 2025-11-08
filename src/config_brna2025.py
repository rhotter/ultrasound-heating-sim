"""
Configuration for replicating Brna et al. 2025 ultrasound neuromodulation study.

Key parameters:
- 1024 CMUT elements in 8×8 mm square (32×32 array)
- Center frequency: 1.8 MHz
- Pulse duration: 0.25 ms (450 cycles)
- Pulse train: 50% duty cycle, 150 ms duration
- Inter-stimulus interval: 10 seconds
- Total exposure: 90 minutes
- Target depth: 2 mm beneath 1.2 mm skull (3.2 mm from surface)
"""

from dataclasses import dataclass, field
from src.config import TissueProperties, GridConfig, AcousticConfig, ThermalConfig, SimulationConfig


@dataclass
class GridConfigBrna2025(GridConfig):
    """Grid configuration for Brna et al. 2025 replication.

    Array is 8×8 mm with 32×32 elements, so pitch = 250 μm.
    Target depth is ~5.2 mm (1mm skin + 1.2mm skull + 3mm into brain).
    """
    # Grid spacing matches element pitch
    dx: float = 250e-6  # [m] 250 μm
    dy: float = 250e-6  # [m] 250 μm
    dz: float = 250e-6  # [m] 250 μm

    # Domain size
    domain_size_x: int = 128  # After PML: 108 points × 250μm = 27 mm (enough for 8mm array + margins)
    domain_size_y: int = 100  # After PML: 108 points × 250μm = 27 mm
    domain_size_z: int = 64   # After PML: 60 points × 250μm = 15 mm depth (enough for focus + margins)

    pml_size: int = 10


@dataclass
class AcousticConfigBrna2025(AcousticConfig):
    """Acoustic configuration for Brna et al. 2025 replication.

    1.8 MHz center frequency, 0.25 ms pulse duration.
    32×32 CMUT array in 8×8 mm.
    """
    # Transducer parameters
    freq: float = 1.8e6  # 1.8 MHz
    num_cycles: int = 450  # 1.8 MHz × 0.25 ms = 450 cycles
    num_elements_x: int = 32  # 32×32 = 1024 elements
    num_elements_y: int = 32
    pitch: float = 250e-6  # 8 mm / 32 = 250 μm
    source_magnitude: float = 0.2536e6  # [Pa]

    # Pulse train parameters
    # PRF = 1/PRI = 1/0.5ms = 2000 Hz
    # Duty cycle within train = 0.25ms / 0.5ms = 50%
    pulse_repetition_freq: float = 2000.0  # [Hz]

    # Time parameters for acoustic simulation (single pulse)
    cfl: float = 0.3
    t_end: float = 0.5e-3  # [s] 0.5 ms to capture full PRI

    # Medium properties
    alpha_power: float = 1.18  # From paper

    # Source position
    source_z_pos: int = 10

    # Focusing
    enable_azimuthal_focusing: bool = True  # 2D phased array focusing


@dataclass
class ThermalConfigBrna2025(ThermalConfig):
    """Thermal configuration for Brna et al. 2025 replication.

    90-minute exposure with pulsed protocol:
    - Within pulse train: 50% duty cycle (already in acoustic PRF)
    - Across pulse trains: 150 ms PTD / 10 s ISI
    - Pulsing is handled in the thermal solver (heat ON/OFF cycling)
    """
    # Time parameters
    dt: float = 0.01  # [s] 10 ms time step for long simulation
    t_end: float = 5400.0  # [s] 90 minutes
    save_every: int = 50  # Save every 0.5 seconds (0.01s × 50)

    # Tissue thermal properties (from paper)
    arterial_temperature: float = 37.0  # [°C]
    blood_density: float = 1000.0  # [kg/m³]
    blood_specific_heat: float = 3640.0  # [J/(kg·K)]

    # Pulsing protocol parameters
    enable_pulsing: bool = True  # Enable pulsed heating
    pulse_duration: float = 0.150  # [s] 150 ms pulse train duration
    isi_duration: float = 10.0  # [s] 10 s inter-stimulus interval


def get_tissue_layers_brna2025():
    """Get tissue layer configuration for Brna et al. 2025.

    Layers from surface:
    0. PML: 2.5 mm (10 grid points, not a tissue layer)
    1. Gel/water coupling: 4 mm (for transducer at z=10 + margins)
    2. Skin: 1 mm
    3. Skull: 1.2 mm
    4. Brain: remainder (target at 2mm beneath skull)

    Transducer position: z=10 (2.5 mm from grid origin, in gel layer)

    Returns:
        List of TissueProperties
    """
    return [
        TissueProperties(
            name="gel",
            sound_speed=1500,  # [m/s] Water/gel
            density=1000,  # [kg/m³]
            absorption_coefficient=0,  # [Np/m] Minimal absorption
            specific_heat=4180,  # [J/(kg·K)] Water
            thermal_conductivity=0.6,  # [W/(m·K)] Water
            thickness=4e-3,  # 4 mm coupling layer
            heat_transfer_rate=0,  # No perfusion
        ),
        TissueProperties(
            name="skin",
            sound_speed=1624,  # [m/s]
            density=1109,  # [kg/m³]
            thickness=1e-3,  # 1 mm (updated from 2mm)
            absorption_coefficient=42.3,  # [Np/m] at 2 MHz
            specific_heat=3391,  # [J/(kg·K)]
            thermal_conductivity=0.37,  # [W/(m·K)]
            heat_transfer_rate=106,  # [ml/min/kg]
        ),
        TissueProperties(
            name="skull",
            sound_speed=2770,  # [m/s]
            density=1908,  # [kg/m³]
            thickness=1.2e-3,  # 1.2 mm (from paper)
            absorption_coefficient=109.1,  # [Np/m] at 2 MHz
            specific_heat=1313,  # [J/(kg·K)]
            thermal_conductivity=0.32,  # [W/(m·K)]
            heat_transfer_rate=10,  # [ml/min/kg]
        ),
        TissueProperties(
            name="brain",
            sound_speed=1468.4,  # [m/s] from paper (cortical tissue)
            density=1000,  # [kg/m³] from paper
            absorption_coefficient=2.4 * (1.8**1.18),  # [Np/m] using paper's formula: a * f^b at 1.8 MHz
            specific_heat=3640,  # [J/(kg·K)] from paper
            thermal_conductivity=0.51,  # [W/(m·K)]
            heat_transfer_rate=568,  # [ml/min/kg]
            thickness=None,  # Fills remainder
        ),
    ]


@dataclass
class SimulationConfigBrna2025(SimulationConfig):
    """Complete simulation configuration for Brna et al. 2025 replication."""

    # Override tissue layers
    tissue_layers: list[TissueProperties] = field(
        default_factory=get_tissue_layers_brna2025
    )

    # Override grid configuration
    grid: GridConfigBrna2025 = field(default_factory=GridConfigBrna2025)

    # Override acoustic configuration
    acoustic: AcousticConfigBrna2025 = field(default_factory=AcousticConfigBrna2025)

    # Override thermal configuration
    thermal: ThermalConfigBrna2025 = field(default_factory=ThermalConfigBrna2025)

    def __post_init__(self):
        """Set up references between configuration objects."""
        # Set references in acoustic config
        self.acoustic._grid = self.grid
        self.acoustic._tissue_layers = self.tissue_layers


def get_effective_intensity_brna2025(acoustic_intensity):
    """Scale acoustic intensity for the pulsed protocol.

    The pulsing protocol has two levels:
    1. Within pulse train: 50% duty cycle (0.25 ms pulse / 0.5 ms PRI)
       → Already accounted for in acoustic simulation (pulse_energy × PRF)
    2. Across pulse trains: 150 ms PTD / 10.15 s (PTD + ISI) ≈ 1.48% duty cycle
       → Applied here

    This function only scales by the across-trains duty cycle (1.48%)

    Args:
        acoustic_intensity: Intensity from acoustic simulation [W/m²]

    Returns:
        Time-averaged intensity for thermal simulation [W/m²]
    """
    PTD = 0.150  # [s] Pulse train duration
    ISI = 10.0   # [s] Inter-stimulus interval

    duty_cycle_across_trains = PTD / (PTD + ISI)

    return acoustic_intensity * duty_cycle_across_trains


# Focal depth calculation for reference:
# Transducer is at z=10 (2.5 mm from grid origin after PML)
# From transducer position:
#   - Remaining gel: 4 - 2.5 = 1.5 mm
#   - Skin: 1 mm
#   - Skull: 1.2 mm
#   - Target: 2 mm beneath skull (into brain)
# Total focal distance from TRANSDUCER: 1.5 + 1 + 1.2 + 2 = 5.7 mm
FOCAL_DEPTH_BRNA2025 = 5.7e-3  # [m]
