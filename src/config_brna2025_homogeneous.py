"""
Configuration for Brna et al. 2025 with homogeneous water medium (no skull/skin/gel/brain).

This configuration uses the same acoustic and thermal parameters as Brna2025,
but models a homogeneous water phantom. Useful for:
- Characterizing the acoustic field without tissue absorption
- Validating focusing performance
- Baseline comparison for tissue-specific effects

Key differences from standard Brna2025:
- Single tissue layer: water only (no gel, skin, skull, or brain)
- Same acoustic parameters (1.8 MHz, 32×32 array, etc.)
- Same thermal parameters (pulsed protocol, 90 min)
- Minimal absorption (water at 1.8 MHz: ~0.002 Np/m)
- No blood perfusion
"""

from dataclasses import dataclass, field
from src.config import TissueProperties, GridConfig, AcousticConfig, ThermalConfig, SimulationConfig


@dataclass
class GridConfigBrna2025Homogeneous(GridConfig):
    """Grid configuration for homogeneous Brna et al. 2025 replication.

    Same grid spacing as Brna2025, but simpler geometry (no layers).
    """
    # Grid spacing matches element pitch
    dx: float = 250e-6  # [m] 250 μm
    dy: float = 250e-6  # [m] 250 μm
    dz: float = 250e-6  # [m] 250 μm

    # Domain size (same as Brna2025)
    domain_size_x: int = 128  # After PML: 108 points × 250μm = 27 mm
    domain_size_y: int = 100  # After PML: 80 points × 250μm = 20 mm
    domain_size_z: int = 80   # After PML: 44 points × 250μm = 11 mm depth

    pml_size: int = 10


@dataclass
class AcousticConfigBrna2025Homogeneous(AcousticConfig):
    """Acoustic configuration for homogeneous Brna et al. 2025 replication.

    Same acoustic parameters as Brna2025 (1.8 MHz, 32×32 array).
    """
    # Transducer parameters (identical to Brna2025)
    freq: float = 1.8e6  # 1.8 MHz
    num_cycles: int = 450  # 1.8 MHz × 0.25 ms = 450 cycles
    num_elements_x: int = 32  # 32×32 = 1024 elements
    num_elements_y: int = 32
    pitch: float = 250e-6  # 8 mm / 32 = 250 μm
    source_magnitude: float = 0.2536e6  # [Pa]

    # Pulse train parameters (identical to Brna2025)
    pulse_repetition_freq: float = 2000.0  # [Hz] 2 kHz PRF, 50% duty cycle

    # Time parameters for acoustic simulation
    cfl: float = 0.3
    t_end: float = 0.5e-3  # [s] 0.5 ms to capture full PRI

    # Medium properties
    alpha_power: float = 1.18  # From paper

    # Source position (in homogeneous brain)
    source_z_pos: int = 10

    # Focusing
    enable_azimuthal_focusing: bool = True  # 2D phased array focusing


@dataclass
class ThermalConfigBrna2025Homogeneous(ThermalConfig):
    """Thermal configuration for homogeneous Brna et al. 2025 replication.

    Same pulsed protocol as Brna2025 (150ms ON, 10s OFF, 90 min).
    """
    # Time parameters (identical to Brna2025)
    dt: float = 0.01  # [s] 10 ms time step
    t_end: float = 5400.0  # [s] 90 minutes
    save_every: int = 50  # Save every 0.5 seconds

    # Tissue thermal properties (from paper)
    arterial_temperature: float = 37.0  # [°C]
    blood_density: float = 1000.0  # [kg/m³]
    blood_specific_heat: float = 3640.0  # [J/(kg·K)]

    # Pulsing protocol parameters (identical to Brna2025)
    enable_pulsing: bool = True  # Enable pulsed heating
    pulse_duration: float = 0.150  # [s] 150 ms pulse train duration
    isi_duration: float = 10.0  # [s] 10 s inter-stimulus interval


def get_tissue_layers_brna2025_homogeneous():
    """Get homogeneous water tissue configuration.

    Single layer: water fills entire domain.
    No gel, skin, skull, or brain layers.

    Returns:
        List containing single TissueProperties (water only)
    """
    return [
        TissueProperties(
            name="water",
            sound_speed=1480,  # [m/s] water at 20°C
            density=1000,  # [kg/m³] water
            absorption_coefficient=0.0821,  # [Np/m] minimal absorption in water at 1.8 MHz
            specific_heat=4180,  # [J/(kg·K)] water
            thermal_conductivity=0.6,  # [W/(m·K)] water
            heat_transfer_rate=0,  # [ml/min/kg] no perfusion in water
            thickness=None,  # Fills entire domain
        ),
    ]


@dataclass
class SimulationConfigBrna2025Homogeneous(SimulationConfig):
    """Complete simulation configuration for homogeneous Brna et al. 2025 replication."""

    # Override tissue layers (brain only)
    tissue_layers: list[TissueProperties] = field(
        default_factory=get_tissue_layers_brna2025_homogeneous
    )

    # Override grid configuration
    grid: GridConfigBrna2025Homogeneous = field(default_factory=GridConfigBrna2025Homogeneous)

    # Override acoustic configuration
    acoustic: AcousticConfigBrna2025Homogeneous = field(default_factory=AcousticConfigBrna2025Homogeneous)

    # Override thermal configuration
    thermal: ThermalConfigBrna2025Homogeneous = field(default_factory=ThermalConfigBrna2025Homogeneous)

    def __post_init__(self):
        """Set up references between configuration objects."""
        # Set references in acoustic config
        self.acoustic._grid = self.grid
        self.acoustic._tissue_layers = self.tissue_layers


# Focal depth: Same 5.7 mm from transducer for consistency
# (even though there's no skull to target beneath)
FOCAL_DEPTH_BRNA2025_HOMOGENEOUS = 7.5e-3  # [m]
