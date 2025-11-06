# %%
"""
Test file comparing BioheatSimulator (custom implementation) with kWaveDiffusion (kWave Python port).

This test loads intensity data and runs both heating simulations with identical parameters,
then compares the maximum temperature evolution over time.
"""

# %%
# Imports
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.heat.simulator import BioheatSimulator
from src.config import SimulationConfig
sys.path.insert(0, str(project_root / "matlab-check"))
from kWaveDiffusion import kWaveDiffusion
from kwave.kgrid import kWaveGrid

# %%
def setup_kwave_diffusion(bioheat_sim, intensity_data, config):
    """
    Setup kWaveDiffusion simulator with parameters matching BioheatSimulator.

    Args:
        bioheat_sim: BioheatSimulator instance (after setup_mesh and setup_tissue_properties)
        intensity_data: Intensity field array [W/m²], shape (Nx, Ny, Nz)
        config: SimulationConfig object

    Returns:
        kWaveDiffusion instance ready to run
    """
    # Create kWaveGrid matching BioheatSimulator grid
    grid_config = config.grid
    kgrid = kWaveGrid(
        [grid_config.Nx, grid_config.Ny, grid_config.Nz],
        [grid_config.dx, grid_config.dy, grid_config.dz]
    )

    # Extract tissue properties from BioheatSimulator (convert torch tensors to numpy)
    # These are spatially-varying fields set up by BioheatSimulator
    density = bioheat_sim.rho.cpu().numpy()  # [kg/m³]
    specific_heat = bioheat_sim.c.cpu().numpy()  # [J/(kg·K)]
    thermal_conductivity = bioheat_sim.k.cpu().numpy()  # [W/(m·K)]

    # Blood perfusion rate: convert from B coefficient back to w_b
    # B = ρ_b * c_b * w_b, so w_b = B / (ρ_b * c_b)
    B_field = bioheat_sim.B.cpu().numpy()
    blood_density = config.thermal.blood_density
    blood_specific_heat = config.thermal.blood_specific_heat
    blood_perfusion_rate = B_field / (blood_density * blood_specific_heat)

    # Blood ambient temperature (arterial temperature)
    arterial_temp = config.thermal.arterial_temperature
    blood_ambient_temperature = np.full_like(density, arterial_temp)

    # Build medium dictionary for kWaveDiffusion
    medium = {
        'density': density,
        'thermal_conductivity': thermal_conductivity,
        'specific_heat': specific_heat,
        'blood_density': blood_density,
        'blood_specific_heat': blood_specific_heat,
        'blood_perfusion_rate': blood_perfusion_rate,
        'blood_ambient_temperature': blood_ambient_temperature,
    }

    # Compute heat source Q = 2 * absorption * intensity
    # Extract absorption coefficient from BioheatSimulator
    absorption = bioheat_sim.absorption.cpu().numpy()  # [Np/m]
    Q_field = 2.0 * absorption * intensity_data  # [W/m³]

    # Build source dictionary for kWaveDiffusion
    source = {
        'Q': Q_field,
        'T0': arterial_temp  # Initial temperature [°C]
    }

    # Create kWaveDiffusion instance
    kdiff = kWaveDiffusion(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=None,  # No sensor mask needed for this comparison
        use_kspace=True,  # Use k-space correction (more accurate)
        display_updates=False  # Suppress console output during test
    )

    return kdiff

# %%
# Configuration
data_dir = project_root / "data"
intensity_file = data_dir / "average_intensity.npy"

# Test parameters (short simulation for testing)
test_duration = 10.0  # seconds
dt = 0.01  # seconds (time step)
save_every = 1  # Save every 10 steps (0.1 seconds)

print("="*70)
print("Heating Simulation Comparison Test")
print("="*70)

# %%
# 1. Load intensity data
print("\n[1/5] Loading intensity data...")
intensity_data = np.load(intensity_file)
print(f"  Intensity shape: {intensity_data.shape}")
print(f"  Intensity range: [{intensity_data.min():.2e}, {intensity_data.max():.2e}] W/m²")
print(f"  Intensity mean: {intensity_data.mean():.2e} W/m²")

# %%
# 2. Setup and run BioheatSimulator
print("\n[2/5] Setting up BioheatSimulator...")

# Create config with test parameters
config = SimulationConfig()
config.thermal.dt = dt
config.thermal.t_end = test_duration
config.thermal.save_every = save_every

# Initialize BioheatSimulator
bioheat_sim = BioheatSimulator(config)
bioheat_sim.setup_mesh()
bioheat_sim.setup_tissue_properties()

# Setup heat source from intensity
intensity_tensor = torch.tensor(intensity_data, dtype=torch.float32, device=bioheat_sim.device)
bioheat_sim.setup_heat_source(intensity_field=intensity_tensor)

print(f"  Grid: {config.grid.Nx} × {config.grid.Ny} × {config.grid.Nz}")
print(f"  Grid spacing: {config.grid.dx*1e6:.1f} µm")
print(f"  Time step: {dt} s")
print(f"  Duration: {test_duration} s")
print(f"  Device: {bioheat_sim.device}")

# %%
print("\n  Running BioheatSimulator...")
T_history_bioheat, times_bioheat, max_temps_bioheat = bioheat_sim.run_simulation(steady_state=False)

# Convert to numpy arrays for comparison
times_bioheat = np.array(times_bioheat)
max_temps_bioheat = np.array(max_temps_bioheat)

print(f"  ✓ Completed {len(times_bioheat)} time points")
print(f"  ✓ Final max temperature: {max_temps_bioheat[-1]:.3f} °C")

# %%
# 3. Setup and run kWaveDiffusion
print("\n[3/5] Setting up kWaveDiffusion...")

kdiff = setup_kwave_diffusion(bioheat_sim, intensity_data, config)

print(f"  Grid: {kdiff.Nx} × {kdiff.Ny} × {kdiff.Nz}")
print(f"  Grid spacing: {kdiff.dx*1e6:.1f} µm")
print(f"  Time step: {dt} s")
print(f"  Duration: {test_duration} s")
print(f"  Diffusion coefficient (ref): {kdiff.diffusion_coeff_ref:.2e} m²/s")
print(f"  Perfusion coefficient (ref): {kdiff.perfusion_coeff_ref:.2e} 1/s")

# Check time step stability
dt_limit = kdiff.dt_limit
print(f"  Time step limit: {dt_limit:.2e} s")
if dt > dt_limit and not np.isinf(dt_limit):
    print(f"  ⚠ Warning: dt ({dt}) > dt_limit ({dt_limit:.2e}), simulation may be unstable!")

# %%
print("\n  Running kWaveDiffusion...")

# Run time-stepping loop
num_steps = int(test_duration / dt)
max_temps_kwave = []
times_kwave = []

for step in tqdm(range(num_steps), desc="kWaveDiffusion steps", unit="step"):
    # Take one time step
    kdiff.takeTimeStep(Nt=1, dt=dt)

    # Calculate time after step (to match BioheatSimulator which increments t before saving)
    t = (step + 1) * dt

    # Save at specified intervals (matching BioheatSimulator: step % save_every == 0 or final step)
    if step % save_every == 0 or step == num_steps - 1:
        max_temp = float(np.max(kdiff.T))
        max_temps_kwave.append(max_temp)
        times_kwave.append(t)

max_temps_kwave = np.array(max_temps_kwave)
times_kwave = np.array(times_kwave)

print(f"  ✓ Completed {len(times_kwave)} time points")
print(f"  ✓ Final max temperature: {max_temps_kwave[-1]:.3f} °C")

# %%
# 4. Compare results
print("\n[4/5] Comparing results...")

# Ensure we have matching time points
assert len(times_bioheat) == len(times_kwave), "Time point mismatch"

# Compute differences
abs_diff = np.abs(max_temps_bioheat - max_temps_kwave)
rel_diff = abs_diff / np.maximum(np.abs(max_temps_bioheat - 37.0), 1e-10)  # Relative to temperature rise

print(f"\n  Temperature comparison:")
print(f"    Mean absolute difference: {abs_diff.mean():.4f} °C")
print(f"    Max absolute difference: {abs_diff.max():.4f} °C")
print(f"    Mean relative difference: {rel_diff.mean()*100:.2f}%")
print(f"    Max relative difference: {rel_diff.max()*100:.2f}%")

# %%
# 5. Visualize and validate
print("\n[5/5] Creating comparison plot...")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Maximum temperature vs time
ax1 = axes[0]
ax1.plot(times_bioheat, max_temps_bioheat, 'b-', linewidth=2, label='BioheatSimulator')
ax1.plot(times_kwave, max_temps_kwave, 'r--', linewidth=2, label='kWaveDiffusion')
ax1.set_xlabel('Time [s]', fontsize=12)
ax1.set_ylabel('Maximum Temperature [°C]', fontsize=12)
ax1.set_title('Heating Simulation Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Absolute difference
ax2 = axes[1]
ax2.plot(times_bioheat, abs_diff, 'k-', linewidth=2)
ax2.set_xlabel('Time [s]', fontsize=12)
ax2.set_ylabel('Absolute Difference [°C]', fontsize=12)
ax2.set_title('Temperature Difference (|BioheatSimulator - kWaveDiffusion|)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()

# Save plot
output_file = project_root / "data" / "heating_comparison.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved comparison plot to: {output_file}")

plt.show()

# %%
