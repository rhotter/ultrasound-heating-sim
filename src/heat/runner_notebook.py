#%%
%load_ext autoreload
%autoreload 2


import torch
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../..")

from src.heat.simulator import BioheatSimulator
from src.heat.visualization import (
    plot_temperature_evolution,
    plot_temperature_field_slices,
    visualize_combined_results,
    make_temperature_video,
)
from src.config import SimulationConfig
import numpy as np

#%%
config = SimulationConfig()
intensity_data = np.load("../../data/average_intensity.npy")


print("\n=== Starting Heat Simulation ===")

# Initialize simulator
print("Initializing bioheat simulator...")
simulator = BioheatSimulator(config)

# Setup mesh
print("Setting up computational mesh...")
simulator.setup_mesh()

# Visualize the layer map
print("Visualizing tissue layer map...")
layer_map = simulator.get_layer_map()
plt.figure(figsize=(10, 5))
plt.imshow(
    layer_map[:, layer_map.shape[1] // 2, :].cpu().numpy().T,
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="Tissue Layer Index")
plt.title("Tissue Layer Map - Thermal Simulation (XZ mid-slice)")
plt.xlabel("X")
plt.ylabel("Z (tissue region only)")
plt.close()

# Setup tissue properties
print("Setting up tissue properties...")
simulator.setup_tissue_properties()

# Setup heat source using acoustic intensity
print("Setting up heat source from acoustic intensity...")
intensity_tensor = torch.tensor(intensity_data, device=simulator.device)
q = simulator.setup_heat_source(intensity_field=intensity_tensor)

#%%
# plot the intensity field
intensity_slice = intensity_tensor[:, config.grid.Ny // 2, :]
plt.imshow(intensity_slice.cpu().numpy().T, origin="lower", cmap="viridis")
plt.colorbar(label="Acoustic Intensity [W/m^2]")
plt.title("Acoustic Intensity (XZ mid-slice)")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()
#%%
# plot the medium
Kt = simulator.Kt.cpu().numpy()
Kt = Kt[:, config.grid.Ny // 2, :]
plt.imshow(Kt.T, origin="lower", cmap="viridis")
plt.colorbar(label="Thermal Conductivity [W/m/K]")
plt.title("Thermal Conductivity")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()

#%%
layer_map = simulator.layer_map.cpu().numpy()
layer_map = layer_map[:, config.grid.Ny // 2, :]
plt.imshow(layer_map.T, origin="lower", cmap="viridis")
plt.colorbar(label="Tissue Layer Index")
plt.title("Tissue Layer Map - Thermal Simulation (XZ mid-slice)")
plt.xlabel("X")
plt.ylabel("Z (tissue region only)")
plt.show()
#%%
# Run simulation
print("Running bioheat simulation...")
T_history, times, max_temps = simulator.run_simulation()

# Visualize results
print("Plotting results...")


# For time-dependent simulation, create all plots
# Temperature evolution
fig, _ = plot_temperature_evolution(times, max_temps)
plt.close()

# Temperature distribution
fig, _ = plot_temperature_field_slices(T_history[-1], config)
plt.close()

# Combined acoustic intensity and temperature visualization
fig, _ = visualize_combined_results(intensity_tensor, T_history[-1], config)
plt.close()

# Create temperature evolution video
# print("Creating temperature evolution video...")
# make_temperature_video(
#     T_history[::5],
#     config,
#     times[::5],
# )

print(f"Temperature history shape: {T_history.shape}")
print(f"Number of time points: {len(times)}")
print("Simulation complete!")
