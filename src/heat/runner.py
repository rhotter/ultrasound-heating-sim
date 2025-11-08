import torch
import matplotlib.pyplot as plt
import os
from typing import Optional
from src.heat.simulator import BioheatSimulator
from src.heat.visualization import (
    plot_temperature_evolution,
    plot_temperature_field_slices,
    visualize_combined_results,
    make_temperature_video,
)
from src.config import SimulationConfig
import numpy as np


def run_heat_simulation(
    config: SimulationConfig,
    intensity_data: np.ndarray,
    output_dir: Optional[str] = None,
    steady_state: bool = False,
    save_properties: bool = False,
):
    """Run the bioheat simulation using provided intensity data.

    Args:
        config: The simulation configuration
        intensity_data: The acoustic intensity field
        output_dir: Directory to save output (plots, data). If None, no files are saved.
        steady_state: If True, use the steady state solver
        save_properties: If True, save tissue property distributions

    Returns:
        dict containing:
            - T_history: Temperature field history (list of tensors or single tensor for steady state)
            - times: Time points for the simulation
            - max_temps: Maximum temperatures over the entire domain at each time point
            - max_temps_skull: Maximum temperatures in skull region at each time point
            - max_temps_brain: Maximum temperatures in brain region at each time point
            - layer_map: Tissue layer map (tensor)
            - simulator: BioheatSimulator instance
    """
    print("\n=== Starting Heat Simulation ===")

    # Initialize simulator
    print("Initializing bioheat simulator...")
    simulator = BioheatSimulator(config)

    # Setup mesh
    print("Setting up computational mesh...")
    simulator.setup_mesh()

    # Visualize the layer map
    if output_dir is not None:
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
        plt.savefig(os.path.join(output_dir, "T0_thermal_tissue_layer_map.png"))
        plt.close()

    # Setup tissue properties
    print("Setting up tissue properties...")
    simulator.setup_tissue_properties()

    # Save property distributions if requested
    if save_properties and output_dir is not None:
        print("Saving tissue property distributions...")
        simulator.save_property_distributions(output_dir)

    # Setup heat source using acoustic intensity
    print("Setting up heat source from acoustic intensity...")
    intensity_tensor = torch.tensor(intensity_data, device=simulator.device)
    simulator.setup_heat_source(intensity_field=intensity_tensor)

    # Run simulation
    print("Running bioheat simulation...")
    T_history, times, max_temps = simulator.run_simulation(steady_state=steady_state)

    # Compute region-specific maximum temperatures (skull and brain)
    layer_map = simulator.get_layer_map()
    skull_mask = (layer_map == 2)  # Index 2 = skull
    brain_mask = (layer_map == 3)  # Index 3 = brain

    max_temps_skull = []
    max_temps_brain = []

    for T_field in T_history:
        T_tensor = torch.tensor(T_field, device=simulator.device) if not isinstance(T_field, torch.Tensor) else T_field

        # Max temperature in skull region
        if skull_mask.any():
            max_temp_skull = float(torch.max(T_tensor[skull_mask]).cpu().numpy())
        else:
            max_temp_skull = 0.0  # No skull voxels in domain
        max_temps_skull.append(max_temp_skull)

        # Max temperature in brain region
        if brain_mask.any():
            max_temp_brain = float(torch.max(T_tensor[brain_mask]).cpu().numpy())
        else:
            max_temp_brain = 0.0  # No brain voxels in domain
        max_temps_brain.append(max_temp_brain)

    # Visualize results
    if output_dir is not None:
        print("Plotting results...")

        if steady_state:
            # For steady state, we only create the final temperature distribution plots
            print("Creating steady state temperature visualizations...")

            # Temperature distribution
            fig, _ = plot_temperature_field_slices(T_history[0], config)
            plt.savefig(os.path.join(output_dir, "T1_steady_state_temperature.png"))
            plt.close()

            # Combined acoustic intensity and temperature visualization
            fig, _ = visualize_combined_results(intensity_tensor, T_history[0], config)
            plt.savefig(os.path.join(output_dir, "T2_steady_state_combined.png"))
            plt.close()

            print(f"Steady state temperature shape: {T_history.shape}")
            print("Simulation complete!")
        else:
            # For time-dependent simulation, create all plots
            # Temperature evolution
            fig, _ = plot_temperature_evolution(
                times,
                max_temps,
                max_temperatures_skull=max_temps_skull,
                max_temperatures_brain=max_temps_brain,
            )
            plt.savefig(os.path.join(output_dir, "T1_temperature_evolution.png"))
            plt.close()

            # Temperature distribution
            fig, _ = plot_temperature_field_slices(T_history[-1], config)
            plt.savefig(os.path.join(output_dir, "T2_temperature_distribution.png"))
            plt.close()

            # Combined acoustic intensity and temperature visualization
            fig, _ = visualize_combined_results(intensity_tensor, T_history[-1], config)
            plt.savefig(os.path.join(output_dir, "T3_acoustic_thermal_combined.png"))
            plt.close()

            # Create temperature evolution video (skip if ffmpeg not available)
            print("Creating temperature evolution video...")
            try:
                make_temperature_video(
                    T_history[::5],
                    config,
                    times[::5],
                    os.path.join(output_dir, "T4_temperature_evolution.mp4"),
                )
            except Exception as e:
                print(f"Skipping temperature video creation: {e}")

            print(f"Temperature history shape: {T_history.shape}")
            print(f"Number of time points: {len(times)}")
            print("Simulation complete!")
    else:
        if steady_state:
            print(f"Steady state temperature shape: {T_history.shape}")
        else:
            print(f"Temperature history shape: {T_history.shape}")
            print(f"Number of time points: {len(times)}")
        print("Simulation complete!")

    # Return simulation results for programmatic use
    return {
        "T_history": T_history,
        "times": times,
        "max_temps": max_temps,
        "max_temps_skull": max_temps_skull,
        "max_temps_brain": max_temps_brain,
        "layer_map": layer_map,
        "simulator": simulator,
    }
