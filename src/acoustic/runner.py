import os
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from src.acoustic.simulator import PressureSimulator
from src.acoustic.visualization import (
    plot_medium_properties,
    plot_intensity_field,
    make_pressure_video,
)
from src.config import SimulationConfig


def run_acoustic_simulation(
    config: SimulationConfig,
    output_dir: Optional[str],
    use_gpu: bool = True,
    focus_depth: Optional[float] = None,
    skip_videos: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the acoustic simulation to generate intensity and max pressure data.

    Args:
        config: Simulation configuration
        output_dir: Directory to save outputs (if None, no files are saved)
        use_gpu: Whether to use GPU for simulation
        focus_depth: Optional focus depth in meters (for elevational focusing)
        skip_videos: If True, skip generating video files

    Returns:
        tuple: (average_intensity, max_pressure, medium_sound_speed, pressure_data) arrays
    """
    print("\n=== Starting Acoustic Simulation ===")

# Initialize simulator
    simulator = PressureSimulator(config, focus_depth=focus_depth)

    # Set up the grid
    print("Setting up grid...")
    simulator.setup_grid()

    # Set up the medium
    print("Setting up medium...")
    simulator.setup_medium()

    # Plot the medium properties
    if output_dir is not None:
        fig, ax = plot_medium_properties(simulator.medium.sound_speed, config, focus_depth=focus_depth)
        plt.savefig(os.path.join(output_dir, "A0_medium_properties.png"))
        plt.close()

    # Set up source and sensor
    print("Setting up source and sensor...")
    simulator.setup_source_sensor()

    # Run the simulation
    print("Running acoustic simulation...")
    sensor_data = simulator.run_simulation(use_gpu=use_gpu)

    # Process and reshape the pressure data
    pressure_data = sensor_data["p"].reshape(
        -1,  # time steps
        config.grid.Nx,
        config.grid.Ny,
        config.grid.Nz,
        order="F",
    )

    # Plot max pressure
    max_pressure = np.max(pressure_data, axis=0)

    # Print maximum pressure analysis
    max_pressure_value = np.max(max_pressure)
    print(f"\n=== Acoustic Pressure Analysis ===")
    print(f"Max pressure (peak): {max_pressure_value/1e6:.3f} MPa")
    print(f"Max pressure (RMS): {max_pressure_value/(np.sqrt(2)*1e6):.3f} MPa")
    print(f"Source pressure: {config.acoustic.source_magnitude/1e6:.3f} MPa")
    print(f"Pressure gain: {max_pressure_value/config.acoustic.source_magnitude:.2f}x")

    if output_dir is not None:
        plt.figure()
        plt.imshow(1e-6 * max_pressure[:, config.grid.Ny // 2, :].T, cmap="coolwarm")
        plt.colorbar(label="Max Pressure [MPa]")
        plt.title("Max Pressure Field")
        plt.xlabel("X position [grid points]")
        plt.ylabel("Z position [grid points]")
        plt.savefig(os.path.join(output_dir, "A1_max_pressure.png"))
        plt.close()

    # Compute intensity fields
    print("Computing intensity fields...")
    average_intensity = simulator.compute_intensity(pressure_data)

    if output_dir is not None:
        # Plot time-averaged intensity field
        fig, ax = plot_intensity_field(
            average_intensity,
            config,
            title="Time-Averaged Intensity Field",
        )
        plt.savefig(os.path.join(output_dir, "A2_intensity_field.png"))
        plt.close()

        # Save intensity data
        intensity_path = os.path.join(output_dir, "average_intensity.npy")
        np.save(intensity_path, average_intensity)
        print(f"Saved intensity data to {intensity_path}")

    # make pressure video (unless skipped or no output directory)
    if not skip_videos and output_dir is not None:
        make_pressure_video(
            pressure_data[:, config.grid.Ny // 2, :],
            config.acoustic.dt,
            downsample=1,
            filename=os.path.join(output_dir, "A3_pressure_video.mp4"),
        )
    elif skip_videos:
        print("Skipping pressure video generation")
    else:
        print("No output directory specified, skipping file outputs")

    return average_intensity, max_pressure, simulator.medium.sound_speed, pressure_data
