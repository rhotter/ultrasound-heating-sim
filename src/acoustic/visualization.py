import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple
from src.config import SimulationConfig
import matplotlib.animation as animation


def plot_medium_properties(
    medium_sound_speed: np.ndarray,
    config: SimulationConfig,
    slice_y: int | None = None,
    transmit_focus: float | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the sound speed distribution in tissue layers.

    Args:
        medium_sound_speed: 3D array of sound speed values
        config: Simulation configuration
        slice_y: Y-plane to slice (defaults to middle)
        transmit_focus: Transmit focus distance in meters (if specified)
    """
    if slice_y is None:
        slice_y = config.grid.Ny // 2

    # Calculate extent in cm
    x_extent_cm = config.grid.Nx * config.grid.dx * 100  # Convert m to cm
    z_extent_cm = config.grid.Nz * config.grid.dz * 100  # Convert m to cm

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        medium_sound_speed[:, slice_y, :].T,
        aspect="auto",
        cmap="viridis",
        extent=[0, x_extent_cm, z_extent_cm, 0]  # [left, right, bottom, top]
    )
    plt.colorbar(im, label="Sound Speed [m/s]")
    ax.set_title("Sound Speed Distribution in Tissue Layers\n(XZ plane at Y = Ny/2)")
    ax.set_xlabel("X position [cm]")
    ax.set_ylabel("Z position [cm]")

    # Add layer annotations
    current_z_cm = 0.0

    # Add horizontal line at tissue start
    ax.axhline(y=0, color="w", linestyle="--", alpha=0.5)

    # Add layer boundaries and labels for each tissue layer except the last one
    for tissue in config.tissue_layers[:-1]:
        if tissue.thickness is not None:
            thickness_cm = tissue.thickness * 100  # Convert m to cm
            if thickness_cm > 0:
                next_z_cm = current_z_cm + thickness_cm
                # Add boundary line
                ax.axhline(y=next_z_cm, color="w", linestyle="--", alpha=0.5)
                # Add layer label
                ax.text(
                    x_extent_cm / 2,
                    current_z_cm + thickness_cm / 2,
                    tissue.name.capitalize(),
                    color="w",
                    ha="center",
                    fontsize=12,
                )
                current_z_cm = next_z_cm

    # Add label for the last (innermost) layer
    ax.text(
        x_extent_cm / 2,
        current_z_cm + 0.25,  # 0.25 cm offset
        config.tissue_layers[-1].name.capitalize(),
        color="w",
        ha="center",
        fontsize=12,
    )

    # Add transducer array visualization
    source_x_size_cm = config.acoustic.num_elements_x * config.acoustic.pitch * 100  # Convert to cm
    x_start_cm = (x_extent_cm - source_x_size_cm) / 2
    source_z_pos_cm = config.acoustic.source_z_pos * config.grid.dz * 100  # Convert to cm
    transducer_height_cm = 3 * config.grid.dz * 100  # Convert to cm
    rect = Rectangle(
        (x_start_cm, source_z_pos_cm),
        source_x_size_cm,
        transducer_height_cm,
        facecolor="red",
        alpha=0.5,
        edgecolor="white",
    )
    ax.add_patch(rect)
    ax.text(
        x_extent_cm / 2,
        source_z_pos_cm - 0.05,  # 0.5 mm above transducer
        "Transducer Array",
        color="red",
        ha="center",
        fontsize=12,
    )

    # Add transmit focus point if specified
    if transmit_focus is not None and not np.isinf(transmit_focus):
        # Convert focus distance from meters to cm
        focus_z_cm = source_z_pos_cm + transmit_focus * 100  # Convert m to cm
        focus_x_cm = x_extent_cm / 2  # On-axis focus

        # Draw focal point
        ax.plot(focus_x_cm, focus_z_cm, 'y*', markersize=20, markeredgecolor='white',
                markeredgewidth=1.5, label=f'Focus ({transmit_focus*1e3:.1f} mm)')
        ax.legend(loc='upper right')

    return fig, ax


def plot_pressure_field(
    pressure_field: np.ndarray,
    config: SimulationConfig,
    z_start: int,
    title: str = "Pressure Field",
    cmap: str = "RdBu",
    vmax: float | None = None,
) -> None:
    """Plot a 2D slice of the pressure field."""
    plt.figure(figsize=(10, 6))
    plt.imshow(
        pressure_field[:, pressure_field.shape[1] // 2, :].T,
        aspect="equal",
        cmap=cmap,
        vmax=vmax,
        vmin=-vmax if vmax is not None else None,
    )

    # Add tissue layer boundaries
    current_z = z_start
    for i, tissue in enumerate(config.tissue_layers[:-1]):  # Skip last layer
        if tissue.thickness is not None:
            points = int(tissue.thickness / config.grid.dz)
            if points > 0:
                next_z = current_z + points
                plt.axhline(y=next_z, color="w", linestyle="--", alpha=0.5)

                # Add tissue label in the middle of each layer
                plt.text(
                    pressure_field.shape[0] + 5,
                    current_z + points // 2,
                    tissue.name.capitalize(),
                    color="white",
                    verticalalignment="center",
                )
                current_z = next_z

    # Add label for the last (innermost) layer
    plt.text(
        pressure_field.shape[0] + 5,
        current_z + 10,
        config.tissue_layers[-1].name.capitalize(),
        color="white",
        verticalalignment="center",
    )

    plt.colorbar(label="Pressure [Pa]")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")


def plot_intensity_field(
    intensity_data: np.ndarray,
    config: SimulationConfig,
    slice_y: int | None = None,
    title: str = "Intensity Field",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the intensity field distribution.

    Args:
        intensity_data: 3D array of intensity values (Nx, Ny, Nz)
        config: Simulation configuration
        slice_y: Y-plane to slice (defaults to middle)
        title: Plot title

    Returns:
        Figure and axes objects
    """
    if slice_y is None:
        slice_y = config.grid.Ny // 2

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        1e-4 * 1e3 * intensity_data[:, slice_y, :].T,  # to mW/cm^2
        cmap="hot",
        aspect="auto",
    )
    plt.colorbar(im, label="Intensity [mW/cm²]")
    ax.set_title(title)
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    return fig, ax


def make_pressure_video(
    pressure_data: np.ndarray,
    dt: float,
    downsample: int = 4,
    filename: str = "pressure_wave.mp4",
):
    """Make a video of the pressure field."""

    # Get global min/max for consistent colorbar
    vmin = pressure_data.min()
    vmax = pressure_data.max()

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pressure_data[0].T, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Pressure [Pa]")
    ax.set_xlabel("X position [grid points]")
    ax.set_ylabel("Z position [grid points]")

    def update(frame):
        im.set_array(pressure_data[downsample * frame].T)
        ax.set_title(
            f"Pressure Field in Tissue Layers - {downsample * frame * dt * 1e6:.2f} μs"
        )
        return [im]

    nt = pressure_data.shape[0]
    anim = animation.FuncAnimation(
        fig, update, frames=nt // downsample, interval=20, blit=True
    )
    anim.save(filename, writer="ffmpeg")
    plt.close()

    print(f"Video saved as {filename}")
