"""
Visualization functions for Modal web API.
These functions generate base64-encoded images for web display.
"""

import io
import base64
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
import torch


def create_base64_image(fig) -> str:
    """Convert a matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def visualize_pressure_field(max_pressure_array: np.ndarray, config, mid_y_index: int) -> str:
    """
    Create pressure field visualization.

    Args:
        max_pressure_array: Max pressure array [Nx, Ny, Nz]
        config: Simulation config with grid parameters
        mid_y_index: Index for Y mid-slice

    Returns:
        Base64-encoded PNG image
    """
    mid_slice_pressure = max_pressure_array[:, mid_y_index, :]
    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [0, config.grid.Lx * 100, config.grid.Lz * 100, 0]
    im = ax.imshow(
        mid_slice_pressure.T * 1e-6,
        aspect="auto",
        cmap="coolwarm",
        origin="upper",
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    ax.set_title("Max Pressure Field")
    plt.colorbar(im, ax=ax, label="Pressure (MPa)")

    return create_base64_image(fig)


def visualize_intensity_field(intensity_array: np.ndarray, config, mid_y_index: int) -> str:
    """
    Create intensity field visualization.

    Args:
        intensity_array: Intensity array [Nx, Ny, Nz]
        config: Simulation config with grid parameters
        mid_y_index: Index for Y mid-slice

    Returns:
        Base64-encoded PNG image
    """
    mid_slice_intensity = intensity_array[:, mid_y_index, :]
    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [0, config.grid.Lx * 100, config.grid.Lz * 100, 0]
    im = ax.imshow(
        mid_slice_intensity.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    ax.set_title("Time-Averaged Intensity")
    plt.colorbar(im, ax=ax, label="Intensity (W/m²)")

    return create_base64_image(fig)


def visualize_medium_properties(medium_sound_speed: np.ndarray, config, focus_depth: Optional[float] = None) -> str:
    """
    Create medium properties visualization.

    Args:
        medium_sound_speed: Sound speed array
        config: Simulation config
        focus_depth: Optional focus depth in meters

    Returns:
        Base64-encoded PNG image
    """
    from src.acoustic.visualization import plot_medium_properties

    fig, ax = plot_medium_properties(medium_sound_speed, config, focus_depth=focus_depth)
    return create_base64_image(fig)


def create_pressure_video(
    pressure_data: np.ndarray,
    config,
    mid_y_slice: int,
    downsample: int = None,
    video_duration: float = 5.0
) -> str:
    """
    Create pressure evolution video.

    Args:
        pressure_data: Pressure data [time, Nx, Ny, Nz]
        config: Simulation config
        mid_y_slice: Y slice index
        downsample: Downsample factor (auto if None)
        video_duration: Video duration in seconds

    Returns:
        Base64-encoded MP4 video
    """
    pressure_slice = pressure_data[:, :, mid_y_slice, :]

    # Downsample to ~100 frames
    if downsample is None:
        downsample = max(1, pressure_slice.shape[0] // 100)
    pressure_frames = pressure_slice[::downsample]

    fps = len(pressure_frames) / video_duration
    vmax = float(np.max(np.abs(pressure_frames)))
    vmin = -vmax

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0, config.grid.Lx * 100, config.grid.Lz * 100, 0]

    im = ax.imshow(
        pressure_frames[0].T,
        aspect="auto",
        cmap="coolwarm",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    plt.colorbar(im, ax=ax, label="Pressure (Pa)")

    def update(frame):
        im.set_array(pressure_frames[frame].T)
        ax.set_title(f"Pressure Field - {frame * downsample * config.acoustic.dt * 1e6:.2f} μs")
        return [im]

    interval_ms = 1000.0 / fps
    anim = animation.FuncAnimation(
        fig, update, frames=len(pressure_frames), interval=interval_ms, blit=True
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        video_path = tmp_video.name

    try:
        fig.tight_layout(pad=0.1)
        anim.save(video_path, writer="ffmpeg", fps=fps)
        plt.close()

        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)


def visualize_temperature_field(T_field, config, mid_y_index: int) -> str:
    """
    Create temperature field visualization.

    Args:
        T_field: Temperature field (torch.Tensor or np.ndarray)
        config: Simulation config
        mid_y_index: Index for Y mid-slice

    Returns:
        Base64-encoded PNG image
    """
    T_np = T_field.cpu().numpy() if isinstance(T_field, torch.Tensor) else np.array(T_field)
    mid_slice = T_np[:, mid_y_index, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0, config.grid.Lx * 100, config.grid.Lz * 100, 0]

    im = ax.imshow(
        mid_slice.T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        vmin=37,
        vmax=np.max(mid_slice),
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    ax.set_title("Temperature Field")
    cbar = plt.colorbar(im, ax=ax, label="Temperature (°C)")
    cbar.formatter.set_useOffset(False)
    cbar.formatter.set_scientific(False)

    return create_base64_image(fig)


def create_temperature_video(
    T_history: list,
    config,
    thermal_dt: float,
    mid_y_slice: int,
    video_duration: float = 5.0
) -> str:
    """
    Create temperature evolution video.

    Args:
        T_history: List of temperature fields
        config: Simulation config
        thermal_dt: Thermal time step
        mid_y_slice: Y slice index
        video_duration: Video duration in seconds

    Returns:
        Base64-encoded MP4 video
    """
    # Convert T_history to numpy arrays and get mid-slices
    T_slices = []
    for T_field in T_history:
        T_np = T_field.cpu().numpy() if isinstance(T_field, torch.Tensor) else np.array(T_field)
        mid_slice = T_np[:, mid_y_slice, :]
        T_slices.append(mid_slice)

    # Get global min/max for consistent colorbar
    all_temps = np.array([t.flatten() for t in T_slices])
    vmin = 37
    vmax = float(np.max(all_temps))

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0, config.grid.Lx * 100, config.grid.Lz * 100, 0]

    im = ax.imshow(
        T_slices[0].T,
        aspect="auto",
        cmap="hot",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    cbar = plt.colorbar(im, ax=ax, label="Temperature (°C)")
    cbar.formatter.set_useOffset(False)
    cbar.formatter.set_scientific(False)

    def update(frame):
        im.set_array(T_slices[frame].T)
        actual_time = frame * thermal_dt * config.thermal.save_every
        ax.set_title(f"Temperature (°C) - Time: {actual_time:.1f}s")
        return [im]

    fps = len(T_slices) / video_duration
    interval_ms = 1000.0 / fps

    anim = animation.FuncAnimation(
        fig, update, frames=len(T_slices), interval=interval_ms, blit=True
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        video_path = tmp_video.name

    try:
        anim.save(video_path, writer="ffmpeg", fps=fps)
        plt.close()

        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
