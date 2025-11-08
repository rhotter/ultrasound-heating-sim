"""
Modal GPU backend for ultrasound heating simulation.
Run simulations on GPU with: modal deploy modal_app.py
"""

import modal
import base64
import io

# Create Modal app
app = modal.App("ultrasound-heating-sim")


# Build image with dependencies and copy source code during build
def build_image():
    # Use Modal's micromamba image with CUDA support for k-Wave
    image = (
        modal.Image.micromamba(python_version="3.11")
        .apt_install(
            "git",
            "pkg-config",
            "libgl1-mesa-glx",  # OpenGL libraries for matplotlib/opencv
            "libglib2.0-0",
            "libhdf5-dev",  # HDF5 libraries including libsz.so.2 for k-Wave CUDA
            "libsz2",  # szip compression library for HDF5
            "libgomp1",  # GCC OpenMP library for k-Wave CUDA
            "ffmpeg",  # Video encoding for matplotlib animations
        )
        .pip_install(
            "k-wave-python==0.4.0",
            "torch",
            "numpy==2.2.3",
            "scipy==1.14.1",
            "matplotlib==3.10.0",
            "pandas==2.2.3",
            "tqdm==4.67.1",
            "beartype==0.19.0",
            "jaxtyping==0.2.36",
            "fastapi",
            "pydantic",
        )
        .run_commands(
            # Create symlinks from system libraries to conda lib directory for k-Wave CUDA
            "ln -sf /usr/lib/x86_64-linux-gnu/libsz.so.2 /opt/conda/lib/libsz.so.2",
            "ln -sf /usr/lib/x86_64-linux-gnu/libgomp.so.1 /opt/conda/lib/libgomp.so.1",
            "ln -sf /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.103 /opt/conda/lib/libhdf5_serial.so.103",
            "mkdir -p /root/src",
        )
    )
    # Copy all source files into the image
    from pathlib import Path

    src_path = Path("./src")
    for py_file in src_path.rglob("*.py"):
        rel_path = py_file.relative_to("src")
        image = image.add_local_file(str(py_file), f"/root/src/{rel_path}")
    return image


image = build_image()


@app.function(
    image=image,
    gpu="A10",  # Use T4 GPU (can upgrade to A100 for faster compute)
    timeout=3600,  # 1 hour timeout
)
def run_simulation(
    # Grid parameters (physical dimensions in meters)
    Lx: float = 0.05,  # 5 cm
    Ly: float = 0.025,  # 2.5 cm
    Lz: float = 0.03,  # 3 cm
    pml_size: int = 10,
    # Acoustic parameters
    freq: float = 2e6,  # 2 MHz
    num_cycles: int = 3,
    num_elements_x: int = 140,
    num_elements_y: int = 64,
    source_magnitude: float = 0.6e6,  # 0.6 MPa
    pulse_repetition_freq: float = 2700,  # 2.7 kHz
    focus_depth: float | None = None,
    enable_azimuthal_focusing: bool = False,
    # Thermal parameters
    thermal_dt: float = 0.01,
    thermal_t_end: float = 1000.0,
    steady_state: bool = False,
    # Options
    acoustic_only: bool = False,
) -> dict:
    """
    Run ultrasound heating simulation on Modal GPU.

    Returns:
        dict with simulation results including:
        - visualizations: dict of base64 encoded PNG images
        - metadata: simulation parameters and results
        - time_series: temperature time series data (if time-dependent simulation)
    """
    import sys

    sys.path.insert(0, "/root")

    # Set matplotlib to use headless backend before any other imports
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np
    import torch

    from src.config import (
        SimulationConfig,
        GridConfig,
        AcousticConfig,
        ThermalConfig,
    )
    from src.acoustic.runner import run_acoustic_simulation
    from src.acoustic.visualization import plot_medium_properties
    from src.heat.runner import run_heat_simulation

    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Build configuration (using default tissue properties from config.py)
    config = SimulationConfig(
        grid=GridConfig(
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            pml_size=pml_size,
        ),
        acoustic=AcousticConfig(
            freq=freq,
            num_cycles=num_cycles,
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            source_magnitude=source_magnitude,
            pulse_repetition_freq=pulse_repetition_freq,
            enable_azimuthal_focusing=enable_azimuthal_focusing,
        ),
        thermal=ThermalConfig(
            dt=thermal_dt,
            t_end=thermal_t_end,
        ),
        # tissue_layers uses default from SimulationConfig
    )

    # Run acoustic simulation
    print("Starting acoustic simulation...")
    # Convert focus_depth = 0 to None (no focusing)
    actual_focus_depth = (
        None if focus_depth == 0 or focus_depth is None else focus_depth
    )
    print(
        f"Focus depth: {actual_focus_depth} meters ({actual_focus_depth * 100 if actual_focus_depth else 'None (no focusing)'} cm)"
    )
    print(f"Enable azimuthal focusing: {enable_azimuthal_focusing}")
    intensity_array, max_pressure_array, medium_sound_speed, pressure_data = (
        run_acoustic_simulation(
            config=config,
            output_dir=None,
            use_gpu=True,
            focus_depth=actual_focus_depth,
        )
    )
    print(f"Acoustic simulation complete, intensity shape: {intensity_array.shape}")
    print(f"Max pressure shape: {max_pressure_array.shape}")
    print(f"Medium sound speed shape: {medium_sound_speed.shape}")
    print(f"Pressure data shape: {pressure_data.shape}")

    # Generate visualizations
    visualizations = {}

    # Max pressure visualization
    mid_slice_pressure = max_pressure_array[:, max_pressure_array.shape[1] // 2, :]
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set extent to physical dimensions in cm
    extent = [
        0,
        config.grid.Lx * 100,
        config.grid.Lz * 100,
        0,
    ]  # [left, right, bottom, top]
    im = ax.imshow(
        mid_slice_pressure.T * 1e-6,
        aspect="auto",
        cmap="coolwarm",
        origin="upper",
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    ax.set_title("Max Pressure (MPa)")
    plt.colorbar(im, ax=ax, label="Pressure (MPa)")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    visualizations["pressure"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Intensity visualization
    mid_slice_intensity = intensity_array[:, intensity_array.shape[1] // 2, :]
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [
        0,
        config.grid.Lx * 100,
        config.grid.Lz * 100,
        0,
    ]  # [left, right, bottom, top]
    im = ax.imshow(
        mid_slice_intensity.T,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        extent=extent,
    )
    ax.set_xlabel("X position (cm)")
    ax.set_ylabel("Z position (depth, cm)")
    ax.set_title("Time-Averaged Acoustic Intensity (W/m²)")
    plt.colorbar(im, ax=ax, label="Intensity (W/m²)")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    visualizations["intensity"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Medium properties visualization
    fig, ax = plot_medium_properties(
        medium_sound_speed, config, focus_depth=actual_focus_depth
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    visualizations["medium"] = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Pressure video
    print("Generating pressure video...")
    import tempfile
    import os
    import matplotlib.animation as animation

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        video_path = tmp_video.name
    try:
        # Take mid-slice in Y direction
        pressure_slice = pressure_data[:, :, pressure_data.shape[2] // 2, :]

        # Downsample to ~100 frames
        downsample = max(1, pressure_slice.shape[0] // 100)
        pressure_frames = pressure_slice[::downsample]

        # Calculate fps to make video exactly 5 seconds
        video_duration = 5.0  # seconds
        fps = len(pressure_frames) / video_duration

        # Get global max for symmetric colorbar
        vmax = float(np.max(np.abs(pressure_frames)))
        vmin = -vmax

        # Create animation with minimal padding
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [
            0,
            config.grid.Lx * 100,
            config.grid.Lz * 100,
            0,
        ]  # [left, right, bottom, top]
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
        cbar = plt.colorbar(im, ax=ax, label="Pressure (Pa)")

        def update(frame):
            im.set_array(pressure_frames[frame].T)
            ax.set_title(f"Pressure Field - {frame * downsample * config.acoustic.dt * 1e6:.2f} μs")
            return [im]

        interval_ms = 1000.0 / fps  # milliseconds between frames
        anim = animation.FuncAnimation(
            fig, update, frames=len(pressure_frames), interval=interval_ms, blit=True
        )
        # Use tight layout before saving to minimize whitespace
        fig.tight_layout(pad=0.1)
        anim.save(video_path, writer="ffmpeg", fps=fps)
        plt.close()

        with open(video_path, "rb") as f:
            visualizations["pressure_video"] = base64.b64encode(f.read()).decode(
                "utf-8"
            )
        print("Pressure video generated successfully")
    except Exception as e:
        print(f"Error generating pressure video: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

    max_intensity = float(np.max(intensity_array))
    mean_intensity = float(np.mean(intensity_array))
    max_pressure = float(np.max(max_pressure_array))

    result = {
        "visualizations": visualizations,
        "metadata": {
            "max_intensity_W_m2": max_intensity,
            "mean_intensity_W_m2": mean_intensity,
            "max_pressure_Pa": max_pressure,
            "grid_size": [config.grid.Nx, config.grid.Ny, config.grid.Nz],
            "frequency_Hz": freq,
        },
    }

    if not acoustic_only:
        # Run thermal simulation
        thermal_result = run_heat_simulation(
            config=config,
            intensity_data=intensity_array,
            output_dir=None,
            steady_state=steady_state,
        )

        # Extract results
        T_history = thermal_result["T_history"]
        times = thermal_result["times"]
        layer_map = thermal_result["layer_map"]
        simulator = thermal_result["simulator"]

        # Convert to numpy
        if steady_state:
            # Steady state: T_history has shape (Nx, Ny, Nz)
            temp_array = (
                T_history.cpu().numpy()
                if isinstance(T_history, torch.Tensor)
                else np.array(T_history)
            )
        else:
            # Time series: convert list of tensors to array
            temp_array = np.array(
                [
                    T.cpu().numpy() if isinstance(T, torch.Tensor) else T
                    for T in T_history
                ]
            )

        print(f"Thermal simulation complete, temperature shape: {temp_array.shape}")

        # Compute metadata
        if steady_state:
            # Steady state returns (1, Nx, Ny, Nz), squeeze to (Nx, Ny, Nz)
            final_temp = (
                temp_array.squeeze()
                if hasattr(temp_array, "squeeze")
                else np.squeeze(temp_array)
            )
        else:
            final_temp = temp_array[-1]

        # Get layer map to compute region-specific temperatures
        layer_map_np = layer_map.cpu().numpy()
        skull_mask = layer_map_np == 2  # Index 2 = skull
        brain_mask = layer_map_np == 3  # Index 3 = brain

        # Handle empty masks (e.g., small test grids without all tissue layers)
        max_temp_rise_skull = (
            float(np.max(final_temp[skull_mask])) if skull_mask.any() else 0.0
        )
        max_temp_rise_brain = (
            float(np.max(final_temp[brain_mask])) if brain_mask.any() else 0.0
        )

        # Generate temperature visualization
        mid_slice_temp = final_temp[:, final_temp.shape[1] // 2, :]
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [
            0,
            config.grid.Lx * 100,
            config.grid.Lz * 100,
            0,
        ]  # [left, right, bottom, top]
        im = ax.imshow(
            mid_slice_temp.T,
            aspect="auto",
            cmap="hot",
            origin="upper",
            vmin=37,
            extent=extent,
        )
        ax.set_xlabel("X position (cm)")
        ax.set_ylabel("Z position (depth, cm)")
        ax.set_title("Temperature (°C)")
        cbar = plt.colorbar(im, ax=ax, label="Temperature (°C)")
        cbar.formatter.set_useOffset(False)
        cbar.formatter.set_scientific(False)
        cbar.update_ticks()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        result["visualizations"]["temperature"] = base64.b64encode(buf.read()).decode(
            "utf-8"
        )
        plt.close()

        # Include time series data for chart
        if not steady_state:
            # Get time series of max temperatures in skull and brain
            skull_temps = []
            brain_temps = []
            for T_field in T_history:
                T_tensor = (
                    T_field
                    if isinstance(T_field, torch.Tensor)
                    else torch.tensor(T_field, device=simulator.device)
                )
                # Handle empty masks
                if not skull_mask.any() or not brain_mask.any():
                    raise ValueError("No skull or brain voxels in domain")

                skull_temps.append(float(torch.max(T_tensor[skull_mask]).cpu().numpy()))
                brain_temps.append(float(torch.max(T_tensor[brain_mask]).cpu().numpy()))

            result["time_series"] = {
                "time": [
                    i * thermal_dt * config.thermal.save_every
                    for i in range(len(T_history))
                ],
                "skull": skull_temps,
                "brain": brain_temps,
            }
            result["has_temperature"] = True

            # Generate temperature video for time-dependent simulations
            print("Generating temperature video...")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                video_path = tmp_video.name
            try:
                import matplotlib.animation as animation

                # Convert T_history to numpy arrays and get mid-slices
                T_slices = []
                for T_field in T_history:
                    T_np = (
                        T_field.cpu().numpy()
                        if isinstance(T_field, torch.Tensor)
                        else np.array(T_field)
                    )
                    # Take mid-slice in Y direction, same as static image
                    mid_slice = T_np[:, T_np.shape[1] // 2, :]
                    T_slices.append(mid_slice)

                # Get global min/max for consistent colorbar
                all_temps = np.array([t.flatten() for t in T_slices])
                vmin = 37  # Same as static image
                vmax = float(np.max(all_temps))

                # Create animation
                fig, ax = plt.subplots(figsize=(8, 6))
                extent = [
                    0,
                    config.grid.Lx * 100,
                    config.grid.Lz * 100,
                    0,
                ]  # [left, right, bottom, top]
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

                # Calculate fps to make video exactly 5 seconds
                video_duration = 5.0  # seconds
                fps = len(T_slices) / video_duration
                interval_ms = 1000.0 / fps  # milliseconds between frames

                anim = animation.FuncAnimation(
                    fig, update, frames=len(T_slices), interval=interval_ms, blit=True
                )
                anim.save(video_path, writer="ffmpeg", fps=fps)
                plt.close()

                with open(video_path, "rb") as f:
                    result["visualizations"]["temperature_video"] = base64.b64encode(
                        f.read()
                    ).decode("utf-8")
                print("Temperature video generated successfully")
            except Exception as e:
                print(f"Error generating temperature video: {e}")
                import traceback

                traceback.print_exc()
            finally:
                if os.path.exists(video_path):
                    os.unlink(video_path)
        else:
            result["has_temperature"] = True
            result["time_series"] = None

        result["metadata"]["max_temp_rise_skull_C"] = max_temp_rise_skull
        result["metadata"]["max_temp_rise_brain_C"] = max_temp_rise_brain
        result["metadata"]["steady_state"] = steady_state

    return result


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Create FastAPI web endpoint for the simulation."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional

    web_app = FastAPI(title="Ultrasound Heating Simulation API")

    # Enable CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class SimulationParams(BaseModel):
        Lx: float = 0.027  # Physical dimensions in meters
        Ly: float = 0.013
        Lz: float = 0.025
        pml_size: int = 10
        freq: float = 2e6
        num_cycles: int = 3
        num_elements_x: int = 1
        num_elements_y: int = 64
        source_magnitude: float = 6e5
        pulse_repetition_freq: float = 1000.0
        focus_depth: Optional[float] = None
        enable_azimuthal_focusing: bool = False
        thermal_dt: float = 0.01
        thermal_t_end: float = 1000.0
        steady_state: bool = False
        acoustic_only: bool = False

    @web_app.post("/api/simulate")
    async def simulate(params: SimulationParams):
        """Start a simulation and return the call_id."""
        # Spawn the simulation asynchronously
        call = run_simulation.spawn(**params.model_dump())
        return {
            "job_id": call.object_id,
            "status": "running",
            "message": "Simulation started successfully",
        }

    @web_app.get("/api/debug/{job_id}")
    async def debug_job(job_id: str):
        """Debug endpoint to see function call details."""
        try:
            from modal.functions import FunctionCall

            function_call = FunctionCall.from_id(job_id)

            # Try to inspect the function call object
            info = {
                "job_id": job_id,
                "function_call_str": str(function_call),
                "function_call_repr": repr(function_call),
                "function_call_type": str(type(function_call)),
                "has_get": hasattr(function_call, "get"),
                "has_get_aio": hasattr(function_call.get, "aio")
                if hasattr(function_call, "get")
                else False,
            }

            # Try to get attributes
            try:
                info["dir"] = [
                    attr for attr in dir(function_call) if not attr.startswith("_")
                ]
            except:
                pass

            return info
        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

    @web_app.get("/api/results/{job_id}")
    async def get_results(job_id: str):
        """Get results for a simulation job."""
        import traceback

        try:
            # Look up the function call
            from modal.functions import FunctionCall
            import modal.exception

            print(f"[get_results] Looking up job {job_id}")
            function_call = FunctionCall.from_id(job_id)
            print(f"[get_results] FunctionCall created: {function_call}")

            # Check if it's finished by trying to get results without waiting
            try:
                # Use .aio for async operation with timeout=0 to poll immediately
                print(f"[get_results] Calling get.aio(timeout=0) for job {job_id}")
                result = await function_call.get.aio(timeout=0)
                print(f"[get_results] get.aio returned, result type: {type(result)}")

                # If result is None, job might still be running
                if result is None:
                    print(f"Job {job_id} returned None result")
                    return {
                        "status": "running",
                        "metadata": {"message": "Simulation still running"},
                    }

                # If we got here with a result, the function completed successfully
                print(f"Job {job_id} completed successfully")
                return {
                    "status": "completed",
                    "metadata": result["metadata"],
                    "visualizations": result["visualizations"],
                    "time_series": result.get("time_series"),
                    "has_temperature": result.get("has_temperature", False),
                }
            except TimeoutError as e:
                # Job is still running - this is the expected exception for incomplete jobs
                print(f"[get_results] Job {job_id} still running (TimeoutError): {e}")
                print(f"[get_results] TimeoutError traceback: {traceback.format_exc()}")
                return {
                    "status": "running",
                    "metadata": {"message": "Simulation still running"},
                }
            except modal.exception.InvalidError as e:
                # This might happen if the function call ID is invalid
                print(f"Invalid job ID {job_id}: {str(e)}")
                return {
                    "status": "error",
                    "metadata": {"message": f"Invalid job ID: {str(e)}"},
                }
            except modal.exception.OutputExpiredError as e:
                # Results expired (older than 7 days)
                print(f"Job {job_id} results expired: {str(e)}")
                return {
                    "status": "error",
                    "metadata": {"message": "Results expired (older than 7 days)"},
                }
            except Exception as e:
                # Job failed with an error or some other exception occurred
                error_msg = str(e)
                exception_type = type(e).__name__
                print(
                    f"Exception {exception_type} while checking job {job_id}: {error_msg}"
                )  # Debug logging

                # Extract the actual error message from Modal's exception format
                if "ValueError" in error_msg:
                    # Extract just the ValueError message
                    import re

                    match = re.search(r"ValueError\('([^']+)'\)", error_msg)
                    if match:
                        error_msg = match.group(1)
                    else:
                        match = re.search(r"ValueError: (.+?)(?:\n|$)", error_msg)
                        if match:
                            error_msg = match.group(1)
                return {
                    "status": "error",
                    "metadata": {"message": f"{exception_type}: {error_msg}"},
                }
        except Exception as e:
            # Failed to look up the function call
            exception_type = type(e).__name__
            print(
                f"Failed to look up job {job_id} ({exception_type}): {str(e)}"
            )  # Debug logging
            return {
                "status": "error",
                "metadata": {
                    "message": f"Failed to check job status: {exception_type}: {str(e)}"
                },
            }

    return web_app


@app.local_entrypoint()
def main():
    """Test the simulation locally."""
    print("Starting simulation on Modal...")
    # Run a small test simulation
    result = run_simulation.remote(
        Lx=0.013,  # 1.3 cm
        Ly=0.007,  # 0.7 cm
        Lz=0.025,  # 2.5 cm
        thermal_t_end=100.0,
        steady_state=True,
    )

    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE!")
    print("=" * 50)
    print(f"Max intensity: {result['metadata']['max_intensity_W_m2']:.2f} W/m²")
    if "max_temp_rise_skull_C" in result["metadata"]:
        print(
            f"Max temperature rise (skull): {result['metadata']['max_temp_rise_skull_C']:.2f} °C"
        )
    if "max_temp_rise_brain_C" in result["metadata"]:
        print(
            f"Max temperature rise (brain): {result['metadata']['max_temp_rise_brain_C']:.2f} °C"
        )
