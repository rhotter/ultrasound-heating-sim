import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst

from src.config import SimulationConfig


class PressureSimulator:
    def __init__(
        self, config: SimulationConfig, transmit_focus: Optional[float] = None
    ):
        self.config = config
        self.kgrid: Optional[kWaveGrid] = None
        self.medium: Optional[kWaveMedium] = None
        self.source: Optional[kSource] = None
        self.sensor: Optional[kSensor] = None
        self.sensor_data: Optional[dict] = None

        if transmit_focus is None:
            self.transmit_focus = np.inf
        else:
            self.transmit_focus = transmit_focus

    def setup_grid(self) -> kWaveGrid:
        """Create and configure the k-Wave grid."""
        self.kgrid = kWaveGrid(
            [self.config.grid.Nx, self.config.grid.Ny, self.config.grid.Nz],
            [self.config.grid.dx, self.config.grid.dy, self.config.grid.dz],
        )

        # Calculate time step using maximum sound speed for stability
        c_max = max(tissue.sound_speed for tissue in self.config.tissue_layers)
        self.kgrid.makeTime(
            c_max, cfl=self.config.acoustic.cfl, t_end=self.config.acoustic.t_end
        )

        return self.kgrid

    def setup_medium(self) -> kWaveMedium:
        """Create and configure the layered medium."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before medium")

        # Initialize medium with default properties
        self.medium = kWaveMedium(
            sound_speed=np.zeros(self.kgrid.k.shape),
            density=np.zeros(self.kgrid.k.shape),
            alpha_coeff=np.zeros(self.kgrid.k.shape),
            alpha_power=self.config.acoustic.alpha_power,
        )

        # Get layer map from config
        layer_map = self.config.layer_map

        # Set properties for each tissue layer based on layer map
        for i, tissue in enumerate(self.config.tissue_layers):
            mask = layer_map == i
            self.medium.sound_speed[mask] = tissue.sound_speed
            self.medium.density[mask] = tissue.density

            # convert from Np/m to dB/cm
            absorption_dB_per_cm = tissue.absorption_coefficient * 8.686 / 100
            freq_MHz = self.config.acoustic.freq / 1e6
            self.medium.alpha_coeff[mask] = absorption_dB_per_cm / freq_MHz

        return self.medium

    def setup_source_sensor(self) -> Tuple[kSource, kSensor]:
        """Create and configure the source and sensor."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before source/sensor")

        # Calculate element positions along y-axis (elevational direction)
        if self.config.acoustic.num_elements_y % 2 != 0:
            y_ids = np.arange(1, self.config.acoustic.num_elements_y + 1) - np.ceil(
                self.config.acoustic.num_elements_y / 2
            )
        else:
            y_ids = (
                np.arange(1, self.config.acoustic.num_elements_y + 1)
                - (self.config.acoustic.num_elements_y + 1) / 2
            )

        # Calculate time delays for elevational focusing
        if not np.isinf(self.transmit_focus):
            # Only calculate delays based on y-distance from center
            y_distances = y_ids * self.config.acoustic.pitch
            c0 = 1500
            cmax = max(tissue.sound_speed for tissue in self.config.tissue_layers)
            cavg = (c0 + cmax) / 2
            time_delays_y = (
                -(
                    np.sqrt(y_distances**2 + self.transmit_focus**2)
                    - self.transmit_focus
                )
                / cavg
            )
            time_delays_y = time_delays_y - np.min(time_delays_y)

            # Repeat the same delay pattern for each column (x-direction)
            time_delays = np.tile(
                time_delays_y[:, np.newaxis], (1, self.config.acoustic.num_elements_x)
            ).flatten()
        else:
            time_delays = np.zeros(
                self.config.acoustic.num_elements_x
                * self.config.acoustic.num_elements_y
            )

        # Create time-delayed source signals
        source_signal = tone_burst(
            1 / self.kgrid.dt,
            self.config.acoustic.freq,
            self.config.acoustic.num_cycles,
            signal_offset=np.round(time_delays / self.kgrid.dt).astype(int),
        )
        source_signal = self.config.acoustic.source_magnitude * source_signal

        # Define source mask for plane wave
        source_x_size = self.config.acoustic.num_elements_x * (
            self.config.acoustic.pitch / self.config.grid.dx
        )
        source_y_size = self.config.acoustic.num_elements_y * (
            self.config.acoustic.pitch / self.config.grid.dy
        )
        x_start = round((self.config.grid.Nx - source_x_size) / 2)
        y_start = round((self.config.grid.Ny - source_y_size) / 2)

        source_mask = np.zeros(self.kgrid.k.shape)
        source_mask[
            x_start : x_start + int(source_x_size),
            y_start : y_start + int(source_y_size),
            self.config.acoustic.source_z_pos,
        ] = 1

        # Create source with signals
        self.source = kSource()
        self.source.p_mask = source_mask
        self.source.p = source_signal

        # Create sensor mask and sensor
        sensor_mask = np.ones(self.kgrid.k.shape)
        self.sensor = kSensor(sensor_mask, record=["p"])

        return self.source, self.sensor

    def run_simulation(self, use_gpu: bool = True) -> dict:
        """Run the k-Wave simulation."""
        if any(x is None for x in [self.kgrid, self.medium, self.source, self.sensor]):
            raise RuntimeError(
                "All simulation components must be set up before running"
            )

        # Set simulation options
        simulation_options = SimulationOptions(
            pml_inside=False,
            pml_size=self.config.grid.pml_size,
            data_cast="single",
            save_to_disk=True,
            data_recast=True,
            save_to_disk_exit=False,
        )

        # Run simulation
        self.sensor_data = kspaceFirstOrder3D(
            medium=self.medium,
            kgrid=self.kgrid,
            source=self.source,
            sensor=self.sensor,
            simulation_options=simulation_options,
            execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu),
        )

        return self.sensor_data

    def compute_intensity(
        self, pressure_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute instantaneous and time-averaged intensity from pressure data.

        Args:
            pressure_data: Pressure data array of shape (time_steps, Nx, Ny, Nz)

        Returns:
            Tuple containing:
            - instantaneous_intensity: Array of shape (time_steps, Nx, Ny, Nz)
            - average_intensity: Array of shape (Nx, Ny, Nz)
        """
        # reset up the medium since it changes for weird reasons
        self.setup_medium()

        # Get local acoustic impedance (ρc)
        Z = self.medium.density * self.medium.sound_speed

        # Compute pulse intensity I = p²/(ρc)
        pulse_energy = np.sum(pressure_data**2, axis=0) * self.config.acoustic.dt / Z

        # Compute time-averaged intensity
        average_intensity = pulse_energy * self.config.acoustic.pulse_repetition_freq

        return average_intensity
