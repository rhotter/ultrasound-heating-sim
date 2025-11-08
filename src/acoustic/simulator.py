import numpy as np
from typing import Tuple, Optional

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
    def __init__(self, config: SimulationConfig, focus_depth: Optional[float] = None):
        self.config = config
        self.kgrid: Optional[kWaveGrid] = None
        self.medium: Optional[kWaveMedium] = None
        self.source: Optional[kSource] = None
        self.sensor: Optional[kSensor] = None
        self.sensor_data: Optional[dict] = None

        if focus_depth is None:
            self.focus_depth = np.inf
        else:
            self.focus_depth = focus_depth

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

            # Don't include absorption in the kwave model to be conservative
            # convert from Np/m to dB/cm
            # absorption_dB_per_cm = tissue.absorption_coefficient * 8.686 / 100
            # freq_MHz = self.config.acoustic.freq / 1e6
            # self.medium.alpha_coeff[mask] = absorption_dB_per_cm / freq_MHz

        return self.medium

    def setup_source_sensor(self) -> Tuple[kSource, kSensor]:
        """Create and configure the source and sensor."""
        if self.kgrid is None:
            raise RuntimeError("Grid must be set up before source/sensor")

        # Calculate element positions
        # Y-axis (elevational direction)
        if self.config.acoustic.num_elements_y % 2 != 0:
            y_ids = np.arange(1, self.config.acoustic.num_elements_y + 1) - np.ceil(
                self.config.acoustic.num_elements_y / 2
            )
        else:
            y_ids = (
                np.arange(1, self.config.acoustic.num_elements_y + 1)
                - (self.config.acoustic.num_elements_y + 1) / 2
            )

        # X-axis (azimuthal direction)
        if self.config.acoustic.num_elements_x % 2 != 0:
            x_ids = np.arange(1, self.config.acoustic.num_elements_x + 1) - np.ceil(
                self.config.acoustic.num_elements_x / 2
            )
        else:
            x_ids = (
                np.arange(1, self.config.acoustic.num_elements_x + 1)
                - (self.config.acoustic.num_elements_x + 1) / 2
            )

        # Calculate time delays for focusing
        if not np.isinf(self.focus_depth):
            c0 = 1500
            cmax = max(tissue.sound_speed for tissue in self.config.tissue_layers)
            cavg = (c0 + cmax) / 2

            if self.config.acoustic.enable_azimuthal_focusing:
                # 2D focusing: calculate delays based on both x and y distances from center
                # Create 2D meshgrid of element positions in physical coordinates
                x_positions, y_positions = np.meshgrid(
                    x_ids * self.config.acoustic.pitch,
                    y_ids * self.config.acoustic.pitch,
                )

                # Calculate 3D Euclidean distance from each element to focal point at (0, 0, focus_depth)
                distances = np.sqrt(
                    x_positions**2 + y_positions**2 + self.focus_depth**2
                )

                # Time delays for spherical wavefront convergence
                time_delays_2d = -(distances - self.focus_depth) / cavg
                time_delays_2d = time_delays_2d - np.min(time_delays_2d)

                # Flatten to 1D array (row-major order: Y varies faster than X)
                time_delays = time_delays_2d.flatten()
            else:
                # 1D focusing: only calculate delays based on y-distance from center (original behavior)
                y_distances = y_ids * self.config.acoustic.pitch
                time_delays_y = (
                    -(np.sqrt(y_distances**2 + self.focus_depth**2) - self.focus_depth)
                    / cavg
                )
                time_delays_y = time_delays_y - np.min(time_delays_y)

                # Repeat the same delay pattern for each column (x-direction)
                time_delays = np.tile(
                    time_delays_y[:, np.newaxis],
                    (1, self.config.acoustic.num_elements_x),
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

        # Define source mask - place individual elements on a grid
        # Each element occupies one grid point, spaced by pitch
        element_spacing_x = max(
            1, int(self.config.acoustic.pitch / self.config.grid.dx)
        )
        element_spacing_y = max(
            1, int(self.config.acoustic.pitch / self.config.grid.dy)
        )

        # Calculate total array size in grid points
        array_size_x = (self.config.acoustic.num_elements_x - 1) * element_spacing_x + 1
        array_size_y = (self.config.acoustic.num_elements_y - 1) * element_spacing_y + 1

        print(
            f"Transducer array: {self.config.acoustic.num_elements_x}x{self.config.acoustic.num_elements_y} elements"
        )
        print(f"Element spacing: {element_spacing_x}x{element_spacing_y} grid points")
        print(f"Array size: {array_size_x}x{array_size_y} grid points")
        print(
            f"Domain size: {self.config.grid.Nx}x{self.config.grid.Ny}x{self.config.grid.Nz} grid points"
        )

        # Check if array fits in domain
        if array_size_x > self.config.grid.Nx or array_size_y > self.config.grid.Ny:
            raise ValueError(
                f"Transducer array (size {array_size_x}x{array_size_y} grid points) is too large for the domain "
                f"(size {self.config.grid.Nx}x{self.config.grid.Ny} grid points). "
                f"With {self.config.acoustic.num_elements_x}x{self.config.acoustic.num_elements_y} elements at "
                f"{element_spacing_x}x{element_spacing_y} spacing, the array needs at least "
                f"{array_size_x}x{array_size_y} grid points. "
                f"Increase domain lateral dimensions (Lx={self.config.grid.Lx * 100:.2f}cm, Ly={self.config.grid.Ly * 100:.2f}cm) "
                f"or reduce number of elements."
            )

        # Center the array in the domain
        x_start = int((self.config.grid.Nx - array_size_x) / 2)
        y_start = int((self.config.grid.Ny - array_size_y) / 2)

        print(f"Array start position: ({x_start}, {y_start})")
        print(
            f"Array end position: ({x_start + array_size_x - 1}, {y_start + array_size_y - 1})"
        )

        source_mask = np.zeros(self.kgrid.k.shape)

        # Place each element at its grid position
        num_elements_placed = 0
        for i in range(self.config.acoustic.num_elements_x):
            for j in range(self.config.acoustic.num_elements_y):
                x_pos = x_start + i * element_spacing_x
                y_pos = y_start + j * element_spacing_y
                # Ensure we're within bounds
                if (
                    0 <= x_pos < self.config.grid.Nx
                    and 0 <= y_pos < self.config.grid.Ny
                ):
                    source_mask[x_pos, y_pos, self.config.acoustic.source_z_pos] = 1
                    num_elements_placed += 1
                else:
                    print(f"Warning: Element at ({x_pos}, {y_pos}) is out of bounds!")

        # Verify the number of source points matches the number of signals
        expected_elements = (
            self.config.acoustic.num_elements_x * self.config.acoustic.num_elements_y
        )
        if num_elements_placed != expected_elements:
            raise ValueError(
                f"Source mask has {num_elements_placed} elements but {expected_elements} signals were created. "
                f"Some elements were out of bounds."
            )

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
