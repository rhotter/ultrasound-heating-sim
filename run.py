#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path

from src.heat.runner import run_heat_simulation
from src.acoustic.runner import run_acoustic_simulation
from src.config import SimulationConfig
from src.config_brna2025 import SimulationConfigBrna2025, get_effective_intensity_brna2025, FOCAL_DEPTH_BRNA2025


def main():
    parser = argparse.ArgumentParser(description="Run acoustic and/or heat simulations")
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Use CPU for computations (default: False). Use --use-cpu to enable.",
    )
    parser.add_argument(
        "--intensity-file",
        type=str,
        help="Path to pre-computed intensity data (.npy file). If not provided, acoustic simulation will be run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save output files (default: data)",
    )
    parser.add_argument(
        "--steady-state",
        action="store_true",
        help="Use steady state solver for heat simulation instead of time stepping",
    )
    parser.add_argument(
        "--save-properties",
        action="store_true",
        help="Save tissue property distributions to npy files",
    )
    parser.add_argument(
        "--transmit-focus",
        type=float,
        default=None,
        help="Transmit focus distance in meters. If not specified, plane wave transmission is used.",
    )
    parser.add_argument(
        "--azimuthal-focusing",
        action="store_true",
        help="Enable 2D focusing in both elevational and azimuthal directions (default: False, elevational only)",
    )
    parser.add_argument(
        "--acoustic-only",
        action="store_true",
        help="Run only acoustic simulation without heat simulation (default: False)",
    )
    parser.add_argument(
        "--config-brna2025",
        action="store_true",
        help="Use Brna et al. 2025 configuration (1.8 MHz, 1024 CMUT elements, 1.2mm skull)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create medium directory if saving properties
    if args.save_properties:
        os.makedirs(os.path.join(args.output_dir, "medium"), exist_ok=True)

    # Initialize configuration
    if args.config_brna2025:
        print("Using Brna et al. 2025 configuration")
        config = SimulationConfigBrna2025()
        # Override transmit focus if not specified
        if args.transmit_focus is None:
            args.transmit_focus = FOCAL_DEPTH_BRNA2025
            print(f"Setting focal depth to {args.transmit_focus*1e3:.1f} mm (2mm below skull)")
    else:
        config = SimulationConfig()

    # Set azimuthal focusing flag (can be overridden by CLI)
    if args.azimuthal_focusing:
        config.acoustic.enable_azimuthal_focusing = True

    # Get intensity data
    if args.intensity_file:
        # Use pre-computed intensity
        intensity_path = Path(args.intensity_file)
        if not intensity_path.exists():
            raise FileNotFoundError(f"Intensity file not found: {args.intensity_file}")
        print(f"Loading pre-computed intensity data from {args.intensity_file}")
        intensity_data = np.load(args.intensity_file)
    else:
        # Run acoustic simulation
        intensity_data = run_acoustic_simulation(
            config,
            args.output_dir,
            use_gpu=not args.use_cpu,
            transmit_focus=args.transmit_focus,
        )

    # Run heat simulation (unless acoustic-only mode)
    if not args.acoustic_only:
        # Apply intensity scaling for Brna2025 pulsing protocol
        if args.config_brna2025:
            print("\nApplying Brna et al. 2025 pulsing protocol:")
            print(f"  Original max intensity: {intensity_data.max():.2e} W/m²")
            intensity_data = get_effective_intensity_brna2025(intensity_data)
            print(f"  Time-averaged intensity: {intensity_data.max():.2e} W/m²")
            print(f"  Duty cycle across trains: 1.48% (150ms PTD / 10.15s)")

        run_heat_simulation(
            config,
            intensity_data,
            args.output_dir,
            steady_state=args.steady_state,
            save_properties=args.save_properties,
        )
    else:
        print("\nSkipping heat simulation (--acoustic-only mode)")
        print(f"Intensity data saved to {args.output_dir}/average_intensity.npy")


if __name__ == "__main__":
    main()
