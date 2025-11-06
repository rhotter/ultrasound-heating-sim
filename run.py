#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path

from src.heat.runner import run_heat_simulation
from src.acoustic.runner import run_acoustic_simulation
from src.config import SimulationConfig


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
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create medium directory if saving properties
    if args.save_properties:
        os.makedirs(os.path.join(args.output_dir, "medium"), exist_ok=True)

    # Initialize configuration
    config = SimulationConfig()

    # Set azimuthal focusing flag
    config.acoustic.enable_azimuthal_focusing = args.azimuthal_focusing

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
