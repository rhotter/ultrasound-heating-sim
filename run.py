#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path

from src.heat.runner import run_heat_simulation
from src.acoustic.runner import run_acoustic_simulation
from src.config import SimulationConfig
from src.config_brna2025 import SimulationConfigBrna2025, get_effective_intensity_brna2025, FOCAL_DEPTH_BRNA2025
from src.config_brna2025_homogeneous import SimulationConfigBrna2025Homogeneous, FOCAL_DEPTH_BRNA2025_HOMOGENEOUS


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
    parser.add_argument(
        "--config-brna2025-homogeneous",
        action="store_true",
        help="Use Brna et al. 2025 configuration with homogeneous brain tissue (no skull/skin/gel layers)",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip generating video files for pressure and temperature evolution (saves time and disk space)",
    )
    parser.add_argument(
        "--transducer-temp",
        type=float,
        default=None,
        help="Transducer surface temperature in °C. If specified, applies constant temperature boundary condition at transducer surface.",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create medium directory if saving properties
    if args.save_properties:
        os.makedirs(os.path.join(args.output_dir, "medium"), exist_ok=True)

    # Initialize configuration
    if args.config_brna2025 and args.config_brna2025_homogeneous:
        raise ValueError("Cannot specify both --config-brna2025 and --config-brna2025-homogeneous")

    if args.config_brna2025:
        print("Using Brna et al. 2025 configuration (with skull/skin layers)")
        config = SimulationConfigBrna2025()
        # Override transmit focus if not specified
        if args.transmit_focus is None:
            args.transmit_focus = FOCAL_DEPTH_BRNA2025
            print(f"Setting focal depth to {args.transmit_focus*1e3:.1f} mm (2mm below skull)")
    elif args.config_brna2025_homogeneous:
        print("Using Brna et al. 2025 configuration (homogeneous brain only)")
        config = SimulationConfigBrna2025Homogeneous()
        # Override transmit focus if not specified
        if args.transmit_focus is None:
            args.transmit_focus = FOCAL_DEPTH_BRNA2025_HOMOGENEOUS
            print(f"Setting focal depth to {args.transmit_focus*1e3:.1f} mm")
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
        # Run acoustic simulation (prints pressure analysis)
        intensity_data = run_acoustic_simulation(
            config,
            args.output_dir,
            use_gpu=not args.use_cpu,
            transmit_focus=args.transmit_focus,
            skip_videos=args.skip_videos,
        )

    # Run heat simulation (unless acoustic-only mode)
    if not args.acoustic_only:
        # Enable transducer heating if temperature specified
        if args.transducer_temp is not None:
            config.thermal.enable_transducer_heating = True
            config.thermal.transducer_temperature = args.transducer_temp
            print(f"\nTransducer heating enabled: {args.transducer_temp}°C")

        # Check if pulsing is enabled (for Brna2025 or if manually set)
        pulsing_enabled = getattr(config.thermal, "enable_pulsing", False)

        # Apply intensity scaling for Brna2025 configs ONLY if pulsing is disabled
        if args.config_brna2025 or args.config_brna2025_homogeneous:
            if pulsing_enabled:
                print("\nUsing pulsed heating protocol (no intensity scaling):")
                print(f"  Intensity during ON phase: {intensity_data.max():.2e} W/m²")
                print(f"  Pulse duration: {config.thermal.pulse_duration*1000:.0f} ms")
                print(f"  Inter-stimulus interval: {config.thermal.isi_duration:.1f} s")
            else:
                print("\nApplying time-averaged Brna et al. 2025 protocol:")
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
            skip_videos=args.skip_videos,
        )
    else:
        print("\nSkipping heat simulation (--acoustic-only mode)")
        print(f"Intensity data saved to {args.output_dir}/average_intensity.npy")


if __name__ == "__main__":
    main()
