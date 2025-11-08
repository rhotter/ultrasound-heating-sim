import { SimulationParams } from '../types/simulation';

export const defaultParams: SimulationParams = {
  Lx: 0.050,  // 5 cm in x
  Ly: 0.025,  // 2.5 cm in y
  Lz: 0.025,  // 2.5 cm in z (enough for all tissue layers + brain)
  pml_size: 10,
  freq: 2000000,
  num_cycles: 3,
  num_elements_x: 140,
  num_elements_y: 64,
  source_magnitude: 600000,
  pulse_repetition_freq: 2700,
  focus_depth: 0,
  enable_azimuthal_focusing: false,
  thermal_dt: 0.01,
  thermal_t_end: 60,  // 1 minute default
  steady_state: false,
  skull_thickness: 0.007,  // 7 mm default
  skull_absorption_db_cm: 9.476,  // 9.476 dB/cm (= 109.1 Np/m)
  pitch: 208e-6,  // 208 µm default
  skip_videos: true,  // Default: skip videos (off)
};