import { SimulationParams } from '../types/simulation';

export const defaultParams: SimulationParams = {
  Lx: 0.050,  // 1.3 cm in x
  Ly: 0.025,  // 0.7 cm in y
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
  thermal_t_end: 100,  // Shorter for faster testing
  steady_state: false,
  acoustic_only: false,
};