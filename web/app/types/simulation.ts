export interface SimulationParams {
  // Grid parameters (physical dimensions in meters)
  Lx: number;  // Physical domain size in x [m]
  Ly: number;  // Physical domain size in y [m]
  Lz: number;  // Physical domain size in z [m]
  pml_size: number;

  // Acoustic parameters
  freq: number;
  num_cycles: number;
  source_magnitude: number;
  pulse_repetition_freq: number;
  focus_depth: number | null;
  enable_azimuthal_focusing: boolean;
  
  // Thermal parameters
  thermal_dt: number;
  thermal_t_end: number;
  steady_state: boolean;
  
  // Tissue parameters
  skull_thickness: number;  // [m]
  skull_absorption_db_cm: number;  // [dB/cm]
  
  // Transducer parameters
  pitch: number;  // [m] - also sets dx/dy/dz grid spacing
  num_elements_x: number;
  num_elements_y: number;
  
  // Options
  skip_videos: boolean;
}

export interface SimulationResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface SimulationMetadata {
  max_intensity_W_m2: number;
  mean_intensity_W_m2: number;
  max_pressure_Pa: number;
  grid_size: number[];
  frequency_Hz: number;
  duty_cycle: number;
  max_temp_rise_skull_C?: number;
  max_temp_rise_brain_C?: number;
  steady_state?: boolean;
}

export interface ResultsResponse {
  status: 'running' | 'completed' | 'error';
  metadata: SimulationMetadata | { message: string };
  visualizations?: {
    pressure?: string;
    intensity?: string;
    medium?: string;
    temperature?: string;
    pressure_video?: string;
    temperature_video?: string;
  };
  time_series?: {
    time: number[];
    skull: number[];
    brain: number[];
  } | null;
  has_temperature?: boolean;
}

