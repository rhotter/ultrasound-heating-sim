'use client';

import { useState } from 'react';
import { SimulationParams } from '../types/simulation';
import { defaultParams } from '../lib/presets';

interface SimulationFormProps {
  onSubmit: (params: SimulationParams) => void;
  isRunning: boolean;
}

export default function SimulationForm({ onSubmit, isRunning }: SimulationFormProps) {
  const [params, setParams] = useState<SimulationParams>(defaultParams);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Display values in user-friendly units
  const [freqMHz, setFreqMHz] = useState(defaultParams.freq / 1e6);
  const [pressureMPa, setPressureMPa] = useState(defaultParams.source_magnitude / 1e6);
  const [LxCm, setLxCm] = useState(parseFloat((defaultParams.Lx * 100).toFixed(2)));
  const [LyCm, setLyCm] = useState(parseFloat((defaultParams.Ly * 100).toFixed(2)));
  const [LzCm, setLzCm] = useState(parseFloat((defaultParams.Lz * 100).toFixed(2)));
  const [focusDepthCm, setFocusDepthCm] = useState(defaultParams.focus_depth ? parseFloat((defaultParams.focus_depth * 100).toFixed(2)) : 0);
  const [durationMin, setDurationMin] = useState(defaultParams.thermal_t_end / 60);
  const [skullThicknessMm, setSkullThicknessMm] = useState(parseFloat((defaultParams.skull_thickness * 1000).toFixed(2)));
  const [skullAbsorptionDbCm, setSkullAbsorptionDbCm] = useState(defaultParams.skull_absorption_db_cm);
  const [pitchUm, setPitchUm] = useState(parseFloat((defaultParams.pitch * 1e6).toFixed(1)));

  const handleChange = (field: keyof SimulationParams, value: any) => {
    setParams(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Convert display units to SI units before submitting
    const submissionParams = {
      ...params,
      freq: freqMHz * 1e6,
      source_magnitude: pressureMPa * 1e6,
      Lx: LxCm / 100,
      Ly: LyCm / 100,
      Lz: LzCm / 100,
      focus_depth: focusDepthCm === 0 ? 0 : focusDepthCm / 100,
      thermal_t_end: durationMin * 60,
      skull_thickness: skullThicknessMm / 1000,
      skull_absorption_db_cm: skullAbsorptionDbCm,
      pitch: pitchUm / 1e6,
    };
    onSubmit(submissionParams);
  };

  const resetForm = () => {
    setParams(defaultParams);
    setFreqMHz(defaultParams.freq / 1e6);
    setPressureMPa(defaultParams.source_magnitude / 1e6);
    setLxCm(parseFloat((defaultParams.Lx * 100).toFixed(2)));
    setLyCm(parseFloat((defaultParams.Ly * 100).toFixed(2)));
    setLzCm(parseFloat((defaultParams.Lz * 100).toFixed(2)));
    setFocusDepthCm(defaultParams.focus_depth ? parseFloat((defaultParams.focus_depth * 100).toFixed(2)) : 0);
    setDurationMin(defaultParams.thermal_t_end / 60);
    setSkullThicknessMm(parseFloat((defaultParams.skull_thickness * 1000).toFixed(2)));
    setSkullAbsorptionDbCm(defaultParams.skull_absorption_db_cm);
    setPitchUm(parseFloat((defaultParams.pitch * 1e6).toFixed(1)));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-10">
      {/* Acoustic Parameters */}
      <div className="space-y-5">
        <h2 className="text-base font-semibold text-neutral-900 pb-2 border-b border-neutral-200">
          Acoustic Parameters
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FormField
            label="Frequency (MHz)"
            value={freqMHz}
            onChange={(v) => setFreqMHz(typeof v === 'number' ? v : parseFloat(v))}
            help="Ultrasound frequency"
            disabled={isRunning}
          />
          <FormField
            label="Number of Pulse Cycles"
            value={params.num_cycles}
            onChange={(v) => handleChange('num_cycles', v)}
            min={1}
            help="Cycles per pulse"
            disabled={isRunning}
          />
          <FormField
            label="Source Pressure (MPa)"
            value={pressureMPa}
            onChange={(v) => setPressureMPa(typeof v === 'number' ? v : parseFloat(v))}
            help="Source pressure"
            disabled={isRunning}
          />
          <FormField
            label="Pulse Repetition Frequency (Hz)"
            value={params.pulse_repetition_freq}
            onChange={(v) => handleChange('pulse_repetition_freq', v)}
            min={100}
            help="Pulse Repetition Frequency"
            disabled={isRunning}
          />
          <SelectField
            label="Focus Type"
            value={params.focus_depth === null || params.focus_depth === 0 ? 'none' : (params.enable_azimuthal_focusing ? 'both' : 'elevational')}
            onChange={(v) => {
              if (v === 'none') {
                handleChange('focus_depth', 0);
                handleChange('enable_azimuthal_focusing', false);
                setFocusDepthCm(0);
              } else if (v === 'elevational') {
                if (params.focus_depth === null || params.focus_depth === 0) {
                  handleChange('focus_depth', 0.015);
                  setFocusDepthCm(1.5);
                }
                handleChange('enable_azimuthal_focusing', false);
              } else if (v === 'both') {
                if (params.focus_depth === null || params.focus_depth === 0) {
                  handleChange('focus_depth', 0.015);
                  setFocusDepthCm(1.5);
                }
                handleChange('enable_azimuthal_focusing', true);
              }
            }}
            options={[
              { value: 'none', label: 'None' },
              { value: 'elevational', label: 'Elevational (1D)' },
              { value: 'both', label: 'Elevational + Azimuthal (2D)' },
            ]}
            help="Focusing mode"
            disabled={isRunning}
          />
          <FormField
            label="Focus Depth (cm)"
            value={focusDepthCm === 0 ? '' : focusDepthCm}
            onChange={(v) => {
              const val = typeof v === 'number' ? v : parseFloat(v) || 0;
              setFocusDepthCm(val);
            }}
            step={0.1}
            help="Depth of focal point"
            disabled={isRunning || params.focus_depth === 0}
          />
        </div>
        <div className="text-xs text-neutral-500 mt-2">
          Duty Cycle: {((params.num_cycles / freqMHz / 1e6) * params.pulse_repetition_freq * 100).toFixed(2)}%
        </div>
      </div>

      {/* Tissue Parameters */}
      <div className="space-y-5">
        <h2 className="text-base font-semibold text-neutral-900 pb-2 border-b border-neutral-200">
          Tissue Parameters
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FormField
            label="Skull Thickness (mm)"
            value={skullThicknessMm}
            onChange={(v) => {
              const val = typeof v === 'number' ? v : parseFloat(v);
              setSkullThicknessMm(val);
            }}
            min={0}
            help="Thickness of skull layer"
            disabled={isRunning}
          />
          <FormField
            label="Skull Absorption (dB/cm)"
            value={skullAbsorptionDbCm}
            onChange={(v) => {
              const val = typeof v === 'number' ? v : parseFloat(v);
              setSkullAbsorptionDbCm(val);
            }}
            help="Acoustic absorption coefficient (0 = no absorption)"
            disabled={isRunning}
          />
        </div>
      </div>

      {/* Thermal Parameters */}
      <div className="space-y-5">
        <h2 className="text-base font-semibold text-neutral-900 pb-2 border-b border-neutral-200">
          Thermal Parameters
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FormField
            label="Duration (min)"
            value={durationMin}
            onChange={(v) => {
              const val = typeof v === 'number' ? v : parseFloat(v);
              setDurationMin(val);
            }}
            help="Total simulation time"
            disabled={isRunning || params.steady_state}
          />
          <CheckboxField
            label="Steady State"
            checked={params.steady_state}
            onChange={(v) => handleChange('steady_state', v)}
            help="Use steady-state solver"
            disabled={isRunning}
          />
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="space-y-5">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm font-medium text-neutral-600 hover:text-neutral-900 flex items-center gap-2 transition-colors"
          disabled={isRunning}
        >
          <svg className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Advanced Settings
        </button>
        {showAdvanced && (
          <div className="space-y-6 pt-2 pl-6 border-l-2 border-neutral-300 ml-2">
            {/* Transducer Settings */}
            <div>
              <h3 className="text-base font-semibold text-neutral-900 mb-3 pb-2 border-b border-neutral-200">
                Transducer
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <FormField
                  label="Transducer Elements X"
                  value={params.num_elements_x}
                  onChange={(v) => handleChange('num_elements_x', v)}
                  min={1}
                  help="Transducer elements in X (azimuthal)"
                  disabled={isRunning}
                />
                <FormField
                  label="Transducer Elements Y"
                  value={params.num_elements_y}
                  onChange={(v) => handleChange('num_elements_y', v)}
                  min={1}
                  help="Transducer elements in Y (elevational)"
                  disabled={isRunning}
                />
                <FormField
                  label="Element Pitch (µm)"
                  value={pitchUm}
                  onChange={(v) => {
                    const val = typeof v === 'number' ? v : parseFloat(v);
                    setPitchUm(val);
                  }}
                  help="Transducer element spacing"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Domain Settings */}
            <div>
              <h3 className="text-base font-semibold text-neutral-900 mb-3 pb-2 border-b border-neutral-200">
                Computational Domain
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <FormField
                  label="Domain Lateral X (cm)"
                  value={LxCm}
                  onChange={(v) => setLxCm(typeof v === 'number' ? v : parseFloat(v))}
                  help="Domain width in X (azimuthal)"
                  disabled={isRunning}
                />
                <FormField
                  label="Domain Lateral Y (cm)"
                  value={LyCm}
                  onChange={(v) => setLyCm(typeof v === 'number' ? v : parseFloat(v))}
                  help="Domain width in Y (elevational)"
                  disabled={isRunning}
                />
                <FormField
                  label="Domain Depth Z (cm)"
                  value={LzCm}
                  onChange={(v) => setLzCm(typeof v === 'number' ? v : parseFloat(v))}
                  help="Domain depth (must fit all tissue layers + brain)"
                  disabled={isRunning}
                />
              </div>
            </div>

            {/* Solver Settings */}
            <div>
              <h3 className="text-base font-semibold text-neutral-900 mb-3 pb-2 border-b border-neutral-200">
                Solver
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <FormField
                  label="PML Boundary Size"
                  value={params.pml_size}
                  onChange={(v) => handleChange('pml_size', v)}
                  help="Perfectly Matched Layer size"
                  disabled={isRunning}
                />
                <FormField
                  label="Thermal Time Step (s)"
                  value={params.thermal_dt}
                  onChange={(v) => handleChange('thermal_dt', v)}
                  help="Thermal solver time step"
                  disabled={isRunning}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Generate Videos Option */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <input
            id="generate-videos"
            type="checkbox"
            checked={!params.skip_videos}
            onChange={(e) => handleChange('skip_videos', !e.target.checked)}
            disabled={isRunning}
            className="w-4 h-4 text-neutral-900 border-neutral-300 rounded focus:ring-neutral-900 disabled:cursor-not-allowed"
          />
          <label htmlFor="generate-videos" className="text-sm text-neutral-700 cursor-pointer">
            Generate videos
          </label>
        </div>
        {!params.skip_videos && (
          <p className="text-xs text-amber-600 ml-6">Warning: significantly slower</p>
        )}
      </div>

      {/* Submit Button */}
      <div className="pt-2">
        <button
          type="submit"
          disabled={isRunning}
          className="px-5 py-2.5 bg-neutral-900 text-white text-sm font-medium rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm disabled:shadow-none"
        >
          {isRunning ? 'Running...' : 'Run Simulation'}
        </button>
      </div>
    </form>
  );
}

interface FormFieldProps {
  label: string;
  value: number | string;
  onChange: (value: number | string) => void;
  min?: number;
  max?: number;
  step?: number;
  help: string;
  placeholder?: string;
  disabled?: boolean;
}

function FormField({ label, value, onChange, min, max, step, help, placeholder, disabled }: FormFieldProps) {
  return (
    <div className="flex flex-col">
      <label className="font-medium text-neutral-700 mb-2 text-sm">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value === '' ? '' : parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
        className="px-3 py-2 border border-neutral-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-neutral-900 disabled:bg-neutral-50 disabled:cursor-not-allowed text-sm transition-all"
      />
    </div>
  );
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  help: string;
  disabled?: boolean;
}

function CheckboxField({ label, checked, onChange, help, disabled }: CheckboxFieldProps) {
  const id = `checkbox-${label.toLowerCase().replace(/\s+/g, '-')}`;
  return (
    <div className="flex items-center gap-2 pt-7">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="w-4 h-4 text-neutral-900 border-neutral-300 rounded focus:ring-neutral-900 disabled:cursor-not-allowed"
      />
      <label htmlFor={id} className="font-medium text-neutral-700 text-sm cursor-pointer">{label}</label>
    </div>
  );
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  help: string;
  disabled?: boolean;
}

function SelectField({ label, value, onChange, options, help, disabled }: SelectFieldProps) {
  return (
    <div className="flex flex-col">
      <label className="font-medium text-neutral-700 mb-2 text-sm">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="px-3 py-2 border border-neutral-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-neutral-900 disabled:bg-neutral-50 disabled:cursor-not-allowed text-sm transition-all"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
