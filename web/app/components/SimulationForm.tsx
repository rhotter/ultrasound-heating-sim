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
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Acoustic Parameters */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Acoustic Parameters
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <FormField
            label="Frequency (MHz)"
            value={freqMHz}
            onChange={(v) => setFreqMHz(typeof v === 'number' ? v : parseFloat(v) || 0)}
            step={0.1}
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
            onChange={(v) => setPressureMPa(typeof v === 'number' ? v : parseFloat(v) || 0)}
            step={0.1}
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
              { value: 'none', label: 'None (Plane Wave)' },
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
            placeholder="1.5"
            disabled={isRunning || params.focus_depth === 0}
          />
        </div>
      </div>

      {/* Thermal Parameters */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Thermal Parameters
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <FormField
            label="Time Step (s)"
            value={params.thermal_dt}
            onChange={(v) => handleChange('thermal_dt', v)}
            min={0.001}
            step={0.001}
            help="Thermal solver time step"
            disabled={isRunning}
          />
          <FormField
            label="Duration (s)"
            value={params.thermal_t_end}
            onChange={(v) => handleChange('thermal_t_end', v)}
            min={10}
            step={10}
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
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm font-medium text-primary-600 hover:text-primary-700 flex items-center gap-2"
          disabled={isRunning}
        >
          <svg className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Advanced Settings
        </button>
        {showAdvanced && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
              label="Domain Lateral X (cm)"
              value={LxCm}
              onChange={(v) => setLxCm(typeof v === 'number' ? v : parseFloat(v) || 0)}
              min={0.5}
              step={0.1}
              help="Domain width in X (azimuthal)"
              disabled={isRunning}
            />
            <FormField
              label="Domain Lateral Y (cm)"
              value={LyCm}
              onChange={(v) => setLyCm(typeof v === 'number' ? v : parseFloat(v) || 0)}
              min={0.5}
              step={0.1}
              help="Domain width in Y (elevational)"
              disabled={isRunning}
            />
            <FormField
              label="Domain Depth Z (cm)"
              value={LzCm}
              onChange={(v) => setLzCm(typeof v === 'number' ? v : parseFloat(v) || 0)}
              min={1.5}
              step={0.1}
              help="Domain depth (must fit all tissue layers + brain)"
              disabled={isRunning}
            />
            <FormField
              label="PML Boundary Size"
              value={params.pml_size}
              onChange={(v) => handleChange('pml_size', v)}
              min={5}
              help="Perfectly Matched Layer size"
              disabled={isRunning}
            />
          </div>
        )}
      </div>

      {/* Submit Button */}
      <div className="flex gap-3 mt-6">
        <button
          type="submit"
          disabled={isRunning}
          className="px-6 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm"
        >
          {isRunning ? 'Running...' : 'Run Simulation'}
        </button>
        <button
          type="button"
          onClick={resetForm}
          disabled={isRunning}
          className="px-6 py-3 bg-white text-gray-700 font-medium rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Reset
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
      <label className="font-medium text-gray-700 mb-2 text-sm">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value === '' ? '' : parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
        className="px-3 py-2 border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
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
        className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 disabled:cursor-not-allowed"
      />
      <label htmlFor={id} className="font-medium text-gray-700 text-sm cursor-pointer">{label}</label>
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
      <label className="font-medium text-gray-700 mb-2 text-sm">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="px-3 py-2 border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
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
