'use client';

import { Loader2 } from 'lucide-react';
import { SimulationMetadata } from '../types/simulation';

interface ResultsDisplayProps {
  status: 'idle' | 'running' | 'completed' | 'error';
  jobId: string | null;
  metadata: SimulationMetadata | null;
  error: string | null;
  hasTemperatureData: boolean;
}

export default function ResultsDisplay({ status, jobId, metadata, error, hasTemperatureData }: ResultsDisplayProps) {
  if (status === 'idle') {
    return null;
  }

  return (
    <div className="mt-6">
      {/* Status Display */}
      {status === 'running' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
            <div>
              <p className="text-gray-900 font-medium">Simulation in progress...</p>
              {jobId && <p className="text-gray-500 text-xs font-mono mt-1">Job ID: {jobId}</p>}
            </div>
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <p className="text-red-900 font-medium">Error</p>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {status === 'completed' && metadata && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
          <p className="text-green-900 font-medium">Simulation completed successfully!</p>
        </div>
      )}

      {/* Results */}
      {status === 'completed' && metadata && 'max_intensity_W_m2' in metadata && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Results</h3>

          <div className="space-y-3">
            {metadata.max_temp_rise_skull_C !== undefined && (
              <>
                <MetricRow
                  label="Max Skull Temp Rise"
                  value={`${metadata.max_temp_rise_skull_C.toFixed(3)} °C`}
                />
                <MetricRow
                  label="Max Brain Temp Rise"
                  value={`${metadata.max_temp_rise_brain_C?.toFixed(3)} °C`}
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200">
      <span className="font-medium text-gray-700 text-sm">{label}</span>
      <span className="text-gray-900 font-semibold font-mono text-sm">{value}</span>
    </div>
  );
}
