'use client';

import { Loader2 } from 'lucide-react';
import { SimulationMetadata } from '../types/simulation';

interface ResultsDisplayProps {
  status: 'idle' | 'running' | 'completed' | 'error';
  jobId: string | null;
  metadata: SimulationMetadata | null;
  error: string | null;
  hasTemperatureData: boolean;
  elapsedTime: number;
}

export default function ResultsDisplay({ status, jobId, metadata, error, hasTemperatureData, elapsedTime }: ResultsDisplayProps) {
  if (status === 'idle') {
    return null;
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
  };

  return (
    <div className="mt-6">
      {/* Status Display */}
      {status === 'running' && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
            <div>
              <p className="text-neutral-900 font-medium text-sm">Simulation in progress... ({formatTime(elapsedTime)})</p>
              {jobId && <p className="text-neutral-500 text-xs font-mono mt-1">Job ID: {jobId}</p>}
            </div>
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-red-900 font-medium text-sm">Error</p>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {status === 'completed' && metadata && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4">
          <p className="text-green-900 font-medium text-sm">Simulation completed successfully</p>
        </div>
      )}
    </div>
  );
}
