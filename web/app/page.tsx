'use client';

import { useState, useEffect, useCallback } from 'react';
import SimulationForm from './components/SimulationForm';
import ResultsDisplay from './components/ResultsDisplay';
import VisualizationPanel from './components/VisualizationPanel';
import { startSimulation, getResults } from './lib/api';
import { SimulationParams, SimulationMetadata } from './types/simulation';

export default function Home() {
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<SimulationMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visualizations, setVisualizations] = useState<any>(null);
  const [timeSeries, setTimeSeries] = useState<any>(null);
  const [hasTemperature, setHasTemperature] = useState(false);

  const checkResults = useCallback(async () => {
    if (!jobId) return;

    try {
      console.log('[checkResults] Polling for results, jobId:', jobId);
      const results = await getResults(jobId);
      console.log('[checkResults] Received results:', results);

      if (results.status === 'completed') {
        console.log('[checkResults] Simulation completed!');
        setStatus('completed');
        setMetadata(results.metadata as SimulationMetadata);
        setVisualizations(results.visualizations);
        setTimeSeries(results.time_series);
        setHasTemperature(results.has_temperature || false);
      } else if (results.status === 'error') {
        console.log('[checkResults] Simulation error:', results.metadata?.message);
        setStatus('error');
        // Extract error message from metadata if available
        const errorMsg = results.metadata?.message || 'Simulation failed';
        setError(errorMsg);
      } else {
        console.log('[checkResults] Still running...');
      }
      // If still running, keep current status
    } catch (err) {
      console.error('[checkResults] Exception:', err);
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to check results');
    }
  }, [jobId]);

  // Poll for results when simulation is running
  useEffect(() => {
    if (status !== 'running') return;

    const interval = setInterval(checkResults, 5000);
    return () => clearInterval(interval);
  }, [status, checkResults]);

  const handleSubmit = async (params: SimulationParams) => {
    setStatus('running');
    setError(null);
    setMetadata(null);
    setJobId(null);
    setVisualizations(null);
    setTimeSeries(null);
    setHasTemperature(false);

    try {
      const response = await startSimulation(params);
      setJobId(response.job_id);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to start simulation');
    }
  };

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-6">
          <h1 className="text-3xl font-semibold text-gray-900">Ultrasound Heating Simulation</h1>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-6">
          <SimulationForm onSubmit={handleSubmit} isRunning={status === 'running'} />

          <ResultsDisplay
            status={status}
            jobId={jobId}
            metadata={metadata}
            error={error}
            hasTemperatureData={hasTemperature}
          />

          {/* Visualizations */}
          {status === 'completed' && visualizations && (
            <div className="mt-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Visualizations</h2>
              <VisualizationPanel
                visualizations={visualizations}
                timeSeries={timeSeries}
                hasTemperature={hasTemperature}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
