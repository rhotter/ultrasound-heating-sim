'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
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
  const [elapsedTime, setElapsedTime] = useState(0);
  const [completionTime, setCompletionTime] = useState<number | null>(null);
  const elapsedTimeRef = useRef(0);

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
        setCompletionTime(elapsedTimeRef.current);
      } else if (results.status === 'error') {
        const errorMsg = results.metadata && 'message' in results.metadata
          ? results.metadata.message
          : 'Simulation failed';
        console.log('[checkResults] Simulation error:', errorMsg);
        setStatus('error');
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

  // Timer for elapsed time
  useEffect(() => {
    if (status !== 'running') return;

    const startTime = Date.now();
    const timer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setElapsedTime(elapsed);
      elapsedTimeRef.current = elapsed;
    }, 1000);

    return () => clearInterval(timer);
  }, [status]);

  const handleSubmit = async (params: SimulationParams) => {
    setStatus('running');
    setError(null);
    setMetadata(null);
    setJobId(null);
    setVisualizations(null);
    setTimeSeries(null);
    setHasTemperature(false);
    setElapsedTime(0);
    setCompletionTime(null);
    elapsedTimeRef.current = 0;

    try {
      const response = await startSimulation(params);
      setJobId(response.job_id);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Failed to start simulation');
    }
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold text-neutral-900 tracking-tight">
            Ultrasound Heating Simulation
          </h1>
          <p className="text-sm text-neutral-600">
            Simulate temperature rise from ultrasound
          </p>
        </div>

        {/* Main Content */}
        <div className="card">
          <div className="card-body space-y-8">
            <SimulationForm onSubmit={handleSubmit} isRunning={status === 'running'} />

            <ResultsDisplay
              status={status}
              jobId={jobId}
              metadata={metadata}
              error={error}
              hasTemperatureData={hasTemperature}
              elapsedTime={elapsedTime}
              completionTime={completionTime}
            />
          </div>
        </div>

        {/* Results */}
        {status === 'completed' && visualizations && (
          <div className="space-y-6">
            <h2 className="text-2xl font-semibold text-neutral-900 tracking-tight">
              Results
            </h2>

            {/* Temperature Results */}
            {metadata && metadata.max_temp_rise_skull_C !== undefined && (
              <div className="card">
                <div className="card-body">
                  <h3 className="text-base font-semibold text-neutral-900 mb-4">Maximum Temperature Rise</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <p className="text-sm font-medium text-neutral-600 mb-2">Skull Region</p>
                      <p className="text-3xl font-semibold text-neutral-900">{metadata.max_temp_rise_skull_C.toFixed(3)} °C</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-neutral-600 mb-2">Brain Region</p>
                      <p className="text-3xl font-semibold text-neutral-900">{(metadata.max_temp_rise_brain_C ?? 0).toFixed(3)} °C</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <VisualizationPanel
              visualizations={visualizations}
              timeSeries={timeSeries}
              hasTemperature={hasTemperature}
            />
          </div>
        )}

        {/* Footer */}
        <div className="text-center py-4">
          <a
            href="https://github.com/rhotter/ultrasound-heating-sim"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-neutral-700 hover:text-neutral-900 underline hover:underline-offset-2 transition-all"
          >
            View source on GitHub
          </a>
        </div>
      </div>
    </div>
  );
}
