import { SimulationParams, SimulationResponse, ResultsResponse } from '../types/simulation';

export async function startSimulation(params: SimulationParams): Promise<SimulationResponse> {
  const response = await fetch('/api/simulate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to start simulation' }));
    throw new Error(error.detail || 'Failed to start simulation');
  }

  return response.json();
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  const response = await fetch(`/api/results/${jobId}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to get results' }));
    throw new Error(error.detail || 'Failed to get results');
  }

  return response.json();
}
