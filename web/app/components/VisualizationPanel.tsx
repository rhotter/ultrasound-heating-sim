'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface VisualizationPanelProps {
  visualizations: {
    pressure?: string;
    intensity?: string;
    medium?: string;
    temperature?: string;
  };
  timeSeries: {
    time: number[];
    skull: number[];
    brain: number[];
  } | null;
  hasTemperature: boolean;
}

export default function VisualizationPanel({ visualizations, timeSeries, hasTemperature }: VisualizationPanelProps) {
  // Prepare data for Recharts
  const chartData = timeSeries
    ? timeSeries.time.map((t, i) => ({
        time: t,
        skull: timeSeries.skull[i],
        brain: timeSeries.brain[i],
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Temperature over time chart */}
      {hasTemperature && timeSeries && (
        <div className="bg-white p-6 border border-gray-200 rounded-lg">
          <h3 className="text-base font-semibold text-gray-900 mb-4">
            Temperature Evolution Over Time
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="time"
                label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                stroke="#6b7280"
              />
              <YAxis
                label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                stroke="#6b7280"
              />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="skull"
                stroke="#f59e0b"
                name="Skull Region"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="brain"
                stroke="#0ea5e9"
                name="Brain Region"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Slice images */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {visualizations.pressure && (
          <div className="bg-white p-4 border border-gray-200 rounded-lg">
            <img
              src={`data:image/png;base64,${visualizations.pressure}`}
              alt="Max Pressure"
              className="w-full rounded"
            />
          </div>
        )}

        {visualizations.intensity && (
          <div className="bg-white p-4 border border-gray-200 rounded-lg">
            <img
              src={`data:image/png;base64,${visualizations.intensity}`}
              alt="Acoustic Intensity"
              className="w-full rounded"
            />
          </div>
        )}

        {visualizations.medium && (
          <div className="bg-white p-4 border border-gray-200 rounded-lg">
            <img
              src={`data:image/png;base64,${visualizations.medium}`}
              alt="Medium Properties"
              className="w-full rounded"
            />
          </div>
        )}

        {visualizations.temperature && (
          <div className="bg-white p-4 border border-gray-200 rounded-lg">
            <img
              src={`data:image/png;base64,${visualizations.temperature}`}
              alt="Temperature"
              className="w-full rounded"
            />
          </div>
        )}
      </div>

      {!hasTemperature && !timeSeries && (
        <div className="bg-gray-50 p-4 border border-gray-200 rounded-lg">
          <p className="text-gray-700 text-sm">
            Time series data not available (steady-state simulation or acoustic only)
          </p>
        </div>
      )}
    </div>
  );
}
