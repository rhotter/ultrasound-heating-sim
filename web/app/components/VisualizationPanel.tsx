'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface VisualizationPanelProps {
  visualizations: {
    pressure?: string;
    intensity?: string;
    medium?: string;
    temperature?: string;
    pressure_video?: string;
    temperature_video?: string;
  };
  timeSeries: {
    time: number[];
    skull: number[];
    brain: number[];
  } | null;
  hasTemperature: boolean;
}

export default function VisualizationPanel({ visualizations, timeSeries, hasTemperature }: VisualizationPanelProps) {
  // Prepare data for Recharts (convert time to minutes)
  const chartData = timeSeries
    ? timeSeries.time.map((t, i) => ({
        time: t / 60, // Convert seconds to minutes
        skull: timeSeries.skull[i],
        brain: timeSeries.brain[i],
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Temperature over time chart */}
      {hasTemperature && timeSeries && (
        <div className="card">
          <div className="card-body">
            <h3 className="text-base font-semibold text-neutral-900 mb-6">
              Temperature Evolution Over Time
            </h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="time"
                label={{ value: 'Time (min)', position: 'insideBottom', offset: -5 }}
                stroke="#6b7280"
                type="number"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis
                label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                stroke="#6b7280"
                domain={['auto', 'auto']}
              />
              <Tooltip
                formatter={(value: number) => value.toFixed(3)}
                labelFormatter={(label: number) => `Time: ${label.toFixed(2)} min`}
              />
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
        </div>
      )}

      {/* Videos */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {visualizations.pressure_video && (
          <div className="card">
            <div className="card-body">
              <h3 className="text-base font-semibold text-neutral-900 mb-4">
                Acoustic Pressure Wave Propagation
              </h3>
              <video
                controls
                loop
                className="w-full rounded-lg"
                src={`data:video/mp4;base64,${visualizations.pressure_video}`}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        )}

        {visualizations.temperature_video && (
          <div className="card">
            <div className="card-body">
              <h3 className="text-base font-semibold text-neutral-900 mb-4">
                Temperature Evolution
              </h3>
              <video
                controls
                loop
                className="w-full rounded-lg"
                src={`data:video/mp4;base64,${visualizations.temperature_video}`}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        )}
      </div>

      {/* Slice images */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {visualizations.medium && (
          <div className="card">
            <div className="card-body">
              <img
                src={`data:image/png;base64,${visualizations.medium}`}
                alt="Medium Properties"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}

        {visualizations.pressure && (
          <div className="card">
            <div className="card-body">
              <img
                src={`data:image/png;base64,${visualizations.pressure}`}
                alt="Max Pressure"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}

        {visualizations.intensity && (
          <div className="card">
            <div className="card-body">
              <img
                src={`data:image/png;base64,${visualizations.intensity}`}
                alt="Acoustic Intensity"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}

        {visualizations.temperature && (
          <div className="card">
            <div className="card-body">
              <img
                src={`data:image/png;base64,${visualizations.temperature}`}
                alt="Temperature"
                className="w-full rounded-lg"
              />
            </div>
          </div>
        )}
      </div>

      {!hasTemperature && !timeSeries && (
        <div className="bg-neutral-50 p-4 border border-neutral-200 rounded-xl">
          <p className="text-neutral-700 text-sm">
            Time series data not available (steady-state simulation or acoustic only)
          </p>
        </div>
      )}
    </div>
  );
}
