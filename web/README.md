# Ultrasound Heating Simulation - Next.js Web Interface

A modern, responsive web interface built with Next.js, TypeScript, and Tailwind CSS for running ultrasound heating simulations on Modal's GPU infrastructure.

## Features

- **Modern UI**: Built with Next.js 14, React 18, and Tailwind CSS
- **Type-Safe**: Full TypeScript support with strict type checking
- **Real-time Updates**: Automatic polling for simulation progress
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Direct Modal Integration**: Next.js API routes call Modal GPU compute directly
- **Interactive Visualizations**: Real-time charts and heatmaps

## Prerequisites

- Node.js 18+ and npm (or yarn/pnpm)
- Modal account with API credentials (get from https://modal.com/settings/tokens)

## Quick Start

### 1. Install Dependencies

```bash
cd web
npm install
```

### 2. Deploy Modal Backend

Deploy the simulation backend to Modal:

```bash
cd ..  # Go to parent directory
modal deploy modal_app.py
```

### 3. Configure Environment

Copy the example environment file:

```bash
cd web
cp .env.local.example .env.local
```

Edit `.env.local` with your Modal API credentials from https://modal.com/settings/tokens:

```bash
MODAL_TOKEN_ID=your_modal_token_id_here
MODAL_TOKEN_SECRET=your_modal_token_secret_here
```

### 4. Start the Development Server

```bash
npm run dev
```

The app will be available at http://localhost:3000

## Project Structure

```
web/
├── app/
│   ├── api/
│   │   ├── simulate/
│   │   │   └── route.ts          # POST /api/simulate - Start simulation
│   │   └── results/
│   │       └── [jobId]/
│   │           └── route.ts      # GET /api/results/:jobId - Get results
│   ├── components/
│   │   ├── SimulationForm.tsx    # Main parameter form
│   │   ├── ResultsDisplay.tsx    # Results visualization
│   │   └── VisualizationPanel.tsx # Charts and heatmaps
│   ├── lib/
│   │   ├── api.ts                # API client functions
│   │   └── presets.ts            # Simulation presets
│   ├── types/
│   │   └── simulation.ts         # TypeScript types
│   ├── globals.css               # Global styles
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Main page
├── public/                       # Static assets
├── .env.local.example            # Environment template
├── .gitignore
├── next.config.js                # Next.js configuration
├── package.json
├── postcss.config.js
├── tailwind.config.js            # Tailwind CSS config
└── tsconfig.json                 # TypeScript config
```

## Available Scripts

- `npm run dev` - Start development server (http://localhost:3000)
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Usage Guide

### Running a Simulation

1. **Select a Preset** (optional):

   - **Quick**: Fast test simulation with reduced resolution
   - **Default**: Standard settings for typical use
   - **Highres**: High-resolution grid for detailed results
   - **Focused**: Configured for focused ultrasound beam

2. **Customize Parameters**:

   - **Grid Parameters**: Computational domain resolution
   - **Acoustic Parameters**: Ultrasound frequency, pressure, transducer config
   - **Thermal Parameters**: Time-stepping and solver options

3. **Submit**: Click "Run Simulation"

4. **Monitor Progress**: Status updates automatically every 5 seconds

5. **View Results**: See metrics, visualizations, and temperature evolution charts

### Parameter Descriptions

#### Grid Parameters

- **Grid Size X/Y/Z**: Resolution of computational domain

  - Larger = more accurate but slower
  - Recommended: 128×64×100 for standard simulations

- **PML Boundary Size**: Absorbing boundary layer
  - Default: 10 points
  - Prevents acoustic reflections at domain edges

#### Acoustic Parameters

- **Frequency**: Ultrasound frequency in Hz

  - Default: 2 MHz (2,000,000 Hz)
  - Range: 100 kHz - 10 MHz

- **Source Pressure**: Acoustic pressure amplitude

  - Default: 0.6 MPa (600,000 Pa)
  - Higher = more heating

- **Focus Distance**: For focused ultrasound

  - Leave empty for plane wave
  - Set distance in meters for focused beam

- **Enable 2D Focusing**: Azimuthal focusing
  - Check for 2D beam focusing
  - Uncheck for 1D focusing (Y direction only)

#### Thermal Parameters

- **Time Step**: Thermal solver time step

  - Default: 0.01 seconds
  - Smaller = more stable but slower

- **Duration**: Total simulation time

  - Default: 1000 seconds
  - Time for temperature to reach steady state

- **Steady State**: Use equilibrium solver

  - Check for fast steady-state solution
  - Uncheck for time-series evolution

- **Acoustic Only**: Skip thermal simulation
  - Check to only compute acoustic field
  - Useful for testing transducer configuration

## Architecture

The application uses a serverless architecture:

```
┌─────────────────┐
│  Web Browser    │
│  (Next.js UI)   │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  Next.js API    │
│  Routes         │
└────────┬────────┘
         │ Modal REST API
         ▼
┌─────────────────┐
│  Modal GPU      │
│  (modal_app.py) │
│  - k-Wave       │
│  - PyTorch      │
│  - Matplotlib   │
└─────────────────┘
```

## API Endpoints

### POST `/api/simulate`

Start a new simulation:

```typescript
const response = await fetch("/api/simulate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(params),
});

const { job_id, status, message } = await response.json();
```

This endpoint calls Modal's REST API to spawn a GPU simulation job.

### GET `/api/results/{job_id}`

Check simulation status and get results:

```typescript
const response = await fetch(`/api/results/${job_id}`);
const data = await response.json();
```

Response when running:

```json
{
  "status": "running",
  "metadata": { "message": "Simulation still running" }
}
```

Response when completed:

```json
{
  "status": "completed",
  "metadata": {
    "max_intensity_W_m2": 1500000,
    "max_pressure_Pa": 600000,
    "max_temp_rise_skull_C": 2.5,
    "max_temp_rise_brain_C": 0.8,
    ...
  },
  "visualizations": {
    "pressure": "base64_encoded_png...",
    "intensity": "base64_encoded_png...",
    "temperature": "base64_encoded_png..."
  },
  "time_series": {
    "time": [0, 0.01, 0.02, ...],
    "skull": [37, 37.1, 37.2, ...],
    "brain": [37, 37.05, 37.08, ...]
  },
  "has_temperature": true
}
```

## Deployment

### Deploy to Vercel

The easiest way to deploy the Next.js app:

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Follow the prompts to link your project.

**Important**: Configure Modal API credentials in Vercel environment variables:

```bash
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

Get these from https://modal.com/settings/tokens

### Deploy Backend on Modal

Make sure the Modal backend is deployed:

```bash
modal deploy modal_app.py
```

This deploys the GPU simulation function that the Next.js app will call via Modal's REST API.

## Customization

### Styling

The project uses Tailwind CSS. Customize colors in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#667eea',    // Your brand color
      secondary: '#764ba2',   // Secondary color
    },
  },
}
```

### Adding Presets

Edit `app/lib/presets.ts`:

```typescript
export const presets: Preset[] = [
  // ... existing presets
  {
    name: "mypreset",
    description: "My custom configuration",
    params: {
      domain_size_x: 200,
      // ... other parameters
    },
  },
];
```

### Tissue Properties

To customize tissue layers, modify `modal_app.py` in the parent directory.

## Troubleshooting

### "Failed to start simulation" error

**Problem**: Can't connect to Modal API

**Solutions**:

- Verify Modal credentials in `.env.local`
- Ensure Modal app is deployed: `modal deploy modal_app.py`
- Check Modal authentication: `modal token set`
- Verify credentials at https://modal.com/settings/tokens

### "Modal app not found"

**Problem**: Modal app not deployed

**Solutions**:

- Deploy Modal app: `modal deploy modal_app.py`
- Verify deployment: `modal app list`

### Simulation times out

**Problem**: Job takes too long

**Solutions**:

- Reduce grid size
- Use steady-state solver
- Increase timeout in `modal_app.py`

### Build errors

**Problem**: TypeScript or build issues

**Solutions**:

```bash
# Clear cache and reinstall
rm -rf .next node_modules
npm install
npm run build
```

## Performance

### Simulation Times (approximate)

On Modal T4 GPU:

- Quick (64×32×50, steady-state): ~2-5 minutes
- Default (128×64×100): ~10-20 minutes
- High-res (256×128×150): ~30-60 minutes

### Frontend Performance

- Initial load: < 1 second
- Polling overhead: ~100 ms every 5 seconds
- Result rendering: < 100 ms

## Contributing

To add new features:

1. Create new components in `app/components/`
2. Add types to `app/types/simulation.ts`
3. Update API functions in `app/lib/api.ts`
4. Test locally before deploying

## License

Same as parent project.

## Support

For issues:

- Frontend issues: Check browser console for errors
- Next.js API issues: Check terminal running `npm run dev`
- Modal simulation issues: `modal app logs ultrasound-heating-sim`

## References

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Modal](https://modal.com/docs)
