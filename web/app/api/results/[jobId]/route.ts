import { NextRequest, NextResponse } from 'next/server';

const MODAL_API_URL = 'https://raffi-60673--ultrasound-heating-sim-fastapi-app.modal.run';

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  try {
    const { jobId } = params;

    // Proxy to Modal FastAPI endpoint
    // Add cache-busting to ensure we always get fresh data
    const response = await fetch(`${MODAL_API_URL}/api/results/${jobId}`, {
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache',
      },
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Modal API error:', error);
      return NextResponse.json(
        { detail: `Modal API error: ${error}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Exception in /api/results:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Failed to get results' },
      { status: 500 }
    );
  }
}
