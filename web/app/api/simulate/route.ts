import { NextRequest, NextResponse } from 'next/server';

const MODAL_API_URL = 'https://raffi-60673--ultrasound-heating-sim-fastapi-app.modal.run';

export async function POST(request: NextRequest) {
  try {
    const params = await request.json();

    // Proxy to Modal FastAPI endpoint
    const response = await fetch(`${MODAL_API_URL}/api/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
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
    console.error('Exception in /api/simulate:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Failed to start simulation' },
      { status: 500 }
    );
  }
}
