# User Interface

This directory contains the React-based frontend application for the MyMind therapeutic AI system, featuring a modern, responsive interface built with TypeScript, Tailwind CSS, and advanced visualization components.

## Architecture

```
ui/
├── dashboard/               # Main analysis dashboard
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/         # Custom React hooks
│   │   │   └── useScatter.ts  # D3 visualization hook
│   │   ├── pages/         # Application pages
│   │   ├── types/         # TypeScript definitions
│   │   └── utils/         # Utility functions
│   ├── public/            # Static assets
│   └── package.json       # Dependencies
├── chat/                   # Interactive chat interface
│   └── README.md          # Chat implementation details
├── profile/               # Client profile management
│   └── README.md          # Profile management details
└── README.md              # This file
```

## Technology Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and builds
- **Styling**: Tailwind CSS for utility-first styling
- **Visualization**: D3.js for interactive charts and graphs
- **State Management**: React Query for server state + Zustand for client state
- **Authentication**: Auth0 or similar JWT-based authentication
- **API Integration**: Axios with React Query for data fetching

## Core Components

### Dashboard Application
The main dashboard provides comprehensive session analysis and client overview:

#### Key Features
- **Session Visualization**: Interactive concept maps using D3.js
- **Real-time Analytics**: Live session processing and insights
- **Progress Tracking**: Multi-session client trajectory analysis
- **Therapeutic Insights**: CBT and schema therapy analysis display
- **Report Generation**: Professional clinical documentation

#### Main Components
```typescript
// Dashboard layout and navigation
src/components/
├── Layout/
│   ├── Sidebar.tsx         # Navigation sidebar
│   ├── Header.tsx          # Top navigation bar
│   └── Layout.tsx          # Main layout wrapper
├── Sessions/
│   ├── SessionList.tsx     # List of therapy sessions
│   ├── SessionCard.tsx     # Individual session preview
│   └── SessionDetail.tsx   # Detailed session view
├── Visualization/
│   ├── ConceptMap.tsx      # D3-based concept visualization
│   ├── ProgressChart.tsx   # Progress tracking charts
│   └── InsightPanel.tsx    # Therapeutic insights display
├── Analysis/
│   ├── KeywordCloud.tsx    # Keyword visualization
│   ├── SentimentChart.tsx  # Sentiment analysis display
│   └── DistortionList.tsx  # Cognitive distortion summary
└── Reports/
    ├── ReportGenerator.tsx # Report creation interface
    └── ReportViewer.tsx    # Report display component
```

### Chat Interface
Interactive chat system for real-time therapeutic assistance:

#### Features
- **Real-time Messaging**: WebSocket-based chat
- **AI-Powered Responses**: Integration with therapeutic AI
- **Session Context**: Contextual responses based on session history
- **Multimedia Support**: Audio and image sharing capabilities

### Profile Management
Comprehensive client profile management system:

#### Features
- **Client Overview**: Demographic and clinical information
- **Progress Tracking**: Long-term therapeutic outcomes
- **Risk Assessment**: Safety and clinical risk indicators
- **Treatment Planning**: Goal setting and intervention tracking

## Implementation Examples

### Concept Visualization Hook
```typescript
// src/hooks/useScatter.ts
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface ConceptNode {
  id: string;
  x: number;
  y: number;
  sentiment: number;
  confidence: number;
  category: string;
}

export const useScatter = (
  data: ConceptNode[],
  width: number,
  height: number
) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.x) as [number, number])
      .range([50, width - 50]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.y) as [number, number])
      .range([height - 50, 50]);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlBu)
      .domain([-1, 1]);

    const sizeScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.confidence) as [number, number])
      .range([5, 20]);

    // Create tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background', 'rgba(0,0,0,0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px');

    // Draw circles
    svg.selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => sizeScale(d.confidence))
      .attr('fill', d => colorScale(d.sentiment))
      .attr('stroke', '#333')
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('mouseover', (event, d) => {
        tooltip.style('visibility', 'visible')
          .html(`
            <strong>${d.id}</strong><br/>
            Sentiment: ${d.sentiment.toFixed(2)}<br/>
            Confidence: ${d.confidence.toFixed(2)}<br/>
            Category: ${d.category}
          `);
      })
      .on('mousemove', (event) => {
        tooltip.style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    // Add labels
    svg.selectAll('text')
      .data(data)
      .enter()
      .append('text')
      .attr('x', d => xScale(d.x))
      .attr('y', d => yScale(d.y) - sizeScale(d.confidence) - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(d => d.id);

    // Cleanup
    return () => {
      tooltip.remove();
    };
  }, [data, width, height]);

  return svgRef;
};
```

### Session Dashboard Component
```typescript
// src/components/Sessions/SessionDashboard.tsx
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ConceptMap } from '../Visualization/ConceptMap';
import { InsightPanel } from '../Analysis/InsightPanel';
import { ProgressChart } from '../Visualization/ProgressChart';

interface SessionDashboardProps {
  sessionId: string;
}

export const SessionDashboard: React.FC<SessionDashboardProps> = ({ sessionId }) => {
  const [selectedInsight, setSelectedInsight] = useState<string | null>(null);

  // Fetch session data
  const { data: sessionData, isLoading } = useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => fetchSessionData(sessionId),
  });

  // Fetch visualization data
  const { data: visualizationData } = useQuery({
    queryKey: ['visualization', sessionId],
    queryFn: () => fetchVisualizationData(sessionId),
    enabled: !!sessionId,
  });

  // Fetch therapeutic insights
  const { data: insights } = useQuery({
    queryKey: ['insights', sessionId],
    queryFn: () => fetchTherapeuticInsights(sessionId),
    enabled: !!sessionId,
  });

  if (isLoading) {
    return <div className="animate-pulse">Loading session data...</div>;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
      {/* Session Overview */}
      <div className="lg:col-span-2">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Session Overview
          </h2>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-800">Duration</h3>
              <p className="text-2xl font-bold text-blue-600">
                {sessionData?.duration} min
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="font-semibold text-green-800">Mood Score</h3>
              <p className="text-2xl font-bold text-green-600">
                {sessionData?.moodScore}/10
              </p>
            </div>
          </div>
          
          {/* Concept Visualization */}
          {visualizationData && (
            <ConceptMap
              data={visualizationData}
              width={600}
              height={400}
              onNodeClick={(nodeId) => setSelectedInsight(nodeId)}
            />
          )}
        </div>
      </div>

      {/* Insights Panel */}
      <div className="lg:col-span-1">
        <InsightPanel
          insights={insights || []}
          selectedInsight={selectedInsight}
          onInsightSelect={setSelectedInsight}
        />
      </div>

      {/* Progress Chart */}
      <div className="lg:col-span-3">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">
            Progress Overview
          </h2>
          <ProgressChart sessionId={sessionId} />
        </div>
      </div>
    </div>
  );
};
```

### API Integration
```typescript
// src/utils/api.ts
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// Request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const apiClient = {
  // Session operations
  getSession: (sessionId: string) => api.get(`/sessions/${sessionId}`),
  getVisualization: (sessionId: string) => api.get(`/sessions/${sessionId}/visualization`),
  getInsights: (sessionId: string) => api.get(`/sessions/${sessionId}/insights`),
  
  // Client operations
  getClientProfile: (clientId: string) => api.get(`/clients/${clientId}/profile`),
  getClientTrajectory: (clientId: string) => api.get(`/clients/${clientId}/trajectory`),
  
  // Analysis operations
  analyzeSession: (sessionId: string) => api.post(`/analysis/sessions/${sessionId}/therapeutic-analysis`),
  querySession: (sessionId: string, query: string) => api.post(`/rag/sessions/${sessionId}/query`, { query }),
  
  // Report operations
  generateReport: (sessionId: string, type: string) => api.get(`/output/sessions/${sessionId}/report?report_type=${type}`),
  streamReport: (sessionId: string) => api.get(`/output/sessions/${sessionId}/stream-report`),
};
```

## Development Setup

### Prerequisites
- Node.js 18+
- npm or yarn
- Git

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ui/dashboard

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Configuration
```typescript
// src/config/app.ts
export const config = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
  wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',
  auth: {
    domain: process.env.REACT_APP_AUTH_DOMAIN,
    clientId: process.env.REACT_APP_AUTH_CLIENT_ID,
  },
  features: {
    enableRealTimeAnalysis: process.env.REACT_APP_ENABLE_REAL_TIME === 'true',
    enableVoiceInput: process.env.REACT_APP_ENABLE_VOICE === 'true',
  },
};
```

## Deployment

### Production Build
```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
```

This modern React interface provides an intuitive, responsive platform for therapeutic analysis, enabling clinicians to efficiently access AI-powered insights and manage client care through a comprehensive digital dashboard.
