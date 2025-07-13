// ui/dashboard/src/components/NeedsProfile/NeedsRadarChart.tsx

import React from 'react';
import { Radar } from 'react-chartjs-2';

interface NeedsRadarChartProps {
  lifeSegmentScores: Record<string, {
    sentiment: number;
    fulfillment: number;
    frequency: number;
  }>;
}

export const NeedsRadarChart: React.FC<NeedsRadarChartProps> = ({ lifeSegmentScores }) => {
  const labels = Object.keys(lifeSegmentScores);

  const data = {
    labels,
    datasets: [
      {
        label: 'Sentiment',
        data: labels.map(l => (lifeSegmentScores[l].sentiment + 1) * 50), // Convert to 0-100
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
      },
      {
        label: 'Need Fulfillment',
        data: labels.map(l => lifeSegmentScores[l].fulfillment * 100),
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
      }
    ]
  };

  return <Radar data={data} />;
};
