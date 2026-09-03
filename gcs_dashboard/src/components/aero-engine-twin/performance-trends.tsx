import React, { useState } from 'react';
import { PerformanceTrend, SparklinePoint } from './mock-data';

interface PerformanceTrendsProps {
  trends: PerformanceTrend[];
}

export const PerformanceTrends: React.FC<PerformanceTrendsProps> = ({ trends }) => {
  const [hoveredPoint, setHoveredPoint] = useState<{ trendId: string; point: SparklinePoint } | null>(null);

  // Helper to generate SVG polyline points string & area fill path
  const generateChartPaths = (data: SparklinePoint[], width: number = 300, height: number = 42) => {
    if (!data || data.length === 0) return { linePath: '', areaPath: '', coords: [] };

    const minVal = Math.min(...data.map((d) => d.val)) * 0.95;
    const maxVal = Math.max(...data.map((d) => d.val)) * 1.05;
    const range = maxVal - minVal || 1;

    const coords = data.map((d, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - ((d.val - minVal) / range) * (height - 8) - 4;
      return { x, y, dataPoint: d };
    });

    const pointsStr = coords.map((c) => `${c.x},${c.y}`).join(' ');
    const firstX = coords[0].x;
    const lastX = coords[coords.length - 1].x;

    const areaPathStr = `M ${firstX},${height} L ${pointsStr} L ${lastX},${height} Z`;

    return { linePath: pointsStr, areaPath: areaPathStr, coords };
  };

  return (
    <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full select-none">
      {/* Card Header */}
      <div className="flex items-center justify-between mb-3 border-b border-[#3a3a3a] pb-2">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide">7-Day Performance Trends</h2>
        <span className="text-[10px] font-mono uppercase tracking-wider text-gray-300 bg-[#383838] border border-[#505050] px-2 py-0.5 rounded">
          Telemetry Micro-Plots
        </span>
      </div>

      {/* Stack of 4 Sparkline Micro Charts */}
      <div className="flex flex-col gap-3 flex-1 justify-between">
        {trends.map((trend) => {
          const { linePath, areaPath, coords } = generateChartPaths(trend.data, 280, 42);

          return (
            <div
              key={trend.id}
              className="bg-[#292929] border border-[#414141] rounded p-2.5 flex flex-col justify-between relative group hover:border-[#606060] transition-colors"
            >
              {/* Sparkline Title & Readout */}
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-300 font-medium">{trend.title}</span>
                <span className="text-xs font-mono font-semibold text-gray-100">
                  {trend.currentValue} <span className="text-[10px] font-normal text-gray-400">{trend.unit}</span>
                </span>
              </div>

              {/* SVG Sparkline Container */}
              <div className="relative w-full h-10 overflow-hidden">
                <svg viewBox="0 0 280 42" preserveAspectRatio="none" className="w-full h-full">
                  <defs>
                    <linearGradient id={`sparkGrad-${trend.id}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#a0a0a0" stopOpacity="0.25" />
                      <stop offset="100%" stopColor="#a0a0a0" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Gradient Area Under Curve */}
                  <path d={areaPath} fill={`url(#sparkGrad-${trend.id})`} />

                  {/* Polyline Curve */}
                  <polyline
                    fill="none"
                    stroke="#b0b0b0"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={linePath}
                  />

                  {/* Interactive Hotspot Circles */}
                  {coords.map((c, i) => (
                    <circle
                      key={i}
                      cx={c.x}
                      cy={c.y}
                      r="3"
                      className="fill-gray-300 opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer hover:r-4"
                      onMouseEnter={() => setHoveredPoint({ trendId: trend.id, point: c.dataPoint })}
                      onMouseLeave={() => setHoveredPoint(null)}
                    />
                  ))}
                </svg>

                {/* Hover Tooltip Overlay */}
                {hoveredPoint && hoveredPoint.trendId === trend.id && (
                  <div className="absolute top-0 right-2 bg-[#383838]/95 text-white text-[10px] font-mono px-1.5 py-0.5 rounded border border-[#606060] shadow z-10 pointer-events-none">
                    {hoveredPoint.point.time}: {hoveredPoint.point.val} {trend.unit}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
