import React from 'react';

interface RadialGaugeProps {
  title: string;
  value: number; // 0 to 100
  unit?: string;
  subText?: string;
}

export const RadialGauge: React.FC<RadialGaugeProps> = ({
  title,
  value,
  unit = '%',
  subText
}) => {
  // Clamp value 0..100
  const clampedVal = Math.min(100, Math.max(0, value));

  // Angles in degrees for semi-circle arc (from 140 deg to 400 deg -> total 260 deg)
  const startAngle = 140;
  const endAngle = 400;
  const totalAngle = endAngle - startAngle;

  const currentAngle = startAngle + (clampedVal / 100) * totalAngle;

  // Polar to Cartesian conversion helper
  const polarToCartesian = (centerX: number, centerY: number, radius: number, angleInDegrees: number) => {
    const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180.0;
    return {
      x: centerX + radius * Math.cos(angleInRadians),
      y: centerY + radius * Math.sin(angleInRadians)
    };
  };

  // SVG Arc generator
  const describeArc = (x: number, y: number, radius: number, startAng: number, endAng: number) => {
    const start = polarToCartesian(x, y, radius, endAng);
    const end = polarToCartesian(x, y, radius, startAng);
    const largeArcFlag = endAng - startAng <= 180 ? '0' : '1';

    return [
      'M', start.x, start.y,
      'A', radius, radius, 0, largeArcFlag, 0, end.x, end.y
    ].join(' ');
  };

  const cx = 110;
  const cy = 115;
  const radius = 70;

  // Ticks at intervals of 10
  const ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

  // Needle tip coordinates
  const needleTip = polarToCartesian(cx, cy, radius - 12, currentAngle);

  return (
    <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full relative overflow-hidden select-none">
      {/* Title Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide">{title}</h2>
        {subText && <span className="text-[11px] font-mono text-gray-400">{subText}</span>}
      </div>

      {/* SVG Radial Gauge Dial */}
      <div className="relative flex items-center justify-center py-2 my-auto">
        <svg viewBox="0 0 220 160" className="w-full max-w-[210px] h-auto overflow-visible">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#777777" />
              <stop offset="50%" stopColor="#b0b0b0" />
              <stop offset="100%" stopColor="#e0e0e0" />
            </linearGradient>
            <filter id="gaugeGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Track Arc */}
          <path
            d={describeArc(cx, cy, radius, startAngle, endAngle)}
            fill="none"
            stroke="#1a1d24"
            strokeWidth="4"
            strokeLinecap="round"
          />

          {/* Scale Ticks & Number Labels */}
          {ticks.map((tickVal) => {
            const tickAngle = startAngle + (tickVal / 100) * totalAngle;
            const innerPt = polarToCartesian(cx, cy, radius + 4, tickAngle);
            const outerPt = polarToCartesian(cx, cy, radius + 8, tickAngle);
            const labelPt = polarToCartesian(cx, cy, radius + 18, tickAngle);

            return (
              <g key={tickVal}>
                {/* Tick line */}
                <line
                  x1={innerPt.x}
                  y1={innerPt.y}
                  x2={outerPt.x}
                  y2={outerPt.y}
                  stroke="#566173"
                  strokeWidth={tickVal % 20 === 0 ? "1.5" : "1"}
                />
                {/* Numeric tick text */}
                <text
                  x={labelPt.x}
                  y={labelPt.y}
                  fill="#788599"
                  fontSize="8.5"
                  fontFamily="monospace"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {tickVal}
                </text>
              </g>
            );
          })}

          {/* Active Arc Fill */}
          {clampedVal > 0 && (
            <path
              d={describeArc(cx, cy, radius, startAngle, currentAngle)}
              fill="none"
              stroke="url(#gaugeGradient)"
              strokeWidth="4"
              strokeLinecap="round"
              filter="url(#gaugeGlow)"
              className="transition-all duration-500 ease-out"
            />
          )}

          {/* Center Pointer Needle */}
          <g className="transition-all duration-500 ease-out">
            <line
              x1={cx}
              y1={cy}
              x2={needleTip.x}
              y2={needleTip.y}
              stroke="#cbd5e1"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            {/* Center Pivot Circle */}
            <circle cx={cx} cy={cy} r="4" fill="#b0b0b0" stroke="#ffffff" strokeWidth="1" />
          </g>

          {/* Floating Center Percentage Pill Badge */}
          <g transform={`translate(${cx - 24}, ${cy - 48})`}>
            <rect
              width="48"
              height="22"
              rx="4"
              fill="#5a5a5a"
              className="drop-shadow-md"
            />
            <text
              x="24"
              y="12.5"
              fill="#ffffff"
              fontSize="11"
              fontWeight="bold"
              fontFamily="monospace"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {Math.round(clampedVal)}{unit}
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
};
