import React, { useState, useEffect } from 'react';
import { EngineSchematic } from './engine-schematic';
import { RadialGauge } from './radial-gauge';
import { PerformanceTrends } from './performance-trends';
import { BottomMetrics } from './bottom-metrics';
import {
  HOTSPOT_CALLOUTS,
  INITIAL_TELEMETRY,
  INITIAL_TRENDS,
  RECOMMENDATIONS,
  SUBSYSTEM_RISKS,
  TelemetryState,
  PerformanceTrend
} from './mock-data';

export const AeroEngineDashboard: React.FC = () => {
  const [telemetry, setTelemetry] = useState<TelemetryState>(INITIAL_TELEMETRY);
  const [trends, setTrends] = useState<PerformanceTrend[]>(INITIAL_TRENDS);
  const [isSimulating, setIsSimulating] = useState<boolean>(true);

  // Periodic Telemetry Simulation Tick
  useEffect(() => {
    if (!isSimulating) return;

    const interval = setInterval(() => {
      setTelemetry((prev) => {
        const rpmNoise = Math.floor((Math.random() - 0.5) * 40);
        const tempNoise = Number(((Math.random() - 0.5) * 1.5).toFixed(1));
        const pressNoise = Number(((Math.random() - 0.5) * 0.4).toFixed(1));

        return {
          ...prev,
          speedRpm: Math.min(6200, Math.max(5400, prev.speedRpm + rpmNoise)),
          cylinderHeadTemp: Math.min(185, Math.max(140, Number((prev.cylinderHeadTemp + tempNoise).toFixed(1)))),
          oilPressure: Math.min(75, Math.max(60, Number((prev.oilPressure + pressNoise).toFixed(1)))),
          fuelConsumptionRate: Math.min(32, Math.max(24, Number((prev.fuelConsumptionRate + pressNoise * 0.2).toFixed(1))))
        };
      });

      // Update Trend Sparklines with latest value
      setTrends((prevTrends) =>
        prevTrends.map((t) => {
          const lastVal = t.data[t.data.length - 1].val;
          const noise = (Math.random() - 0.5) * (lastVal * 0.05);
          const newVal = Number((lastVal + noise).toFixed(2));

          const updatedData = [...t.data.slice(1), { time: 'Now', val: newVal }];
          return {
            ...t,
            currentValue: newVal,
            data: updatedData
          };
        })
      );
    }, 2500);

    return () => clearInterval(interval);
  }, [isSimulating]);

  return (
    <div className="flex h-screen w-screen bg-[#282828] text-[#d0d0d0] overflow-hidden select-none font-sans">
      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Dashboard Scrollable Grid Container */}
        <main className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Upper Grid (Row 1) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-auto lg:h-[480px]">
            {/* Left Column: Schematic / 3D Model Viewport (Col span 6) */}
            <div className="lg:col-span-6 h-full">
              <EngineSchematic telemetry={telemetry} hotspots={HOTSPOT_CALLOUTS} />
            </div>

            {/* Center Column: Stacked 2 Radial Gauges (Col span 3) */}
            <div className="lg:col-span-3 grid grid-rows-2 gap-4 h-full">
              <RadialGauge
                title="Thermal Efficiency"
                value={telemetry.thermalEfficiency}
                subText="Nominal: >80%"
              />
              <RadialGauge
                title="Health Score"
                value={telemetry.healthScore}
                subText="Overall UAV RUL"
              />
            </div>

            {/* Right Column: Performance Trends Sparklines (Col span 3) */}
            <div className="lg:col-span-3 h-full">
              <PerformanceTrends trends={trends} />
            </div>
          </div>

          {/* Lower Grid (Row 2: 4 Footer Cards) */}
          <BottomMetrics
            telemetry={telemetry}
            subsystemRisks={SUBSYSTEM_RISKS}
            recommendations={RECOMMENDATIONS}
          />
        </main>
      </div>
    </div>
  );
};

export default AeroEngineDashboard;
