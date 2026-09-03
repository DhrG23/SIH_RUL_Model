import React from 'react';
import { RecommendationItem, RiskSubsystem, TelemetryState } from './mock-data';
import { AlertOctagon, Wrench, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface BottomMetricsProps {
  telemetry: TelemetryState;
  subsystemRisks: RiskSubsystem[];
  recommendations: RecommendationItem[];
}

export const BottomMetrics: React.FC<BottomMetricsProps> = ({
  telemetry,
  subsystemRisks,
  recommendations
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-4 select-none">
      {/* Card 1: Operational Parameters */}
      <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide mb-3 border-b border-[#3a3a3a] pb-2">
          Operational Parameters
        </h2>
        <div className="space-y-4">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-gray-400 font-sans">Operational Hours</span>
            <div className="text-right">
              <span className="text-2xl font-bold font-mono text-white tracking-tight">
                {telemetry.operatingHours.toLocaleString()}
              </span>
              <span className="text-[10px] text-gray-500 font-mono ml-1">hrs/year</span>
            </div>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="text-xs text-gray-400 font-sans">YTD Fuel Consumed</span>
            <div className="text-right">
              <span className="text-2xl font-bold font-mono text-white tracking-tight">
                {telemetry.fuelConsumed.toLocaleString()}
              </span>
              <span className="text-xs text-gray-400 font-mono ml-1">L</span>
            </div>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="text-xs text-gray-400 font-sans">Fuel Consumption Rate</span>
            <div className="text-right">
              <span className="text-xl font-bold font-mono text-white tracking-tight">
                {telemetry.fuelConsumptionRate}
              </span>
              <span className="text-[10px] text-gray-500 font-mono ml-1">L/hr</span>
            </div>
          </div>
        </div>
      </div>

      {/* Card 2: Operational Safety Intelligence */}
      <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide mb-3 border-b border-[#3a3a3a] pb-2 flex items-center justify-between">
          <span>Operational Safety Intelligence</span>
          <ShieldAlert className="w-4 h-4 text-gray-300" />
        </h2>
        <div className="space-y-2.5 text-xs">
          <div>
            <div className="font-semibold text-gray-300 mb-0.5">Hazard Description</div>
            <p className="text-gray-400 leading-snug text-[11px]">
              High temperature is likely to cause thermal stress on cylinder head 2, which may be dangerous during high-power cruise altitude.
            </p>
          </div>

          <div>
            <div className="font-semibold text-gray-300 mb-0.5">Control Measure</div>
            <p className="text-gray-400 leading-snug text-[11px]">
              Ensure cooling baffle clearance and optimize fuel-air mixture settings for high-altitude UAV profile.
            </p>
          </div>

          <div className="flex items-center justify-between pt-1 border-t border-[#3a3a3a]/50">
            <span className="font-semibold text-gray-300">Probability</span>
            <div className="flex items-center gap-1.5 bg-[#292929] border border-[#414141] px-2.5 py-0.5 rounded">
              <span className="w-2.5 h-2.5 rounded-full bg-gray-300 animate-pulse" />
              <span className="font-mono text-xs text-white font-medium">High</span>
            </div>
          </div>
        </div>
      </div>

      {/* Card 3: Subsystem Risk */}
      <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide mb-3 border-b border-[#3a3a3a] pb-2">
          Risk Assessment
        </h2>
        <div className="space-y-3">
          {subsystemRisks.map((item, idx) => (
            <div key={idx} className="flex items-center justify-between text-xs py-1 border-b border-[#2d3440] last:border-0">
              <span className="text-gray-300 font-medium">{item.name}</span>
              <div className="flex items-center gap-2">
                <span
                  className={`w-3 h-3 rounded-full ${
                    item.statusColor === 'green'
                      ? 'bg-gray-300'
                      : item.statusColor === 'amber'
                      ? 'bg-gray-400'
                      : 'bg-gray-500'
                  }`}
                />
                <span className="font-mono text-gray-400 w-12 text-right">{item.level}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Card 4: Recommendations */}
      <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full">
        <h2 className="text-sm font-semibold text-gray-200 tracking-wide mb-3 border-b border-[#3a3a3a] pb-2">
          Recommendations
        </h2>
        <div className="space-y-2 overflow-y-auto max-h-[140px] pr-1">
          {recommendations.map((rec) => (
            <div
              key={rec.id}
              className="bg-[#292929] border border-[#414141] rounded p-2 text-xs flex gap-2.5 items-start hover:border-[#606060] transition-colors"
            >
              <div className="p-1 rounded bg-[#383838] text-gray-300 border border-[#505050] shrink-0 mt-0.5">
                {rec.type === 'anomaly' ? (
                  <AlertOctagon className="w-3.5 h-3.5" />
                ) : (
                  <Wrench className="w-3.5 h-3.5" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-semibold text-gray-200 truncate">{rec.title}</div>
                <div className="text-[11px] text-gray-400 leading-tight mt-0.5 line-clamp-2">
                  {rec.description}
                </div>
                <div className="text-[9px] font-mono text-gray-500 mt-1">{rec.timestamp}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
