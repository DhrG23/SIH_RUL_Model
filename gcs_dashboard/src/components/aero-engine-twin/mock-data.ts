export interface TelemetryCallout {
  id: string;
  name: string;
  locationLabel: string;
  vibration: string;
  vibrationValue: number;
  temperature: string;
  temperatureValue: number;
  status: 'normal' | 'warning' | 'critical';
  position3D: [number, number, number];
  screenPos2D: { x: number; y: number }; // Percentage position on SVG
}

export interface SummaryReadout {
  label: string;
  value: string;
  unit: string;
  status?: 'normal' | 'warning' | 'critical';
}

export interface TelemetryState {
  speedRpm: number;
  throttle: number;
  manifoldPressure: number;
  oilTemp: number;
  oilPressure: number;
  boostPressure: number;
  cylinderHeadTemp: number;
  egt: number;
  thermalEfficiency: number;
  healthScore: number;
  operatingHours: number;
  fuelConsumed: number;
  fuelConsumptionRate: number;
}

export interface SparklinePoint {
  time: string;
  val: number;
}

export interface PerformanceTrend {
  id: string;
  title: string;
  unit: string;
  currentValue: number;
  status: 'normal' | 'warning' | 'critical';
  data: SparklinePoint[];
}

export interface RiskSubsystem {
  name: string;
  level: 'Low' | 'Medium' | 'High';
  statusColor: 'green' | 'amber' | 'red';
}

export interface RecommendationItem {
  id: string;
  type: 'anomaly' | 'warning' | 'maintenance';
  title: string;
  description: string;
  timestamp: string;
}

export const INITIAL_TELEMETRY: TelemetryState = {
  speedRpm: 5800,
  throttle: 82,
  manifoldPressure: 38.2,
  oilTemp: 92,
  oilPressure: 68.5,
  boostPressure: 1.42,
  cylinderHeadTemp: 165,
  egt: 680,
  thermalEfficiency: 84,
  healthScore: 92,
  operatingHours: 4860,
  fuelConsumed: 11483,
  fuelConsumptionRate: 28.4
};

export const HOTSPOT_CALLOUTS: TelemetryCallout[] = [
  {
    id: 'de-cyl1',
    name: 'Drive End (DE) - Cyl 1',
    locationLabel: 'DE',
    vibration: '0.32 mA/g',
    vibrationValue: 0.32,
    temperature: '85 °C',
    temperatureValue: 85,
    status: 'warning',
    position3D: [-0.8, 0.4, 0.5],
    screenPos2D: { x: 18, y: 32 }
  },
  {
    id: 'de-cyl2',
    name: 'Drive End (DE) - Cyl 2',
    locationLabel: 'DE',
    vibration: '0.16 mA/g',
    vibrationValue: 0.16,
    temperature: '98 °C',
    temperatureValue: 98,
    status: 'warning',
    position3D: [-0.2, 0.6, -0.4],
    screenPos2D: { x: 26, y: 15 }
  },
  {
    id: 'nde-fuel',
    name: 'Non-Drive End (NDE) - Fuel Rail',
    locationLabel: 'NDE',
    vibration: '0.24 mA/g',
    vibrationValue: 0.24,
    temperature: '70 °C',
    temperatureValue: 70,
    status: 'normal',
    position3D: [0.6, 0.1, -0.6],
    screenPos2D: { x: 42, y: 72 }
  },
  {
    id: 'nde-turbo',
    name: 'Non-Drive End (NDE) - Turbo Intake',
    locationLabel: 'NDE',
    vibration: '0.17 mA/g',
    vibrationValue: 0.17,
    temperature: '71 °C',
    temperatureValue: 71,
    status: 'normal',
    position3D: [0.8, -0.2, 0.4],
    screenPos2D: { x: 62, y: 48 }
  }
];

export const INITIAL_TRENDS: PerformanceTrend[] = [
  {
    id: 'engine-vibe',
    title: 'Motor / Engine Vibration',
    unit: 'mA/g',
    currentValue: 0.24,
    status: 'normal',
    data: [
      { time: '00:00', val: 0.18 },
      { time: '04:00', val: 0.22 },
      { time: '08:00', val: 0.35 },
      { time: '12:00', val: 0.28 },
      { time: '16:00', val: 0.19 },
      { time: '20:00', val: 0.31 },
      { time: '24:00', val: 0.24 }
    ]
  },
  {
    id: 'pump-vibe',
    title: 'Cylinder / Pump Vibration',
    unit: 'mA/g',
    currentValue: 0.31,
    status: 'normal',
    data: [
      { time: '00:00', val: 0.15 },
      { time: '04:00', val: 0.21 },
      { time: '08:00', val: 0.23 },
      { time: '12:00', val: 0.22 },
      { time: '16:00', val: 0.26 },
      { time: '20:00', val: 0.29 },
      { time: '24:00', val: 0.31 }
    ]
  },
  {
    id: 'flow-rate',
    title: 'Fuel Flow Rate',
    unit: 'L/min',
    currentValue: 32.85,
    status: 'normal',
    data: [
      { time: '00:00', val: 28.1 },
      { time: '04:00', val: 27.9 },
      { time: '08:00', val: 28.5 },
      { time: '12:00', val: 34.2 },
      { time: '16:00', val: 33.8 },
      { time: '20:00', val: 32.5 },
      { time: '24:00', val: 32.85 }
    ]
  },
  {
    id: 'discharge-press',
    title: 'Discharge / Oil Pressure',
    unit: 'kPa',
    currentValue: 91.0,
    status: 'warning',
    data: [
      { time: '00:00', val: 88.2 },
      { time: '04:00', val: 89.1 },
      { time: '08:00', val: 89.5 },
      { time: '12:00', val: 90.2 },
      { time: '16:00', val: 88.9 },
      { time: '20:00', val: 90.1 },
      { time: '24:00', val: 91.0 }
    ]
  }
];

export const SUBSYSTEM_RISKS: RiskSubsystem[] = [
  { name: 'Ignition / Bearing', level: 'Low', statusColor: 'green' },
  { name: 'Mechanical Seal', level: 'Low', statusColor: 'green' },
  { name: 'Fuel Injection / Impeller', level: 'Medium', statusColor: 'amber' },
  { name: 'Overall Anomaly Score', level: 'Low', statusColor: 'green' }
];

export const RECOMMENDATIONS: RecommendationItem[] = [
  {
    id: 'rec-1',
    type: 'anomaly',
    title: 'Aero Engine Discharge Anomaly',
    description: 'Anomaly found on discharge manifold pressure delta threshold (+2.4 kPa)',
    timestamp: '04/03/2026 09:50 AM'
  },
  {
    id: 'rec-2',
    type: 'warning',
    title: 'Cylinder Head 2 Temp Alert',
    description: 'Cylinder head 2 temperature exceeding nominal limits (98 °C), with delta of 8 °C',
    timestamp: '03/03/2026 08:46 AM'
  }
];
