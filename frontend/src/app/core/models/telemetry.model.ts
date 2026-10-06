export type ConnectionState = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'STALE' | 'ERROR';

export type ParameterSeverity = 'NORMAL' | 'CAUTION' | 'WARNING' | 'CRITICAL' | 'UNAVAILABLE';

export type ParameterTrend = 'RISING' | 'FALLING' | 'STABLE';

export interface EngineHealth {
  overall_score: number;
  cooling: number;
  lubrication: number;
  injector: number;
  combustion: number;
}

export interface DiagnosticEvent {
  fault: string;
  severity: number;
  confidence: number;
  evidence: string[];
  affected_parameter: string;
  timestamp_ms: number;
  status: 'ACTIVE' | 'RESOLVED' | 'PENDING';
}

export interface MissionInfo {
  mission_id: string;
  mission_elapsed_ms: number;
  altitude_ft: number;
  ambient_temp_c: number;
  engine_load_pct: number;
  engine_operating_state: string;
  mission_phase: string;
}

export interface SystemStatus {
  backend_available: boolean;
  ml_available: boolean;
  llm_available: boolean;
  simulator_running: boolean;
}

export interface TelemetryEnvelope {
  timestamp_ms: number;
  engine_id: string;
  mission_id: string;
  variables: Record<string, number | null | undefined>;
  health: EngineHealth;
  diagnostics: DiagnosticEvent[];
  rul_predictions?: Record<string, number> | null;
  mission?: MissionInfo;
  status?: SystemStatus;
  reception_wall_clock?: number;
}

export interface ConnectionMetrics {
  state: ConnectionState;
  lastTelemetryTimestamp: number | null;
  lastPacketReceivedTime: number | null;
  reconnectAttempts: number;
  telemetryRate: number; // Hz
  latencyMs: number;
  packetsReceived: number;
}

export interface TelemetryParameterMeta {
  key: string;
  label: string;
  unit: string;
  decimals: number;
  category: 'powertrain' | 'thermal' | 'lubrication' | 'combustion' | 'environment' | 'electrical';
  nominalMin?: number;
  nominalMax?: number;
  cautionMin?: number;
  cautionMax?: number;
  criticalMin?: number;
  criticalMax?: number;
}

export interface ProcessedParameter {
  key: string;
  label: string;
  unit: string;
  value: number | null;
  formattedValue: string;
  trend: ParameterTrend;
  status: ParameterSeverity;
  category: string;
  recentSeries?: number[];
}
