import { Injectable, signal, computed, inject, OnDestroy } from '@angular/core';
import { Subscription } from 'rxjs';
import {
  DiagnosticEvent,
  EngineHealth,
  MissionInfo,
  ParameterSeverity,
  ParameterTrend,
  ProcessedParameter,
  TelemetryEnvelope,
  TelemetryParameterMeta,
} from '../models/telemetry.model';
import { TelemetryWebSocketService } from './telemetry-websocket.service';
import { PARAMETER_METADATA } from '../constants/parameter.constants';

export { PARAMETER_METADATA };

export interface TelemetryHistoryPoint {
  timestamp_ms: number;
  wall_clock: number;
  values: Record<string, number>;
}

@Injectable({
  providedIn: 'root',
})
export class EngineStateStore implements OnDestroy {
  private readonly wsService = inject(TelemetryWebSocketService);
  private subscription: Subscription | null = null;

  // Primary Signals
  public readonly latestEnvelope = signal<TelemetryEnvelope | null>(null);
  public readonly history = signal<TelemetryHistoryPoint[]>([]);
  public readonly selectedParameterKey = signal<string>('engine_rpm');
  public readonly maxHistoryPoints = 12000; // 10 minutes at 20 Hz

  // Derived signals
  public readonly engineId = computed(() => this.latestEnvelope()?.engine_id ?? 'DRDO_MALE_UAV_ENG_01');
  public readonly missionId = computed(() => this.latestEnvelope()?.mission_id ?? 'RECON_MISSION_ALPHA');
  public readonly simulationTimeMs = computed(() => this.latestEnvelope()?.timestamp_ms ?? 0);

  public readonly health = computed<EngineHealth>(() => {
    const env = this.latestEnvelope();
    return (
      env?.health ?? {
        overall_score: 0,
        cooling: 0,
        lubrication: 0,
        injector: 0,
        combustion: 0,
      }
    );
  });

  public readonly diagnostics = computed<DiagnosticEvent[]>(() => {
    return this.latestEnvelope()?.diagnostics ?? [];
  });

  public readonly rulPredictions = computed<Record<string, number>>(() => {
    return this.latestEnvelope()?.rul_predictions ?? {};
  });

  public readonly mission = computed<MissionInfo | null>(() => {
    return this.latestEnvelope()?.mission ?? null;
  });

  public readonly activeAlertCount = computed(() => {
    return this.diagnostics().filter(d => d.status === 'ACTIVE').length;
  });

  public readonly processedParameters = computed<ProcessedParameter[]>(() => {
    const env = this.latestEnvelope();
    const variables = env?.variables ?? {};
    const hist = this.history();
    const histLength = hist.length;

    // Compare with point from ~20 steps ago (1.0 sec at 20 Hz)
    const prevPoint = histLength > 20 ? hist[histLength - 20] : null;

    return PARAMETER_METADATA.map(meta => {
      let val = variables[meta.key];
      if (val === undefined || val === null || isNaN(val as number)) {
        return {
          key: meta.key,
          label: meta.label,
          unit: meta.unit,
          value: null,
          formattedValue: 'N/A',
          trend: 'STABLE',
          status: 'UNAVAILABLE',
          category: meta.category,
        };
      }

      let numVal = Number(val);
      // Unit adjustment for brake power from Watts to kW if large
      if (meta.key === 'brake_power_w') {
        numVal = numVal / 1000.0;
      }

      const formatted = numVal.toFixed(meta.decimals);

      // Determine Trend
      let trend: ParameterTrend = 'STABLE';
      if (prevPoint && prevPoint.values[meta.key] !== undefined) {
        let prevVal = prevPoint.values[meta.key];
        if (meta.key === 'brake_power_w') prevVal = prevVal / 1000.0;

        const diff = numVal - prevVal;
        const relativeThreshold = Math.max(0.01, Math.abs(numVal) * 0.015);
        if (diff > relativeThreshold) {
          trend = 'RISING';
        } else if (diff < -relativeThreshold) {
          trend = 'FALLING';
        }
      }

      // Determine Status/Severity
      let status: ParameterSeverity = 'NORMAL';
      if (meta.criticalMax !== undefined && numVal >= meta.criticalMax) {
        status = 'CRITICAL';
      } else if (meta.criticalMin !== undefined && numVal <= meta.criticalMin) {
        status = 'CRITICAL';
      } else if (meta.cautionMax !== undefined && numVal >= meta.cautionMax) {
        status = 'CAUTION';
      } else if (meta.cautionMin !== undefined && numVal <= meta.cautionMin) {
        status = 'CAUTION';
      }

      // Extract recent series for sparkline (last 25 points)
      const sparklineSlice = hist.slice(-25);
      const recentSeries = sparklineSlice.map(h => {
        let v = h.values[meta.key];
        if (v === undefined || isNaN(v)) return numVal;
        if (meta.key === 'brake_power_w') v = v / 1000.0;
        return v;
      });
      if (recentSeries.length === 0 && numVal !== null) {
        recentSeries.push(numVal);
      }

      return {
        key: meta.key,
        label: meta.label,
        unit: meta.unit,
        value: numVal,
        formattedValue: formatted,
        trend,
        status,
        category: meta.category,
        recentSeries,
      };
    });
  });

  constructor() {
    this.subscription = this.wsService.packet$.subscribe(packet => {
      this.handleIncomingPacket(packet);
    });
  }

  private handleIncomingPacket(packet: TelemetryEnvelope): void {
    this.latestEnvelope.set(packet);

    // Extract numeric values for fast chart history
    const numericValues: Record<string, number> = {};
    for (const [k, v] of Object.entries(packet.variables)) {
      if (typeof v === 'number' && !isNaN(v)) {
        numericValues[k] = v;
      }
    }

    const historyPoint: TelemetryHistoryPoint = {
      timestamp_ms: packet.timestamp_ms,
      wall_clock: packet.reception_wall_clock ?? Date.now(),
      values: numericValues,
    };

    this.history.update(arr => {
      const next = [...arr, historyPoint];
      if (next.length > this.maxHistoryPoints) {
        return next.slice(next.length - this.maxHistoryPoints);
      }
      return next;
    });
  }

  /**
   * Retrieves a decimation-optimized slice of history for real-time charting.
   */
  public getHistoryWindow(durationMs: number): TelemetryHistoryPoint[] {
    const hist = this.history();
    if (hist.length === 0) return [];

    const latestTime = hist[hist.length - 1].timestamp_ms;
    const startTime = latestTime - durationMs;

    // Filter points in window
    let i = hist.length - 1;
    while (i >= 0 && hist[i].timestamp_ms >= startTime) {
      i--;
    }
    const startIndex = Math.max(0, i);
    return hist.slice(startIndex);
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }
}
