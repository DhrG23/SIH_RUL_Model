import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EngineStateStore } from '../../core/services/engine-state.store';

export interface GaugeConfig {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  nominalStart: number;
  nominalEnd: number;
  cautionStart: number;
  cautionEnd: number;
  criticalStart: number;
  criticalEnd: number;
  ticks: number[];
  decimals: number;
}

@Component({
  selector: 'app-gauge-cluster',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './gauge-cluster.component.html',
  styleUrls: ['./gauge-cluster.component.css'],
})
export class GaugeClusterComponent {
  private readonly store = inject(EngineStateStore);

  public readonly gauges: GaugeConfig[] = [
    {
      key: 'engine_rpm',
      label: 'Engine Speed',
      unit: 'RPM',
      min: 0,
      max: 4000,
      nominalStart: 1800,
      nominalEnd: 3200,
      cautionStart: 3200,
      cautionEnd: 3500,
      criticalStart: 3500,
      criticalEnd: 4000,
      ticks: [0, 1000, 2000, 3000, 4000],
      decimals: 0,
    },
    {
      key: 'cht',
      label: 'Cylinder Head',
      unit: '°C',
      min: 50,
      max: 250,
      nominalStart: 90,
      nominalEnd: 195,
      cautionStart: 195,
      cautionEnd: 220,
      criticalStart: 220,
      criticalEnd: 250,
      ticks: [50, 100, 150, 200, 250],
      decimals: 1,
    },
    {
      key: 'vibration_g_rms',
      label: 'Vibration',
      unit: 'g RMS',
      min: 0,
      max: 1.5,
      nominalStart: 0.1,
      nominalEnd: 0.6,
      cautionStart: 0.6,
      cautionEnd: 0.85,
      criticalStart: 0.85,
      criticalEnd: 1.5,
      ticks: [0, 0.5, 1, 1.5],
      decimals: 2,
    },
    {
      key: 'fuel_mass_flow_gps',
      label: 'Fuel Flow Rate',
      unit: 'g/s',
      min: 0,
      max: 25,
      nominalStart: 2,
      nominalEnd: 18,
      cautionStart: 18,
      cautionEnd: 22,
      criticalStart: 22,
      criticalEnd: 25,
      ticks: [0, 5, 10, 15, 20, 25],
      decimals: 2,
    },
  ];

  // Preserved for compatibility
  public readonly rpmConfig = this.gauges[0];
  public readonly chtConfig = this.gauges[1];

  public readonly latestVars = computed(() => {
    return this.store.latestEnvelope()?.variables ?? {};
  });

  public getParamValue(key: string): number {
    const val = this.latestVars()[key];
    if (val === undefined || val === null || isNaN(val as number)) return 0;
    return Number(val);
  }

  public getSeries(key: string): number[] {
    const history = this.store.history().slice(-48);
    const series = history
      .map(point => point.values[key])
      .filter((value): value is number => value !== undefined && !isNaN(value));
    const current = this.getParamValue(key);
    return series.length > 1 ? series : [current, current];
  }

  public getChartPath(values: number[], min: number, max: number): string {
    const width = 480;
    const height = 150;
    const padding = 8;
    const range = this.getChartRange(values, min, max);
    const span = Math.max(0.0001, range.max - range.min);
    const step = (width - padding * 2) / Math.max(1, values.length - 1);
    const points = values.map((value, index) => {
      const clamped = Math.max(range.min, Math.min(range.max, value));
      const x = padding + index * step;
      const y = height - padding - ((clamped - range.min) / span) * (height - padding * 2);
      return `${x.toFixed(1)} ${y.toFixed(1)}`;
    });
    return points.length ? `M ${points.join(' L ')}` : '';
  }

  public getLastPoint(values: number[], min: number, max: number): { x: number; y: number } {
    const width = 480;
    const height = 150;
    const padding = 8;
    const range = this.getChartRange(values, min, max);
    const span = Math.max(0.0001, range.max - range.min);
    const last = values[values.length - 1] ?? range.min;
    const clamped = Math.max(range.min, Math.min(range.max, last));
    return {
      x: padding + Math.max(0, values.length - 1) * ((width - padding * 2) / Math.max(1, values.length - 1)),
      y: height - padding - ((clamped - range.min) / span) * (height - padding * 2),
    };
  }

  private getChartRange(values: number[], min: number, max: number): { min: number; max: number } {
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const dataSpan = dataMax - dataMin;
    const padding = Math.max(dataSpan * 0.15, (max - min) * 0.02, 0.0001);
    return {
      min: Math.max(min, dataMin - padding),
      max: Math.min(max, dataMax + padding),
    };
  }

  public getGaugeStatus(cfg: GaugeConfig, val: number): 'normal' | 'caution' | 'critical' {
    if (val >= cfg.criticalStart) return 'critical';
    if (val >= cfg.cautionStart) return 'caution';
    return 'normal';
  }

  public getNeedleRotation(val: number, min: number, max: number): number {
    const clamped = Math.max(min, Math.min(max, val));
    const ratio = (clamped - min) / (max - min);
    return -120 + ratio * 240;
  }

  public valToAngle(val: number, min: number, max: number): number {
    const clamped = Math.max(min, Math.min(max, val));
    const ratio = (clamped - min) / (max - min);
    return 210 + ratio * 240;
  }

  public describeArc(
    cx: number,
    cy: number,
    r: number,
    startAngleDeg: number,
    endAngleDeg: number
  ): string {
    const toRad = (deg: number) => ((deg - 90) * Math.PI) / 180.0;
    const start = {
      x: cx + r * Math.cos(toRad(startAngleDeg)),
      y: cy + r * Math.sin(toRad(startAngleDeg)),
    };
    const end = {
      x: cx + r * Math.cos(toRad(endAngleDeg)),
      y: cy + r * Math.sin(toRad(endAngleDeg)),
    };

    const delta = endAngleDeg - startAngleDeg;
    const largeArcFlag = delta <= 180 ? '0' : '1';

    return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
  }

  public describeActiveArc(cx: number, cy: number, r: number, val: number, min: number, max: number): string {
    const angle = this.valToAngle(val, min, max);
    const endAngle = Math.max(210.5, angle);
    return this.describeArc(cx, cy, r, 210, endAngle);
  }

}

