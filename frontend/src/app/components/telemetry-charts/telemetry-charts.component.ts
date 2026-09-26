import {
  Component,
  ElementRef,
  ViewChild,
  OnInit,
  OnDestroy,
  inject,
  signal,
  computed,
  effect,
  NgZone,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { EngineStateStore, PARAMETER_METADATA, TelemetryHistoryPoint } from '../../core/services/engine-state.store';
import { ThemeService } from '../../core/services/theme.service';
import { TelemetryParameterMeta } from '../../core/models/telemetry.model';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-telemetry-charts',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './telemetry-charts.component.html',
  styleUrls: ['./telemetry-charts.component.css'],
})
export class TelemetryChartsComponent implements OnInit, OnDestroy {
  private readonly store = inject(EngineStateStore);
  private readonly route = inject(ActivatedRoute);
  public readonly themeService = inject(ThemeService);
  private readonly ngZone = inject(NgZone);
  private readonly http = inject(HttpClient);

  private readonly seededKeys = new Set<string>();

  @ViewChild('chartCanvas', { static: true })
  canvasRef!: ElementRef<HTMLCanvasElement>;

  // Controls
  public readonly selectedKey = this.store.selectedParameterKey;
  public readonly timeWindowMs = signal<number>(60000); // 1 minute default
  public readonly availableParameters = PARAMETER_METADATA;

  public readonly timeWindows = [
    { label: '30s', ms: 30000 },
    { label: '1m', ms: 60000 },
    { label: '5m', ms: 300000 },
    { label: '10m', ms: 600000 },
  ];

  // Stats for the active window
  public readonly currentValue = signal<number | null>(null);
  public readonly minValue = signal<number | null>(null);
  public readonly maxValue = signal<number | null>(null);
  public readonly avgValue = signal<number | null>(null);

  public readonly activeMeta = computed<TelemetryParameterMeta>(() => {
    const key = this.selectedKey();
    return PARAMETER_METADATA.find(m => m.key === key) ?? PARAMETER_METADATA[0];
  });

  public readonly activeParameter = computed(() =>
    this.store.processedParameters().find(param => param.key === this.selectedKey()) ?? null
  );

  public readonly axisScale = computed(() => {
    const values = this.activeParameter()?.recentSeries ?? [];
    if (!values.length) return { min: '—', mid: '—', max: '—' };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mid = min + (max - min) / 2;
    const decimals = this.activeMeta().decimals;
    return {
      min: min.toFixed(decimals),
      mid: mid.toFixed(decimals),
      max: max.toFixed(decimals),
    };
  });

  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  constructor() {
    const parameter = this.route.snapshot.queryParamMap.get('parameter');
    if (parameter && PARAMETER_METADATA.some(meta => meta.key === parameter)) {
      this.store.selectedParameterKey.set(parameter);
    }

    effect(() => {
      this.selectedKey();
      this.timeWindowMs();
      this.themeService.theme();
      this.updateStats();
      this.renderFrame();
    });

    // Backfill from the backend's real trend buffer (see
    // Simulated_engine/engine_simulator/trend_buffer.py) so switching
    // parameters or opening this page mid-session shows real recent
    // history immediately instead of an empty chart that only fills in
    // live from here on.
    effect(() => {
      const key = this.selectedKey();
      void this.seedHistoryFromServer(key);
    });
  }

  ngOnInit(): void {
    this.setupResizeObserver();
    this.startChartLoop();
  }

  ngOnDestroy(): void {
    if (this.animationFrameId !== null) cancelAnimationFrame(this.animationFrameId);
    this.resizeObserver?.disconnect();
  }

  private updateStats(): void {
    const key = this.selectedKey();
    const points = this.store.getHistoryWindow(this.timeWindowMs());
    const values = points.map(point => {
      const value = point.values[key];
      return key === 'brake_power_w' && value !== undefined ? value / 1000 : value;
    }).filter((value): value is number => value !== undefined && !isNaN(value));

    this.currentValue.set(values.at(-1) ?? null);
    this.minValue.set(values.length ? Math.min(...values) : null);
    this.maxValue.set(values.length ? Math.max(...values) : null);
    this.avgValue.set(values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);
  }

  setWindow(ms: number): void {
    this.timeWindowMs.set(ms);
  }

  setParameter(key: string): void {
    this.store.selectedParameterKey.set(key);
  }

  private async seedHistoryFromServer(key: string): Promise<void> {
    if (this.seededKeys.has(key)) return;
    this.seededKeys.add(key);
    try {
      const res = await firstValueFrom(
        this.http.get<{ key: string; points: Array<{ t_ms: number; value: number }> }>(
          `${environment.httpUrl}/trend/${key}?n=600`,
        ),
      );
      if (!res.points?.length) return;

      const existing = this.store.history();
      const existingTimestamps = new Set(existing.map((p) => p.timestamp_ms));
      const earliestLiveTs = existing[0]?.timestamp_ms ?? Infinity;

      const backfill: TelemetryHistoryPoint[] = res.points
        .filter((p) => p.t_ms < earliestLiveTs && !existingTimestamps.has(p.t_ms))
        .map((p) => ({ timestamp_ms: p.t_ms, wall_clock: p.t_ms, values: { [key]: p.value } }));

      if (backfill.length === 0) return;
      this.store.history.update((current) => [...backfill, ...current]);
      this.renderFrame();
    } catch {
      // Backend not reachable yet, or no trend recorded for this key so
      // far -- the chart just starts from live data only, same as before.
    }
  }

  private setupResizeObserver(): void {
    if (typeof ResizeObserver === 'undefined') return;
    const canvas = this.canvasRef.nativeElement;
    this.resizeObserver = new ResizeObserver(() => {
      this.resizeCanvas();
      this.renderFrame();
    });
    if (canvas.parentElement) {
      this.resizeObserver.observe(canvas.parentElement);
    }
    this.resizeCanvas();
  }

  private resizeCanvas(): void {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
  }

  private startChartLoop(): void {
    this.ngZone.runOutsideAngular(() => {
      const loop = () => {
        this.renderFrame();
        this.animationFrameId = requestAnimationFrame(loop);
      };
      this.animationFrameId = requestAnimationFrame(loop);
    });
  }

  private renderFrame(): void {
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    if (width === 0 || height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const paddingLeft = 55 * dpr;
    const paddingRight = 20 * dpr;
    const paddingTop = 20 * dpr;
    const paddingBottom = 30 * dpr;

    const plotW = width - paddingLeft - paddingRight;
    const plotH = height - paddingTop - paddingBottom;

    const isDark = this.themeService.theme() === 'dark';

    // Clear background
    ctx.fillStyle = isDark ? '#161b22' : '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Get slice of history
    const windowMs = this.timeWindowMs();
    const key = this.selectedKey();
    const meta = this.activeMeta();
    const points = this.store.getHistoryWindow(windowMs);

    if (points.length === 0) {
      this.drawEmptyState(ctx, width, height, dpr, isDark);
      return;
    }

    const latestTime = points[points.length - 1].timestamp_ms;
    const startTime = Math.max(0, latestTime - windowMs);

    // Extract values
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;
    let latestVal: number | null = null;

    const filtered: Array<{ t: number; v: number }> = [];

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      let val = p.values[key];
      if (val !== undefined && !isNaN(val)) {
        if (key === 'brake_power_w') val = val / 1000.0;
        filtered.push({ t: p.timestamp_ms, v: val });
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
        count++;
        latestVal = val;
      }
    }

    if (filtered.length === 0) {
      this.drawEmptyState(ctx, width, height, dpr, isDark);
      return;
    }

    this.currentValue.set(latestVal);
    this.minValue.set(min);
    this.maxValue.set(max);
    this.avgValue.set(count > 0 ? sum / count : null);

    const dataMin = min;
    const dataMax = max;
    const dataSpan = Math.max(1.0, dataMax - dataMin);
    const thresholdWindow = dataSpan * 3;
    const scaleMin = dataMin - thresholdWindow;
    const scaleMax = dataMax + thresholdWindow;

    const nearbyThresholds = [
      meta.cautionMax,
      meta.criticalMax,
      meta.cautionMin,
      meta.criticalMin,
    ].filter((threshold): threshold is number =>
      threshold !== undefined && threshold >= scaleMin && threshold <= scaleMax
    );

    const yMin = Math.min(dataMin, ...nearbyThresholds);
    const yMax = Math.max(dataMax, ...nearbyThresholds);
    const visibleSpan = Math.max(1.0, yMax - yMin);
    const paddedSpan = visibleSpan * 0.1;
    const paddedYMin = yMin - paddedSpan;
    const paddedYMax = yMax + paddedSpan;
    const totalSpan = paddedYMax - paddedYMin;

    // Grid & Axes
    this.drawGrid(ctx, paddingLeft, paddingTop, plotW, plotH, startTime, latestTime, paddedYMin, paddedYMax, dpr, isDark);

    // Thresholds
    this.drawThresholds(ctx, paddingLeft, paddingTop, plotW, plotH, paddedYMin, totalSpan, meta, dpr, isDark);

    // Data Curve
    const displayData = this.aggregateForDisplay(filtered, Math.max(120, Math.floor(plotW / dpr)));
    this.drawLinePlot(ctx, displayData, startTime, latestTime, paddedYMin, totalSpan, paddingLeft, paddingTop, plotW, plotH, dpr, isDark);
  }

  private aggregateForDisplay(data: Array<{ t: number; v: number }>, maxPoints: number): Array<{ t: number; v: number }> {
    if (data.length <= maxPoints) return data;

    const bucketSize = Math.ceil(data.length / maxPoints);
    const aggregated: Array<{ t: number; v: number }> = [];
    for (let index = 0; index < data.length; index += bucketSize) {
      const bucket = data.slice(index, index + bucketSize);
      aggregated.push({
        t: bucket[Math.floor(bucket.length / 2)].t,
        v: bucket.reduce((sum, point) => sum + point.v, 0) / bucket.length,
      });
    }
    return aggregated;
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D,
    pl: number,
    pt: number,
    pw: number,
    ph: number,
    startTime: number,
    latestTime: number,
    yMin: number,
    yMax: number,
    dpr: number,
    isDark: boolean
  ): void {
    ctx.strokeStyle = isDark ? '#21262d' : '#f0f2f5';
    ctx.lineWidth = 1 * dpr;

    ctx.fillStyle = isDark ? '#8b949e' : '#8c95a3';
    ctx.font = `${10 * dpr}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    // Horizontal grid
    const yDivs = 4;
    for (let i = 0; i <= yDivs; i++) {
      const y = pt + (ph * i) / yDivs;
      const val = yMax - ((yMax - yMin) * i) / yDivs;

      ctx.beginPath();
      ctx.moveTo(pl, y);
      ctx.lineTo(pl + pw, y);
      ctx.stroke();

      ctx.fillText(val.toFixed(1), pl - 8 * dpr, y);
    }

    // Vertical time grid
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xDivs = 4;
    const timeSpan = latestTime - startTime;

    for (let i = 0; i <= xDivs; i++) {
      const x = pl + (pw * i) / xDivs;
      const tMs = startTime + (timeSpan * i) / xDivs;

      ctx.beginPath();
      ctx.moveTo(x, pt);
      ctx.lineTo(x, pt + ph);
      ctx.stroke();

      const sec = Math.floor(tMs / 1000);
      const m = Math.floor(sec / 60);
      const s = sec % 60;
      const label = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      ctx.fillText(label, x, pt + ph + 8 * dpr);
    }
  }

  private drawThresholds(
    ctx: CanvasRenderingContext2D,
    pl: number,
    pt: number,
    pw: number,
    ph: number,
    yMin: number,
    totalSpan: number,
    meta: TelemetryParameterMeta,
    dpr: number,
    isDark: boolean
  ): void {
    const drawLine = (val: number, color: string, label: string) => {
      if (val < yMin || val > yMin + totalSpan) return;
      const y = pt + ph - ((val - yMin) / totalSpan) * ph;

      ctx.save();
      ctx.setLineDash([3 * dpr, 3 * dpr]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(pl, y);
      ctx.lineTo(pl + pw, y);
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.font = `${9 * dpr}px -apple-system, sans-serif`;
      ctx.textAlign = 'right';
      ctx.fillText(label, pl + pw - 6 * dpr, y - 4 * dpr);
      ctx.restore();
    };

    const cautionColor = isDark ? '#d29922' : '#d97706';
    const critColor = isDark ? '#f85149' : '#dc2626';

    if (meta.cautionMax !== undefined) {
      drawLine(meta.cautionMax, cautionColor, 'Caution');
    }
    if (meta.criticalMax !== undefined) {
      drawLine(meta.criticalMax, critColor, 'Limit');
    }
    if (meta.cautionMin !== undefined) {
      drawLine(meta.cautionMin, cautionColor, 'Low Caution');
    }
    if (meta.criticalMin !== undefined) {
      drawLine(meta.criticalMin, critColor, 'Low Limit');
    }
  }

  private drawLinePlot(
    ctx: CanvasRenderingContext2D,
    data: Array<{ t: number; v: number }>,
    startTime: number,
    latestTime: number,
    yMin: number,
    totalSpan: number,
    pl: number,
    pt: number,
    pw: number,
    ph: number,
    dpr: number,
    isDark: boolean
  ): void {
    if (data.length === 0) return;
    const timeSpan = Math.max(1, latestTime - startTime);

    const getX = (t: number) => pl + ((t - startTime) / timeSpan) * pw;
    const getY = (v: number) => pt + ph - ((v - yMin) / totalSpan) * ph;

    const strokeColor = isDark ? '#38bdf8' : '#2563eb';
    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2 * dpr;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    data.forEach((point, index) => {
      const x = getX(point.t);
      const y = getY(point.v);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.restore();
  }

  private drawEmptyState(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    dpr: number,
    isDark: boolean
  ): void {
    ctx.fillStyle = isDark ? '#6e7681' : '#9ca3af';
    ctx.font = `${13 * dpr}px -apple-system, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Awaiting live telemetry...', w / 2, h / 2);
  }
}
