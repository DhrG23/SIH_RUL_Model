import { Component, Input, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ParameterSeverity } from '../../core/models/telemetry.model';

@Component({
  selector: 'app-sparkline',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './sparkline.component.html',
  styleUrls: ['./sparkline.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SparklineComponent {
  private readonly _data = signal<number[]>([]);
  @Input()
  set data(v: number[]) {
    this._data.set(v ?? []);
  }
  get data(): number[] {
    return this._data();
  }

  private readonly _status = signal<ParameterSeverity>('NORMAL');
  @Input()
  set status(v: ParameterSeverity) {
    this._status.set(v ?? 'NORMAL');
  }
  get status(): ParameterSeverity {
    return this._status();
  }

  private readonly _width = signal<number>(72);
  @Input()
  set width(v: number) {
    this._width.set(v ?? 72);
  }
  get width(): number {
    return this._width();
  }

  private readonly _height = signal<number>(24);
  @Input()
  set height(v: number) {
    this._height.set(v ?? 24);
  }
  get height(): number {
    return this._height();
  }

  public readonly pathData = computed(() => {
    const pts = this._data();
    if (!pts || pts.length < 2) return { line: '', area: '', lastX: 0, lastY: 0 };

    const w = this._width();
    const h = this._height();
    const padY = 2;
    const effH = Math.max(1, h - padY * 2);

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      const v = pts[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const span = Math.max(0.0001, max - min);
    const stepX = w / (pts.length - 1);

    const coords = pts.map((v, i) => {
      const x = Number((i * stepX).toFixed(2));
      const y = Number((h - padY - ((v - min) / span) * effH).toFixed(2));
      return { x, y };
    });

    // Build SVG line
    let line = `M ${coords[0].x} ${coords[0].y}`;
    for (let i = 1; i < coords.length; i++) {
      line += ` L ${coords[i].x} ${coords[i].y}`;
    }

    // Build SVG area
    const last = coords[coords.length - 1];
    const area = `${line} L ${last.x} ${h} L ${coords[0].x} ${h} Z`;

    return { line, area, lastX: last.x, lastY: last.y };
  });

  get strokeColor(): string {
    const s = this._status();
    if (s === 'CRITICAL' || s === 'WARNING') {
      return 'var(--status-critical)';
    }
    if (s === 'CAUTION') {
      return 'var(--status-caution)';
    }
    return 'var(--accent-primary)';
  }

  get fillColor(): string {
    const s = this._status();
    if (s === 'CRITICAL' || s === 'WARNING') {
      return 'var(--status-critical-bg)';
    }
    if (s === 'CAUTION') {
      return 'var(--status-caution-bg)';
    }
    return 'var(--accent-subtle)';
  }
}

