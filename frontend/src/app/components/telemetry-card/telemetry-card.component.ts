import { Component, Input, Output, EventEmitter, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ProcessedParameter } from '../../core/models/telemetry.model';

@Component({
  selector: 'app-telemetry-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './telemetry-card.component.html',
  styleUrls: ['./telemetry-card.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TelemetryCardComponent {
  @Input({ required: true }) param!: ProcessedParameter;
  @Input() isSelected = false;
  @Input() listMode = false;
  @Output() cardSelect = new EventEmitter<string>();

  onSelect(): void {
    this.cardSelect.emit(this.param.key);
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onSelect();
    }
  }

  get trendSymbol(): string {
    switch (this.param.trend) {
      case 'RISING':
        return '↑';
      case 'FALLING':
        return '↓';
      default:
        return '→';
    }
  }

  get trendClass(): string {
    return `trend-${this.param.trend.toLowerCase()}`;
  }

  get statusClass(): string {
    return `status-${this.param.status.toLowerCase()}`;
  }
}
