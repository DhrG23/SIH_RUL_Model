import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EngineStateStore, PARAMETER_METADATA } from '../../core/services/engine-state.store';
import { DiagnosticEvent } from '../../core/models/telemetry.model';

@Component({
  selector: 'app-alert-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './alert-panel.component.html',
  styleUrls: ['./alert-panel.component.css'],
})
export class AlertPanelComponent {
  private readonly store = inject(EngineStateStore);

  public readonly diagnostics = this.store.diagnostics;
  public readonly filter = signal<'ALL' | 'ACTIVE' | 'PENDING'>('ALL');

  public readonly activeAlerts = computed<DiagnosticEvent[]>(() => {
    return this.diagnostics().filter(d => d.status === 'ACTIVE');
  });

  public readonly pendingAlerts = computed<DiagnosticEvent[]>(() => {
    return this.diagnostics().filter(d => d.status === 'PENDING');
  });

  public readonly visibleDiagnostics = computed(() => {
    const selectedFilter = this.filter();
    if (selectedFilter === 'ACTIVE') return this.activeAlerts();
    if (selectedFilter === 'PENDING') return this.pendingAlerts();
    return this.diagnostics();
  });

  public getSeverityLevel(severity: number): 'INFO' | 'WARNING' | 'CRITICAL' {
    if (severity >= 0.7) return 'CRITICAL';
    if (severity >= 0.3) return 'WARNING';
    return 'INFO';
  }

  public formatFaultName(fault: string): string {
    // Model-classified faults ("Misfire", "Injector Abnormality") already
    // arrive human-readable; detector-generated codes ("SENSOR_DRIFT",
    // "cht_overheat_threshold_exceeded") are snake/upper case -- title-case
    // them here rather than a single string.replace() that only touches
    // the first underscore.
    if (/[a-z]/.test(fault) && !fault.includes('_')) return fault;
    return fault
      .split('_')
      .map((word) => (word.length ? word[0].toUpperCase() + word.slice(1).toLowerCase() : word))
      .join(' ');
  }

  public parameterLabel(key: string): string {
    return PARAMETER_METADATA.find((p) => p.key === key)?.label ?? this.formatFaultName(key);
  }

  public formatTime(ms: number): string {
    const sec = Math.floor(ms / 1000);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    const millis = ms % 1000;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
  }
}
