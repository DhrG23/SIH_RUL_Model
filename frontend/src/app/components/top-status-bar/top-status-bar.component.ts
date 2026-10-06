import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EngineStateStore } from '../../core/services/engine-state.store';
import { TelemetryWebSocketService } from '../../core/services/telemetry-websocket.service';

@Component({
  selector: 'app-top-status-bar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './top-status-bar.component.html',
  styleUrls: ['./top-status-bar.component.css'],
})
export class TopStatusBarComponent {
  private readonly store = inject(EngineStateStore);
  private readonly wsService = inject(TelemetryWebSocketService);

  public readonly engineId = this.store.engineId;
  public readonly missionId = this.store.missionId;
  public readonly simTimeMs = this.store.simulationTimeMs;
  public readonly health = this.store.health;
  public readonly activeAlertCount = this.store.activeAlertCount;
  public readonly connectionState = this.wsService.connectionState;
  public readonly telemetryRate = this.wsService.telemetryRate;
  public readonly mission = this.store.mission;

  public readonly formattedSimTime = computed(() => {
    const ms = this.simTimeMs();
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    const millis = ms % 1000;

    const pad = (n: number, z = 2) => String(n).padStart(z, '0');
    return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}.${pad(Math.floor(millis / 100), 1)}`;
  });

  public readonly connectionLabel = computed(() => {
    switch (this.connectionState()) {
      case 'CONNECTED':
        return 'Live';
      case 'STALE':
        return 'Stale';
      case 'CONNECTING':
        return 'Connecting';
      default:
        return 'Offline';
    }
  });

  public readonly operatingState = computed(() => {
    return this.mission()?.engine_operating_state ?? 'Awaiting telemetry';
  });

  public readonly hasTelemetry = computed(() => this.store.latestEnvelope() !== null);
  public readonly healthLabel = computed(() => this.hasTelemetry() ? `${this.health().overall_score.toFixed(0)}%` : '—');
}
