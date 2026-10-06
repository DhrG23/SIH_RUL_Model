import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TelemetryWebSocketService } from '../../core/services/telemetry-websocket.service';
import { EngineStateStore } from '../../core/services/engine-state.store';

@Component({
  selector: 'app-system-status',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './system-status.component.html',
  styleUrls: ['./system-status.component.css'],
})
export class SystemStatusComponent {
  private readonly wsService = inject(TelemetryWebSocketService);
  private readonly store = inject(EngineStateStore);

  public readonly connectionState = this.wsService.connectionState;
  public readonly telemetryRate = this.wsService.telemetryRate;
  public readonly packetsReceived = this.wsService.packetsReceived;
  public readonly reconnectAttempts = this.wsService.reconnectAttempts;

  public readonly lastReceivedText = computed(() => {
    const last = this.wsService.lastPacketReceivedTime();
    if (!last) return 'Never';
    const elapsed = Date.now() - last;
    return `${elapsed} ms ago`;
  });

  public readonly backendStatus = computed(() => {
    const s = this.connectionState();
    return s === 'CONNECTED' ? 'ONLINE' : (s === 'STALE' ? 'UNRESPONSIVE' : 'OFFLINE');
  });

  public reconnectManually(): void {
    this.wsService.disconnect();
    setTimeout(() => {
      this.wsService.connect();
    }, 200);
  }
}
