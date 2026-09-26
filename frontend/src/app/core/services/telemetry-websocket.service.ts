import { Injectable, signal, computed, OnDestroy, inject, NgZone } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ConnectionMetrics, ConnectionState, TelemetryEnvelope } from '../models/telemetry.model';

@Injectable({
  providedIn: 'root',
})
export class TelemetryWebSocketService implements OnDestroy {
  private readonly ngZone = inject(NgZone);

  // Raw streams
  private readonly packetSubject = new Subject<TelemetryEnvelope>();
  public readonly packet$: Observable<TelemetryEnvelope> = this.packetSubject.asObservable();

  // Reactive state signals
  public readonly connectionState = signal<ConnectionState>('DISCONNECTED');
  public readonly lastTelemetryTimestamp = signal<number | null>(null);
  public readonly lastPacketReceivedTime = signal<number | null>(null);
  public readonly reconnectAttempts = signal<number>(0);
  public readonly telemetryRate = signal<number>(0);
  public readonly latencyMs = signal<number>(0);
  public readonly packetsReceived = signal<number>(0);

  public readonly metrics = computed<ConnectionMetrics>(() => ({
    state: this.connectionState(),
    lastTelemetryTimestamp: this.lastTelemetryTimestamp(),
    lastPacketReceivedTime: this.lastPacketReceivedTime(),
    reconnectAttempts: this.reconnectAttempts(),
    telemetryRate: this.telemetryRate(),
    latencyMs: this.latencyMs(),
    packetsReceived: this.packetsReceived(),
  }));

  // Internal connection state
  private socket: WebSocket | null = null;
  private reconnectTimer: any = null;
  private staleCheckTimer: any = null;
  private rateCheckTimer: any = null;
  private packetsInCurrentSecond = 0;
  private isIntentionallyClosed = false;

  constructor() {
    this.startRateMonitor();
    this.startStaleMonitor();
    this.connect();
  }

  public connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isIntentionallyClosed = false;
    this.connectionState.set('CONNECTING');

    try {
      this.socket = new WebSocket(environment.wsUrl);

      this.socket.onopen = () => {
        this.ngZone.run(() => {
          this.connectionState.set('CONNECTED');
          this.reconnectAttempts.set(0);
          console.info('[WebSocket] Connected to', environment.wsUrl);
        });
      };

      this.socket.onmessage = (event: MessageEvent) => {
        const arrivalTime = Date.now();
        this.packetsInCurrentSecond++;

        try {
          const parsed = JSON.parse(event.data) as TelemetryEnvelope;
          if (parsed && typeof parsed.timestamp_ms === 'number' && parsed.variables) {
            parsed.reception_wall_clock = arrivalTime;

            // Approximate packet latency if backend includes timestamp or estimated ping
            this.lastTelemetryTimestamp.set(parsed.timestamp_ms);
            this.lastPacketReceivedTime.set(arrivalTime);
            this.packetsReceived.update(count => count + 1);

            // If we were STALE, restore to CONNECTED
            if (this.connectionState() === 'STALE') {
              this.connectionState.set('CONNECTED');
            }

            this.packetSubject.next(parsed);
          } else {
            console.warn('[WebSocket] Malformed telemetry envelope received:', event.data);
          }
        } catch (err) {
          console.error('[WebSocket] Failed to parse message JSON:', err);
        }
      };

      this.socket.onerror = (err: Event) => {
        this.ngZone.run(() => {
          console.warn('[WebSocket] Error on connection:', err);
          if (this.connectionState() !== 'STALE') {
            this.connectionState.set('ERROR');
          }
        });
      };

      this.socket.onclose = () => {
        this.ngZone.run(() => {
          this.socket = null;
          if (!this.isIntentionallyClosed) {
            this.connectionState.set('DISCONNECTED');
            this.scheduleReconnect();
          }
        });
      };
    } catch (e) {
      this.connectionState.set('ERROR');
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.isIntentionallyClosed = true;
    this.clearTimers();
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.connectionState.set('DISCONNECTED');
  }

  private scheduleReconnect(): void {
    if (this.isIntentionallyClosed || this.reconnectTimer) {
      return;
    }

    const currentAttempts = this.reconnectAttempts();
    if (currentAttempts >= environment.maxReconnectAttempts) {
      console.warn(`[WebSocket] Max reconnect attempts reached (${environment.maxReconnectAttempts}).`);
      return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s, up to maxReconnectDelayMs
    const delay = Math.min(
      environment.initialReconnectDelayMs * Math.pow(2, currentAttempts),
      environment.maxReconnectDelayMs
    );

    this.reconnectAttempts.update(n => n + 1);
    console.info(`[WebSocket] Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts()})...`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  private startStaleMonitor(): void {
    // Check every 500ms if telemetry packets have stopped arriving
    this.staleCheckTimer = setInterval(() => {
      const lastReceived = this.lastPacketReceivedTime();
      const state = this.connectionState();

      if (state === 'CONNECTED' && lastReceived !== null) {
        const elapsed = Date.now() - lastReceived;
        if (elapsed > environment.staleThresholdMs) {
          this.connectionState.set('STALE');
          this.telemetryRate.set(0);
        }
      }
    }, 500);
  }

  private startRateMonitor(): void {
    // Computes packet frequency (Hz) every 1000ms
    this.rateCheckTimer = setInterval(() => {
      this.telemetryRate.set(this.packetsInCurrentSecond);
      this.packetsInCurrentSecond = 0;
    }, 1000);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.staleCheckTimer) {
      clearInterval(this.staleCheckTimer);
      this.staleCheckTimer = null;
    }
    if (this.rateCheckTimer) {
      clearInterval(this.rateCheckTimer);
      this.rateCheckTimer = null;
    }
  }

  ngOnDestroy(): void {
    this.disconnect();
    this.packetSubject.complete();
  }
}
