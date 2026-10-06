import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../../environments/environment';

interface ReplaySample {
  timeMs: number;
  rpm: number;
  fuelFlowKgH: number;
  oilTemp: number;
  oilPressure: number;
  egt: number;
  cht: number;
  vibration: number;
  altitude: number;
  overallHealth: number; // 0-1
  coolingHealth: number;
  lubricationHealth: number;
  injectorHealth: number;
  combustionHealth: number;
  primaryFault: string;
  faultSeverity: number;
}

interface MissionSummary {
  id: string;
  label: string;
  durationMs: number;
  samples: number;
  primaryFaultBreakdown: Record<string, number>;
}

interface ReplayMission extends MissionSummary {
  data: ReplaySample[];
}

interface FaultTimelineEntry {
  fault: string;
  start_ms: number;
  end_ms: number;
  peak_severity: number;
}

interface HealthSummaryEntry {
  label: string;
  start: number;
  end: number;
  min: number;
}

interface SensorSummaryEntry {
  label: string;
  min: number;
  max: number;
  mean: number;
  final: number;
  trend_direction: string;
  trend_r_squared: number;
}

interface AiDiagnosisEvent {
  source: string;
  fault?: string;
  status?: string;
  affected_parameter?: string;
  [key: string]: unknown;
}

interface AdvisoryEntry {
  fault: string;
  meaning: string;
  color: string;
}

interface MissionReport {
  mission_id: string;
  label: string;
  duration_ms: number;
  samples: number;
  ground_truth_fault_timeline: FaultTimelineEntry[];
  health_summary: Record<string, HealthSummaryEntry>;
  sensor_summary: Record<string, SensorSummaryEntry>;
  ai_diagnosis_events: AiDiagnosisEvent[];
  maintenance_advisory: AdvisoryEntry[];
  note: string;
}

const FAULT_LABELS: Record<string, string> = {
  nominal: 'Nominal',
  cooling_degradation: 'Cooling Degradation',
  lubrication_degradation: 'Lubrication Degradation',
  oil_pump_degradation: 'Oil Pump Degradation',
  injector_degradation: 'Injector Degradation',
  combustion_abnormality: 'Combustion Abnormality',
};

@Component({
  selector: 'app-mission-replay-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './mission-replay-page.component.html',
  styleUrls: ['./mission-replay-page.component.css'],
})
export class MissionReplayPageComponent implements OnInit, OnDestroy {
  private readonly http = inject(HttpClient);

  readonly playbackSpeeds = [0.25, 0.5, 1, 2, 4];
  readonly missionList = signal<MissionSummary[]>([]);
  readonly selectedMissionId = signal<string | null>(null);
  readonly missionData = signal<ReplayMission | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly currentIndex = signal(0);
  readonly playbackSpeed = signal(1);
  readonly isPlaying = signal(false);

  private playbackTimer: ReturnType<typeof setInterval> | null = null;

  readonly reportOpen = signal(false);
  readonly reportLoading = signal(false);
  readonly reportError = signal<string | null>(null);
  readonly report = signal<MissionReport | null>(null);

  readonly confirmedAiFaults = computed(() => {
    const events = this.report()?.ai_diagnosis_events ?? [];
    const seen = new Map<string, AiDiagnosisEvent>();
    for (const event of events) {
      if (event.source === 'fault_classifier' && event.fault && event.fault !== 'Normal') {
        seen.set(event.fault, event); // keep the latest occurrence of each fault type
      }
    }
    return [...seen.values()];
  });

  readonly driftAlerts = computed(() =>
    (this.report()?.ai_diagnosis_events ?? []).filter((e) => e.source === 'sensor_drift_detector'),
  );

  readonly currentSample = computed<ReplaySample | null>(() => {
    const mission = this.missionData();
    if (!mission || mission.data.length === 0) return null;
    return mission.data[Math.min(this.currentIndex(), mission.data.length - 1)];
  });

  readonly progress = computed(() => {
    const mission = this.missionData();
    if (!mission || mission.data.length < 2) return 0;
    return (this.currentIndex() / (mission.data.length - 1)) * 100;
  });

  readonly healthLinePoints = computed(() => {
    const mission = this.missionData();
    if (!mission || mission.data.length < 2) return '';
    const n = mission.data.length;
    return mission.data
      .map((sample, i) => `${(i / (n - 1)) * 100},${100 - sample.overallHealth * 100}`)
      .join(' ');
  });

  readonly healthAreaPoints = computed(() => {
    const line = this.healthLinePoints();
    return line ? `0,100 ${line} 100,100` : '';
  });

  readonly playheadX = computed(() => this.progress());
  readonly playheadY = computed(() => 100 - (this.currentSample()?.overallHealth ?? 1) * 100);

  ngOnInit(): void {
    void this.loadMissionList();
  }

  ngOnDestroy(): void {
    this.stopPlayback();
  }

  async loadMissionList(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const res = await firstValueFrom(
        this.http.get<{ missions: any[] }>(`${environment.httpUrl}/missions`),
      );
      const list: MissionSummary[] = (res.missions || []).map((m) => ({
        id: m.id,
        label: m.label,
        durationMs: m.duration_ms,
        samples: m.samples,
        primaryFaultBreakdown: m.primary_fault_breakdown ?? {},
      }));
      this.missionList.set(list);
      if (list.length > 0) await this.selectMission(list[0].id);
    } catch {
      this.error.set(
        'Could not reach the simulator backend for mission data. Start it with: ' +
          'python3 -m engine_simulator.ws_server --port 8001',
      );
    } finally {
      this.loading.set(false);
    }
  }

  async selectMission(missionId: string): Promise<void> {
    this.stopPlayback();
    this.isPlaying.set(false);
    this.currentIndex.set(0);
    this.selectedMissionId.set(missionId);
    this.loading.set(true);
    this.error.set(null);
    try {
      const res: any = await firstValueFrom(
        this.http.get(`${environment.httpUrl}/missions/${missionId}`),
      );
      const groundTruthByTime = new Map<number, any>();
      (res.ground_truth || []).forEach((g: any) => groundTruthByTime.set(g.time_ms, g));

      const data: ReplaySample[] = (res.telemetry || []).map((t: any) => {
        const gt = groundTruthByTime.get(t.time_ms) ?? {};
        return {
          timeMs: t.time_ms ?? 0,
          rpm: t.engine_rpm ?? 0,
          fuelFlowKgH: (t.fuel_mass_flow_gps ?? 0) * 3.6,
          oilTemp: t.oil_temperature ?? 0,
          oilPressure: t.oil_pressure ?? 0,
          egt: t.egt ?? 0,
          cht: t.cht ?? 0,
          vibration: t.vibration_g_rms ?? 0,
          altitude: t.altitude ?? 0,
          overallHealth: t.overall_engine_health ?? gt.true_overall_health ?? 1,
          coolingHealth: t.cooling_health_index ?? gt.true_cooling_health ?? 1,
          lubricationHealth: t.lubrication_health_index ?? gt.true_lubrication_health ?? 1,
          injectorHealth: t.injector_health_index ?? gt.true_injector_health ?? 1,
          combustionHealth: t.combustion_health_index ?? gt.true_combustion_health ?? 1,
          primaryFault: gt.primary_fault ?? 'nominal',
          faultSeverity: gt.max_fault_severity ?? 0,
        };
      });

      const summary = this.missionList().find((m) => m.id === missionId);
      this.missionData.set({
        id: missionId,
        label: summary?.label ?? res.label ?? missionId,
        durationMs: summary?.durationMs ?? 0,
        samples: data.length,
        primaryFaultBreakdown: summary?.primaryFaultBreakdown ?? {},
        data,
      });
    } catch {
      this.error.set('Failed to load telemetry for this mission.');
    } finally {
      this.loading.set(false);
    }
  }

  setSpeed(speed: number): void {
    this.playbackSpeed.set(speed);
    if (this.isPlaying()) this.startPlayback();
  }

  togglePlayback(): void {
    const mission = this.missionData();
    if (!mission) return;
    if (this.isPlaying()) {
      this.isPlaying.set(false);
      this.stopPlayback();
      return;
    }
    if (this.currentIndex() >= mission.data.length - 1) this.currentIndex.set(0);
    this.isPlaying.set(true);
    this.startPlayback();
  }

  skip(steps: number): void {
    const mission = this.missionData();
    if (!mission) return;
    const lastIndex = mission.data.length - 1;
    this.currentIndex.update((current) => Math.max(0, Math.min(lastIndex, current + steps)));
    if (this.currentIndex() === lastIndex) {
      this.isPlaying.set(false);
      this.stopPlayback();
    }
  }

  seek(event: Event): void {
    this.currentIndex.set(Number((event.target as HTMLInputElement).value));
  }

  formatTime(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  formatNumber(value: number, decimals = 0): string {
    return value.toLocaleString('en-US', { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
  }

  faultLabel(fault: string | undefined): string {
    if (!fault) return 'Nominal';
    return FAULT_LABELS[fault] ?? fault.replace(/_/g, ' ');
  }

  async openReport(): Promise<void> {
    const missionId = this.selectedMissionId();
    if (!missionId) return;
    this.reportOpen.set(true);
    this.reportError.set(null);
    this.reportLoading.set(true);
    try {
      const data = await firstValueFrom(
        this.http.get<MissionReport>(`${environment.httpUrl}/missions/${missionId}/report`),
      );
      this.report.set(data);
    } catch {
      this.reportError.set(
        'Could not generate the mission report. The backend replays every fault-classifier and ' +
          'drift-detector tick for the full mission, so this can take a little while on long missions.',
      );
    } finally {
      this.reportLoading.set(false);
    }
  }

  closeReport(): void {
    this.reportOpen.set(false);
  }

  printReport(): void {
    window.print();
  }

  healthLabel(key: string, fallback: string): string {
    return this.report()?.health_summary[key]?.label ?? fallback;
  }

  asPercent(value: number | undefined): string {
    return value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
  }

  trendIcon(direction: string): string {
    return direction === 'rising' ? '↑' : direction === 'falling' ? '↓' : '→';
  }

  private startPlayback(): void {
    this.stopPlayback();
    this.playbackTimer = setInterval(() => this.skip(1), 1000 / this.playbackSpeed());
  }

  private stopPlayback(): void {
    if (this.playbackTimer !== null) clearInterval(this.playbackTimer);
    this.playbackTimer = null;
  }
}
