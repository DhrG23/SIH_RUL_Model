import { TestBed } from '@angular/core/testing';
import { EngineStateStore } from './engine-state.store';
import { TelemetryWebSocketService } from './telemetry-websocket.service';
import { TelemetryEnvelope } from '../models/telemetry.model';

describe('EngineStateStore', () => {
  let store: EngineStateStore;
  let wsService: TelemetryWebSocketService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [EngineStateStore, TelemetryWebSocketService],
    });
    store = TestBed.inject(EngineStateStore);
    wsService = TestBed.inject(TelemetryWebSocketService);
  });

  it('should initialize with default states and empty diagnostics', () => {
    expect(store.engineId()).toBe('DRDO_MALE_UAV_ENG_01');
    expect(store.missionId()).toBe('RECON_MISSION_ALPHA');
    expect(store.simulationTimeMs()).toBe(0);
    expect(store.activeAlertCount()).toBe(0);
    expect(store.health().overall_score).toBe(0);
  });

  it('should process telemetry packet into parameters and health', () => {
    const mockPacket: TelemetryEnvelope = {
      timestamp_ms: 15400,
      engine_id: 'DRDO_MALE_UAV_ENG_01',
      mission_id: 'RECON_MISSION_ALPHA',
      variables: {
        engine_rpm: 3150,
        cht: 165.4,
        egt: 742.0,
        oil_pressure: 48.5,
        oil_temperature: 92.1,
      },
      health: {
        overall_score: 94.5,
        cooling: 96.0,
        lubrication: 93.0,
        injector: 99.0,
        combustion: 98.0,
      },
      diagnostics: [
        {
          fault: 'cooling_degradation',
          severity: 0.35,
          confidence: 0.88,
          evidence: ['CHT rising above cruise nominal'],
          affected_parameter: 'cht',
          timestamp_ms: 15400,
          status: 'ACTIVE',
        },
      ],
    };

    // Inject packet via private subject
    (wsService as any).packetSubject.next(mockPacket);

    expect(store.simulationTimeMs()).toBe(15400);
    expect(store.health().overall_score).toBe(94.5);
    expect(store.activeAlertCount()).toBe(1);

    const params = store.processedParameters();
    const rpmParam = params.find(p => p.key === 'engine_rpm');
    expect(rpmParam?.value).toBe(3150);
    expect(rpmParam?.status).toBe('NORMAL');

    const chtParam = params.find(p => p.key === 'cht');
    expect(chtParam?.value).toBe(165.4);
    expect(chtParam?.status).toBe('NORMAL');
  });

  it('should gracefully handle missing variables with UNAVAILABLE status', () => {
    const mockPacket: TelemetryEnvelope = {
      timestamp_ms: 1000,
      engine_id: 'DRDO_MALE_UAV_ENG_01',
      mission_id: 'RECON_MISSION_ALPHA',
      variables: {
        // missing all other variables
        engine_rpm: 3000,
      },
      health: {
        overall_score: 100,
        cooling: 100,
        lubrication: 100,
        injector: 100,
        combustion: 100,
      },
      diagnostics: [],
    };

    (wsService as any).packetSubject.next(mockPacket);

    const params = store.processedParameters();
    const chtParam = params.find(p => p.key === 'cht');
    expect(chtParam?.value).toBeNull();
    expect(chtParam?.formattedValue).toBe('N/A');
    expect(chtParam?.status).toBe('UNAVAILABLE');
  });
});
