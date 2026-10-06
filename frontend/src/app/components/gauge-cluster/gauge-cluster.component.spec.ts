import { ComponentFixture, TestBed } from '@angular/core/testing';
import { GaugeClusterComponent } from './gauge-cluster.component';
import { EngineStateStore } from '../../core/services/engine-state.store';
import { TelemetryWebSocketService } from '../../core/services/telemetry-websocket.service';

describe('GaugeClusterComponent', () => {
  let component: GaugeClusterComponent;
  let fixture: ComponentFixture<GaugeClusterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GaugeClusterComponent],
      providers: [EngineStateStore, TelemetryWebSocketService],
    }).compileComponents();

    fixture = TestBed.createComponent(GaugeClusterComponent);
    component = fixture.componentInstance;
  });

  it('should create gauge cluster', () => {
    expect(component).toBeTruthy();
  });

  it('should correctly calculate needle rotation angle', () => {
    // Min should be -120 deg
    expect(component.getNeedleRotation(0, 0, 4000)).toBe(-120);
    // Mid should be 0 deg
    expect(component.getNeedleRotation(2000, 0, 4000)).toBe(0);
    // Max should be +120 deg
    expect(component.getNeedleRotation(4000, 0, 4000)).toBe(120);
  });

  it('should generate valid SVG arc path string', () => {
    const arc = component.describeArc(80, 80, 58, 210, 450);
    expect(arc).toContain('M ');
    expect(arc).toContain(' A 58 58');
  });
});
