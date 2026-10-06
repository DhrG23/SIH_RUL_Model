import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { App } from './app';
import { routes } from './app.routes';
import { EngineStateStore } from './core/services/engine-state.store';
import { TelemetryWebSocketService } from './core/services/telemetry-websocket.service';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [EngineStateStore, TelemetryWebSocketService, provideRouter(routes)],
    }).compileComponents();
  });

  it('should create the GCS dashboard shell', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render sidebar and top status bar', async () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    await fixture.whenStable();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.sidebar')).toBeTruthy();
    expect(compiled.querySelector('app-top-status-bar')).toBeTruthy();
  });

  it('should render overview components on default view', async () => {
    const fixture = TestBed.createComponent(App);
    await TestBed.inject(Router).navigateByUrl('/overview');
    fixture.detectChanges();
    await fixture.whenStable();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-gauge-cluster')).toBeTruthy();
    expect(compiled.querySelector('app-engine-health')).toBeTruthy();
    expect(compiled.querySelector('app-mission-panel')).toBeTruthy();
  });
});
