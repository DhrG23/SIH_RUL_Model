import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { AlertPanelComponent } from '../../components/alert-panel/alert-panel.component';
import { Engine3dViewComponent } from '../../components/engine-3d-view/engine-3d-view.component';
import { GaugeClusterComponent } from '../../components/gauge-cluster/gauge-cluster.component';
import { RulPredictionsComponent } from '../../components/rul-predictions/rul-predictions.component';
import { ModelAccuracyComponent } from '../../components/model-accuracy/model-accuracy.component';
import { EngineHealthComponent } from '../../components/engine-health/engine-health.component';
import { MissionPanelComponent } from '../../components/mission-panel/mission-panel.component';
import { TelemetryChartsComponent } from '../../components/telemetry-charts/telemetry-charts.component';
import { TelemetryGridComponent } from '../../components/telemetry-grid/telemetry-grid.component';
import { EngineStateStore } from '../../core/services/engine-state.store';
import { TelemetryWebSocketService } from '../../core/services/telemetry-websocket.service';

@Component({
  selector: 'app-dashboard-page',
  standalone: true,
  imports: [
    CommonModule, AlertPanelComponent, Engine3dViewComponent,
    GaugeClusterComponent,
    TelemetryChartsComponent, TelemetryGridComponent, RulPredictionsComponent, ModelAccuracyComponent,
    EngineHealthComponent, MissionPanelComponent,
  ],
  templateUrl: './dashboard-page.component.html',
  styleUrls: ['./dashboard-page.component.css'],
})
export class DashboardPageComponent {
  private readonly route = inject(ActivatedRoute);
  readonly store = inject(EngineStateStore);
  readonly wsService = inject(TelemetryWebSocketService);

  readonly view = computed(() => this.route.snapshot.data['view'] as string);
  readonly pageTitle = computed(() => ({
    overview: 'Propulsion health overview',
    telemetry: 'Telemetry',
    gauges: 'Engine instruments',
    charts: 'Telemetry trends',
    diagnostics: 'Diagnostics',
  }[this.view()] ?? 'Mission overview'));
  readonly pageDescription = computed(() => ({
    overview: 'Live digital-twin state, engine health, mission context, predictive diagnostics, and maintenance indicators.',
    telemetry: 'Browse all live channels in one focused monitoring view.',
    gauges: 'Compare the readings that matter most to engine operating margins.',
    charts: 'Investigate a signal over time and validate changes against its thresholds.',
    diagnostics: '',
  }[this.view()] ?? 'Operational engine monitoring.'));
  readonly health = this.store.health;
  readonly connectionState = this.wsService.connectionState;
}
