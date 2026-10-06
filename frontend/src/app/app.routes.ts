import { Routes } from '@angular/router';
import { DashboardPageComponent } from './pages/dashboard-page/dashboard-page.component';
import { AssistantPageComponent } from './pages/assistant-page/assistant-page.component';
import { MissionReplayPageComponent } from './pages/mission-replay-page/mission-replay-page.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'overview' },
  { path: 'overview', component: DashboardPageComponent, data: { view: 'overview' } },
  { path: 'telemetry', component: DashboardPageComponent, data: { view: 'telemetry' } },
  { path: 'gauges', component: DashboardPageComponent, data: { view: 'gauges' } },
  { path: 'charts', component: DashboardPageComponent, data: { view: 'charts' } },
  { path: 'diagnostics', component: DashboardPageComponent, data: { view: 'diagnostics' } },
  { path: 'replay', component: MissionReplayPageComponent },
  { path: 'assistant', component: AssistantPageComponent },
  { path: '**', redirectTo: 'overview' },
];
