import { Injectable, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface ScenarioSummary {
  name: string;
  label: string;
  fault_files: string[];
  mode: 'Nominal' | 'Fault demonstration';
}

interface ScenarioListResponse {
  active_scenario: string;
  available_scenarios: ScenarioSummary[];
}

interface ScenarioSelectionResponse {
  active_scenario: string;
}

@Injectable({ providedIn: 'root' })
export class ScenarioControlService {
  private readonly http = inject(HttpClient);

  public readonly scenarios = signal<ScenarioSummary[]>([]);
  public readonly activeScenario = signal('');
  public readonly isLoading = signal(false);
  public readonly isSwitching = signal(false);
  public readonly error = signal('');

  async load(): Promise<void> {
    this.isLoading.set(true);
    this.error.set('');
    try {
      const response = await firstValueFrom(
        this.http.get<ScenarioListResponse>(`${environment.httpUrl}/scenarios`),
      );
      this.scenarios.set(response.available_scenarios);
      this.activeScenario.set(response.active_scenario);
    } catch {
      this.error.set('Scenario controls are unavailable while the telemetry server is offline.');
    } finally {
      this.isLoading.set(false);
    }
  }

  async select(name: string): Promise<void> {
    if (!name || name === this.activeScenario() || this.isSwitching()) return;

    this.isSwitching.set(true);
    this.error.set('');
    try {
      const response = await firstValueFrom(
        this.http.post<ScenarioSelectionResponse>(`${environment.httpUrl}/scenario/select`, { name }),
      );
      this.activeScenario.set(response.active_scenario);
    } catch {
      this.error.set('Could not switch the live scenario.');
    } finally {
      this.isSwitching.set(false);
    }
  }
}
