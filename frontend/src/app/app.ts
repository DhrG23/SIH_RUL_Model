import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { TopStatusBarComponent } from './components/top-status-bar/top-status-bar.component';
import { EngineStateStore } from './core/services/engine-state.store';
import { ThemeService } from './core/services/theme.service';

export type DashboardView = 'overview' | 'telemetry' | 'gauges' | 'charts' | 'diagnostics' | 'replay' | 'assistant';

export interface ViewDef {
  id: DashboardView;
  label: string;
  route: string;
  icon: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    RouterLinkActive,
    RouterOutlet,
    TopStatusBarComponent,
  ],
  templateUrl: './app.html',
  styleUrls: ['./app.css'],
})
export class App {
  public readonly store = inject(EngineStateStore);
  public readonly themeService = inject(ThemeService);

  public readonly mobileNavOpen = signal<boolean>(false);

  public readonly views: ViewDef[] = [
    { id: 'overview', label: 'Mission overview', route: '/overview', icon: 'OV' },
    { id: 'telemetry', label: 'Live telemetry', route: '/telemetry', icon: 'TL' },
    { id: 'charts', label: 'Trends & signals', route: '/charts', icon: 'TR' },
    { id: 'diagnostics', label: 'Diagnostics & faults', route: '/diagnostics', icon: 'DG' },
    { id: 'replay', label: 'Mission replay', route: '/replay', icon: 'RP' },
    { id: 'assistant', label: 'AI maintenance assistant', route: '/assistant', icon: 'AI' },
  ];

  closeMobileNav(): void {
    this.mobileNavOpen.set(false);
  }

  toggleMobileNav(): void {
    this.mobileNavOpen.update((open) => !open);
  }
}
