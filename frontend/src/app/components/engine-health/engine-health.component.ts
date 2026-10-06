import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EngineStateStore } from '../../core/services/engine-state.store';

@Component({
  selector: 'app-engine-health',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './engine-health.component.html',
  styleUrls: ['./engine-health.component.css'],
})
export class EngineHealthComponent {
  private readonly store = inject(EngineStateStore);

  public readonly health = this.store.health;

  public readonly overallScore = computed(() => {
    return this.health().overall_score;
  });

  public readonly healthState = computed<'NOMINAL' | 'DEGRADED' | 'CRITICAL'>(() => {
    const s = this.overallScore();
    if (s >= 80) return 'NOMINAL';
    if (s >= 55) return 'DEGRADED';
    return 'CRITICAL';
  });

  public readonly healthColor = computed(() => {
    const state = this.healthState();
    if (state === 'NOMINAL') return '#10b981';
    if (state === 'DEGRADED') return '#f59e0b';
    return '#f43f5e';
  });

  // Calculate SVG circular gauge stroke dash
  public readonly strokeDashoffset = computed(() => {
    const score = Math.max(0, Math.min(100, this.overallScore()));
    const circumference = 2 * Math.PI * 45; // r = 45
    return circumference - (score / 100) * circumference;
  });

  public getSubsystemClass(val: number): string {
    if (val >= 80) return 'sub-nominal';
    if (val >= 55) return 'sub-caution';
    return 'sub-critical';
  }
}
