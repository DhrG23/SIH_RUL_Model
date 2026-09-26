import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { EngineStateStore } from '../../core/services/engine-state.store';

@Component({
  selector: 'app-mission-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './mission-panel.component.html',
  styleUrls: ['./mission-panel.component.css'],
})
export class MissionPanelComponent {
  private readonly store = inject(EngineStateStore);

  public readonly mission = this.store.mission;

  public readonly elapsedFormatted = computed(() => {
    const ms = this.mission()?.mission_elapsed_ms ?? 0;
    const sec = Math.floor(ms / 1000);
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  });
}
