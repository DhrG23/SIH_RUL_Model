import { Component, computed, inject, signal, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { EngineStateStore } from '../../core/services/engine-state.store';
import { TelemetryCardComponent } from '../telemetry-card/telemetry-card.component';

@Component({
  selector: 'app-telemetry-grid',
  standalone: true,
  imports: [CommonModule, TelemetryCardComponent],
  templateUrl: './telemetry-grid.component.html',
  styleUrls: ['./telemetry-grid.component.css'],
})
export class TelemetryGridComponent {
  private readonly store = inject(EngineStateStore);
  private readonly router = inject(Router);

  @Output() parameterSelected = new EventEmitter<string>();

  public readonly activeCategory = signal<string>('all');
  public readonly searchQuery = signal<string>('');
  public readonly selectedKey = this.store.selectedParameterKey;

  public readonly categories = [
    { id: 'all', label: 'All Telemetry' },
    { id: 'powertrain', label: 'Powertrain' },
    { id: 'thermal', label: 'Thermal' },
    { id: 'lubrication', label: 'Lubrication' },
    { id: 'combustion', label: 'Combustion' },
    { id: 'electrical', label: 'Electrical' },
  ];

  public readonly filteredParameters = computed(() => {
    const cat = this.activeCategory();
    const query = this.searchQuery().trim().toLowerCase();
    const all = this.store.processedParameters();
    return all.filter(p => {
      const matchesCategory = cat === 'all' || p.category === cat;
      const matchesSearch = !query || p.label.toLowerCase().includes(query) || p.key.toLowerCase().includes(query);
      return matchesCategory && matchesSearch;
    });
  });

  public readonly selectedParameter = computed(() =>
    this.store.processedParameters().find(param => param.key === this.selectedKey()) ?? null
  );

  setCategory(cat: string): void {
    this.activeCategory.set(cat);
    const firstParameter = this.filteredParameters()[0];
    if (firstParameter) this.store.selectedParameterKey.set(firstParameter.key);
  }

  onCardSelect(key: string): void {
    this.store.selectedParameterKey.set(key);
    this.parameterSelected.emit(key);
  }

  onParameterSelect(key: string): void {
    this.onCardSelect(key);
  }

  searchTelemetry(): void {
    const match = this.filteredParameters()[0];
    if (!match) return;
    this.store.selectedParameterKey.set(match.key);
    this.router.navigate(['/charts'], { queryParams: { parameter: match.key } });
  }
}
