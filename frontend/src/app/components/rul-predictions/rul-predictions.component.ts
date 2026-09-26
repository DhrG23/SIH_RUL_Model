import { CommonModule } from '@angular/common';
import { Component, computed, inject } from '@angular/core';
import { EngineStateStore } from '../../core/services/engine-state.store';

interface RulModel {
  name: string;
  shortName: string;
  color: string;
}

@Component({
  selector: 'app-rul-predictions',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './rul-predictions.component.html',
  styleUrls: ['./rul-predictions.component.css'],
})
export class RulPredictionsComponent {
  readonly store = inject(EngineStateStore);
  readonly minimumSafeRulHours = 24;

  readonly models: RulModel[] = [
    { name: 'Linear Regression', shortName: 'Linear regression', color: '#2563eb' },
    { name: 'Gradient Boosting', shortName: 'Gradient boosting', color: '#d97706' },
    { name: 'Neural Network (MLP)', shortName: 'Neural network', color: '#7c3aed' },
    { name: 'Random Forest', shortName: 'Random forest', color: '#059669' },
    { name: 'Support Vector Regression', shortName: 'Support vector regression', color: '#e11d48' },
  ];

  value(modelName: string): number | null {
    const value = this.store.rulPredictions()[modelName];
    return typeof value === 'number' ? value : null;
  }

  barWidth(modelName: string): string {
    const value = this.value(modelName);
    return `${Math.min(100, Math.max(0, ((value ?? 0) / 130) * 100))}%`;
  }

  readonly minimumRul = computed(() => {
    const values = Object.values(this.store.rulPredictions());
    return values.length > 0 ? Math.min(...values) : null;
  });

  readonly healthStatus = computed<'healthy' | 'attention' | 'critical'>(() => {
    const diagnostics = this.store.diagnostics();
    const minimumRul = this.minimumRul();
    if (diagnostics.some((diagnostic) => diagnostic.status === 'ACTIVE' && diagnostic.severity >= 0.7)) {
      return 'critical';
    }
    if (diagnostics.length > 0 || (minimumRul !== null && minimumRul < this.minimumSafeRulHours)) {
      return 'attention';
    }
    return 'healthy';
  });

  readonly diagnosticText = computed(() => {
    const diagnostics = this.store.diagnostics();
    const diagnostic = diagnostics.find((item) => item.status === 'ACTIVE') ?? diagnostics[0];
    if (!diagnostic) {
      return 'Engine is operating well within safe limits. No maintenance action needed yet.';
    }

    const fault = diagnostic.fault.replaceAll('_', ' ');
    const evidence = diagnostic.evidence?.[0];
    return evidence ? `${fault}: ${evidence}` : `${fault} requires attention.`;
  });
}
