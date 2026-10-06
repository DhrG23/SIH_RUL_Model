import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

interface ModelMetric {
  name: string;
  shortName: string;
  rmse: number;
  mae: number;
  r2: number;
}

@Component({
  selector: 'app-model-accuracy',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './model-accuracy.component.html',
  styleUrls: ['./model-accuracy.component.css'],
})
export class ModelAccuracyComponent {
  readonly models: ModelMetric[] = [
    { name: 'Linear Regression', shortName: 'Linear regression', rmse: 12.88, mae: 10.09, r2: 0.897 },
    { name: 'Support Vector Regression', shortName: 'Support vector regression', rmse: 13.11, mae: 8.59, r2: 0.894 },
    { name: 'Gradient Boosting', shortName: 'Gradient boosting', rmse: 14, mae: 8.73, r2: 0.879 },
    { name: 'Random Forest', shortName: 'Random forest', rmse: 14.05, mae: 7.41, r2: 0.878 },
    { name: 'Neural Network (MLP)', shortName: 'Neural network (MLP)', rmse: 20.22, mae: 12.77, r2: 0.747 },
  ];

  readonly maxError = 22;

  barWidth(value: number, max: number): string {
    return `${Math.min((value / max) * 100, 100)}%`;
  }
}