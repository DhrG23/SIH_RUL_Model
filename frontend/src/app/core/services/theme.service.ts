import { Injectable, signal, effect } from '@angular/core';

export type AppTheme = 'light' | 'dark';

@Injectable({
  providedIn: 'root',
})
export class ThemeService {
  public readonly theme = signal<AppTheme>('dark');

  constructor() {
    // Check saved theme or system preference
    if (typeof window !== 'undefined' && window.localStorage) {
      const saved = localStorage.getItem('gcs_theme') as AppTheme;
      if (saved === 'light' || saved === 'dark') {
        this.theme.set(saved);
      }
    }

    effect(() => {
      const current = this.theme();
      if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-theme', current);
      }
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem('gcs_theme', current);
      }
    });
  }

  public toggleTheme(): void {
    this.theme.update(t => (t === 'light' ? 'dark' : 'light'));
  }
}
