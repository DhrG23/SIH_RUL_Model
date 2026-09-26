import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SparklineComponent } from './sparkline.component';

describe('SparklineComponent', () => {
  let component: SparklineComponent;
  let fixture: ComponentFixture<SparklineComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SparklineComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(SparklineComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should compute SVG path data when points are provided', () => {
    component.data = [10, 20, 15, 30, 25];
    component.width = 60;
    component.height = 20;
    fixture.detectChanges();

    const paths = component.pathData();
    expect(paths.line).toContain('M ');
    expect(paths.area).toContain('Z');
    expect(paths.lastX).toBe(60);
  });

  it('should return empty paths when data is insufficient', () => {
    component.data = [10];
    fixture.detectChanges();

    const paths = component.pathData();
    expect(paths.line).toBe('');
    expect(paths.area).toBe('');
  });
});
