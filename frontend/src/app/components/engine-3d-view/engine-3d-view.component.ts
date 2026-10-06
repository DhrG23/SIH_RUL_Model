import {
  Component,
  ElementRef,
  ViewChild,
  OnInit,
  OnDestroy,
  inject,
  signal,
  computed,
  NgZone,
  effect,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EngineStateStore } from '../../core/services/engine-state.store';
import { ThemeService } from '../../core/services/theme.service';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-engine-3d-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './engine-3d-view.component.html',
  styleUrls: ['./engine-3d-view.component.css'],
})
export class Engine3dViewComponent implements OnInit, OnDestroy {
  private readonly store = inject(EngineStateStore);
  private readonly ngZone = inject(NgZone);
  public readonly themeService = inject(ThemeService);

  @ViewChild('webglContainer', { static: true })
  containerRef!: ElementRef<HTMLDivElement>;

  // UI state signals
  public readonly loadState = signal<'loading' | 'ready' | 'error'>('loading');
  public readonly errorMessage = signal<string>('');
  public readonly isWireframe = signal<boolean>(false);
  public readonly activeView = signal<'iso' | 'front' | 'top' | 'exhaust' | 'oil'>('iso');
  public readonly thermalOverlay = signal<boolean>(true);
  public readonly activeFaultCount = this.store.activeAlertCount;

  // Three.js instances
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private modelRoot: THREE.Group | null = null;
  private gridHelper!: THREE.GridHelper;

  // Tracked engine sub-nodes
  // Only the Propeller node is animated. The imported GLB keeps the hub and
  // engine assembly stationary, so the rotor cannot accidentally take its
  // parent/bar/shaft with it.
  private propellerNode: THREE.Object3D | null = null;
  private thermalHeads: THREE.Mesh[] = [];
  private exhaustParts: THREE.Mesh[] = [];
  private oilParts: THREE.Mesh[] = [];
  private componentPlaceholders: Map<string, THREE.Mesh[]> = new Map();

  // Animation & sizing
  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private lastTime = performance.now();
  private originalMaterials: Map<string, THREE.Material | THREE.Material[]> = new Map();

  // Idle "showcase" auto-rotate: on by default, pauses the moment the operator
  // grabs the model and quietly resumes a couple of seconds after they let go.
  private autoRotateResumeTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly AUTO_ROTATE_RESUME_DELAY_MS = 2200;

  constructor() {
    // React to theme changes smoothly
    effect(() => {
      const theme = this.themeService.theme();
      if (this.scene) {
        const isDark = theme === 'dark';
        this.scene.background = new THREE.Color(isDark ? 0x0d1117 : 0x2f2f2f);
        if (this.gridHelper) {
          this.scene.remove(this.gridHelper);
          this.gridHelper = new THREE.GridHelper(
            12,
            24,
            isDark ? 0x30363d : 0x414141,
            isDark ? 0x1f242c : 0x383838
          );
          this.gridHelper.position.y = -1.5;
          this.scene.add(this.gridHelper);
        }
      }
    });
  }

  ngOnInit(): void {
    this.ngZone.runOutsideAngular(() => {
      this.initThree();
    });
    this.setupResize();
  }

  ngOnDestroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.autoRotateResumeTimer) {
      clearTimeout(this.autoRotateResumeTimer);
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    this.controls?.dispose();
    if (this.modelRoot) {
      this.disposeObject(this.modelRoot);
    }
    this.disposeObject(this.scene);
    if (this.renderer) {
      this.renderer.dispose();
    }
  }

  public retryLoad(): void {
    if (!this.renderer || !this.scene) return;
    this.loadState.set('loading');
    this.errorMessage.set('');
    if (this.modelRoot) {
      this.scene.remove(this.modelRoot);
      this.disposeObject(this.modelRoot);
      this.modelRoot = null;
    }
    this.loadModel();
  }

  private disposeObject(object: THREE.Object3D): void {
    object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      child.geometry.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        material.dispose();
      }
    });
  }

  private initThree(): void {
    const container = this.containerRef.nativeElement;
    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;
    const isDark = this.themeService.theme() === 'dark';

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(isDark ? 0x0d1117 : 0x2f2f2f);

    // Subtle natural floor grid
    this.gridHelper = new THREE.GridHelper(
      12,
      24,
      isDark ? 0x30363d : 0x414141,
      isDark ? 0x1f242c : 0x383838
    );
    this.gridHelper.position.y = -1.5;
    this.scene.add(this.gridHelper);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(4.5, 3.2, 5.0);

    // 3. Balanced Natural Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight1.position.set(6, 12, 8);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xe2e8f0, 0.9);
    dirLight2.position.set(-6, 5, -6);
    this.scene.add(dirLight2);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    fillLight.position.set(0, -4, 6);
    this.scene.add(fillLight);

    // Soft key-light shadow, grounding the model on the presentation floor --
    // subtle, not a hard studio shadow, so it reads as "premium product shot"
    // rather than a harsh spotlight.
    dirLight1.castShadow = true;
    dirLight1.shadow.mapSize.set(1024, 1024);
    dirLight1.shadow.camera.near = 1;
    dirLight1.shadow.camera.far = 30;
    dirLight1.shadow.camera.left = -8;
    dirLight1.shadow.camera.right = 8;
    dirLight1.shadow.camera.top = 8;
    dirLight1.shadow.camera.bottom = -8;
    dirLight1.shadow.bias = -0.0005;
    dirLight1.shadow.radius = 4;

    // 4. WebGL Renderer
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
      this.renderer.setSize(width, height);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.05;
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      // Procedural studio-room environment map -- no external HDR asset needed --
      // gives metal/glass surfaces real reflections instead of flat shading,
      // which is most of what separates a "toy" 3D viewer from a polished one.
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      pmrem.dispose();

      container.appendChild(this.renderer.domElement);

      // 5. Controls
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.minDistance = 1.5;
      this.controls.maxDistance = 18;
      this.controls.target.set(0, 0, 0);

      // Gentle idle showcase spin -- pauses instantly on user interaction and
      // eases back in shortly after release, like a modern product page.
      this.controls.autoRotate = true;
      this.controls.autoRotateSpeed = 0.6;
      this.controls.addEventListener('start', () => {
        this.controls.autoRotate = false;
        if (this.autoRotateResumeTimer) clearTimeout(this.autoRotateResumeTimer);
      });
      this.controls.addEventListener('end', () => {
        if (this.autoRotateResumeTimer) clearTimeout(this.autoRotateResumeTimer);
        this.autoRotateResumeTimer = setTimeout(() => {
          this.controls.autoRotate = true;
        }, this.AUTO_ROTATE_RESUME_DELAY_MS);
      });

      this.startAnimationLoop();
      this.loadModel();
    } catch (e: any) {
      this.ngZone.run(() => {
        console.warn('[3D Twin] WebGL renderer not available:', e);
        this.loadState.set('error');
        this.errorMessage.set('WebGL rendering context unavailable. Telemetry monitoring remains fully active.');
      });
    }
  }

  private loadModel(): void {
    const loader = new GLTFLoader();
    const modelUrl = environment.modelPath;

    loader.load(
      modelUrl,
      (gltf: any) => {
        this.ngZone.run(() => {
          this.modelRoot = gltf.scene;
          this.processModelHierarchy(gltf.scene);

          // Center model
          const box = new THREE.Box3().setFromObject(gltf.scene);
          const center = box.getCenter(new THREE.Vector3());
          gltf.scene.position.sub(center);

          this.scene.add(gltf.scene);
          this.loadState.set('ready');
          console.info('[3D Twin] Model loaded successfully with natural studio shading.');
        });
      },
      undefined,
      (err: any) => {
        this.ngZone.run(() => {
          console.error('[3D Twin] Failed to load 3D GLB model:', err);
          this.loadState.set('error');
          this.errorMessage.set('Engine 3D model could not be loaded. Telemetry monitoring remains active.');
        });
      }
    );
  }

  private processModelHierarchy(root: THREE.Object3D): void {
    this.propellerNode = null;
    this.thermalHeads = [];
    this.exhaustParts = [];
    this.oilParts = [];
    this.componentPlaceholders.clear();

    root.traverse((node: THREE.Object3D) => {
      const name = node.name.toLowerCase();

      if (name.includes('placeholder')) {
        node.visible = false;
      }

      // The GLB has an explicit Propeller parent containing both blades.
      // Its local Y axis is aligned with the propeller shaft. Keep all
      // parent groups and the hub fixed; only this node is rotated.
      if (name === 'propeller') {
        this.propellerNode = node;
      }

      if (node instanceof THREE.Mesh) {
        // Cheap, real depth cue: everything casts, the presentation floor
        // (below) receives -- this is what makes the model feel like it's
        // actually sitting on a surface instead of floating in a render.
        node.castShadow = true;
        node.receiveShadow = false;

        if (node.material) {
          this.originalMaterials.set(node.uuid, node.material);
          if (Array.isArray(node.material)) {
            node.material = node.material.map((m: THREE.Material) => m.clone());
          } else {
            node.material = node.material.clone();
          }

          if (name === 'presentationfloor') {
            node.castShadow = false;
            node.receiveShadow = true;
            const floorMaterials = Array.isArray(node.material) ? node.material : [node.material];
            for (const material of floorMaterials) {
              const floorMaterial = material as THREE.MeshStandardMaterial;
              floorMaterial.color?.setHex(0xdfe4e8);
              floorMaterial.roughness = 0.9;
              floorMaterial.metalness = 0;
            }
          }
        }

        if (name.includes('cylinderhead') || name.includes('headbody') || name.includes('coolingfin')) {
          this.thermalHeads.push(node);
        }

        if (name.includes('exhaust')) {
          this.exhaustParts.push(node);
        }

        if (name.includes('oil')) {
          this.oilParts.push(node);
        }

        const extras = (node as any).userData || {};
        const compTag = extras.telemetryComponent;
        if (compTag) {
          if (!this.componentPlaceholders.has(compTag)) {
            this.componentPlaceholders.set(compTag, []);
          }
          this.componentPlaceholders.get(compTag)!.push(node);
        }
      }
    });
  }

  private startAnimationLoop(): void {
    const animate = (time: number) => {
      const dt = Math.min(0.1, (time - this.lastTime) / 1000.0);
      this.lastTime = time;

      this.updateEngineVisuals(dt, time);

      this.controls.update();
      this.renderer.render(this.scene, this.camera);

      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  private updateEngineVisuals(dt: number, timeMs: number): void {
    if (!this.modelRoot) return;

    const env = this.store.latestEnvelope();
    const vars = env?.variables ?? {};
    const diagnostics = this.store.diagnostics();

    const rpm = Number(vars['engine_rpm'] ?? 0);
    if (rpm > 1 && this.propellerNode) {
      // Visual time-scale keeps a 2,000–3,500 RPM propeller readable while
      // preserving the real rotation direction and shaft axis.
      const visualTimeScale = 0.06;
      const angleDelta = (rpm / 60.0) * (Math.PI * 2) * dt * visualTimeScale;
      this.propellerNode.rotateY(angleDelta);
    }

    // Vibration is represented by telemetry and diagnostics rather than
    // translating the entire model. This keeps the physical geometry fixed.

    // Realistic Thermal Emissive (CHT)
    const cht = Number(vars['cht'] ?? 90.0);
    const chtNorm = Math.min(1.0, Math.max(0.0, (cht - 100.0) / 130.0));
    const chtEmissiveColor = new THREE.Color().setHSL(0.06, 0.85, chtNorm * 0.4);

    for (const mesh of this.thermalHeads) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat && mat.emissive) {
        mat.emissive.copy(this.thermalOverlay() ? chtEmissiveColor : new THREE.Color(0x000000));
        mat.emissiveIntensity = this.thermalOverlay() ? chtNorm * 0.65 : 0;
      }
    }

    // 4. Exhaust Gas Temperature (EGT)
    const egt = Number(vars['egt'] ?? 600.0);
    const egtNorm = Math.min(1.0, Math.max(0.0, (egt - 650.0) / 350.0));
    const egtEmissiveColor = new THREE.Color().setHSL(0.07, 0.9, egtNorm * 0.5);

    for (const mesh of this.exhaustParts) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat && mat.emissive) {
        mat.emissive.copy(this.thermalOverlay() ? egtEmissiveColor : new THREE.Color(0x000000));
        mat.emissiveIntensity = this.thermalOverlay() ? egtNorm * 0.6 : 0;
      }
    }

    // 5. Active Diagnostic Fault Indicator (only reveal focused marker)
    for (const targets of this.componentPlaceholders.values()) {
      for (const mesh of targets) {
        mesh.visible = false;
        const mat = mesh.material as THREE.MeshStandardMaterial;
        if (mat?.emissive) {
          mat.emissive.setHex(0x000000);
          mat.emissiveIntensity = 0;
        }
      }
    }

    const activeFaults = diagnostics.filter(d => d.status === 'ACTIVE');
    if (activeFaults.length > 0) {
      const pulse = (Math.sin(timeMs * 0.005) + 1.0) * 0.5;
      for (const fault of activeFaults) {
        let tag = '';
        if (fault.fault.includes('cooling') || fault.fault.includes('radiator')) tag = 'radiator';
        else if (fault.fault.includes('oil') || fault.fault.includes('lubrication')) tag = 'oil_pump';
        else if (fault.fault.includes('injector')) tag = 'spark_plug';
        else if (fault.fault.includes('combustion')) tag = 'bearing';

        if (tag && this.componentPlaceholders.has(tag)) {
          const targets = this.componentPlaceholders.get(tag)!;
          for (const m of targets) {
            m.visible = true;
            const mat = m.material as THREE.MeshStandardMaterial;
            if (mat?.emissive) {
              mat.emissive.setHex(0xdc2626);
              mat.emissiveIntensity = 0.3 + pulse * 0.9;
            }
          }
        }
      }
    }
  }

  public getLiveValue(key: string): string {
    const value = this.store.latestEnvelope()?.variables?.[key];
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    if (key === 'engine_rpm') return Math.round(Number(value)).toLocaleString();
    if (key === 'cht') return Number(value).toFixed(0);
    if (key === 'egt') return Number(value).toFixed(0);
    if (key === 'oil_pressure') return Number(value).toFixed(1);
    return Number(value).toFixed(1);
  }

  // Camera Presets
  public setViewPreset(preset: 'iso' | 'front' | 'top' | 'exhaust' | 'oil'): void {
    this.activeView.set(preset);
    if (!this.controls) return;

    switch (preset) {
      case 'iso':
        this.camera.position.set(4.5, 3.2, 5.0);
        this.controls.target.set(0, 0, 0);
        break;
      case 'front':
        this.camera.position.set(0, 0.5, 6.0);
        this.controls.target.set(0, 0, 0);
        break;
      case 'top':
        this.camera.position.set(0, 7.0, 0.1);
        this.controls.target.set(0, 0, 0);
        break;
      case 'exhaust':
        this.camera.position.set(-5.0, 1.2, -2.5);
        this.controls.target.set(0, 0, 0);
        break;
      case 'oil':
        this.camera.position.set(2.5, -2.0, 3.5);
        this.controls.target.set(0, -0.5, 0);
        break;
    }
    this.controls.update();
  }

  public toggleWireframe(): void {
    const next = !this.isWireframe();
    this.isWireframe.set(next);

    if (this.modelRoot) {
      this.modelRoot.traverse((node: THREE.Object3D) => {
        if (node instanceof THREE.Mesh && node.material) {
          if (Array.isArray(node.material)) {
            node.material.forEach((m: THREE.Material) => ((m as any).wireframe = next));
          } else {
            (node.material as any).wireframe = next;
          }
        }
      });
    }
  }

  public resetCamera(): void {
    this.setViewPreset('iso');
  }

  private setupResize(): void {
    if (typeof ResizeObserver === 'undefined') return;
    const container = this.containerRef.nativeElement;
    this.resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w > 0 && h > 0 && this.camera && this.renderer) {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
      }
    });
    this.resizeObserver.observe(container);
  }
}
