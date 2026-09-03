"use client";

import React, { Suspense, useState, useMemo, Component, ReactNode } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import { TelemetryCallout, TelemetryState } from './mock-data';
import { Box, RefreshCw } from 'lucide-react';

interface EngineSchematicProps {
  telemetry: TelemetryState;
  hotspots?: TelemetryCallout[];
  modelPath?: string;
}

// 3D Callout Pin Data
interface PinCalloutData {
  id: string;
  locationLabel: string;
  name: string;
  vibration: string;
  temperature: string;
  tempVal: number;
  status: 'normal' | 'warning' | 'critical';
  position: [number, number, number];
}

const DEFAULT_CALLOUTS: PinCalloutData[] = [
  {
    id: 'cyl-1',
    locationLabel: 'DE - Cyl 1',
    name: 'Cylinder 1',
    vibration: '0.32 mA/g',
    temperature: '85 °C',
    tempVal: 85,
    status: 'warning',
    position: [-0.9, 0.5, 0.4]
  },
  {
    id: 'cyl-3-alert',
    locationLabel: 'DE - Cyl 2 (Alert)',
    name: 'Cylinder 3 Alert',
    vibration: '0.16 mA/g',
    temperature: '98 °C',
    tempVal: 98,
    status: 'warning',
    position: [-0.2, 0.75, -0.4]
  },
  {
    id: 'crankcase',
    locationLabel: 'Crankcase Block',
    name: 'Crankcase',
    vibration: '0.21 mA/g',
    temperature: '76 °C',
    tempVal: 76,
    status: 'normal',
    position: [0.1, -0.3, 0.5]
  },
  {
    id: 'fuel-rail',
    locationLabel: 'NDE - Fuel Rail',
    name: 'Fuel Rail',
    vibration: '0.24 mA/g',
    temperature: '70 °C',
    tempVal: 70,
    status: 'normal',
    position: [0.8, 0.2, -0.5]
  }
];

// Fallback procedural engine mesh
const ProceduralEngineMesh: React.FC<{ isWireframe: boolean }> = ({ isWireframe }) => {
  const mat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#f2f2f2',
        metalness: 0.8,
        roughness: 0.3,
        wireframe: isWireframe
      }),
    [isWireframe]
  );

  return (
    <group>
      {/* Main Crankcase */}
      <mesh material={mat} position={[0, 0, 0]}>
        <boxGeometry args={[1.8, 0.8, 1.0]} />
      </mesh>
      {/* Cylinders */}
      {[-0.6, -0.2, 0.2, 0.6].map((x, i) => (
        <mesh key={i} material={mat} position={[x, 0.6, i % 2 === 0 ? 0.25 : -0.25]}>
          <cylinderGeometry args={[0.2, 0.2, 0.7, 16]} />
        </mesh>
      ))}
      {/* Turbo housing */}
      <mesh material={mat} position={[1.0, 0.1, 0]} rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[0.3, 0.1, 16, 32]} />
      </mesh>
    </group>
  );
};

// React Error Boundary for 3D GLTF Loader
interface ErrorBoundaryProps {
  fallback: ReactNode;
  children: ReactNode;
}
interface ErrorBoundaryState {
  hasError: boolean;
}

class ModelErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.warn('GLTF Model load error, rendering fallback procedural mesh:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

// Inner GLTF Model Loader (Hooks called strictly at top-level)
const GLTFModelContent: React.FC<{ modelPath: string; isWireframe: boolean }> = ({
  modelPath,
  isWireframe
}) => {
  const gltf = useGLTF(modelPath);

  const clonedScene = useMemo(() => {
    if (!gltf || !gltf.scene) return null;
    const clone = gltf.scene.clone(true);

    // Center and scale model
    const box = new THREE.Box3().setFromObject(clone);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = 2.5 / maxDim;

    clone.scale.setScalar(scale);
    clone.position.sub(center.multiplyScalar(scale));

    // Apply material properties
    clone.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        mesh.material = new THREE.MeshStandardMaterial({
          color: '#f2f2f2',
          metalness: 0.75,
          roughness: 0.35,
          wireframe: isWireframe
        });
      }
    });

    return clone;
  }, [gltf, isWireframe]);

  if (!clonedScene) {
    return <ProceduralEngineMesh isWireframe={isWireframe} />;
  }

  return <primitive object={clonedScene} />;
};

// Main Engine Model wrapper with ErrorBoundary
const EngineModel: React.FC<{ modelPath: string; isWireframe: boolean }> = ({
  modelPath,
  isWireframe
}) => {
  return (
    <ModelErrorBoundary fallback={<ProceduralEngineMesh isWireframe={isWireframe} />}>
      <GLTFModelContent modelPath={modelPath} isWireframe={isWireframe} />
    </ModelErrorBoundary>
  );
};

// Subtle dark loading spinner for Suspense fallback
const CanvasLoader: React.FC = () => {
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center bg-[#242424]/90 border border-[#454545] px-4 py-3 rounded shadow-2xl backdrop-blur select-none">
        <RefreshCw className="w-5 h-5 text-gray-300 animate-spin mb-2" />
        <span className="text-xs font-mono text-gray-300 tracking-wide">
          Loading 3D Engine Model...
        </span>
      </div>
    </Html>
  );
};

export const EngineSchematic: React.FC<EngineSchematicProps> = ({
  telemetry,
  modelPath = '/3D_models/engine1.glb'
}) => {
  const [isWireframe, setIsWireframe] = useState(false);

  return (
    <div className="bg-[#303030] border border-[#454545] rounded p-4 flex flex-col justify-between h-full relative select-none">
      {/* Card Header & Controls */}
      <div className="flex items-center justify-between mb-3 border-b border-[#3a3a3a] pb-2 z-20">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-gray-200 tracking-wide">
            3D Digital Twin Viewport
          </h2>
          <span className="text-[10px] text-gray-400 font-mono">
            DRDO MALE UAV Aero Piston Engine
          </span>
        </div>

        {/* Wireframe Shader Toggle */}
        <button
          onClick={() => setIsWireframe(!isWireframe)}
          className={`px-2.5 py-1 text-xs font-mono rounded border flex items-center gap-1.5 transition-colors ${
            isWireframe
              ? 'bg-[#383838] border-[#5a5a5a] text-gray-200 shadow'
              : 'bg-[#292929] border-[#454545] text-gray-400 hover:text-white'
          }`}
          title="Toggle Wireframe Shader"
        >
          <Box className="w-3.5 h-3.5" />
          <span>{isWireframe ? 'Wireframe ON' : 'Wireframe OFF'}</span>
        </button>
      </div>

      {/* Main Viewport Container */}
      <div className="relative flex-1 w-full h-full min-h-[400px] bg-[#2b2b2b] rounded border border-[#3d3d3d] overflow-hidden flex items-center justify-center">
        {/* R3F WebGL Canvas Container */}
        <Canvas
          className="w-full h-full relative min-h-[400px]"
          camera={{ position: [3.2, 2.0, 3.8], fov: 45 }}
          gl={{ alpha: true, antialias: true }}
          style={{ background: 'transparent' }}
        >
          {/* Studio Lighting */}
          <ambientLight intensity={0.7} />
          <directionalLight position={[10, 10, 5]} intensity={1.2} />
          <pointLight position={[-10, -10, -5]} intensity={0.5} />

          {/* Interactive OrbitControls */}
          <OrbitControls
            enableZoom={true}
            enablePan={false}
            maxPolarAngle={Math.PI / 2}
            minDistance={2.8}
            maxDistance={8}
          />

          {/* Suspense boundary for 3D GLTF Model */}
          <Suspense fallback={<CanvasLoader />}>
            <EngineModel modelPath={modelPath} isWireframe={isWireframe} />
          </Suspense>
        </Canvas>

        {/* Right-Anchored Floating Readout Sidebar Overlay */}
        <div className="absolute inset-0 pointer-events-none z-10 flex flex-col justify-between p-3">
          <div className="flex justify-end">
            <div className="w-44 bg-[#262626]/95 backdrop-blur border border-[#454545] rounded p-2.5 space-y-2 text-xs font-mono shadow-2xl pointer-events-auto">
              <div className="text-[11px] font-sans font-bold text-gray-300 border-b border-[#2d3440] pb-1 uppercase tracking-wider">
                Motor Readouts
              </div>

              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-gray-400">Speed</span>
                  <span className="text-white font-bold">{telemetry.speedRpm} rpm</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Throttle</span>
                  <span className="text-gray-100 font-bold">{telemetry.throttle} %</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Manifold Press</span>
                  <span className="text-white font-bold">
                    {telemetry.manifoldPressure} inHg
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Oil Temp</span>
                  <span className="text-white font-bold">{telemetry.oilTemp} °C</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Oil Press</span>
                  <span className="text-white font-bold">{telemetry.oilPressure} psi</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Boost Press</span>
                  <span className="text-white font-bold">{telemetry.boostPressure} bar</span>
                </div>
                <div className="flex justify-between border-t border-[#2d3440] pt-1">
                  <span className="text-gray-400">Cyl Temp</span>
                  <span className="text-gray-100 font-bold">
                    {telemetry.cylinderHeadTemp} °C
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">EGT</span>
                  <span className="text-white font-bold">{telemetry.egt} °C</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-between items-end pointer-events-auto">
            <div className="bg-[#292929]/90 border border-[#454545] rounded px-3 py-1.5 text-xs font-mono shadow-lg">
              <span className="text-gray-400 mr-2">Inlet Pressure:</span>
              <span className="text-gray-100 font-bold">13.22 psi (91.0 kPa)</span>
            </div>
            <div className="text-[10px] font-mono text-gray-400 bg-[#1b1f26]/80 px-2 py-1 rounded border border-[#2b313c]">
              Rotate: Left Drag | Pan: Right Drag | Zoom: Scroll
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
