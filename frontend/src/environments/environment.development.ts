export const environment = {
  production: false,
  wsUrl: 'ws://localhost:8001/ws/telemetry',
  httpUrl: 'http://localhost:8001',
  staleThresholdMs: 2000,
  maxReconnectAttempts: 10,
  initialReconnectDelayMs: 1000,
  maxReconnectDelayMs: 10000,
  modelPath: '/models/engine.glb',
};
