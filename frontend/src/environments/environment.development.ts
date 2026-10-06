export const environment = {
  production: false,
  wsUrl: 'wss://sih-rul-model.onrender.com/ws/telemetry',
  httpUrl: 'https://sih-rul-model.onrender.com/',
  staleThresholdMs: 2000,
  maxReconnectAttempts: 10,
  initialReconnectDelayMs: 1000,
  maxReconnectDelayMs: 10000,
  modelPath: '/models/engine.glb',
};
