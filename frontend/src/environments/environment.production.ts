// Used automatically by `ng build` (production is the default configuration --
// see angular.json's fileReplacements). This is what a Vercel/Netlify-deployed
// build actually ships to every visitor's browser, so it CANNOT point at
// localhost -- that would only ever work on your own machine.
//
// Before deploying: replace the two placeholder values below with your real
// Render backend URL. Use wss:// (not ws://) and https:// (not http://) --
// a page served over https can't open an insecure ws:// connection, browsers
// block it as mixed content.
export const environment = {
  production: true,
  wsUrl: 'wss://sih-rul-model.onrender.com//ws/telemetry',
  httpUrl: 'https://sih-rul-model.onrender.com/',
  staleThresholdMs: 2000,
  maxReconnectAttempts: 10,
  initialReconnectDelayMs: 1000,
  maxReconnectDelayMs: 10000,
  modelPath: '/models/engine1.glb',
};
