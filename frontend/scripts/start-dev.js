const { spawn } = require('node:child_process');
const path = require('node:path');

const appRoot = path.resolve(__dirname, '..');
const simulatorRoot = path.resolve(appRoot, '..', 'Simulated_engine');
const pythonCommand = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const enginePort = process.env.GCS_WS_PORT || '8001';
const children = [];

function startProcess(command, args, cwd, label) {
  const child = spawn(command, args, {
    cwd,
    env: process.env,
    stdio: 'inherit',
    windowsHide: false,
  });

  child.on('error', error => {
    console.error(`[${label}] Could not start: ${error.message}`);
    process.exitCode = 1;
  });

  child.on('exit', (code, signal) => {
    if (code !== 0 && signal === null) {
      console.error(`[${label}] Stopped with exit code ${code}`);
    }
  });

  children.push(child);
  return child;
}

console.log(`[startup] Starting engine telemetry server on ws://localhost:${enginePort}/ws/telemetry`);
startProcess(pythonCommand, ['-m', 'engine_simulator.ws_server', '--port', enginePort], simulatorRoot, 'engine');

console.log('[startup] Starting Angular development server on http://localhost:4200/');
startProcess(process.execPath, [path.join(appRoot, 'node_modules', '@angular', 'cli', 'bin', 'ng.js'), 'serve'], appRoot, 'angular');

function stopChildren() {
  for (const child of children) {
    if (!child.killed) {
      child.kill();
    }
  }
}

process.on('SIGINT', () => {
  stopChildren();
  process.exit(0);
});

process.on('SIGTERM', () => {
  stopChildren();
  process.exit(0);
});

process.on('exit', stopChildren);