import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import fs from 'fs';
import path from 'path';

// Middleware plugin to stream large 3D GLB assets without Vite file-watcher memory buffering
const serve3DModelsPlugin = () => ({
  name: 'serve-3d-models-plugin',
  configureServer(server: any) {
    server.middlewares.use((req: any, res: any, next: any) => {
      if (req.url && (req.url.endsWith('.glb') || req.url.includes('/models/') || req.url.includes('/3D_models/'))) {
        const modelFile = path.resolve(__dirname, '3D_models', 'engine1.glb');
        if (fs.existsSync(modelFile)) {
          res.setHeader('Content-Type', 'model/gltf-binary');
          return fs.createReadStream(modelFile).pipe(res);
        }
      }
      next();
    });
  }
});

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), serve3DModelsPlugin()],
  build: {
    chunkSizeWarningLimit: 2000
  },
  server: {
    port: 3000,
    open: true,
    watch: {
      ignored: ['**/3D_models/**', '**/public/**']
    }
  }
});
