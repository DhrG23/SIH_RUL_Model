# AeroTwin GCS

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 22.1.7.

## Development server

To start a local development server, run:

```bash
npm start
```

This starts both the Angular dashboard and the engine telemetry WebSocket server. Once they are running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

The telemetry server requires the Python dependencies in `../Simulated_engine/requirements.txt`. To run only the frontend, use `npm run start:frontend`.

### Gemini engine assistant

The **AI assistant** item in the left sidebar sends questions to Google's Gemini API through the telemetry server, along with the full conversation so far. It grounds every answer in the current health, diagnostics, mission state, and key telemetry values from the live snapshot — nothing else. The assistant cannot control the engine, run commands, or touch any file; it can only read the supplied telemetry and answer in natural language.

Before using it, export a Gemini API key (create one for free at [Google AI Studio](https://aistudio.google.com/apikey)) on the machine running the telemetry server:

```bash
export GEMINI_API_KEY="your-key-here"
# optional, defaults to gemini-3.5-flash:
export GEMINI_MODEL="gemini-3.5-flash"
```

If the key isn't set, the assistant page shows a clear "not configured" error instead of failing silently.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Vitest](https://vitest.dev/) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
