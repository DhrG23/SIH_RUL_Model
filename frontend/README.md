# AeroTwin GCS

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 22.1.7.

## Development server

To start a local development server, run:

```bash
npm start
```

This starts both the Angular dashboard and the engine telemetry WebSocket server. Once they are running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

The telemetry server requires the Python dependencies in `../Simulated_engine/requirements.txt`. To run only the frontend, use `npm run start:frontend`.

### Codex engine assistant

The **AI assistant** item in the left sidebar sends questions to the local Codex CLI through the telemetry server. It includes the current health, diagnostics, mission state, and key telemetry values with each question. Install and authenticate the Codex CLI on the machine running the telemetry server before using it; questions run in Codex's read-only sandbox.

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
