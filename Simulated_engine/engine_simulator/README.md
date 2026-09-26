# Physics-Driven Engine Digital-Twin Simulator

A high-fidelity, physics-driven Python time-series simulation engine built for aircraft/automotive internal combustion powertrain digital-twinning, progressive fault diagnosis, and ML telemetry generation.

---

## Core Principle

$$\text{YAML defines scenario} \longrightarrow \text{Python physics generates data} \longrightarrow \text{Sensor model creates observations} \longrightarrow \text{ML detects patterns}$$

- **No Hardcoded Dataset Generation**: All values are computed from physical/empirical differential equations and state updates rather than static tables or random noise envelopes.
- **Dynamic Variable Discovery**: Automatically scans existing project assets (`index.html`, `script.js`, `data/physics/`) to discover provided variables and units without assuming a fixed schema.
- **Continuous Stateful Dynamics**: Evolving state variables (Cylinder Head Temperature, Oil Temperature, Exhaust Gas Temperature, Engine RPM, and Lubrication Hydraulics) use continuous ODE integration so parameters evolve realistically without artificial jumps.
- **Latent Physical Faults**: Faults modify underlying physical parameters (e.g. fin fouling factor, pump volumetric wear, injector clogging, misfire ratio) rather than directly overriding sensor values.
- **Milliseconds Time Base**: All scenario durations, event timelines, fault progressions, and simulation loops operate strictly in **milliseconds** (`time_ms`).
- **Separation of Concerns**: Realistic sensor noise, drift, and quantization are added **strictly after** the true physical state has been computed.
- **ML Training Ready**: Dual synchronized output streams: `telemetry.csv` (noisy observed features) and `ground_truth.csv` (true states, active fault labels, latent severity curves, and component health indices).

---

## Timestep Simulation Loop

Each simulation step executes the formal sequence:

$$\text{time (ms)} \longrightarrow \text{read scenario} \longrightarrow \text{calculate physics} \longrightarrow \text{update state} \longrightarrow \text{derive variables} \longrightarrow \text{apply sensor model} \longrightarrow \text{store telemetry}$$

1. **`time`**: Advances $t_{ms} = \text{step} \times dt_{ms}$.
2. **`read scenario`**: Evaluates operating-condition changes, environmental shifts, and calculates progressive fault severities $S(t_{ms}) \in [0, 1]$ active at time $t_{ms}$.
3. **`calculate physics`**: Evaluates airflow, fueling, combustion efficiency, indicated and brake torque, oil hydraulics, heat dissipation, and vibration.
4. **`update engine state`**: Integrates continuous ODE states (CHT, Oil Temp, EGT, RPM) forward across $\Delta t = dt_{ms}/1000$ seconds.
5. **`calculate derived variables`**: Dynamic DAG resolution computes all mapped and derived variables (Brake Power, Vibration g-RMS, Viscosity, Mass Flows, Health Indices). Skips unavailable variables gracefully without breaking.
6. **`apply sensor model`**: Applies Gaussian measurement noise, slow drift, digital ADC quantization, and sensor clipping bounds to the true physical state.
7. **`store telemetry`**: Records dual streams (`telemetry.csv` and `ground_truth.csv`).

---

## Directory Structure

```
engine_simulator/
├── __init__.py               # Package entry points and exports
├── scanner.py                # Dynamic variable scanner and resilient DAG derivation engine
├── schema.py                 # Dataclasses & YAML schema specifications (times in milliseconds)
├── physics_engine.py         # ODE integration wrapper using data/physics equations
├── fault_manager.py          # Progressive fault loader, progression curves, latent parameter injection
├── sensor_model.py           # Post-physics realistic noise, drift, quantization, clipping
├── simulator.py              # Timestep loop orchestrator
├── exporter.py               # Synchronized CSV and JSON dataset generator for ML
├── cli.py                    # Command-line interface
├── faults/                   # Standalone progressive fault YAML files
│   ├── cooling_degradation.yaml
│   ├── lubrication_degradation.yaml
│   ├── oil_pump_degradation.yaml
│   ├── injector_degradation.yaml
│   └── combustion_abnormality.yaml
├── scenarios/                # Mission scenario YAML files
│   ├── nominal_cruise.yaml
│   ├── overheat_cooling_failure.yaml
│   ├── oil_loss_pump_failure.yaml
│   ├── injector_clog_misfire.yaml
│   └── multi_fault_stress_test.yaml
├── tests/                    # Automated verification test suite
│   ├── test_scanner.py
│   ├── test_faults.py
│   ├── test_physics_continuity.py
│   └── test_simulator_pipeline.py
└── README.md
```

---

## Progressive Fault YAML Definitions

Faults are defined in separate YAML files in `engine_simulator/faults/`:

| Fault File | Fault Type | Affected Latent Parameters | Physical Response |
|---|---|---|---|
| `cooling_degradation.yaml` | `cooling_degradation` | `radiator_blockage_ratio`, `fin_fouling_factor`, `coolant_leak_fraction` | Convective heat rejection decreases $\to$ CHT & Oil Temp rise continuously $\to$ Cooling health drops |
| `lubrication_degradation.yaml` | `lubrication_degradation` | `viscosity_loss_ratio`, `bearing_clearance_wear`, `boundary_scuffing_severity` | Viscosity drops, friction power rises $\to$ Oil heating increases, vibration increases |
| `oil_pump_degradation.yaml` | `oil_pump_degradation` | `pump_wear_factor`, `bearing_clearance_wear` | Pump volumetric delivery drops $\to$ Oil pressure collapses $\to$ Lubrication health drops |
| `injector_degradation.yaml` | `injector_degradation` | `clogging_fraction`, `leakage_fraction` | Fuel delivery restricted $\to$ Lean mixture (high AFR) $\to$ Power loss, injector health drops |
| `combustion_abnormality.yaml` | `combustion_abnormality` | `misfire_fraction`, `timing_retard_deg` | Combustion efficiency drops $\to$ Severe torque drop $\to$ Spike in vibration ($g_{\text{RMS}}$) |

---

## CLI Usage

### 1. Scan Discovered Project Variables
```powershell
python -m engine_simulator.cli scan
```

### 2. List Available Scenarios & Faults
```powershell
python -m engine_simulator.cli list-scenarios
python -m engine_simulator.cli list-faults
```

### 3. Run a Simulation Scenario
```powershell
python -m engine_simulator.cli run --scenario overheat_cooling_failure.yaml
python -m engine_simulator.cli run --scenario oil_loss_pump_failure.yaml
python -m engine_simulator.cli run --scenario injector_clog_misfire.yaml
python -m engine_simulator.cli run --scenario multi_fault_stress_test.yaml
```

Output files are stored in `output/`:
- `<scenario>_telemetry.csv`: Features for ML models (e.g. Random Forest, LSTM, Transformer, Autoencoders).
- `<scenario>_ground_truth.csv`: Targets and labels for classification, regression, and Remaining Useful Life (RUL) estimation.
- `<scenario>_metadata.json`: Run summary and column manifests.

---

## Running Automated Tests

Run the complete test suite:
```powershell
python -m unittest discover -s engine_simulator/tests -p "test_*.py" -v
```


