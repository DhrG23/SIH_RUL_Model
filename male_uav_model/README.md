# MALE UAV — Physics, Thermodynamics & Environment Model

A simulation package for a Medium-Altitude Long-Endurance (MALE) UAV,
covering flight dynamics, engine thermodynamics, weather/turbulence
effects, propeller performance, engine sensor data, and a live/streaming
data pipeline that ties it all together. Driven by **your own aircraft and
engine data**, not hardcoded values.

## Package contents

### Core models
| File                    | Purpose |
|--------------------------|---------|
| `physics_model.py`       | ISA atmosphere, aerodynamic drag-polar model, 3-DOF point-mass flight dynamics (climb/cruise), RK4 integrator. |
| `thermo_model.py`        | Engine thermodynamics: Brayton-cycle turboprop/turboshaft model (compressor → combustor → turbine) and a piston-engine altitude power-lapse model. |
| `dof6_model.py`          | **Full 6-DOF rigid-body dynamics** — roll/pitch/yaw moments from aileron/elevator/rudder deflections, stability derivatives, 12-state nonlinear equations of motion. |
| `propeller_model.py`     | **Propeller performance map** — thrust/power coefficients vs. advance ratio (replaces the constant-efficiency assumption), solves for the power-matched operating point. |
| `loiter_model.py`        | **Loiter/station-keeping endurance simulation** — integrates fuel burn second-by-second around a banked holding pattern until fuel is actually exhausted (not a single formula estimate), including weather and wind drift. |
| `atmosphere_extended.py` | **Weather effects (hot/cold day)** via ISA+ΔT and density-altitude; **turbulence/gusts** via a simplified Dryden-type gust model; steady-wind model. |
| `sensor_model.py`        | **Engine sensor simulation** — generates realistic noisy RPM/EGT/fuel-flow/torque/oil-temp readings from the true thermodynamic state, plus sensor-only power estimation methods (torque-based, fuel-flow-based) to cross-check against physics truth. |
| `live_simulation.py`     | **Live/streaming pipeline** — ties every module together; processes one input at a time (interactive prompt or a streamed CSV) and emits full computed output immediately for each one, rather than batch-computing everything up front. |

### Reporting & data
| File                          | Purpose |
|--------------------------------|---------|
| `performance_report.py`        | Computes headline performance numbers (stall speed, best-range/endurance speed, fuel flow, endurance, range, service ceiling). |
| `demo_mission.py`               | Simple climb-to-cruise example wiring physics + thermo + report together. |
| `data_loader.py`                | Loads your aircraft/engine parameters from JSON or CSV. |
| `aircraft_config.json`, `turboprop_config.json`, `piston_config.json` | Editable parameter templates — put your real numbers here. |
| `aircraft_config_example.csv`   | Same data in CSV form. |
| `live_inputs_example.csv`       | Example "live" input stream (climb → hot-day cruise → turbulence → cold-day loiter) for `live_simulation.py --stream`. |

## Quick start

```bash
# Basic climb/cruise + performance summary (original pipeline)
python3 demo_mission.py

# Full-featured live pipeline, streaming a sequence of inputs
python3 live_simulation.py --stream live_inputs_example.csv --out live_outputs.csv

# Or feed it inputs interactively, one at a time
python3 live_simulation.py
> 1,0.7,4,20,moderate      # dt, throttle, climb angle(deg), temp offset(C), turbulence
> 1,0.6,0,-10,light
> quit

# Loiter endurance under different weather
python3 loiter_model.py

# 6-DOF roll/turn response to an aileron input
python3 dof6_model.py

# Propeller efficiency vs. advance ratio
python3 propeller_model.py

# Engine sensor readings + sensor-based power estimation
python3 sensor_model.py
```

Every module also has a runnable `if __name__ == "__main__":` demo at the
bottom — run any file directly to see a worked example.

## How the pieces connect

```
atmosphere_extended.py  ─┐  (weather offset, turbulence/gusts, wind)
                          │
physics_model.py  ───────┼──▶  live_simulation.py  ──▶  sensor_model.py
   (3-DOF trajectory)     │     (ties it all together,      (synthetic
                          │      one input at a time)        engine sensors)
thermo_model.py  ────────┤
  (engine thermodynamics) │
                          │
propeller_model.py ──────┘  (shaft power → thrust, real efficiency curve)

dof6_model.py       -- standalone full 6-DOF (roll/pitch/yaw), same
                        atmosphere/thermo building blocks, for handling-
                        qualities / autopilot-level work

loiter_model.py      -- standalone endurance-to-fuel-exhaustion simulation,
                        using thermo_model + propeller_model + weather

performance_report.py -- point-estimate summary (fast, single numbers)
                          for range/endurance/ceiling from physics+thermo
```

`live_simulation.py` is the main "put it all together" entry point: each
call to `LiveUAVSimulator.step()` takes one new input (throttle, commanded
climb angle, weather, turbulence) and returns the full computed output for
that instant — flight state, engine performance, and simulated sensor
readings — using only the state carried over from the previous step. That
is the "live data entry → immediate output" pattern requested: nothing is
precomputed in a batch, and you can feed it a live stream (a file being
appended to, a socket, a REPL) exactly as shown in `run_from_stream()` and
`run_interactive()`.

## Loading your own data

Copy `aircraft_config.json` / `turboprop_config.json` (or the `.csv`
version), fill in your own numbers, and point `demo_mission.py` /
`live_simulation.py` / `loiter_model.py` at your files (or edit their
`if __name__` sections to call `data_loader.load_aircraft_params(...)`
etc.). Any field you omit falls back to the class default, and
`data_loader.py` tells you which fields fell back so nothing is silently
wrong.

### Aircraft parameters (`physics_model.AircraftParams`)
| Field | Meaning | Units |
|---|---|---|
| `mass` | Dry (zero-fuel) mass | kg |
| `wing_area` | Reference wing area | m² |
| `aspect_ratio` | Wing aspect ratio | — |
| `oswald_efficiency` | Span efficiency factor | — |
| `CD0` | Zero-lift drag coefficient | — |
| `CLmax` | Max usable lift coefficient | — |

### Turboprop engine (`thermo_model.TurbopropParams`)
| Field | Meaning | Units |
|---|---|---|
| `pressure_ratio` | Overall compressor pressure ratio | — |
| `turbine_inlet_temp` | Max TIT | K |
| `eta_compressor`, `eta_turbine`, `eta_combustor`, `eta_mechanical` | Component efficiencies | — |
| `mass_flow_design` | Sea-level design-point air mass flow at full throttle | kg/s |

### Piston engine (`thermo_model.PistonEngineParams`) — alternative
| Field | Meaning | Units |
|---|---|---|
| `sea_level_power_W` | Rated sea-level power | W |
| `bsfc_sea_level` | Brake-specific fuel consumption | kg/(W·s) |
| `critical_altitude_m` | Altitude up to which a turbocharger holds sea-level power | m |
| `altitude_lapse_exponent` | Exponent in the density-ratio power-lapse law | — |

### 6-DOF aircraft (`dof6_model.Aircraft6DOF` / `StabilityDerivatives`)
Adds `Ixx`, `Iyy`, `Izz` (moments of inertia, kg·m²), `wingspan`, `mean_chord`,
and the linear stability & control derivatives (`CL_alpha`, `Cm_alpha`,
`Cl_beta`, `Cn_beta`, control-surface effectiveness, damping terms). These
are the numbers a stability & control analysis (VLM/CFD sweep or wind
tunnel) would give you for a real airframe — the bundled defaults are
representative round numbers for a conventional high-AR UAV, not a
specific design.

### Propeller (`propeller_model.PropellerMap`)
Supply your own `(J, CT, CP)` table via `PropellerMap.from_table(diameter_m,
table)` from a manufacturer chart or blade-element analysis, instead of the
bundled generic curve.

## Weather, turbulence & sensor inputs — how to drive them

- **Hot/cold day**: pass `dT_isa` (or `temp_offset_C` in the live
  simulator) — degrees above/below the ISA standard temperature at every
  altitude. `atmosphere_extended.WEATHER_PRESETS` has named presets
  (`cold_day` = -20 °C, `hot_day` = +20 °C, etc.). This is standard
  "ISA+dT" notation and is what makes hot/high conditions reduce engine
  power and air density, and cold days increase it.
- **Turbulence**: pass `turbulence` = `'none' | 'light' | 'moderate' |
  'severe'`. Internally this drives a simplified Dryden-type gust filter
  (correlated random gusts, not just noise) that perturbs the effective
  airspeed each step — see `atmosphere_extended.DrydenGustModel` for the
  math and its stated simplifications.
- **Engine sensors**: `sensor_model.EngineSensorSuite` turns the physics
  truth into realistic RPM/EGT/fuel-flow/torque/oil-temp readings with
  per-channel noise and slow bias drift, and provides two independent
  ways to estimate power from sensors alone (torque×RPM, or fuel-flow ×
  assumed thermal efficiency) so you can see how sensor-based estimates
  compare to the underlying truth.

## Assumptions & limitations (read before using for real decisions)

- **3-DOF vs 6-DOF**: `physics_model.py` is longitudinal-only (climb/cruise,
  no turning); `dof6_model.py` adds full roll/pitch/yaw but uses **linear**
  stability derivatives (valid near trim, not for large-angle/post-stall
  maneuvers) and currently isn't wired into the fuel/thermo loop the way
  the 3-DOF path is — treat it as a separate dynamics module for
  handling-qualities/autopilot work, not a drop-in fuel-burn replacement.
- **6-DOF is numerically stiff**: the roll mode responds fast relative to
  the other axes: use a small time step (≤0.05–0.1 s) with the explicit
  RK4 integrator provided, or the simulation will diverge. A production
  tool would use a stiffer-aware integrator or trim the derivatives to your
  real airframe's actual (typically slower) roll mode.
- **Off-standard-day atmosphere** shifts ISA temperature uniformly with
  altitude and recomputes density from the standard pressure profile — a
  standard first-order approximation, not a real meteorological sounding.
- **Turbulence model** is a simplified first-order (Ornstein-Uhlenbeck)
  approximation of the Dryden spectrum, not the full MIL-HDBK-1797
  second-order shaping filters — good for engineering-level gust response,
  not certification-grade handling-qualities analysis.
- **Propeller map** is a representative generic fixed-pitch curve, not a
  specific manufacturer's propeller — supply your own `(J, CT, CP)` data
  for a real design.
- **Loiter simulation** assumes a constant-CL, constant-bank steady turn
  and recomputes the power-matched throttle at each step — it does not
  model spiral/Dutch-roll dynamics or a full 6-DOF turn.
- **Engine sensor model** is a plausible noise/bias/correlation model
  illustrating how sensor-based estimation differs from physics truth —
  it is not calibrated to any specific real sensor's datasheet.
- **live_simulation.py's weather coupling to the engine** applies a
  first-order density-ratio correction to the thermo_model's standard-day
  output rather than re-deriving the full Brayton cycle at the offset
  temperature — reasonable for engineering estimates, but a rigorous
  hot/cold-day engine deck would recompute the compressor/turbine cycle at
  the actual inlet temperature directly.
- Replace all bundled example numbers (mass ≈1000 kg, ~150 hp turboprop
  class, generic stability derivatives) with your own data before drawing
  conclusions about a specific airframe.

## Natural next steps beyond this package

- Couple `dof6_model.py` into the same fuel-burn/thermo loop as the 3-DOF
  path, so full 6-DOF missions also track fuel and engine state
- Full second-order MIL-HDBK-1797 Dryden filters if you need
  certification-grade turbulence response
- A real blade-element propeller model (variable pitch, RPM governor) in
  place of the table lookup
- Icing, precipitation drag, and electrical/avionics thermal effects for
  a more complete "weather effects" picture
- Wire actual live telemetry (a socket, MAVLink stream, or hardware sensor
  feed) into `LiveUAVSimulator.step()` in place of the CSV/stdin examples
