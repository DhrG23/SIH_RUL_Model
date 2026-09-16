"""
physics_model.py
=================
Physics (flight-dynamics) model for a generic Medium-Altitude Long-Endurance
(MALE) UAV, in the class of RQ-1/MQ-1 Predator or MQ-9 Reaper style airframes.

Contents
--------
1. ISA (International Standard Atmosphere) model, valid 0-20 km, for
   density/pressure/temperature/speed-of-sound as a function of altitude.
2. Aerodynamic model: simple parabolic drag polar with altitude/Mach-aware
   dynamic pressure.
3. 3-DOF point-mass equations of motion in the vertical (climb/cruise) plane:
   states = [V (true airspeed, m/s), gamma (flight path angle, rad),
             h (altitude, m), x (downrange distance, m)]
   This is the standard "energy-state" formulation used for MALE UAV
   performance/mission studies (climb, cruise, descent, endurance).
4. RK4 integrator to propagate the states forward in time given a thrust
   and lift-coefficient (or bank/AoA) command from an autopilot/guidance
   layer -- not included here, only the plant model.

Units: SI throughout (meters, seconds, kg, radians, Newtons, Kelvin, Pascal).
"""

from dataclasses import dataclass, field
import math

G0 = 9.80665            # standard gravity, m/s^2
R_AIR = 287.05287        # specific gas constant for dry air, J/(kg*K)
GAMMA_AIR = 1.4           # ratio of specific heats for air


# ---------------------------------------------------------------------------
# 1. ISA Atmosphere Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ISALayer:
    base_alt: float      # m
    base_temp: float     # K
    lapse_rate: float    # K/m (negative = temperature decreasing with height)
    base_press: float    # Pa


# Standard ISA layers up to 20 km (covers all MALE UAV operating altitudes,
# typically service ceiling 15,000-25,000 ft / ~4,500-7,600 m, occasionally
# higher for some platforms).
_ISA_LAYERS = [
    ISALayer(0.0,     288.15, -0.0065, 101325.0),
    ISALayer(11000.0, 216.65,  0.0000,  22632.06),
    ISALayer(20000.0, 216.65,  0.0010,   5474.89),
]


def isa_atmosphere(h_m: float):
    """
    Return (temperature_K, pressure_Pa, density_kg_m3, speed_of_sound_m_s)
    at geopotential altitude h_m using the ISA model.
    """
    h_m = max(0.0, h_m)
    layer = _ISA_LAYERS[0]
    for candidate in _ISA_LAYERS:
        if h_m >= candidate.base_alt:
            layer = candidate
        else:
            break

    dh = h_m - layer.base_alt
    if abs(layer.lapse_rate) > 1e-12:
        T = layer.base_temp + layer.lapse_rate * dh
        P = layer.base_press * (T / layer.base_temp) ** (-G0 / (layer.lapse_rate * R_AIR))
    else:
        T = layer.base_temp
        P = layer.base_press * math.exp(-G0 * dh / (R_AIR * layer.base_temp))

    rho = P / (R_AIR * T)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)
    return T, P, rho, a


# ---------------------------------------------------------------------------
# 2. Aircraft definition & aerodynamic model
# ---------------------------------------------------------------------------
@dataclass
class AircraftParams:
    """Representative MALE-class UAV parameters (Predator/Reaper order of
    magnitude). Replace with your specific airframe's data."""
    mass: float = 1020.0          # kg, typical operating mass
    wing_area: float = 11.5       # m^2
    aspect_ratio: float = 19.0    # high-AR wing typical of MALE UAVs
    oswald_efficiency: float = 0.80
    CD0: float = 0.028            # zero-lift drag coefficient
    CLmax: float = 1.4            # max lift coefficient (clean)

    @property
    def k_induced(self) -> float:
        """Induced drag factor K in CD = CD0 + K*CL^2."""
        return 1.0 / (math.pi * self.aspect_ratio * self.oswald_efficiency)


def aero_forces(V: float, h: float, CL: float, ac: AircraftParams):
    """
    Compute lift (N) and drag (N) at true airspeed V (m/s), altitude h (m),
    for a commanded lift coefficient CL.
    """
    _, _, rho, _ = isa_atmosphere(h)
    q = 0.5 * rho * V * V                      # dynamic pressure, Pa
    CL = max(-ac.CLmax, min(ac.CLmax, CL))     # saturate
    CD = ac.CD0 + ac.k_induced * CL * CL       # parabolic drag polar
    L = q * ac.wing_area * CL
    D = q * ac.wing_area * CD
    return L, D, q, rho


def cl_for_level_flight(V: float, h: float, ac: AircraftParams) -> float:
    """Lift coefficient required for steady, level (gamma=0) flight."""
    _, _, rho, _ = isa_atmosphere(h)
    q = 0.5 * rho * V * V
    W = ac.mass * G0
    return W / (q * ac.wing_area) if q > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# 3. 3-DOF point-mass equations of motion (vertical plane)
# ---------------------------------------------------------------------------
@dataclass
class FlightState:
    V: float       # true airspeed, m/s
    gamma: float   # flight path angle, rad (+ = climbing)
    h: float       # altitude, m
    x: float       # downrange distance, m


def state_derivatives(state: FlightState, thrust_N: float, CL: float,
                       ac: AircraftParams) -> FlightState:
    """
    Point-mass equations of motion:
        m*Vdot        = T - D - m*g*sin(gamma)
        m*V*gammadot  = L - m*g*cos(gamma)
        hdot          = V*sin(gamma)
        xdot          = V*cos(gamma)
    """
    L, D, _, _ = aero_forces(state.V, state.h, CL, ac)
    m, g = ac.mass, G0

    V_dot = (thrust_N - D - m * g * math.sin(state.gamma)) / m
    if state.V > 1.0:
        gamma_dot = (L - m * g * math.cos(state.gamma)) / (m * state.V)
    else:
        gamma_dot = 0.0
    h_dot = state.V * math.sin(state.gamma)
    x_dot = state.V * math.cos(state.gamma)

    return FlightState(V_dot, gamma_dot, h_dot, x_dot)


def rk4_step(state: FlightState, dt: float, thrust_N: float, CL: float,
             ac: AircraftParams) -> FlightState:
    """Classic 4th-order Runge-Kutta integration of one time step."""

    def add(a: FlightState, b: FlightState, scale: float) -> FlightState:
        return FlightState(a.V + scale * b.V, a.gamma + scale * b.gamma,
                            a.h + scale * b.h, a.x + scale * b.x)

    k1 = state_derivatives(state, thrust_N, CL, ac)
    k2 = state_derivatives(add(state, k1, dt / 2), thrust_N, CL, ac)
    k3 = state_derivatives(add(state, k2, dt / 2), thrust_N, CL, ac)
    k4 = state_derivatives(add(state, k3, dt), thrust_N, CL, ac)

    return FlightState(
        state.V + (dt / 6) * (k1.V + 2 * k2.V + 2 * k3.V + k4.V),
        state.gamma + (dt / 6) * (k1.gamma + 2 * k2.gamma + 2 * k3.gamma + k4.gamma),
        state.h + (dt / 6) * (k1.h + 2 * k2.h + 2 * k3.h + k4.h),
        state.x + (dt / 6) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
    )


if __name__ == "__main__":
    # Quick sanity check: steady, level cruise at 15,000 ft.
    ac = AircraftParams()
    h_cruise = 4572.0  # 15,000 ft in m
    V_cruise = 55.0     # m/s (~107 kt), typical MALE UAV cruise speed
    CL_trim = cl_for_level_flight(V_cruise, h_cruise, ac)
    L, D, q, rho = aero_forces(V_cruise, h_cruise, CL_trim, ac)
    print(f"Cruise trim @ {h_cruise:.0f} m, V={V_cruise} m/s:")
    print(f"  rho={rho:.4f} kg/m^3, q={q:.1f} Pa, CL_trim={CL_trim:.3f}")
    print(f"  L={L:.1f} N (should ~= weight {ac.mass*G0:.1f} N), D={D:.1f} N")
