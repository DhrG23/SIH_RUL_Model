"""
dof6_model.py
==============
Full 6-degree-of-freedom rigid-body flight dynamics: adds roll, yaw and
pitch MOMENTS (not just the longitudinal climb/cruise physics in
physics_model.py) driven by control-surface deflections (aileron, elevator,
rudder), so the aircraft can bank, turn, and be perturbed in all three axes
-- the natural extension needed for autopilot/handling-qualities work,
turn/loiter dynamics, and turbulence response.

State vector (12 states, standard body-axis formulation):
    position:        x, y, z          (NED, m)
    attitude:        phi, theta, psi  (roll, pitch, yaw Euler angles, rad)
    body velocity:   u, v, w          (m/s, along body x/y/z)
    body rates:      p, q, r          (rad/s, roll/pitch/yaw rate)

Aerodynamics use a standard linearized stability-derivative model
(dimensional/non-dimensional coefficients times dynamic pressure, angle of
attack, sideslip, rates and control deflections) -- the same modeling
approach used for early-stage UAV flight control design. Default
derivatives are representative of a MALE-class UAV (high aspect ratio,
conventional tail) -- replace with your own stability & control data
(from a VLM/CFD sweep or wind-tunnel test) for a specific airframe.
"""

import math
from dataclasses import dataclass, field

from physics_model import isa_atmosphere, G0, AircraftParams


@dataclass
class StabilityDerivatives:
    """Non-dimensional linear stability & control derivatives.
    Representative values for a conventional, high-aspect-ratio MALE UAV;
    replace with your own aircraft's data for real analysis."""
    # Longitudinal
    CL_alpha: float = 5.0        # per rad
    CL0: float = 0.30
    Cm0: float = 0.02
    Cm_alpha: float = -0.8       # per rad (negative = statically stable)
    Cm_q: float = -12.0          # per rad/s (pitch damping)
    Cm_delta_e: float = -1.1     # per rad elevator

    # Lateral-directional
    CY_beta: float = -0.6        # per rad sideslip
    Cl_beta: float = -0.08       # per rad (dihedral effect, roll from sideslip)
    Cl_p: float = -0.45          # per rad/s (roll damping)
    Cl_r: float = 0.05           # per rad/s
    Cl_delta_a: float = 0.12     # per rad aileron
    Cn_beta: float = 0.10        # per rad (weathercock stability)
    Cn_r: float = -0.15          # per rad/s (yaw damping)
    Cn_delta_r: float = -0.08    # per rad rudder


@dataclass
class Aircraft6DOF:
    mass: float = 1020.0
    # Representative inertias for a ~1000 kg, 14.8 m span MALE UAV (radius
    # of gyration roughly b/4 to b/3 for roll -- tail/nacelle mass raises
    # Iyy/Izz somewhat above the pure-wing estimate). Undersized inertias
    # make the roll mode unrealistically stiff/fast and force a tiny
    # integration step -- use real inertia data for your airframe if you
    # have it (e.g. from a CAD mass properties report).
    Ixx: float = 8000.0    # kg*m^2, roll inertia
    Iyy: float = 12000.0   # kg*m^2, pitch inertia
    Izz: float = 18000.0   # kg*m^2, yaw inertia
    wing_area: float = 11.5
    wingspan: float = 14.8
    mean_chord: float = 0.80
    aero: StabilityDerivatives = field(default_factory=StabilityDerivatives)
    CD0: float = 0.028
    k_induced: float = 0.021


@dataclass
class State6DOF:
    x: float = 0.0; y: float = 0.0; z: float = 0.0            # NED position, m (z down)
    phi: float = 0.0; theta: float = 0.0; psi: float = 0.0     # rad
    u: float = 45.0; v: float = 0.0; w: float = 0.0            # body velocity, m/s
    p: float = 0.0; q: float = 0.0; r: float = 0.0             # body rates, rad/s

    def as_list(self):
        return [self.x, self.y, self.z, self.phi, self.theta, self.psi,
                self.u, self.v, self.w, self.p, self.q, self.r]

    @classmethod
    def from_list(cls, vec):
        return cls(*vec)


@dataclass
class Controls:
    throttle: float = 0.5      # 0-1
    elevator: float = 0.0      # rad, +ve = trailing edge down (nose-up moment convention here)
    aileron: float = 0.0       # rad, +ve = right roll moment
    rudder: float = 0.0        # rad, +ve = right yaw moment


def _airdata(state: State6DOF):
    """True airspeed, angle of attack, sideslip from body velocities."""
    V = math.sqrt(state.u ** 2 + state.v ** 2 + state.w ** 2)
    alpha = math.atan2(state.w, state.u) if abs(state.u) > 1e-6 else 0.0
    beta = math.asin(max(-1.0, min(1.0, state.v / V))) if V > 1e-6 else 0.0
    return V, alpha, beta


def aero_forces_moments(state: State6DOF, controls: Controls, thrust_N: float,
                         ac: Aircraft6DOF, rho: float):
    """Compute body-axis forces (Fx,Fy,Fz) and moments (L,M,N) including
    gravity, aerodynamics, and thrust."""
    V, alpha, beta = _airdata(state)
    q_bar = 0.5 * rho * V * V
    S, b, c = ac.wing_area, ac.wingspan, ac.mean_chord
    d = ac.aero

    # Nondimensional rates (standard normalization)
    p_hat = state.p * b / (2 * V) if V > 1 else 0.0
    q_hat = state.q * c / (2 * V) if V > 1 else 0.0
    r_hat = state.r * b / (2 * V) if V > 1 else 0.0

    # --- Longitudinal aero ---
    CL = d.CL0 + d.CL_alpha * alpha
    CD = ac.CD0 + ac.k_induced * CL * CL
    Cm = d.Cm0 + d.Cm_alpha * alpha + d.Cm_q * q_hat + d.Cm_delta_e * controls.elevator

    L_aero = q_bar * S * CL      # lift, wind-ish axis magnitude
    D_aero = q_bar * S * CD      # drag
    M = q_bar * S * c * Cm       # pitching moment

    # --- Lateral-directional aero ---
    CY = d.CY_beta * beta
    Cl = d.Cl_beta * beta + d.Cl_p * p_hat + d.Cl_r * r_hat + d.Cl_delta_a * controls.aileron
    Cn = d.Cn_beta * beta + d.Cn_r * r_hat + d.Cn_delta_r * controls.rudder

    Y = q_bar * S * CY
    L_mom = q_bar * S * b * Cl
    N = q_bar * S * b * Cn

    # Resolve lift/drag (wind axes, small-angle) into body axes and add thrust + gravity
    Fx = thrust_N - D_aero * math.cos(alpha) + L_aero * math.sin(alpha)
    Fz = -D_aero * math.sin(alpha) - L_aero * math.cos(alpha)
    Fy = Y

    W = ac.mass * G0
    Fx += -W * math.sin(state.theta)
    Fy += W * math.sin(state.phi) * math.cos(state.theta)
    Fz += W * math.cos(state.phi) * math.cos(state.theta)

    return Fx, Fy, Fz, L_mom, M, N


def state_derivatives_6dof(state: State6DOF, controls: Controls, thrust_N: float,
                            ac: Aircraft6DOF, h_for_density: float = None):
    """Full 12-state nonlinear equations of motion (body axes, Ixz=0
    assumed -- valid for a conventional symmetric airframe)."""
    if h_for_density is None:
        h_for_density = -state.z
    _, _, rho, _ = isa_atmosphere(h_for_density)

    Fx, Fy, Fz, Lm, M, N = aero_forces_moments(state, controls, thrust_N, ac, rho)
    m = ac.mass
    p, q, r = state.p, state.q, state.r

    u_dot = Fx / m - q * state.w + r * state.v
    v_dot = Fy / m - r * state.u + p * state.w
    w_dot = Fz / m - p * state.v + q * state.u

    p_dot = Lm / ac.Ixx - ((ac.Izz - ac.Iyy) / ac.Ixx) * q * r
    q_dot = M / ac.Iyy - ((ac.Ixx - ac.Izz) / ac.Iyy) * p * r
    r_dot = N / ac.Izz - ((ac.Iyy - ac.Ixx) / ac.Izz) * p * q

    phi, theta = state.phi, state.theta
    phi_dot = p + q * math.sin(phi) * math.tan(theta) + r * math.cos(phi) * math.tan(theta)
    theta_dot = q * math.cos(phi) - r * math.sin(phi)
    cos_theta = math.cos(theta) if abs(math.cos(theta)) > 1e-6 else 1e-6
    psi_dot = q * math.sin(phi) / cos_theta + r * math.cos(phi) / cos_theta

    # Body -> NED velocity transform (standard 3-2-1 Euler DCM)
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(state.psi), math.sin(state.psi)

    x_dot = (cth * cps) * state.u + (sph * sth * cps - cph * sps) * state.v + \
            (cph * sth * cps + sph * sps) * state.w
    y_dot = (cth * sps) * state.u + (sph * sth * sps + cph * cps) * state.v + \
            (cph * sth * sps - sph * cps) * state.w
    z_dot = (-sth) * state.u + (sph * cth) * state.v + (cph * cth) * state.w

    return State6DOF(x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot,
                      u_dot, v_dot, w_dot, p_dot, q_dot, r_dot)


def rk4_step_6dof(state: State6DOF, dt: float, controls: Controls, thrust_N: float,
                   ac: Aircraft6DOF) -> State6DOF:
    def add(a: State6DOF, b: State6DOF, scale: float) -> State6DOF:
        av, bv = a.as_list(), b.as_list()
        return State6DOF.from_list([av[i] + scale * bv[i] for i in range(12)])

    k1 = state_derivatives_6dof(state, controls, thrust_N, ac)
    k2 = state_derivatives_6dof(add(state, k1, dt / 2), controls, thrust_N, ac)
    k3 = state_derivatives_6dof(add(state, k2, dt / 2), controls, thrust_N, ac)
    k4 = state_derivatives_6dof(add(state, k3, dt), controls, thrust_N, ac)

    sv, k1v, k2v, k3v, k4v = state.as_list(), k1.as_list(), k2.as_list(), k3.as_list(), k4.as_list()
    new_vec = [sv[i] + (dt / 6) * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) for i in range(12)]
    return State6DOF.from_list(new_vec)


if __name__ == "__main__":
    ac = Aircraft6DOF()
    state = State6DOF(u=45.0)
    controls = Controls(throttle=0.6, aileron=math.radians(5.0))  # 5 deg aileron -> roll

    print(f"{'t(s)':>5} {'phi(deg)':>9} {'p(deg/s)':>9} {'psi(deg)':>9} {'V(m/s)':>7}")
    thrust_N = 2500.0
    dt = 0.05  # roll mode is fast -- use a small step for explicit RK4 stability
    n_steps = 200  # 10 s of simulated time
    for i in range(n_steps):
        state = rk4_step_6dof(state, dt, controls, thrust_N, ac)
        V, alpha, beta = _airdata(state)
        if (i + 1) % 20 == 0:
            print(f"{(i+1)*dt:5.1f} {math.degrees(state.phi):9.2f} "
                  f"{math.degrees(state.p):9.2f} {math.degrees(state.psi):9.2f} {V:7.2f}")
