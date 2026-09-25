"""
atmosphere_extended.py
=======================
Extends physics_model.isa_atmosphere() with two real-world effects the plain
ISA standard day doesn't capture:

1. Off-standard ("hot day" / "cold day") atmospheres -- ambient temperature
   offset from the ISA standard, which changes air density, engine power,
   and true airspeed for a given indicated airspeed. Expressed as ISA+dT
   (e.g. "ISA+20" = a hot day, "ISA-15" = a cold day), the standard way
   performance charts handle weather effects.

2. Atmospheric turbulence / gusts -- a simplified Dryden-type continuous
   turbulence model that adds randomly varying gust velocity components to
   the steady wind, so the aircraft doesn't fly through perfectly still air.

Both feed into the rest of the model chain: thermo_model uses the offset
temperature (hot day = less dense air = less engine power), and
physics_model / dof6_model can add the gust velocity to the aircraft's
airspeed to see its effect on lift/drag/handling.
"""

import math
import random
from dataclasses import dataclass

from .physics_model import R_AIR, GAMMA_AIR, G0, isa_atmosphere as _isa_standard


# ---------------------------------------------------------------------------
# 1. Off-standard-day atmosphere (weather effect: hot / cold)
# ---------------------------------------------------------------------------
def isa_atmosphere_offset(h_m: float, dT_isa: float = 0.0):
    """
    Atmosphere at altitude h_m on a day that is dT_isa degrees warmer (+) or
    colder (-) than the ISA standard day at every altitude (the standard
    "ISA+dT" convention used in aircraft performance work, e.g. a 35 C hot
    day at sea level is roughly ISA+20).

    Approximation: shift the standard-day temperature by dT_isa at every
    altitude, then recompute density from the ideal gas law using the
    standard-day pressure profile. This is the common first-order way to
    model hot/cold-day performance without re-deriving a full non-standard
    hydrostatic profile, and is accurate enough for performance estimation
    (it is NOT a substitute for real meteorological sounding data).

    Returns (T_K, P_Pa, rho_kg_m3, a_m_s).
    """
    T_std, P_std, _, _ = _isa_standard(h_m)
    T = T_std + dT_isa
    rho = P_std / (R_AIR * T)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)
    return T, P_std, rho, a


def density_altitude(h_m: float, dT_isa: float) -> float:
    """
    Density altitude: the altitude in the STANDARD atmosphere that has the
    same air density as the actual (off-standard) condition. This is the
    number that actually governs aircraft/engine performance on a hot day
    -- "hot and high" airfields perform as if they were much higher up.
    Found by searching the standard atmosphere for matching density.
    """
    _, _, rho_actual, _ = isa_atmosphere_offset(h_m, dT_isa)
    lo, hi = -1000.0, 15000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        _, _, rho_std, _ = _isa_standard(mid)
        if rho_std > rho_actual:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Common named weather conditions, as ISA temperature offsets (deg C)
WEATHER_PRESETS = {
    "standard_day": 0.0,
    "cold_day": -20.0,     # e.g. winter/high-latitude ops
    "hot_day": 20.0,        # e.g. desert summer ops
    "extreme_cold": -35.0,
    "extreme_hot": 35.0,
}


# ---------------------------------------------------------------------------
# 2. Wind & turbulence (gust) model
# ---------------------------------------------------------------------------
TURBULENCE_INTENSITY_SIGMA = {
    # Gust RMS velocity (m/s) per axis, loosely following the "light /
    # moderate / severe" turbulence categories used in flight-sim and
    # dispatch weather briefings. These are engineering round numbers,
    # not a specific regulatory table.
    "none": 0.0,
    "light": 0.5,
    "moderate": 1.5,
    "severe": 3.5,
}

# Turbulence length scale (m) -- how "gusty" vs "smooth" the disturbance is
# in space. ~530 m (1750 ft) is the standard medium/high-altitude Dryden
# scale length per MIL-F-8785C.
TURBULENCE_LENGTH_SCALE_M = 530.0


@dataclass
class GustState:
    """Running gust velocity state (body-ish axes: along-track, cross,
    vertical), updated each timestep by gust_model.update()."""
    u_gust: float = 0.0   # m/s, along flight direction
    v_gust: float = 0.0   # m/s, lateral
    w_gust: float = 0.0   # m/s, vertical


class DrydenGustModel:
    """
    Simplified (first-order Markov / Ornstein-Uhlenbeck) approximation of
    Dryden continuous turbulence. This reproduces the key qualitative
    behavior of the full Dryden spectrum -- gusts correlated over the
    turbulence length scale, RMS amplitude set by intensity -- using a
    first-order shaping filter per axis:

        gust_dot = -(V/L) * gust + sigma * sqrt(2V/L) * white_noise

    This is a standard simplification for engineering simulations; a
    control-system-grade Dryden model uses full second-order filters for
    the lateral/vertical axes (see MIL-HDBK-1797) if you need spectral
    fidelity for handling-qualities work.
    """

    def __init__(self, intensity: str = "moderate", length_scale_m: float = TURBULENCE_LENGTH_SCALE_M,
                 seed: int = None):
        if intensity not in TURBULENCE_INTENSITY_SIGMA:
            raise ValueError(f"Unknown intensity '{intensity}'. Choose from {list(TURBULENCE_INTENSITY_SIGMA)}")
        self.sigma = TURBULENCE_INTENSITY_SIGMA[intensity]
        self.L = length_scale_m
        self.state = GustState()
        self._rng = random.Random(seed)

    def update(self, V_true_airspeed: float, dt: float) -> GustState:
        """Advance the gust state by dt given current true airspeed (m/s),
        which sets how fast the aircraft "burns through" the turbulence
        length scale. Returns the updated GustState."""
        V = max(V_true_airspeed, 1.0)
        tau = self.L / V                      # correlation time, s
        alpha = math.exp(-dt / tau) if tau > 1e-6 else 0.0
        noise_gain = self.sigma * math.sqrt(max(0.0, 1 - alpha ** 2))

        self.state.u_gust = alpha * self.state.u_gust + noise_gain * self._rng.gauss(0, 1)
        self.state.v_gust = alpha * self.state.v_gust + noise_gain * self._rng.gauss(0, 1)
        self.state.w_gust = alpha * self.state.w_gust + noise_gain * self._rng.gauss(0, 1)
        return self.state


@dataclass
class SteadyWind:
    """Constant background wind, e.g. from a met report."""
    speed_m_s: float = 0.0
    from_heading_deg: float = 0.0  # meteorological convention: wind FROM this heading

    def components(self):
        """Return (wind_north_m_s, wind_east_m_s)."""
        ang = math.radians(self.from_heading_deg)
        # Wind blows FROM from_heading_deg, i.e. velocity points opposite
        wn = -self.speed_m_s * math.cos(ang)
        we = -self.speed_m_s * math.sin(ang)
        return wn, we


if __name__ == "__main__":
    print("Hot day (ISA+20) vs cold day (ISA-20) at 4572 m:")
    for label, dT in [("standard", 0.0), ("hot (+20)", 20.0), ("cold (-20)", -20.0)]:
        T, P, rho, a = isa_atmosphere_offset(4572.0, dT)
        print(f"  {label:12s}: T={T:6.1f} K  rho={rho:.4f} kg/m^3  "
              f"density_alt={density_altitude(4572.0, dT):7.0f} m")

    print("\nGust model sample (moderate turbulence, V=50 m/s, 10 steps):")
    gust = DrydenGustModel(intensity="moderate", seed=42)
    for i in range(10):
        g = gust.update(V_true_airspeed=50.0, dt=1.0)
        print(f"  t={i+1:2d}s  u_gust={g.u_gust:+.2f}  v_gust={g.v_gust:+.2f}  w_gust={g.w_gust:+.2f} m/s")
