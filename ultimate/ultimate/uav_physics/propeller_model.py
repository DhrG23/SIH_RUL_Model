"""
propeller_model.py
====================
Replaces the constant "prop_eff = 0.80" assumption used in the earlier demo
with a proper propeller performance map: thrust and power coefficients as a
function of advance ratio J = V / (n*D), which is how real propeller charts
are built and how efficiency actually varies with flight condition.

The table below is a representative, generically-shaped fixed-pitch
general-aviation/UAV propeller curve (peak efficiency ~0.82 around J~0.6-0.7,
falling off at both low and high advance ratio) -- NOT a specific
manufacturer's propeller. Swap in your own (J, CT, CP) table from a wind
tunnel test or manufacturer chart via `PropellerMap.from_table()` for a
real design.

Definitions
-----------
    J  = V / (n * D)              advance ratio (V: m/s, n: rev/s, D: m)
    CT = T / (rho * n^2 * D^4)    thrust coefficient
    CP = P / (rho * n^3 * D^5)    power coefficient
    eta = J * CT / CP             propulsive efficiency
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# Representative generic fixed-pitch propeller performance table.
_DEFAULT_TABLE = [
    # J,    CT,     CP
    (0.0,  0.120,  0.100),
    (0.2,  0.115,  0.098),
    (0.4,  0.105,  0.093),
    (0.6,  0.090,  0.082),
    (0.8,  0.068,  0.068),
    (1.0,  0.042,  0.052),
    (1.2,  0.012,  0.033),
    (1.35, 0.000,  0.020),   # zero-thrust (windmilling) advance ratio
]


@dataclass
class PropellerMap:
    diameter_m: float = 1.9          # representative MALE-UAV prop diameter
    table: List[Tuple[float, float, float]] = field(default_factory=lambda: list(_DEFAULT_TABLE))

    @classmethod
    def from_table(cls, diameter_m: float, table: List[Tuple[float, float, float]]):
        """Build a PropellerMap from your own (J, CT, CP) data points,
        sorted by ascending J."""
        return cls(diameter_m=diameter_m, table=sorted(table, key=lambda r: r[0]))

    def _interp(self, J: float, col: int) -> float:
        rows = self.table
        J = max(rows[0][0], min(rows[-1][0], J))  # clamp to table range
        for i in range(len(rows) - 1):
            j0, j1 = rows[i][0], rows[i + 1][0]
            if j0 <= J <= j1:
                frac = (J - j0) / (j1 - j0) if j1 > j0 else 0.0
                return rows[i][col] + frac * (rows[i + 1][col] - rows[i][col])
        return rows[-1][col]

    def coefficients(self, J: float):
        """Return (CT, CP, eta) at advance ratio J."""
        CT = self._interp(J, 1)
        CP = self._interp(J, 2)
        eta = (J * CT / CP) if CP > 1e-9 else 0.0
        return CT, CP, max(0.0, eta)

    def thrust_and_power(self, V_m_s: float, n_rps: float, rho: float):
        """Given true airspeed, prop rotational speed (rev/s) and air
        density, return (thrust_N, power_absorbed_W, eta)."""
        D = self.diameter_m
        J = V_m_s / (n_rps * D) if n_rps > 1e-6 else 0.0
        CT, CP, eta = self.coefficients(J)
        T = CT * rho * n_rps ** 2 * D ** 4
        P = CP * rho * n_rps ** 3 * D ** 5
        return T, P, eta

    def thrust_from_shaft_power(self, shaft_power_W: float, V_m_s: float, rho: float,
                                 n_min_rps: float = 5.0, n_max_rps: float = 60.0):
        """
        Given available shaft power (from the engine model) and current
        flight condition, solve for the prop rotational speed that absorbs
        exactly that power, then return the resulting thrust and
        efficiency. This replaces "thrust = power*const_eff/V" with a
        proper power-matched operating point on the propeller map.
        """
        if shaft_power_W <= 0:
            return 0.0, 0.0, 0.0

        lo, hi = n_min_rps, n_max_rps
        for _ in range(40):
            mid = (lo + hi) / 2
            _, P_mid, _ = self.thrust_and_power(V_m_s, mid, rho)
            if P_mid < shaft_power_W:
                lo = mid
            else:
                hi = mid
        n_op = (lo + hi) / 2
        T, P, eta = self.thrust_and_power(V_m_s, n_op, rho)
        return T, eta, n_op


if __name__ == "__main__":
    prop = PropellerMap()
    print("Advance ratio sweep (D=1.9 m):")
    print(f"{'J':>5} {'CT':>7} {'CP':>7} {'eta':>6}")
    for J in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        CT, CP, eta = prop.coefficients(J)
        print(f"{J:5.2f} {CT:7.4f} {CP:7.4f} {eta:6.3f}")

    print("\nPower-matched operating point example:")
    rho_sl = 1.225
    T, eta, n_op = prop.thrust_from_shaft_power(shaft_power_W=60_000.0, V_m_s=50.0, rho=rho_sl)
    print(f"  60 kW available @ V=50 m/s, sea level: "
          f"n={n_op*60:.0f} RPM, thrust={T:.0f} N, eta={eta:.3f}")
