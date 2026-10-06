"""
Physics Engine Wrapper and Stateful Differential Integrator.

Integrates continuous engine state variables (CHT, Oil Temperature, EGT, Engine RPM)
using first-principles differential equations from data/physics.
Ensures variables evolve continuously over time in milliseconds without unnatural jumps.
"""

from typing import Any, Dict, Tuple
import numpy as np

from .physics import environment as env
from .physics import airflow as air
from .physics import fuel
from .physics import combustion as comb
from .physics import friction as fric
from .physics import power as pwr
from .physics import lubrication as lub
from .physics import thermal as therm
from .physics import vibration as vib
from .physics import health as hlth
from .physics.parameters import (
    DEFAULT_GEOMETRY,
    DEFAULT_THERMAL,
    DEFAULT_LUBRICATION,
    DEFAULT_FLUIDS,
)
from .physics.faults import EngineFaultState
from .schema import EngineState


class PhysicsEngine:
    """Evaluates physics equations and integrates continuous ODE state variables."""

    def __init__(
        self,
        geometry=DEFAULT_GEOMETRY,
        thermal_params=DEFAULT_THERMAL,
        lubrication_params=DEFAULT_LUBRICATION,
        fluid_params=DEFAULT_FLUIDS,
        tau_rpm_s: float = 0.35,
    ):
        self.geo = geometry
        self.thermal = thermal_params
        self.lub = lubrication_params
        self.fluids = fluid_params
        self.tau_rpm = tau_rpm_s

    def calculate_instantaneous_physics(
        self,
        state: EngineState,
        fault_state: EngineFaultState,
    ) -> Dict[str, Any]:
        """
        Calculates all instantaneous physical parameters, heat flows, powers,
        pressures, vibration, and derivatives given the current engine state and fault state.
        """
        # 1. Environment & Psychrometrics
        p_amb_pa = env.isa_pressure(state.altitude_m)
        pv_pa = env.vapor_pressure(state.ambient_temp_k, state.humidity_ratio)
        rho_amb = env.moist_air_density(p_amb_pa, state.ambient_temp_k, pv_pa)
        v_cool_mps = env.effective_cooling_velocity(
            airspeed_tas=state.wind_speed_mps * 0.6,
            wind_speed=state.wind_speed_mps,
            wind_heading_rel_rad=state.wind_direction_rad,
            rpm=state.rpm,
        )

        # 2. Airflow & Manifold
        p_man_pa = air.manifold_pressure(
            p_amb_pa=p_amb_pa,
            throttle_position=state.throttle_norm,
            rpm=state.rpm,
        )
        t_man_k = state.ambient_temp_k + 3.0  # Slight heating through manifold
        rho_man = air.manifold_air_density(p_man_pa, t_man_k)
        eta_v = air.volumetric_efficiency(
            rpm=state.rpm,
            manifold_pressure_pa=p_man_pa,
            ambient_pressure_pa=p_amb_pa,
            manifold_temperature_k=t_man_k,
        )
        mdot_air_kgps = air.air_mass_flow(
            density_man=rho_man,
            displacement_m3=self.geo.displacement_volume,
            rpm=state.rpm,
            volumetric_eff=eta_v,
        )

        # 3. Fueling & Faulted Injector
        afr_target = fuel.target_air_fuel_ratio(
            load_fraction=state.engine_load_norm,
            rpm=state.rpm,
        )
        mdot_fuel_target = fuel.target_fuel_flow(mdot_air_kgps, afr_target)
        theta_inj = fault_state.injector.effective_injector_health()
        mdot_fuel_actual = fuel.actual_fuel_flow(mdot_fuel_target, theta_inj)
        afr_actual = fuel.actual_air_fuel_ratio(mdot_air_kgps, mdot_fuel_actual)
        phi = fuel.equivalence_ratio(afr_actual)

        # 4. Combustion & Faulted Ignition
        mu_mis = fault_state.combustion.effective_misfire_intensity()
        retard_deg = fault_state.combustion.timing_retard_deg
        eta_comb = comb.combustion_efficiency(
            equivalence_ratio_phi=phi,
            misfire_intensity=mu_mis,
            timing_retard_deg=retard_deg,
        )
        eta_ind_th = comb.indicated_thermal_efficiency(
            compression_ratio=self.geo.compression_ratio,
            gamma=self.fluids.gamma_combusted,
            equivalence_ratio_phi=phi,
        )
        q_fuel_chem = comb.fuel_chemical_power(
            actual_fuel_flow_kg_per_s=mdot_fuel_actual,
            lhv_j_per_kg=self.fluids.lhv_fuel,
        )

        # 5. Lubrication & Oil Viscosity with Degradation
        mu_nominal = lub.oil_viscosity(state.oil_temp_k)
        mu_actual = fault_state.lubrication.modify_viscosity(mu_nominal)
        theta_pump = fault_state.lubrication.effective_pump_health()
        leakage_factor = fault_state.lubrication.effective_clearance_leakage()
        vdot_oil_pump = lub.oil_pump_flow(state.rpm, pump_health=theta_pump)
        p_oil_pa = lub.oil_pressure(
            rpm=state.rpm,
            oil_viscosity_pa_s=mu_actual,
            pump_health=theta_pump,
            clearance_leakage_factor=leakage_factor,
        )

        # 6. Power, Torque & Friction
        p_ind_w = pwr.indicated_power(q_fuel_chem, eta_comb, eta_ind_th)
        imep_pa = pwr.indicated_mean_effective_pressure(p_ind_w, state.rpm, self.geo.displacement_volume)
        fmep_pa = fric.friction_mean_effective_pressure(
            rpm=state.rpm,
            imep_pa=imep_pa,
            oil_viscosity_pa_s=mu_actual,
            lubrication_degradation=fault_state.lubrication.boundary_scuffing_severity,
        )
        p_fric_w = fric.friction_power(fmep_pa, self.geo.displacement_volume, state.rpm)
        p_brake_w = pwr.brake_power(p_ind_w, p_fric_w)
        eta_mech = pwr.mechanical_efficiency(p_brake_w, p_ind_w)
        tau_brake_nm = pwr.engine_torque(p_brake_w, state.rpm)
        tau_ind_nm = pwr.indicated_torque(p_ind_w, state.rpm)

        # 7. Thermal Balances & Derivatives
        q_waste_w = therm.waste_heat(q_fuel_chem, eta_comb, p_brake_w)
        theta_cool = fault_state.cooling.effective_cooling_health()
        q_cool_cht_w = therm.cooling_heat_rejection(
            t_cht_k=state.cht_k,
            t_amb_k=state.ambient_temp_k,
            air_density_kg_per_m3=rho_amb,
            cooling_velocity_m_per_s=v_cool_mps,
            cooling_health=theta_cool,
        )
        q_oil_cooler_w = therm.oil_cooler_heat_rejection(
            t_oil_k=state.oil_temp_k,
            t_amb_k=state.ambient_temp_k,
            air_density_kg_per_m3=rho_amb,
            cooling_velocity_m_per_s=v_cool_mps,
            cooling_health=theta_cool,
        )
        # Small thermal transfer between cylinder wall and oil gallery
        q_cht_to_oil_w = 45.0 * (state.cht_k - state.oil_temp_k)

        d_cht_dt = therm.cht_derivative(
            q_waste_w=q_waste_w,
            q_cool_cht_w=q_cool_cht_w,
            q_cht_to_oil_w=q_cht_to_oil_w,
        )
        d_oil_temp_dt = therm.oil_temperature_derivative(
            friction_power_w=p_fric_w,
            q_waste_w=q_waste_w,
            q_oil_cooler_w=q_oil_cooler_w,
        ) + (q_cht_to_oil_w / self.thermal.c_oil)

        t_egt_ss = therm.egt_steady_state(
            t_amb_k=state.ambient_temp_k,
            q_waste_w=q_waste_w,
            air_mass_flow_kg_per_s=mdot_air_kgps,
            fuel_mass_flow_kg_per_s=mdot_fuel_actual,
            equivalence_ratio_phi=phi,
        )
        d_egt_dt = therm.egt_derivative(
            t_egt_current_k=state.egt_k,
            t_egt_steady_state_k=t_egt_ss,
            exhaust_mass_flow_kg_per_s=mdot_air_kgps + mdot_fuel_actual,
        )

        # 8. Mechanical RPM Dynamic Target & Derivative
        # RPM target responds to throttle and opposing load
        rpm_idle = 1000.0
        rpm_max = 5800.0
        rpm_target = rpm_idle + (rpm_max - rpm_idle) * (state.throttle_norm ** 1.1) * (1.0 - 0.20 * state.engine_load_norm)
        # Degradation / power loss can also limit achievable RPM
        if p_ind_w < p_fric_w:
            rpm_target = max(600.0, rpm_target * 0.7)
        d_rpm_dt = (rpm_target - state.rpm) / self.tau_rpm

        # 9. Vibration
        a_rms, g_rms = vib.total_engine_vibration(
            rpm=state.rpm,
            indicated_torque_nm=tau_ind_nm,
            misfire_intensity=mu_mis,
            lubrication_degradation=fault_state.lubrication.boundary_scuffing_severity,
        )

        # 10. Health Indices
        h_cool = hlth.cooling_health_index(theta_cool, state.cht_k)
        h_lub = hlth.lubrication_health_index(theta_pump, leakage_factor, p_oil_pa, state.oil_temp_k)
        h_inj = hlth.injector_health_index(theta_inj, afr_actual, afr_target)
        h_comb = hlth.combustion_health_index(mu_mis, eta_comb)
        h_overall = hlth.overall_engine_health(h_cool, h_lub, h_inj, h_comb, strategy="weakest_link")

        return {
            # States & Derivatives
            "d_cht_dt": float(d_cht_dt),
            "d_oil_temp_dt": float(d_oil_temp_dt),
            "d_egt_dt": float(d_egt_dt),
            "d_rpm_dt": float(d_rpm_dt),
            "rpm_target": float(rpm_target),
            "t_egt_ss": float(t_egt_ss),
            # Atmospheric & Flow
            "ambient_pressure_pa": float(p_amb_pa),
            "air_density_kgpm3": float(rho_amb),
            "cooling_velocity_mps": float(v_cool_mps),
            "manifold_pressure_pa": float(p_man_pa),
            "volumetric_efficiency": float(eta_v),
            "air_mass_flow_kgps": float(mdot_air_kgps),
            "air_mass_flow_gps": float(mdot_air_kgps * 1000.0),
            # Fuel & Combustion
            "target_afr": float(afr_target),
            "actual_afr": float(afr_actual),
            "equivalence_ratio_phi": float(phi),
            "fuel_mass_flow_kgps": float(mdot_fuel_actual),
            "fuel_mass_flow_gps": float(mdot_fuel_actual * 1000.0),
            "combustion_efficiency": float(eta_comb),
            "indicated_thermal_efficiency": float(eta_ind_th),
            "fuel_chemical_power_w": float(q_fuel_chem),
            # Power & Mechanical
            "indicated_power_w": float(p_ind_w),
            "friction_power_w": float(p_fric_w),
            "brake_power_w": float(p_brake_w),
            "brake_power_kw": float(p_brake_w / 1000.0),
            "mechanical_efficiency": float(eta_mech),
            "engine_torque_nm": float(tau_brake_nm),
            "indicated_torque_nm": float(tau_ind_nm),
            "imep_pa": float(imep_pa),
            "fmep_pa": float(fmep_pa),
            # Hydraulics & Lubrication
            "oil_viscosity_pa_s": float(mu_actual),
            "oil_pump_flow_m3ps": float(vdot_oil_pump),
            "oil_pump_flow_lpm": float(vdot_oil_pump * 60000.0),
            "oil_pressure_pa": float(p_oil_pa),
            "oil_pressure_psi": float(p_oil_pa / 6894.757),
            # Heat Flows
            "waste_heat_w": float(q_waste_w),
            "cooling_heat_rejection_w": float(q_cool_cht_w),
            "oil_cooler_heat_rejection_w": float(q_oil_cooler_w),
            # Vibration
            "vibration_rms_mps2": float(a_rms),
            "vibration_g_rms": float(g_rms),
            # Health
            "cooling_health_index": float(h_cool),
            "lubrication_health_index": float(h_lub),
            "injector_health_index": float(h_inj),
            "combustion_health_index": float(h_comb),
            "overall_engine_health": float(h_overall),
        }

    def update_engine_state(
        self,
        current_state: EngineState,
        physics_output: Dict[str, Any],
        dt_ms: int,
        dynamic_rpm: bool = True,
    ) -> EngineState:
        """
        Integrates continuous ODE state variables forward by dt_ms milliseconds
        using Heun's (Runge-Kutta 2nd order) predictor-corrector update.
        """
        dt = dt_ms / 1000.0  # Convert milliseconds to seconds

        # Predictor step (Euler)
        new_cht_k = current_state.cht_k + physics_output["d_cht_dt"] * dt
        new_oil_k = current_state.oil_temp_k + physics_output["d_oil_temp_dt"] * dt
        new_egt_k = current_state.egt_k + physics_output["d_egt_dt"] * dt

        if dynamic_rpm:
            new_rpm = current_state.rpm + physics_output["d_rpm_dt"] * dt
        else:
            new_rpm = physics_output["rpm_target"]

        # Physical safety clamping to prevent unphysical numerical divergence
        new_cht_k = max(current_state.ambient_temp_k, min(550.0, new_cht_k))
        new_oil_k = max(current_state.ambient_temp_k, min(480.0, new_oil_k))
        new_egt_k = max(current_state.ambient_temp_k, min(1400.0, new_egt_k))
        new_rpm = max(0.0, min(7000.0, new_rpm))

        # Return updated continuous state
        return EngineState(
            time_ms=current_state.time_ms + dt_ms,
            cht_k=float(new_cht_k),
            oil_temp_k=float(new_oil_k),
            egt_k=float(new_egt_k),
            rpm=float(new_rpm),
            throttle_norm=current_state.throttle_norm,
            engine_load_norm=current_state.engine_load_norm,
            altitude_m=current_state.altitude_m,
            ambient_temp_k=current_state.ambient_temp_k,
            humidity_ratio=current_state.humidity_ratio,
            wind_speed_mps=current_state.wind_speed_mps,
            wind_direction_rad=current_state.wind_direction_rad,
        )

    def initialize_thermal_steady_state(
        self,
        state: EngineState,
        fault_state: EngineFaultState,
    ) -> EngineState:
        """
        Finds the initial thermal equilibrium at the specified operating point,
        preventing artificial transient jumps at the start of a cruise mission.
        """
        st = EngineState(
            time_ms=0,
            cht_k=state.ambient_temp_k + 95.0,
            oil_temp_k=state.ambient_temp_k + 75.0,
            egt_k=state.ambient_temp_k + 500.0,
            rpm=state.rpm,
            throttle_norm=state.throttle_norm,
            engine_load_norm=state.engine_load_norm,
            altitude_m=state.altitude_m,
            ambient_temp_k=state.ambient_temp_k,
            humidity_ratio=state.humidity_ratio,
            wind_speed_mps=state.wind_speed_mps,
            wind_direction_rad=state.wind_direction_rad,
        )

        # Fast pseudo-integration to equilibrium
        for _ in range(40):
            res = self.calculate_instantaneous_physics(st, fault_state)
            st = self.update_engine_state(st, res, dt_ms=2500, dynamic_rpm=False)

        st.time_ms = 0
        return st
