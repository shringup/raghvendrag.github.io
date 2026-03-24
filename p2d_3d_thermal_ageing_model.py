#!/usr/bin/env python3
"""
Physics-inspired reduced P2D electrochemical + pseudo-3D thermal + ageing model
for Samsung INR18650-29E.

Standard-library only script (no numpy/matplotlib dependency), designed to run in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from typing import Dict, List

R_GAS = 8.314462618


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("nan")
    return math.sqrt(sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n)


def linear_interp(xp: List[float], fp: List[float], x: float) -> float:
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    lo, hi = 0, len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    a = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] * (1 - a) + fp[hi] * a


@dataclass
class CellGeometry:
    radius_m: float = 9.3e-3
    height_m: float = 65.0e-3
    mass_kg: float = 0.048
    cp_j_kgk: float = 1000.0

    @property
    def volume_m3(self) -> float:
        return math.pi * self.radius_m**2 * self.height_m


@dataclass
class ElectrochemParams:
    q_nominal_ah: float = 2.9
    v_min: float = 2.5
    r_ohm_ref: float = 0.040
    r_ct_ref: float = 0.020
    ea_ohm_j_mol: float = 18000.0
    ea_ct_j_mol: float = 24000.0
    dUdT_v_k: float = 0.00015


@dataclass
class AgeingParams:
    k_sei_ref: float = 2.0e-12
    ea_sei_j_mol: float = 35000.0
    alpha_sei: float = 0.6
    k_gas_ref: float = 3.0e-9
    eta_gas_threshold_v: float = 0.08
    shell_compliance_pa_per_m3: float = 2.8e11
    headspace_m3: float = 2.0e-7


@dataclass
class ModelOptions:
    nx: int = 4
    ny: int = 4
    nz: int = 6
    dt_s: float = 1.0


class ReducedP2DElectrochem:
    def __init__(self, p: ElectrochemParams):
        self.p = p

    @staticmethod
    def ocv_n(sto: float) -> float:
        x = clamp(sto, 1e-5, 1 - 1e-5)
        return 0.12 + 0.9 * x + 0.1 * math.tanh((x - 0.2) / 0.05)

    @staticmethod
    def ocv_p(sto: float) -> float:
        y = clamp(sto, 1e-5, 1 - 1e-5)
        return 4.25 - 0.9 * y + 0.06 * math.tanh((y - 0.5) / 0.08)

    def r_ohm(self, temp_k: float, q_loss_frac: float) -> float:
        base = self.p.r_ohm_ref * math.exp(self.p.ea_ohm_j_mol / R_GAS * (1 / temp_k - 1 / 298.15))
        return base * (1 + 1.8 * q_loss_frac)

    def r_ct(self, temp_k: float, q_loss_frac: float) -> float:
        base = self.p.r_ct_ref * math.exp(self.p.ea_ct_j_mol / R_GAS * (1 / temp_k - 1 / 298.15))
        return base * (1 + 2.1 * q_loss_frac)

    def step(self, soc: float, current_a: float, temp_k: float, q_loss_frac: float) -> Dict[str, float]:
        soc = clamp(soc, 1e-5, 0.9999)
        ocv = self.ocv_p(1 - soc) - self.ocv_n(soc)

        r_ohm = self.r_ohm(temp_k, q_loss_frac)
        r_ct = self.r_ct(temp_k, q_loss_frac)

        i0_scale = max(1e-4, soc * (1 - soc))
        eta_act = current_a * r_ct / (0.6 + 8.0 * i0_scale)
        eta_conc = 0.015 * current_a * (1 + 2.0 * abs(0.5 - soc))

        voltage = ocv - current_a * r_ohm - eta_act - eta_conc

        q_ohmic = current_a * current_a * r_ohm
        q_irrev = current_a * (eta_act + eta_conc)
        q_rev = current_a * temp_k * self.p.dUdT_v_k

        return {
            "voltage": voltage,
            "eta_side": eta_act + eta_conc,
            "q_ohmic": q_ohmic,
            "q_irrev": q_irrev,
            "q_rev": q_rev,
            "q_total": q_ohmic + q_irrev + q_rev,
            "r_ohm": r_ohm,
        }


class ThermalPseudo3D:
    """Stable pseudo-3D lumped finite-volume thermal network."""

    def __init__(self, g: CellGeometry, o: ModelOptions):
        self.g, self.o = g, o
        self.nx, self.ny, self.nz = o.nx, o.ny, o.nz
        self.dx = (2 * g.radius_m) / o.nx
        self.dy = (2 * g.radius_m) / o.ny
        self.dz = g.height_m / o.nz

        self.kx = 0.35
        self.ky = 0.35
        self.kz = 22.0

        self.node_volume = g.volume_m3 / (o.nx * o.ny * o.nz)
        self.node_heatcap = g.mass_kg * g.cp_j_kgk / (o.nx * o.ny * o.nz)

        self.T = [[[298.15 for _ in range(self.nz)] for _ in range(self.ny)] for _ in range(self.nx)]

    def reset(self, t_k: float) -> None:
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    self.T[i][j][k] = t_k

    def mean_temp(self) -> float:
        s = 0.0
        n = self.nx * self.ny * self.nz
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    s += self.T[i][j][k]
        return s / n

    def max_temp(self) -> float:
        m = -1e9
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    m = max(m, self.T[i][j][k])
        return m

    def step(self, q_gen_w: float, h_w_m2k: float, t_amb_k: float, dt: float) -> None:
        q_per_node = q_gen_w / (self.nx * self.ny * self.nz)
        Tn = [[[self.T[i][j][k] for k in range(self.nz)] for j in range(self.ny)] for i in range(self.nx)]

        ax = self.dy * self.dz
        ay = self.dx * self.dz
        az = self.dx * self.dy

        gx = self.kx * ax / self.dx
        gy = self.ky * ay / self.dy
        gz = self.kz * az / self.dz

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    Tc = self.T[i][j][k]
                    q_cond = 0.0
                    q_conv = 0.0

                    for di, dj, dk, gcond, area in [
                        (-1, 0, 0, gx, ax), (1, 0, 0, gx, ax),
                        (0, -1, 0, gy, ay), (0, 1, 0, gy, ay),
                        (0, 0, -1, gz, az), (0, 0, 1, gz, az),
                    ]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                            q_cond += gcond * (self.T[ni][nj][nk] - Tc)
                        else:
                            q_conv += h_w_m2k * area * (t_amb_k - Tc)

                    dT = (q_cond + q_conv + q_per_node) * dt / self.node_heatcap
                    Tn[i][j][k] = Tc + dT

        self.T = Tn


class AgeingSubmodel:
    def __init__(self, p: AgeingParams, echem: ElectrochemParams):
        self.p, self.echem = p, echem
        self.q_loss_ah = 0.0
        self.gas_mol = 0.0

    def step(self, current_a: float, eta_side_v: float, temp_k: float, dt: float) -> Dict[str, float]:
        arr = math.exp(-self.p.ea_sei_j_mol / (R_GAS * temp_k))
        eta_factor = math.exp(self.p.alpha_sei * abs(eta_side_v) / max(0.01, 0.026 * temp_k / 298.15))
        dsei_dt = self.p.k_sei_ref * arr * eta_factor * (1 + abs(current_a) / 3.0)

        i_side = 1.2 * abs(current_a) * min(0.45, dsei_dt / 1e-12)
        self.q_loss_ah += i_side * dt / 3600.0

        eta_excess = max(0.0, abs(eta_side_v) - self.p.eta_gas_threshold_v)
        temp_factor = math.exp(clamp((temp_k - 298.15) / 18.0, -6.0, 6.0))
        self.gas_mol += self.p.k_gas_ref * eta_excess * temp_factor * dt

        q_loss_frac = clamp(self.q_loss_ah / self.echem.q_nominal_ah, 0.0, 0.35)

        pressure_pa = (self.gas_mol * R_GAS * temp_k) / self.p.headspace_m3
        expanded_volume = self.gas_mol * 24.5e-3
        if expanded_volume > self.p.headspace_m3:
            pressure_pa += self.p.shell_compliance_pa_per_m3 * (expanded_volume - self.p.headspace_m3)

        return {"q_loss_frac": q_loss_frac, "pressure_pa": pressure_pa}


class CoupledBatteryModel:
    def __init__(self):
        self.geom = CellGeometry()
        self.ep = ElectrochemParams()
        self.ap = AgeingParams()
        self.opt = ModelOptions()
        self.echem = ReducedP2DElectrochem(self.ep)
        self.th = ThermalPseudo3D(self.geom, self.opt)
        self.age = AgeingSubmodel(self.ap, self.ep)
        self.soc = 0.99

    def simulate_discharge(self, c_rate: float, t_amb_c: float, n_cycles: int, h_w_m2k: float = 10.0) -> Dict[str, List[float]]:
        current = c_rate * self.ep.q_nominal_ah
        dt = self.opt.dt_s
        t_amb_k = t_amb_c + 273.15
        t_global = 0.0

        rec: Dict[str, List[float]] = {k: [] for k in [
            "time_s", "cycle", "soc", "voltage_v", "t_mean_c", "t_max_c", "q_ohmic_w", "q_irrev_w", "q_rev_w",
            "q_total_w", "capacity_ah", "gas_mol", "pressure_kpa", "r_ohm"
        ]}

        for cyc in range(1, n_cycles + 1):
            self.soc = 0.99
            q_avail = max(0.5, self.ep.q_nominal_ah - self.age.q_loss_ah)

            while True:
                temp_k = self.th.mean_temp()
                q_loss_frac = clamp(self.age.q_loss_ah / self.ep.q_nominal_ah, 0.0, 0.35)

                ec = self.echem.step(self.soc, current, temp_k, q_loss_frac)
                age = self.age.step(current, ec["eta_side"], temp_k, dt)
                self.th.step(ec["q_total"], h_w_m2k, t_amb_k, dt)

                self.soc -= current * dt / 3600.0 / q_avail
                t_global += dt

                rec["time_s"].append(t_global)
                rec["cycle"].append(float(cyc))
                rec["soc"].append(self.soc)
                rec["voltage_v"].append(ec["voltage"])
                rec["t_mean_c"].append(self.th.mean_temp() - 273.15)
                rec["t_max_c"].append(self.th.max_temp() - 273.15)
                rec["q_ohmic_w"].append(ec["q_ohmic"])
                rec["q_irrev_w"].append(ec["q_irrev"])
                rec["q_rev_w"].append(ec["q_rev"])
                rec["q_total_w"].append(ec["q_total"])
                rec["capacity_ah"].append(self.ep.q_nominal_ah - self.age.q_loss_ah)
                rec["gas_mol"].append(self.age.gas_mol)
                rec["pressure_kpa"].append(age["pressure_pa"] / 1000.0)
                rec["r_ohm"].append(ec["r_ohm"])

                if ec["voltage"] <= self.ep.v_min or self.soc <= 0.01:
                    break

            self.th.reset(t_amb_k)

        return rec


def validate_against_gupta_29e(sim: Dict[str, List[float]]) -> Dict[str, float]:
    # Digitized-checkpoint scaffold inspired by IIT Delhi publication figures.
    exp_t = [0, 600, 1200, 1800, 2400, 3000, 3400]
    exp_v = [4.18, 3.95, 3.79, 3.67, 3.50, 3.20, 2.82]
    exp_tc = [25.0, 28.2, 31.6, 35.9, 41.5, 48.0, 53.8]

    sim_t0 = sim["time_s"][0]
    sim_t = [t - sim_t0 for t in sim["time_s"]]

    pred_v = [linear_interp(sim_t, sim["voltage_v"], t) for t in exp_t]
    pred_tc = [linear_interp(sim_t, sim["t_max_c"], t) for t in exp_t]

    return {
        "rmse_voltage_v": rmse(exp_v, pred_v),
        "rmse_temperature_c": rmse(exp_tc, pred_tc),
        "max_temp_rise_c": max(sim["t_max_c"]) - 25.0,
    }


def write_csv(path: str, data: Dict[str, List[float]]) -> None:
    keys = list(data.keys())
    n = len(data[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([data[k][i] for k in keys])


def main() -> None:
    m_ref = CoupledBatteryModel()
    sim_25 = m_ref.simulate_discharge(c_rate=1.0, t_amb_c=25.0, n_cycles=1, h_w_m2k=10.0)
    validation = validate_against_gupta_29e(sim_25)

    m_hot = CoupledBatteryModel()
    sim_45 = m_hot.simulate_discharge(c_rate=1.0, t_amb_c=45.0, n_cycles=40, h_w_m2k=8.0)

    print("=== Validation scaffold (Gupta IIT Delhi, Samsung 29E) ===")
    for k, v in validation.items():
        print(f"{k:>22s}: {v:.4f}")

    final_capacity = sim_45["capacity_ah"][-1]
    fade_pct = (1 - final_capacity / 2.9) * 100
    print("\n=== Ageing summary @45°C ambient (40 cycles) ===")
    print(f"Final capacity [Ah]    : {final_capacity:.3f}")
    print(f"Capacity fade [%]      : {fade_pct:.2f}")
    print(f"Final gas [mmol]       : {sim_45['gas_mol'][-1] * 1e3:.4f}")
    print(f"Final pressure [kPa]   : {sim_45['pressure_kpa'][-1]:.2f}")
    print(f"Final ohmic R [mOhm]   : {sim_45['r_ohm'][-1] * 1e3:.2f}")
    print(f"Peak Tmax [°C]         : {max(sim_45['t_max_c']):.2f}")

    write_csv("simulation_25C.csv", sim_25)
    write_csv("simulation_45C_40cycles.csv", sim_45)
    print("Saved: simulation_25C.csv")
    print("Saved: simulation_45C_40cycles.csv")


if __name__ == "__main__":
    main()
