"""
Simulation script for Zr-doped AlN heater ceramics using the DEVSIM API.
"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    from devsim import (
        add_1d_interface,
        add_1d_mesh_line,
        add_1d_contact,
        create_1d_mesh,
        create_device,
        create_gmsh_mesh,
        create_region,
        delete_device,
        get_contact_charge,
        get_contact_current,
        get_node_model_values,
        node_model,
        element_model,
        equation,
        finalize_mesh,
        get_parameter,
        set_parameter,
        solve,
        add_drift_diffusion,
        set_contact_node_model,
        set_node_values,
        get_contact_current,
        get_node_model_list,
        get_node_model_values,
    )
except ImportError as exc:  # pragma: no cover - safety for environments without DEVSIM
    raise SystemExit(
        "The devsim module is required to run this script. "
        "Install DEVSIM (https://devsim.org/) before executing." 
    ) from exc


# -----------------------------------------------------------------------------
# Data classes describing physical configuration
# -----------------------------------------------------------------------------


@dataclass
class TrapParameters:
    density_cm3: float = 1e15
    energy_below_ec_ev: float = 0.85
    capture_cross_section_cm2: float = 1e-14
    alpha: float = 0.0  # Homotopy ramp factor


@dataclass
class DopingProfile:
    zr_mol_percent: float
    activation_fraction: float = 1e-7
    max_active_cm3: float = 5e18
    aln_atomic_density_cm3: float = 4.4e22  # Approximate atomic density of AlN

    def active_donor_concentration(self) -> float:
        raw_conc = (
            self.aln_atomic_density_cm3
            * (self.zr_mol_percent / 100.0)
            * self.activation_fraction
        )
        return min(raw_conc, self.max_active_cm3)


@dataclass
class SolverControl:
    rel_error: float = 1e-5
    abs_error: float = 1e-12
    max_iters: int = 50
    poisson_only_iters: int = 50
    trap_ramp_steps: int = 20
    trap_density_min: float = 1e14
    trap_density_max: float = 1e17


@dataclass
class SimulationConfig:
    thickness_cm: float = 1e-3
    mesh_points: int = 200
    temperature_start_k: float = 400.0
    temperature_final_k: float = 300.0
    temperature_steps: int = 5
    bias_stop_v: float = 50.0
    bias_initial_step_v: float = 2.0
    bias_growth_factor: float = 1.5
    bias_max_step_v: float = 10.0
    trap_parameters: TrapParameters = field(default_factory=TrapParameters)
    solver: SolverControl = field(default_factory=SolverControl)
    zr_mol_percents: Tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)


# -----------------------------------------------------------------------------
# Utility functions for device construction and physics models
# -----------------------------------------------------------------------------


def create_output_directory(base: str = "aln_zr_outputs") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def create_mesh(device: str, region: str, config: SimulationConfig) -> None:
    """Create a 1D vertical mesh spanning the AlN body."""
    create_1d_mesh(mesh="aln_mesh")
    add_1d_mesh_line(mesh="aln_mesh", pos=0.0, ps=1e-6)
    add_1d_mesh_line(mesh="aln_mesh", pos=config.thickness_cm, ps=config.thickness_cm / config.mesh_points)
    add_1d_interface(mesh="aln_mesh", name="aln_interface", pos=0.0)
    finalize_mesh(mesh="aln_mesh")
    create_device(mesh="aln_mesh", device=device)
    create_region(device=device, region=region, material="AlN", mesh="aln_mesh")
    add_1d_contact(
        device=device,
        region=region,
        name="top_contact",
        material="metal",
        location=config.thickness_cm,
    )
    add_1d_contact(
        device=device,
        region=region,
        name="bottom_contact",
        material="metal",
        location=0.0,
    )


def configure_material_parameters(device: str, region: str, config: SimulationConfig) -> None:
    """Set material constants for AlN and global parameters."""
    q = 1.60217662e-19
    k_b = 8.617333262145e-5  # eV/K
    eps0 = 8.854187817e-14  # F/cm

    set_parameter(device=device, name="Permittivity", value=eps0 * 8.5)
    set_parameter(device=device, region=region, name="dielectric_constant", value=8.5)
    set_parameter(device=device, region=region, name="relative_permittivity", value=8.5)
    set_parameter(device=device, region=region, name="IntrinsicDensity", value=1e6)
    set_parameter(device=device, region=region, name="Nc", value=2.3e18)
    set_parameter(device=device, region=region, name="Nv", value=1.8e19)
    set_parameter(device=device, name="q", value=q)
    set_parameter(device=device, name="k_B", value=k_b)
    set_parameter(device=device, name="T", value=config.temperature_start_k)
    set_parameter(device=device, name="Eg", value=6.2)
    set_parameter(device=device, region=region, name="ElectronAffinity", value=2.1)
    set_parameter(device=device, name="n_min", value=1e5)
    set_parameter(device=device, name="p_min", value=1e5)
    set_parameter(device=device, region=region, name="trap_alpha", value=0.0)
    set_parameter(device=device, region=region, name="trap_density", value=config.trap_parameters.density_cm3)
    set_parameter(device=device, region=region, name="trap_cross_section", value=config.trap_parameters.capture_cross_section_cm2)
    set_parameter(device=device, region=region, name="trap_energy", value=config.trap_parameters.energy_below_ec_ev)


def add_initial_node_models(device: str, region: str) -> None:
    """Add baseline electrostatic node models for the semiconductor."""
    node_model(device=device, region=region, name="Potential", equation="Potential")
    set_node_values(device=device, region=region, name="Potential", value=0.0)
    node_model(device=device, region=region, name="n", equation="IntrinsicDensity")
    node_model(device=device, region=region, name="p", equation="IntrinsicDensity")
    node_model(device=device, region=region, name="NetDoping", equation="0")


def add_carrier_models(device: str, region: str) -> None:
    """Define carrier densities with numerical clamping to maintain stability."""
    node_model(
        device=device,
        region=region,
        name="n_eff",
        equation="max(n, @n_min@)",
    )
    node_model(
        device=device,
        region=region,
        name="p_eff",
        equation="max(p, @p_min@)",
    )


def add_trap_models(device: str, region: str) -> None:
    """Define trap occupancy, charge, and recombination terms using SRH statistics."""
    # Thermal velocity approximation [cm/s]
    node_model(device=device, region=region, name="thermal_velocity", equation="1e7")

    # n1 and p1 for SRH statistics
    node_model(
        device=device,
        region=region,
        name="n1",
        equation="@Nc@ * exp(-(@trap_energy@)/(k_B*T))",
    )
    node_model(
        device=device,
        region=region,
        name="p1",
        equation="@Nv@ * exp(-(Eg-@trap_energy@)/(k_B*T))",
    )

    # Capture coefficients
    node_model(
        device=device,
        region=region,
        name="cn",
        equation="@trap_cross_section@ * thermal_velocity",
    )
    node_model(
        device=device,
        region=region,
        name="cp",
        equation="@trap_cross_section@ * thermal_velocity",
    )

    # Trap occupancy fraction (SRH occupancy)
    node_model(
        device=device,
        region=region,
        name="trap_occupancy",
        equation="(p1 + p_eff) / (p1 + p_eff + n_eff + n1)",
    )

    # Trap charge (negative when filled)
    node_model(
        device=device,
        region=region,
        name="trap_charge_density",
        equation="-q * @trap_alpha@ * @trap_density@ * trap_occupancy",
    )

    # SRH recombination rate (for reference)
    node_model(
        device=device,
        region=region,
        name="srh_denominator",
        equation="cn * (n_eff + n1) + cp * (p_eff + p1)",
    )
    node_model(
        device=device,
        region=region,
        name="U_SRH",
        equation="@trap_alpha@ * (cn * cp * (n_eff * p_eff - IntrinsicDensity^2)) / srh_denominator",
    )


def add_poisson_equation(device: str, region: str) -> None:
    """Add the Poisson equation with trap charge contribution."""
    node_model(
        device=device,
        region=region,
        name="NetCharge",
        equation="q * (p - n + NetDoping) + trap_charge_density",
    )
    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        equation="div(-Permittivity * grad(Potential)) + NetCharge",
    )


def add_drift_diffusion_equations(device: str, region: str) -> None:
    """Activate electron and hole continuity equations for drift-diffusion."""
    add_drift_diffusion(device=device, region=region, electrons=True, holes=True)


def set_contact_biases(device: str, region: str) -> None:
    set_contact_node_model(device=device, contact="top_contact", name="Potential@top_contact", equation="top_bias")
    set_contact_node_model(device=device, contact="top_contact", name="Potential:top_contact", equation="0")
    set_contact_node_model(device=device, contact="bottom_contact", name="Potential@bottom_contact", equation="0")
    set_contact_node_model(device=device, contact="bottom_contact", name="Potential:bottom_contact", equation="0")
    set_parameter(device=device, name="top_bias", value=0.0)


def apply_doping(device: str, region: str, doping_cm3: float) -> None:
    node_model(device=device, region=region, name="NetDoping", equation=f"{doping_cm3}")


def apply_temperature(device: str, temp_k: float) -> None:
    set_parameter(device=device, name="T", value=temp_k)


def run_poisson_presolve(device: str, solver: SolverControl) -> None:
    solve(
        type="dc",
        absolute_error=solver.abs_error,
        relative_error=solver.rel_error,
        maximum_iterations=solver.poisson_only_iters,
    )


def run_dd_solve(device: str, solver: SolverControl) -> None:
    solve(
        type="dc",
        absolute_error=solver.abs_error,
        relative_error=solver.rel_error,
        maximum_iterations=solver.max_iters,
    )


def ramp_temperature(device: str, config: SimulationConfig, solver: SolverControl) -> None:
    temps = [
        config.temperature_start_k
        - i * (config.temperature_start_k - config.temperature_final_k) / max(config.temperature_steps - 1, 1)
        for i in range(config.temperature_steps)
    ]
    for temp in temps:
        apply_temperature(device, temp)
        run_dd_solve(device, solver)


def ramp_traps(device: str, region: str, config: SimulationConfig) -> None:
    solver = config.solver
    for step in range(1, solver.trap_ramp_steps + 1):
        alpha = step / solver.trap_ramp_steps
        set_parameter(device=device, region=region, name="trap_alpha", value=alpha)
        try:
            run_dd_solve(device, solver)
        except Exception:
            # Fallback to Poisson stabilization before retrying
            run_poisson_presolve(device, solver)
            run_dd_solve(device, solver)


def sweep_bias(
    device: str,
    config: SimulationConfig,
    results: List[Dict[str, float]],
    trap_stats: List[Dict[str, float]],
    potential_profiles: List[Tuple[float, List[float], List[float]]],
    trap_profiles: List[Tuple[float, List[float], List[float]]],
) -> None:
    solver = config.solver
    bias = 0.0
    step = config.bias_initial_step_v
    last_successful_bias = 0.0

    while bias <= config.bias_stop_v + 1e-9:
        set_parameter(device=device, name="top_bias", value=bias)
        try:
            run_dd_solve(device, solver)
            last_successful_bias = bias
        except Exception:
            # Roll back and reduce step size
            bias = last_successful_bias
            step *= 0.5
            if step < 1e-3:
                raise RuntimeError("Bias sweep failed to converge even after step reduction")
            bias += step
            continue

        current = get_contact_current(device=device, contact="top_contact")
        electron_mean = sum(get_node_model_values(device=device, region="aln_body", name="n")) / config.mesh_points
        hole_mean = sum(get_node_model_values(device=device, region="aln_body", name="p")) / config.mesh_points
        trap_mean = sum(get_node_model_values(device=device, region="aln_body", name="trap_occupancy")) / config.mesh_points

        results.append(
            {
                "bias_V": bias,
                "current_Acm2": current,
                "mean_electrons_cm3": electron_mean,
                "mean_holes_cm3": hole_mean,
                "mean_trap_occupancy": trap_mean,
            }
        )

        y_positions = get_node_model_values(device=device, region="aln_body", name="Position")
        potentials = get_node_model_values(device=device, region="aln_body", name="Potential")
        trap_occ = get_node_model_values(device=device, region="aln_body", name="trap_occupancy")
        trap_charge = get_node_model_values(device=device, region="aln_body", name="trap_charge_density")

        potential_profiles.append((bias, y_positions, potentials))
        trap_profiles.append((bias, trap_occ, trap_charge))

        bias += step
        step = min(step * config.bias_growth_factor, config.bias_max_step_v)


def export_results(
    output_dir: str,
    doping_label: str,
    iv_data: List[Dict[str, float]],
    potential_profiles: List[Tuple[float, List[float], List[float]]],
    trap_profiles: List[Tuple[float, List[float], List[float]]],
) -> None:
    iv_path = os.path.join(output_dir, "iv_results.csv")
    file_exists = os.path.isfile(iv_path)
    with open(iv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "case",
                "bias_V",
                "current_Acm2",
                "mean_electrons_cm3",
                "mean_holes_cm3",
                "mean_trap_occupancy",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for row in iv_data:
            row_with_case = {"case": doping_label, **row}
            writer.writerow(row_with_case)

    for bias, positions, potentials in potential_profiles:
        fname = f"potential_profile_{doping_label}_{bias:.2f}V.csv"
        with open(os.path.join(output_dir, fname), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["position_cm", "potential_V"])
            for pos, pot in zip(positions, potentials):
                writer.writerow([pos, pot])

    for bias, occupancy, charge in trap_profiles:
        fname = f"trap_profile_{doping_label}_{bias:.2f}V.csv"
        with open(os.path.join(output_dir, fname), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["trap_occupancy", "trap_charge_Ccm3"])
            for occ, chg in zip(occupancy, charge):
                writer.writerow([occ, chg])


def plot_results(output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import glob

    # Plot IV curves
    iv_path = os.path.join(output_dir, "iv_results.csv")
    case_data: Dict[str, List[Tuple[float, float]]] = {}
    with open(iv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            case = row["case"]
            bias = float(row["bias_V"])
            current = float(row["current_Acm2"])
            case_data.setdefault(case, []).append((bias, current))

    plt.figure(figsize=(6, 4))
    for case, data in case_data.items():
        data.sort()
        biases, currents = zip(*data)
        plt.semilogy(biases, [abs(c) + 1e-30 for c in currents], label=f"Zr {case}")
    plt.xlabel("Bias (V)")
    plt.ylabel("Current density (A/cm$^2$)")
    plt.title("AlN Heater I-V Characteristics")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iv_curves.png"), dpi=300)
    plt.close()

    # Potential profiles
    potential_files = glob.glob(os.path.join(output_dir, "potential_profile_*.csv"))
    plt.figure(figsize=(6, 4))
    for pfile in potential_files:
        with open(pfile, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            positions = []
            potentials = []
            for row in reader:
                positions.append(float(row["position_cm"]))
                potentials.append(float(row["potential_V"]))
            label = os.path.basename(pfile).replace("potential_profile_", "").replace(".csv", "")
            plt.plot(positions, potentials, label=label)
    plt.xlabel("Depth (cm)")
    plt.ylabel("Potential (V)")
    plt.title("Potential profiles across AlN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "potential_profiles.png"), dpi=300)
    plt.close()

    # Trap profiles
    trap_files = glob.glob(os.path.join(output_dir, "trap_profile_*.csv"))
    plt.figure(figsize=(6, 4))
    for tfile in trap_files:
        with open(tfile, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            occupancy = []
            charge = []
            for row in reader:
                occupancy.append(float(row["trap_occupancy"]))
                charge.append(float(row["trap_charge_Ccm3"]))
            label = os.path.basename(tfile).replace("trap_profile_", "").replace(".csv", "")
            plt.plot(occupancy, charge, label=label)
    plt.xlabel("Trap occupancy")
    plt.ylabel("Trap charge (C/cm$^3$)")
    plt.title("Trap charge distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trap_profiles.png"), dpi=300)
    plt.close()


def run_case(device: str, region: str, config: SimulationConfig, doping_case: DopingProfile, output_dir: str) -> None:
    doping_cm3 = doping_case.active_donor_concentration()
    trap_density = config.trap_parameters.density_cm3 * max(doping_case.zr_mol_percent, 1e-6)
    trap_density = min(max(trap_density, config.solver.trap_density_min), config.solver.trap_density_max)

    # Device setup
    create_mesh(device, region, config)
    configure_material_parameters(device, region, config)
    add_initial_node_models(device, region)
    add_carrier_models(device, region)
    add_trap_models(device, region)
    apply_doping(device, region, doping_cm3)
    set_parameter(device=device, region=region, name="trap_density", value=trap_density)
    add_poisson_equation(device, region)
    add_drift_diffusion_equations(device, region)
    set_contact_biases(device, region)

    solver = config.solver
    run_poisson_presolve(device, solver)
    ramp_temperature(device, config, solver)
    ramp_traps(device, region, config)

    iv_data: List[Dict[str, float]] = []
    potential_profiles: List[Tuple[float, List[float], List[float]]] = []
    trap_profiles: List[Tuple[float, List[float], List[float]]] = []

    sweep_bias(device, config, iv_data, [], potential_profiles, trap_profiles)
    export_results(output_dir, f"{doping_case.zr_mol_percent:.1f}molpct", iv_data, potential_profiles, trap_profiles)
    delete_device(device=device)


def main() -> None:
    config = SimulationConfig()
    output_dir = create_output_directory()

    for zr_percent in config.zr_mol_percents:
        device_name = f"aln_device_{zr_percent:.1f}"
        doping_case = DopingProfile(zr_mol_percent=zr_percent)
        run_case(device_name, "aln_body", config, doping_case, output_dir)

    plot_results(output_dir)
    print("✅ DEVSIM simulation completed successfully")


if __name__ == "__main__":
    main()
