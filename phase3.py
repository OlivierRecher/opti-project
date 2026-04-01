import argparse
import os
import random
import re

import gurobipy as gp
from gurobipy import GRB

from salbp_parser import parse_alb_file


def solve_m1(J, K, P, t, time_limit=1800, output_flag=0):
    """Solve SALBP-2 baseline model M1 (minimize CT)."""
    model = gp.Model("M1_CT")

    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")
    s = model.addVars(J, vtype=GRB.INTEGER, lb=min(K), ub=max(K), name="s")
    CT = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="CT")

    model.addConstrs((x.sum(j, "*") == 1 for j in J), name="assign")
    model.addConstrs(
        (s[j] == gp.quicksum(k * x[j, k] for k in K) for j in J),
        name="server_index",
    )
    model.addConstrs((s[i] <= s[j] for i, j in P), name="precedence")
    model.addConstrs(
        (gp.quicksum(t[j] * x[j, k] for j in J) <= CT for k in K),
        name="bottleneck",
    )

    model.setObjective(CT, GRB.MINIMIZE)
    model.setParam("OutputFlag", output_flag)
    model.setParam("TimeLimit", time_limit)
    model.optimize()
    return model, x, s, CT


def solve_evnfp_mccormick(J, K, P, t, e, cbar, time_limit=1800, output_flag=0):
    """
    Solve EVNF-P exactly as MILP with McCormick linearization.

    z[j,k] = E[k] * x[j,k], with x binary and E[k] bounded in [Emin, Emax_bound].
    """
    model = gp.Model("EVNFP_McCormick")

    emin = min(e.values())
    emax_bound = max(e.values())

    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")
    s = model.addVars(J, vtype=GRB.INTEGER, lb=min(K), ub=max(K), name="s")
    L = model.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name="L")
    E = model.addVars(K, vtype=GRB.CONTINUOUS, lb=emin, ub=emax_bound, name="E")
    Emax = model.addVar(vtype=GRB.CONTINUOUS, lb=emin, ub=emax_bound, name="Emax")
    z = model.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0.0, ub=emax_bound, name="z")

    model.addConstrs((x.sum(j, "*") == 1 for j in J), name="assign")
    model.addConstrs(
        (s[j] == gp.quicksum(k * x[j, k] for k in K) for j in J),
        name="server_index",
    )
    model.addConstrs((s[i] <= s[j] for i, j in P), name="precedence")

    model.addConstrs(
        (L[k] == gp.quicksum(t[j] * x[j, k] for j in J) for k in K),
        name="load_def",
    )
    model.addConstrs((L[k] <= cbar for k in K), name="latency_bound")

    # Exact McCormick envelope for z[j,k] = E[k] * x[j,k] since x is binary.
    model.addConstrs((z[j, k] >= emin * x[j, k] for j in J for k in K), name="mc1")
    model.addConstrs(
        (z[j, k] <= emax_bound * x[j, k] for j in J for k in K),
        name="mc2",
    )
    model.addConstrs(
        (z[j, k] >= E[k] - emax_bound * (1 - x[j, k]) for j in J for k in K),
        name="mc3",
    )
    model.addConstrs(
        (z[j, k] <= E[k] - emin * (1 - x[j, k]) for j in J for k in K),
        name="mc4",
    )

    model.addConstrs(
        (
            gp.quicksum(t[j] * z[j, k] for j in J)
            == gp.quicksum(e[j] * t[j] * x[j, k] for j in J)
            for k in K
        ),
        name="energy_ratio_link",
    )
    model.addConstrs((E[k] <= Emax for k in K), name="peak_energy")

    model.setObjective(Emax, GRB.MINIMIZE)
    model.setParam("OutputFlag", output_flag)
    model.setParam("TimeLimit", time_limit)
    model.optimize()
    return model, x, s, L, E, Emax


def get_placement(J, K, x):
    """Return placement map {server: [tasks]} from optimized x vars."""
    placement = {}
    for k in K:
        tasks = [j for j in J if x[j, k].X > 0.5]
        if tasks:
            placement[k] = tasks
    return placement


def compute_server_metrics(placement, t, e):
    """Compute load Lk and weighted energy Ek for each active server."""
    metrics = {}
    for k, tasks in placement.items():
        load = sum(t[j] for j in tasks)
        weighted = sum(e[j] * t[j] for j in tasks)
        energy = weighted / load if load > 0 else None
        metrics[k] = {"load": load, "weighted": weighted, "energy": energy}
    return metrics


def generate_energy_scores(J, rng):
    """Generate e_j in {1,...,15} with a deterministic RNG."""
    return {j: rng.randint(1, 15) for j in J}


def _status_label(status):
    if status == GRB.OPTIMAL:
        return "OPT"
    if status == GRB.TIME_LIMIT:
        return "TIME_LIMIT"
    if status == GRB.SUBOPTIMAL:
        return "SUBOPT"
    return str(status)


def _instance_sort_key(filename):
    match = re.search(r"n=(\d+)_(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 999, 999


def _print_solution_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_evnfp_solution(model, J, K, t, e, x, Emax):
    """Pretty print EVNF-P solution."""
    placement = get_placement(J, K, x)
    metrics = compute_server_metrics(placement, t, e)

    print(f"Status: {_status_label(model.status)}")
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        print(f"Emax: {Emax.X:.4f}")
        print(f"Runtime: {model.Runtime:.3f}s")
        if model.SolCount > 0 and model.status != GRB.OPTIMAL:
            print(f"Best bound: {model.ObjBound:.4f}")

    print("Placement et metriques par serveur:")
    for k in K:
        tasks = placement.get(k, [])
        if not tasks:
            print(f"  S{k}: []")
            continue
        one_based = [j + 1 for j in tasks]
        load = metrics[k]["load"]
        energy = metrics[k]["energy"]
        print(f"  S{k}: {one_based} | Lk={load:5.1f} | Ek={energy:7.4f}")


def evaluate_fixed_placement(placement, t, e):
    """Evaluate Emax for a fixed placement with 1-based task IDs."""
    emax = 0.0
    loads = {}
    energies = {}
    for k, tasks_1b in placement.items():
        tasks_0b = [j - 1 for j in tasks_1b]
        load = sum(t[j] for j in tasks_0b)
        weighted = sum(e[j] * t[j] for j in tasks_0b)
        energy = weighted / load if load > 0 else 0.0
        loads[k] = load
        energies[k] = energy
        emax = max(emax, energy)
    return loads, energies, emax


def validate_illustrative_example():
    """Reproduce the two target values from Section 2.15."""
    _print_solution_header("Validation numerique - instance illustrative (Section 2.15)")

    # Data from the statement, indexed from 0 in code.
    t = {0: 8, 1: 10, 2: 1, 3: 9, 4: 9, 5: 3, 6: 2, 7: 1}
    e = {0: 7, 1: 3, 2: 12, 3: 10, 4: 5, 5: 14, 6: 9, 7: 4}

    placement_m1 = {
        1: [2],
        2: [1, 4],
        3: [3, 5, 6, 7, 8],
    }
    placement_evnf = {
        1: [1],
        2: [2, 3, 4, 5, 6, 7],
        3: [8],
    }

    loads_a, energies_a, emax_a = evaluate_fixed_placement(placement_m1, t, e)
    loads_b, energies_b, emax_b = evaluate_fixed_placement(placement_evnf, t, e)

    print("Cible A (placement optimal M1):")
    print(f"  Charges: {loads_a}")
    print(
        "  Energies: "
        f"{{k: round(v, 4) for k, v in energies_a.items()}} -> "
        f"{ {k: round(v, 4) for k, v in energies_a.items()} }"
    )
    print(f"  Emax calcule = {emax_a:.4f} (attendu ~8.5882)")

    print("\nCible B (placement optimal EVNF-P):")
    print(f"  Charges: {loads_b}")
    print(
        "  Energies: "
        f"{{k: round(v, 4) for k, v in energies_b.items()}} -> "
        f"{ {k: round(v, 4) for k, v in energies_b.items()} }"
    )
    print(f"  Emax calcule = {emax_b:.4f} (attendu 7.0000)")


def run_phase3_benchmark(
    instances_dir,
    seed=42,
    cbar_factor=1.05,
    time_limit_m1=1800,
    time_limit_evnf=1800,
    include_n100=True,
    include_n50=True,
):
    """Run M1 then EVNF-P on all benchmark instances and print a comparison table."""
    files = sorted(
        [
            f
            for f in os.listdir(instances_dir)
            if f.endswith(".alb")
            and (include_n100 or "n=100" not in f)
            and (include_n50 or "n=50" not in f)
        ],
        key=_instance_sort_key,
    )

    _print_solution_header("Benchmark Phase 3 - EVNF-P vs Phase 1")
    print(f"Instances: {len(files)} | seed energetique={seed} | Cbar factor={cbar_factor}")

    rng = random.Random(seed)
    results = []

    for filename in files:
        path = os.path.join(instances_dir, filename)
        J, K, P, t, ct_limit, order_strength = parse_alb_file(path)

        e = generate_energy_scores(J, rng)

        m1_model, m1_x, _, m1_CT = solve_m1(
            J,
            K,
            P,
            t,
            time_limit=time_limit_m1,
            output_flag=0,
        )

        ct_star = m1_CT.X if m1_model.SolCount > 0 else None
        if ct_star is not None:
            cbar = cbar_factor * ct_star
        else:
            cbar = ct_limit

        ev_model, ev_x, _, _, _, ev_Emax = solve_evnfp_mccormick(
            J,
            K,
            P,
            t,
            e,
            cbar,
            time_limit=time_limit_evnf,
            output_flag=0,
        )

        placement_m1 = get_placement(J, K, m1_x) if m1_model.SolCount > 0 else {}
        m1_metrics = compute_server_metrics(placement_m1, t, e) if placement_m1 else {}
        m1_emax = (
            max(v["energy"] for v in m1_metrics.values()) if m1_metrics else None
        )

        ev_placement = get_placement(J, K, ev_x) if ev_model.SolCount > 0 else {}
        ev_metrics = compute_server_metrics(ev_placement, t, e) if ev_placement else {}
        ev_emax_check = (
            max(v["energy"] for v in ev_metrics.values()) if ev_metrics else None
        )

        row = {
            "instance": filename,
            "n": len(J),
            "m": len(K),
            "os": order_strength,
            "ct_star": ct_star,
            "cbar": cbar,
            "m1_emax": m1_emax,
            "ev_emax": (ev_Emax.X if ev_model.SolCount > 0 else None),
            "ev_emax_check": ev_emax_check,
            "m1_runtime": m1_model.Runtime,
            "ev_runtime": ev_model.Runtime,
            "m1_status": _status_label(m1_model.status),
            "ev_status": _status_label(ev_model.status),
        }
        results.append(row)

        ct_str = f"{ct_star:.2f}" if ct_star is not None else "N/A"
        m1_e = f"{m1_emax:.3f}" if m1_emax is not None else "N/A"
        ev_e = f"{row['ev_emax']:.3f}" if row["ev_emax"] is not None else "N/A"
        print(
            f"{filename:<24} n={len(J):3d} m={len(K):2d} "
            f"OS={order_strength:.3f} CT*={ct_str:>7} Cbar={cbar:7.2f} "
            f"Emax(M1)={m1_e:>7} Emax(EV)={ev_e:>7} "
            f"t_M1={m1_model.Runtime:7.2f}s t_EV={ev_model.Runtime:7.2f}s "
            f"[{row['m1_status']}/{row['ev_status']}]"
        )

    print("-" * 88)
    _print_benchmark_summary(results)
    return results


def _print_benchmark_summary(results):
    """Print aggregate stats for quick report integration."""
    by_n = sorted({r["n"] for r in results})
    for n in by_n:
        subset = [r for r in results if r["n"] == n]
        gains = []
        ev_vals = []
        m1_vals = []
        for r in subset:
            if r["m1_emax"] is not None and r["ev_emax"] is not None:
                gains.append(r["m1_emax"] - r["ev_emax"])
                ev_vals.append(r["ev_emax"])
                m1_vals.append(r["m1_emax"])

        print(f"n={n} ({len(subset)} instances)")
        if gains:
            print(f"  Emax moyen M1      : {_mean(m1_vals):.4f}")
            print(f"  Emax moyen EVNF-P  : {_mean(ev_vals):.4f}")
            print(f"  Gain moyen (M1-EV) : {_mean(gains):.4f}")
        print(f"  Runtime M1 moyen   : {_mean([r['m1_runtime'] for r in subset]):.3f}s")
        print(f"  Runtime EV moyen   : {_mean([r['ev_runtime'] for r in subset]):.3f}s")


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Phase 3 EVNF-P")
    parser.add_argument("--instances-dir", default="./Instances")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cbar-factor", type=float, default=1.05)
    parser.add_argument("--time-limit-m1", type=int, default=1800)
    parser.add_argument("--time-limit-ev", type=int, default=1800)
    parser.add_argument("--exclude-n100", action="store_true")
    parser.add_argument("--exclude-n50-n100", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    print("PHASE 3 - EVNF-P (McCormick exact MILP)")
    print("Regle de generation des scores energetiques: uniforme {1..15}, seed=42")

    validate_illustrative_example()

    if not args.skip_benchmark:
        exclude_large = args.exclude_n50_n100 or args.exclude_n100
        run_phase3_benchmark(
            instances_dir=args.instances_dir,
            seed=args.seed,
            cbar_factor=args.cbar_factor,
            time_limit_m1=args.time_limit_m1,
            time_limit_evnf=args.time_limit_ev,
            include_n100=not exclude_large,
            include_n50=not args.exclude_n50_n100,
        )


if __name__ == "__main__":
    main()
