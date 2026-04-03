import argparse
import os
import random
import re
import math

import gurobipy as gp
from gurobipy import GRB

from salbp_parser import parse_alb_file


def solve_salbp(J, K, P, t):
    """
    Résout le SALBP-1 (minimisation du cycle time) par MILP.

    Retourne (model, x, s, CT) après optimisation.
    """
    model = gp.Model("VNF_Placement")

    x = model.addVars(J, K, vtype=GRB.BINARY, name="x")
    s = model.addVars(J, vtype=GRB.INTEGER, lb=1, ub=len(K), name="s")
    CT = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="CT")

    # Chaque tâche est assignée à exactement un serveur
    model.addConstrs((x.sum(j, '*') == 1 for j in J), name="unicite")

    # Lien entre x et s : s[j] = somme(k * x[j,k])
    model.addConstrs(
        (s[j] == gp.quicksum(k * x[j, k] for k in K) for j in J),
        name="def_s",
    )

    # Précédences
    model.addConstrs((s[i] <= s[j] for i, j in P), name="precedences")

    # Cycle time >= charge de chaque serveur
    model.addConstrs(
        (gp.quicksum(t[j] * x[j, k] for j in J) <= CT for k in K),
        name="bottleneck",
    )

    model.setObjective(CT, GRB.MINIMIZE)
    return model, x, s, CT


def get_placement(J, K, x):
    """Retourne {serveur: [liste de tâches]} à partir des variables x."""
    placement = {}
    for k in K:
        vnfs = [j for j in J if x[j, k].X > 0.5]
        if vnfs:
            placement[k] = vnfs
    return placement


def print_solution(model, J, K, P, t, x, s):
    """Affiche le détail d'une solution optimale."""
    print(f"\nCycle time optimal : {model.objVal:.1f}")

    placement = get_placement(J, K, x)
    print("\nPlacement des tâches :")
    for k, vnfs in placement.items():
        load = sum(t[j] for j in vnfs)
        print(f"  Serveur {k}: tâches {vnfs} (charge {load})")

    violated = [(i, j) for i, j in P if s[i].X > s[j].X]
    if violated:
        for i, j in violated:
            print(f"  VIOLATION {i}->{j} (s[{i}]={s[i].X}, s[{j}]={s[j].X})")
    else:
        print(f"\nPrécédences : {len(P)}/{len(P)} respectées")

    print(f"\nTemps de résolution : {model.Runtime:.3f}s")
    print(f"Borne inférieure : {model.ObjBound:.1f} (gap {model.MIPGap*100:.2f}%)")
    print(f"Nœuds Branch & Bound explorés : {int(model.NodeCount)}")

    print("\nDistribution de charge :")
    for k, vnfs in placement.items():
        load = sum(t[j] for j in vnfs)
        pct = load / model.objVal * 100 if model.objVal > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  Serveur {k}: {bar} {load:6.1f} / {model.objVal:.1f} ({pct:5.1f}%)")


def solve_single_instance(filepath):
    """Charge et résout une instance. Affiche les résultats."""
    J, K, P, t, ct_limit, order_strength = parse_alb_file(filepath)

    print("=" * 70)
    print(f"Instance : {os.path.basename(filepath)}")
    print(f"  n={len(J)}, m={len(K)}, précédences={len(P)}")
    print(f"  t_min={min(t.values())}, t_max={max(t.values())}, ct_limit={ct_limit}")
    print(f"  order_strength={order_strength}")
    print("=" * 70)

    model, x, s, CT = solve_salbp(J, K, P, t)
    model.optimize()

    print()
    if model.status == GRB.OPTIMAL:
        print_solution(model, J, K, P, t, x, s)
    else:
        print(f"Pas de solution optimale (status {model.status})")
    print("=" * 70)


def _instance_sort_key(filename):
    """Tri par taille (n) puis par index."""
    match = re.search(r"n=(\d+)_(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (999, 0)


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


def _print_solution_header(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


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


def validate_phase3_illustrative_example():
    """Reproduce the two target values from Section 2.15."""
    _print_solution_header("Validation numerique - instance illustrative (Section 2.15)")

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


def solve_m1_phase3(J, K, P, t, time_limit=1800, output_flag=0):
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


def run_phase3_benchmark(
    instances_dir,
    seed=42,
    cbar_factor=1.05,
    time_limit_m1=1800,
    time_limit_evnf=1800,
    include_n100=True,
    include_n50=True,
):
    """Run M1 then EVNF-P on benchmark instances and print a comparison table."""
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

        m1_model, m1_x, _, m1_CT = solve_m1_phase3(
            J,
            K,
            P,
            t,
            time_limit=time_limit_m1,
            output_flag=0,
        )

        ct_star = m1_CT.X if m1_model.SolCount > 0 else None
        cbar = cbar_factor * ct_star if ct_star is not None else ct_limit

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
        m1_emax = max(v["energy"] for v in m1_metrics.values()) if m1_metrics else None

        row = {
            "instance": filename,
            "n": len(J),
            "m": len(K),
            "os": order_strength,
            "ct_star": ct_star,
            "cbar": cbar,
            "m1_emax": m1_emax,
            "ev_emax": (ev_Emax.X if ev_model.SolCount > 0 else None),
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
    _print_phase3_summary(results)
    return results


def _print_phase3_summary(results):
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


def run_benchmark(instances_dir):
    """Résout toutes les instances et retourne les résultats."""
    files = sorted(
        [f for f in os.listdir(instances_dir)
         if f.endswith(".alb") and "n=100" not in f],
        key=_instance_sort_key,
    )
    print(f"\nBenchmark sur {len(files)} instances")
    print("=" * 100)

    results = []
    for filename in files:
        J, K, P, t, ct_limit, order_strength = parse_alb_file(
            os.path.join(instances_dir, filename)
        )

        model, x, s, CT = solve_salbp(J, K, P, t)
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", 1800)  # 30 minutes
        model.optimize()

        total_time = sum(t.values())
        optimal_CT = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else None
        
        # Calcul robuste du gap
        gap_opt = None
        if model.status == GRB.OPTIMAL:
            gap_opt = 0.0
        elif model.MIPGap is not None and model.MIPGap < float('inf'):
            if not math.isnan(model.MIPGap):
                gap_opt = model.MIPGap * 100

        results.append({
            "instance": filename,
            "n": len(J),
            "m": len(K),
            "order_strength": order_strength,
            "num_precedences": len(P),
            "total_time": total_time,
            "optimal_CT": optimal_CT,
            "runtime": model.Runtime,
            "lower_bound": model.ObjBound,
            "gap_optimality": gap_opt,
            "node_count": int(model.NodeCount),
        })

        if model.status == GRB.OPTIMAL:
            status = "OK"
        elif model.status == GRB.TIME_LIMIT:
            status = "TIMEOUT"
        else:
            status = "FAIL"
        ct_str = f"{optimal_CT:.0f}" if optimal_CT else "N/A"
        gap = results[-1]['gap_optimality']
        gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
        print(
            f"  {filename:<25} n={len(J):3d}  m={len(K):2d}  "
            f"os={order_strength:.3f}  CT={ct_str:>4s}  "
            f"time={model.Runtime:.3f}s  nodes={int(model.NodeCount):>6d}  gap={gap_str:>7s}  [{status}]"
        )

    print("=" * 100)
    return results


def print_stats(results):
    """Affiche les statistiques agrégées par taille."""
    sizes = sorted({r["n"] for r in results})

    for n in sizes:
        subset = [r for r in results if r["n"] == n]
        runtimes = [r["runtime"] for r in subset]
        ct_vals = [r["optimal_CT"] for r in subset if r["optimal_CT"]]
        gaps = [r["gap_optimality"] for r in subset if r["gap_optimality"] is not None]
        nodes = [r["node_count"] for r in subset if r["node_count"]]

        print(f"\nn={n} ({len(subset)} instances) :")
        print(f"  OS moyen        : {_mean([r['order_strength'] for r in subset]):.3f}")
        print(f"  Précédences moy : {_mean([r['num_precedences'] for r in subset]):.1f}")
        if ct_vals:
            print(f"  CT optimal moy  : {_mean(ct_vals):.0f}")
        print(f"  Runtime moyen   : {_mean(runtimes):.4f}s")
        if gaps:
            print(f"  Gap moyen (%)   : {_mean(gaps):.2f}%")
        if nodes:
            print(f"  Nœuds B&B moyen : {_mean(nodes):.0f}")


def save_runtime_plot(results, output_path="resol_time_analysis.png"):
    """Génère et sauvegarde le graphique temps de résolution."""
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    sizes = sorted({r["n"] for r in results})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for n in sizes:
        subset = [r for r in results if r["n"] == n]
        runtimes = [r["runtime"] for r in subset]

        ax1.scatter([n] * len(subset), runtimes, s=100, label=f"n={n}",
                    alpha=0.7, edgecolors="black")
        ax2.scatter([r["order_strength"] for r in subset], runtimes, s=100,
                    label=f"n={n}", alpha=0.7, edgecolors="black")

    ax1.set_xlabel("Nombre de tâches (n)")
    ax1.set_ylabel("Temps de résolution (s)")
    ax1.set_title("Temps vs taille")
    ax1.set_xticks(sizes)
    ax1.legend()

    ax2.set_xlabel("Order Strength")
    ax2.set_ylabel("Temps de résolution (s)")
    ax2.set_title("Temps vs Order Strength")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nGraphique sauvegardé : {output_path}")


def _mean(values):
    return sum(values) / len(values) if values else 0


def main():
    parser = argparse.ArgumentParser(description="SALBP + Phase 3 EVNF-P")
    parser.add_argument("--instances-dir", default="./Instances")
    parser.add_argument("--run-phase3", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cbar-factor", type=float, default=1.05)
    parser.add_argument("--time-limit-m1", type=int, default=1800)
    parser.add_argument("--time-limit-ev", type=int, default=1800)
    parser.add_argument("--exclude-n100", action="store_true")
    parser.add_argument("--exclude-n50-n100", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    if args.run_phase3:
        print("PHASE 3 - EVNF-P (McCormick exact MILP)")
        print("Regle de generation des scores energetiques: uniforme {1..15}, seed=42")
        validate_phase3_illustrative_example()

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
        return

    solve_single_instance("./Instances/instance_n=20_1.alb")
    results = run_benchmark(args.instances_dir)
    print_stats(results)
    try:
        save_runtime_plot(results)
    except ImportError:
        print("matplotlib non installé, graphique ignoré.")


if __name__ == "__main__":
    main()
