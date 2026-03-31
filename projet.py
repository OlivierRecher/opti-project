import os
import re

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

        results.append({
            "instance": filename,
            "n": len(J),
            "m": len(K),
            "order_strength": order_strength,
            "num_precedences": len(P),
            "total_time": total_time,
            "optimal_CT": optimal_CT,
            "runtime": model.Runtime,
        })

        if model.status == GRB.OPTIMAL:
            status = "OK"
        elif model.status == GRB.TIME_LIMIT:
            status = "TIMEOUT"
        else:
            status = "FAIL"
        ct_str = f"{optimal_CT:.0f}" if optimal_CT else "N/A"
        print(
            f"  {filename:<25} n={len(J):3d}  m={len(K):2d}  "
            f"OS={order_strength:.3f}  prec={len(P):2d}  "
            f"total={total_time:5d}  CT={ct_str:>4s}  "
            f"time={model.Runtime:.3f}s  [{status}]"
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

        print(f"\nn={n} ({len(subset)} instances) :")
        print(f"  OS moyen        : {_mean([r['order_strength'] for r in subset]):.3f}")
        print(f"  Précédences moy : {_mean([r['num_precedences'] for r in subset]):.1f}")
        print(f"  Total time moy  : {_mean([r['total_time'] for r in subset]):.0f}")
        if ct_vals:
            print(f"  CT optimal moy  : {_mean(ct_vals):.0f}")
        print(f"  Runtime moyen   : {_mean(runtimes):.4f}s")


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


if __name__ == "__main__":
    # 1. Résoudre une instance unique avec affichage détaillé
    solve_single_instance("./Instances/instance_n=20_1.alb")

    # 2. Benchmark sur toutes les instances
    results = run_benchmark("./Instances")
    print_stats(results)

    # 3. Graphique
    try:
        save_runtime_plot(results)
    except ImportError:
        print("matplotlib non installé, graphique ignoré.")
