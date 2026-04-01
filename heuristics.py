"""
Phase 2 — Heuristiques et métaheuristiques pour le placement de VNF
====================================================================
Trois méthodes implémentées :
  H1 — Greedy constructive (heuristique déterministe)
  H2 — Simulated Annealing / Recuit simulé (métaheuristique)
  H3 — Algorithme Génétique (métaheuristique à population)

Dépendances : Python 3.10+, numpy
Usage       : python phase2_heuristics.py
"""

import os
import time
import random
import math
import csv
import numpy as np
from salbp_parser import parse_alb_file


# =============================================================================
# Utilitaires communs
# =============================================================================

def compute_CT(assignment: dict, J: list, K: list, t: dict) -> float:
    """Calcule le cycle time (charge max) pour un assignment {j: k}."""
    loads = {k: 0.0 for k in K}
    for j in J:
        loads[assignment[j]] += t[j]
    return max(loads.values())


def is_feasible(assignment: dict, P: list) -> bool:
    """Vérifie que toutes les précédences (i,j) sont respectées : s[i] <= s[j]."""
    for i, j in P:
        if assignment[i] > assignment[j]:
            return False
    return True


def topological_order(J: list, P: list) -> list:
    """Tri topologique de J selon P (Kahn's algorithm)."""
    in_degree = {j: 0 for j in J}
    successors = {j: [] for j in J}
    for i, j in P:
        in_degree[j] += 1
        successors[i].append(j)
    queue = [j for j in J if in_degree[j] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    return order


def random_feasible_neighbor(assignment: dict, J: list, K: list, P: list) -> dict:
    """
    Génère un voisin en déplaçant une VNF aléatoire vers un autre serveur,
    puis répare les violations de précédence par propagation.
    """
    new_assign = dict(assignment)
    j = random.choice(J)
    k_new = random.choice(K)
    new_assign[j] = k_new

    # Réparation : pour chaque précédence violée, déplacer le successeur vers
    # le serveur du prédécesseur (ou supérieur). Itéré jusqu'à stabilité.
    changed = True
    iterations = 0
    while changed and iterations < len(J) * 2:
        changed = False
        iterations += 1
        for i, jj in P:
            if new_assign[i] > new_assign[jj]:
                new_assign[jj] = new_assign[i]
                changed = True
    return new_assign


# =============================================================================
# H1 — Greedy constructive
# =============================================================================

def greedy_constructive(J: list, K: list, P: list, t: dict) -> tuple[dict, float]:
    """
    Heuristique gloutonne : parcourt les VNF dans l'ordre topologique et
    affecte chacune au serveur dont l'ajout minimise la charge maximale
    courante, sous contrainte de précédence.

    Complexité : O(n * m)
    """
    order = topological_order(J, P)
    assignment = {}
    loads = {k: 0.0 for k in K}

    for j in order:
        # Serveurs candidats : tous les k >= max(s[i]) pour tout prédécesseur i
        min_k = min(K)
        for i, jj in P:
            if jj == j and i in assignment:
                min_k = max(min_k, assignment[i])

        candidates = [k for k in K if k >= min_k]
        if not candidates:
            candidates = K  # fallback de sécurité

        # Choisir le serveur qui minimise l'augmentation de charge max
        best_k = min(candidates, key=lambda k: loads[k] + t[j])
        assignment[j] = best_k
        loads[best_k] += t[j]

    ct = max(loads.values())
    return assignment, ct


# =============================================================================
# H2 — Simulated Annealing (Recuit Simulé)
# =============================================================================

def simulated_annealing(
    J: list,
    K: list,
    P: list,
    t: dict,
    T_init: float = 500.0,
    T_min: float = 0.1,
    alpha: float = 0.995,
    max_iter: int = 50_000,
    seed: int = 42,
) -> tuple[dict, float]:
    """
    Métaheuristique de recuit simulé pour minimiser CT.

    Paramètres
    ----------
    T_init   : température initiale (contrôle l'acceptation de solutions dégradantes)
    T_min    : température d'arrêt
    alpha    : taux de refroidissement géométrique (T <- alpha * T à chaque pas)
    max_iter : nombre maximum d'itérations
    seed     : graine aléatoire pour reproductibilité

    L'exploration est assurée par des déplacements aléatoires d'une VNF vers un
    nouveau serveur, avec réparation des précédences (cf. random_feasible_neighbor).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Solution initiale : greedy
    current, current_ct = greedy_constructive(J, K, P, t)
    best, best_ct = dict(current), current_ct

    T = T_init
    it = 0

    while T > T_min and it < max_iter:
        neighbor = random_feasible_neighbor(current, J, K, P)
        neighbor_ct = compute_CT(neighbor, J, K, t)

        delta = neighbor_ct - current_ct
        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_ct = neighbor, neighbor_ct
            if current_ct < best_ct:
                best, best_ct = dict(current), current_ct

        T *= alpha
        it += 1

    return best, best_ct


# =============================================================================
# H3 — Algorithme Génétique
# =============================================================================

def _random_feasible_individual(J: list, K: list, P: list, t: dict) -> dict:
    """Génère un individu faisable : greedy avec ordre topologique perturbé."""
    order = topological_order(J, P)
    # Perturbation : shuffle local des tâches sans précédence immédiate entre elles
    perturbed = list(order)
    random.shuffle(perturbed)
    # Re-trier pour respecter les précédences (sort stable topologique)
    topo_rank = {j: i for i, j in enumerate(order)}
    perturbed.sort(key=lambda j: topo_rank[j])

    assignment = {}
    loads = {k: 0.0 for k in K}
    for j in perturbed:
        min_k = min(K)
        for i, jj in P:
            if jj == j and i in assignment:
                min_k = max(min_k, assignment[i])
        candidates = [k for k in K if k >= min_k]
        best_k = min(candidates, key=lambda k: loads[k] + t[j])
        assignment[j] = best_k
        loads[best_k] += t[j]
    return assignment


def _crossover(p1: dict, p2: dict, J: list, K: list, P: list) -> dict:
    """
    Croisement uniforme : pour chaque VNF, hérite du parent 1 ou 2 avec prob 0.5,
    puis répare les violations de précédence.
    """
    child = {}
    for j in J:
        child[j] = p1[j] if random.random() < 0.5 else p2[j]
    # Réparation topologique
    order = topological_order(J, P)
    for j in order:
        for i, jj in P:
            if jj == j and i in child:
                if child[i] > child[j]:
                    child[j] = child[i]
    return child


def _mutate(individual: dict, J: list, K: list, P: list, mutation_rate: float) -> dict:
    """Mutation : déplace chaque VNF avec probabilité mutation_rate, puis répare."""
    new_ind = dict(individual)
    for j in J:
        if random.random() < mutation_rate:
            new_ind[j] = random.choice(K)
    # Réparation
    order = topological_order(J, P)
    for j in order:
        for i, jj in P:
            if jj == j:
                if new_ind[i] > new_ind[j]:
                    new_ind[j] = new_ind[i]
    return new_ind


def genetic_algorithm(
    J: list,
    K: list,
    P: list,
    t: dict,
    pop_size: int = 60,
    n_generations: int = 300,
    mutation_rate: float = 0.05,
    tournament_size: int = 5,
    elite_size: int = 4,
    seed: int = 42,
) -> tuple[dict, float]:
    """
    Algorithme génétique à état permanent (steady-state GA) pour minimiser CT.

    Paramètres
    ----------
    pop_size       : taille de la population
    n_generations  : nombre de générations
    mutation_rate  : probabilité de mutation par gène (VNF)
    tournament_size: taille du tournoi pour la sélection
    elite_size     : nombre d'élites conservés sans modification (élitisme)
    seed           : graine pour reproductibilité

    Opérateurs : sélection par tournoi, croisement uniforme avec réparation,
    mutation aléatoire avec réparation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Initialisation de la population
    population = [_random_feasible_individual(J, K, P, t) for _ in range(pop_size - 1)]
    # Inclure la solution greedy comme amorce de qualité
    greedy_ind, _ = greedy_constructive(J, K, P, t)
    population.append(greedy_ind)

    def fitness(ind):
        return compute_CT(ind, J, K, t)

    for _ in range(n_generations):
        # Tri par fitness croissante
        population.sort(key=fitness)

        # Élitisme : garder les meilleurs
        new_population = population[:elite_size]

        # Remplir le reste par sélection + croisement + mutation
        while len(new_population) < pop_size:
            # Sélection par tournoi (parent 1)
            t1 = random.sample(population, tournament_size)
            p1 = min(t1, key=fitness)
            # Sélection par tournoi (parent 2)
            t2 = random.sample(population, tournament_size)
            p2 = min(t2, key=fitness)

            child = _crossover(p1, p2, J, K, P)
            child = _mutate(child, J, K, P, mutation_rate)
            new_population.append(child)

        population = new_population

    population.sort(key=fitness)
    best = population[0]
    return best, fitness(best)


# =============================================================================
# Benchmark — évaluation sur toutes les instances
# =============================================================================

PHASE1_RESULTS = {
    # CT* obtenus en phase 1 (à mettre à jour si nécessaire)
    "instance_n=20_1.alb":  962,
    "instance_n=20_2.alb":  954,
    "instance_n=20_3.alb":  930,
    "instance_n=20_4.alb":  909,
    "instance_n=20_5.alb":  950,
    "instance_n=20_6.alb":  973,
    "instance_n=20_7.alb":  930,
    "instance_n=20_8.alb":  996,
    "instance_n=20_9.alb":  976,
    "instance_n=20_10.alb": 945,
    "instance_n=20_11.alb": 924,
    "instance_n=20_12.alb": 977,
    "instance_n=20_13.alb": 967,
    "instance_n=20_14.alb": 994,
    "instance_n=20_15.alb": 994,
    "instance_n=20_16.alb": 981,
    "instance_n=20_17.alb": 969,
    "instance_n=20_18.alb": 960,
    "instance_n=20_19.alb": 1000,
    "instance_n=20_20.alb": 1000,
    "instance_n=50_1.alb":  910,
    "instance_n=50_2.alb":  931,
    # n=100 : pas de CT* prouvé optimal, on laisse None
    "instance_n=100_1.alb": None,
    "instance_n=100_2.alb": None,
}

HEURISTICS = [
    ("H1_Greedy",   lambda J, K, P, t: greedy_constructive(J, K, P, t)),
    ("H2_SA",       lambda J, K, P, t: simulated_annealing(J, K, P, t)),
    ("H3_GA",       lambda J, K, P, t: genetic_algorithm(J, K, P, t)),
]


def run_benchmark(instances_dir: str, output_csv: str = "phase2_results.csv"):
    """Évalue les trois heuristiques sur toutes les instances et sauvegarde les résultats."""
    files = sorted(
        [f for f in os.listdir(instances_dir) if f.endswith(".alb")],
        key=lambda f: (
            int(f.split("n=")[1].split("_")[0]) if "n=" in f else 0,
            int(f.split("_")[-1].replace(".alb", "")) if "_" in f else 0,
        ),
    )

    rows = []
    header = ["instance", "n", "m", "OS",
              "H1_Greedy_CT", "H1_Greedy_gap%", "H1_Greedy_time_s",
              "H2_SA_CT",     "H2_SA_gap%",     "H2_SA_time_s",
              "H3_GA_CT",     "H3_GA_gap%",     "H3_GA_time_s",
              "CT_star"]

    print(f"\n{'Instance':<25} {'n':>4} {'m':>3} {'OS':>5}  "
          f"{'H1_CT':>6} {'H1_gap':>7} {'H1_t':>6}  "
          f"{'H2_CT':>6} {'H2_gap':>7} {'H2_t':>6}  "
          f"{'H3_CT':>6} {'H3_gap':>7} {'H3_t':>6}  "
          f"{'CT*':>6}")
    print("=" * 120)

    for filename in files:
        filepath = os.path.join(instances_dir, filename)
        J, K, P, t, ct_limit, order_strength = parse_alb_file(filepath)
        ct_star = PHASE1_RESULTS.get(filename)

        row = {
            "instance": filename,
            "n": len(J),
            "m": len(K),
            "OS": round(order_strength, 3),
            "CT_star": ct_star if ct_star else "N/A",
        }

        results_line = f"{filename:<25} {len(J):>4} {len(K):>3} {order_strength:>5.3f}  "

        for name, heuristic in HEURISTICS:
            t0 = time.perf_counter()
            assignment, ct_h = heuristic(J, K, P, t)
            elapsed = time.perf_counter() - t0

            # Vérification de faisabilité
            feasible = is_feasible(assignment, P)
            if not feasible:
                ct_h = float("inf")

            gap = ((ct_h - ct_star) / ct_star * 100) if ct_star else None
            gap_str = f"{gap:.2f}%" if gap is not None else "N/A"

            row[f"{name}_CT"] = round(ct_h, 1)
            row[f"{name}_gap%"] = round(gap, 4) if gap is not None else "N/A"
            row[f"{name}_time_s"] = round(elapsed, 4)

            results_line += f"{ct_h:>6.0f} {gap_str:>7} {elapsed:>6.3f}  "

        results_line += f"{ct_star if ct_star else 'N/A':>6}"
        print(results_line)
        rows.append(row)

    print("=" * 120)

    # Sauvegarde CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nRésultats sauvegardés dans : {output_csv}")
    return rows


def print_summary(rows: list):
    """Affiche un résumé statistique par heuristique."""
    print("\n=== Résumé par heuristique ===")
    for name in ["H1_Greedy", "H2_SA", "H3_GA"]:
        gaps = [r[f"{name}_gap%"] for r in rows if isinstance(r.get(f"{name}_gap%"), float)]
        times = [r[f"{name}_time_s"] for r in rows]
        n_optimal = sum(1 for g in gaps if g <= 0.01)
        avg_gap = sum(gaps) / len(gaps) if gaps else float("nan")
        avg_time = sum(times) / len(times)
        print(f"  {name:12s} | gap moyen : {avg_gap:6.2f}%  "
              f"| instances à l'optimal : {n_optimal}/{len(gaps)}  "
              f"| temps moyen : {avg_time:.3f}s")


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    INSTANCES_DIR = "./Instances"

    # --- Test rapide sur une instance unique ---
    print("=== Test sur instance_n=20_1 ===")
    J, K, P, t, ct_limit, os_ = parse_alb_file(
        os.path.join(INSTANCES_DIR, "instance_n=20_1.alb")
    )

    for name, heuristic in HEURISTICS:
        t0 = time.perf_counter()
        assignment, ct_h = heuristic(J, K, P, t)
        elapsed = time.perf_counter() - t0
        feasible = is_feasible(assignment, P)
        print(f"  {name}: CT={ct_h:.0f}  faisable={feasible}  temps={elapsed:.4f}s")

    # --- Benchmark complet ---
    rows = run_benchmark(INSTANCES_DIR)
    print_summary(rows)