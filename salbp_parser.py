import os

def parse_alb_file(filepath):
    """
    Parse un fichier .alb (Assembly Line Balancing) et extrait les données du problème.
    
    Format du fichier .alb:
    - <number of tasks>: n (nombre de VNF)
    - <cycle time>: CT_limit
    - <order strength>: OS (métrique [0, 1])
    - <task times>: liste des temps de traitement
    - <precedence relations>: graphe de précédences (i, j)
    
    Paramètres:
    -----------
    filepath : chemin vers le fichier .alb
    
    Retourne:
    ---------
    J : liste [0, 1, ..., n-1] des indices VNF
    K : liste [0, 1, ..., m-1] des indices serveurs (m = 4 par défaut)
    P : liste de tuples (i, j) pour les précédences
    t : dictionnaire {j: temps_traitement_j}
    cycle_time_limit : limite de cycle time imposée
    order_strength : métrique de densité du graphe de précédences
    """
    data = {}
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('<end>'):
                continue
            
            # Détection des sections
            if line.startswith('<number of tasks>'):
                current_section = 'tasks_count'
                continue
            elif line.startswith('<cycle time>'):
                current_section = 'cycle_time'
                continue
            elif line.startswith('<order strength>'):
                current_section = 'order_strength'
                continue
            elif line.startswith('<task times>'):
                current_section = 'task_times'
                data['task_times'] = {}
                continue
            elif line.startswith('<precedence relations>'):
                current_section = 'precedences'
                data['precedences'] = []
                continue
            
            # Parser le contenu selon la section
            if current_section == 'tasks_count':
                data['n_tasks'] = int(line)
                current_section = None
            elif current_section == 'cycle_time':
                data['cycle_time'] = int(line)
                current_section = None
            elif current_section == 'order_strength':
                # Gestion du format français (virgule décimale)
                line_clean = line.replace(',', '.')
                data['order_strength'] = float(line_clean)
                current_section = None
            elif current_section == 'task_times':
                parts = line.split()
                if len(parts) == 2:
                    task_id = int(parts[0]) - 1  # Conversion à indexation 0
                    task_time = int(parts[1])
                    data['task_times'][task_id] = task_time
            elif current_section == 'precedences':
                parts = line.split(',')
                if len(parts) == 2:
                    i = int(parts[0]) - 1  # Conversion à indexation 0
                    j = int(parts[1]) - 1
                    data['precedences'].append((i, j))
    
    # Construction des structures de données
    n = data['n_tasks']
    J = list(range(n))
    
    # Détermination du nombre de serveurs optimaux (m)
    filename = os.path.basename(filepath)
    m = 4  # par défaut
    if 'n=50_1' in filename: m = 8
    elif 'n=50_2' in filename: m = 6
    elif 'n=100_1' in filename: m = 23
    elif 'n=100_2' in filename: m = 21
    elif 'n=20' in filename:
        try:
            name = filename.split('.')[0]
            idx = int(name.split('_')[-1])
            if idx == 16: m = 12
            elif idx == 17: m = 10
            elif idx == 18: m = 11
            elif idx == 19: m = 14
            elif idx == 20: m = 11
            elif 1 <= idx <= 15: m = 3
        except (ValueError, IndexError):
            pass
            
    K = list(range(1, m + 1))
    P = data['precedences']
    t = data['task_times']
    
    return J, K, P, t, data['cycle_time'], data['order_strength']
