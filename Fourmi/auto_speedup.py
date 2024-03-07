import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time

def execute_command_and_collect_data(cmd, max_data_count=20):
    """Exécute une commande et collecte jusqu'à max_data_count valeurs de FPS."""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps_values = []
    try:
        while len(fps_values) < max_data_count:
            line = proc.stdout.readline()
            if not line:
                break  # Fin de la sortie
            if "FPS" in line:
                fps_str = line.split()[2].replace(',', '').replace(':', '')
                try:
                    fps_values.append(float(fps_str))
                except ValueError:
                    print(f"Erreur de conversion: '{fps_str}' n'est pas un float valide.")
    finally:
        proc.kill()  # Assurez-vous que le processus est arrêté

    return np.mean(fps_values) if fps_values else 0  # Retourner 0 si aucune valeur n'a été collectée

nb_procs = range(2, 5)  # De 2 à 10 processeurs, en ajustant comme nécessaire
times_sequential = []
times_parallel = []
valid_nb_procs = []  # Liste pour garder une trace des processeurs valides

for nb_proc in nb_procs:
    # Exécute le programme séquentiel
    cmd_seq = f'python3 ants_sequentiel.py'
    average_fps_seq = execute_command_and_collect_data(cmd_seq)
    if average_fps_seq:  # Vérifie si une valeur moyenne a été retournée
        times_sequential.append(1 / average_fps_seq)
        valid_nb_procs.append(nb_proc)  # Ajouter le nombre de processeurs à la liste des valides

    # Exécute le programme parallèle
    cmd_par = f'mpiexec -n {nb_proc} python3 ants_gather.py'
    average_fps_par = execute_command_and_collect_data(cmd_par)
    if average_fps_par:  # Vérifie si une valeur moyenne a été retournée
        times_parallel.append(1 / average_fps_par)

# Calcul du speedup pour chaque nombre de processeurs valide
speedups = [t_seq / t_par for t_seq, t_par in zip(times_sequential, times_parallel)]

# Tracé du speedup
plt.plot(valid_nb_procs, speedups, marker='o')
plt.xlabel('Nombre de processeurs')
plt.ylabel('Speedup')
plt.title('Speedup par rapport au nombre de processeurs')
plt.grid(True)
plt.savefig('speedup_graph.png', dpi=300)  # Sauvegarder le graphique
plt.show()
