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

nb_procs = range(2, 5)  # De 2 à 4 processeurs pour l'exemple
times_sequential = []
times_parallel_gather = []
times_parallel_send = []

# Exécute le programme séquentiel une seule fois car le temps ne change pas avec le nombre de processeurs
cmd_seq = f'python3 ants_sequentiel.py'
average_fps_seq = execute_command_and_collect_data(cmd_seq)
if average_fps_seq :    
    time_sequential = 1 / average_fps_seq

for nb_proc in nb_procs:
    # Exécute le programme parallèle gather
    cmd_par_gather = f'mpiexec -n {nb_proc} python3 ants_gather.py'
    average_fps_par_gather = execute_command_and_collect_data(cmd_par_gather)
    if average_fps_par_gather:
        times_parallel_gather.append(1 / average_fps_par_gather)

    # Exécute le programme parallèle send
    cmd_par_send = f'mpiexec -n {nb_proc} python3 ants_send.py'  # Assurez-vous que ants_send.py est le nom correct
    average_fps_par_send = execute_command_and_collect_data(cmd_par_send)
    if average_fps_par_send:
        times_parallel_send.append(1 / average_fps_par_send)

# Calcul du speedup pour gather et send
speedups_gather = [time_sequential / t_par for t_par in times_parallel_gather]
speedups_send = [time_sequential / t_par for t_par in times_parallel_send]

# Tracé du speedup
plt.figure(figsize=(10, 5))
plt.plot(nb_procs, speedups_gather, marker='o', label='ants_gather')
plt.plot(nb_procs, speedups_send, marker='x', label='ants_send')
plt.xlabel('Nombre de processeurs')
plt.ylabel('Speedup')
plt.title('Speedup par rapport au nombre de processeurs')
plt.legend()
plt.grid(True)
plt.savefig('speedup_graph_comparison.png', dpi=300)  # Sauvegarder le graphique
plt.show()
