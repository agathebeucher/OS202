import subprocess
import matplotlib.pyplot as plt

def run_program(num_processors):
    command = f"mpiexec -np {num_processors} python3 ants.py"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    return float(output.strip())

def main():
    # Nombre de processeurs à tester
    num_processors_list = [1, 2, 4, 8, 16]

    # Temps pour un processeur séquentiel
    t1 = run_program(1)

    # Temps pour plusieurs processeurs
    speedup_values = []
    for num_processors in num_processors_list:
        t2 = run_program(num_processors)
        speedup = t1 / t2
        speedup_values.append(speedup)

    # Tracé de la courbe de speedup
    plt.plot(num_processors_list, speedup_values, marker='o')
    plt.xlabel('Nombre de processeurs')
    plt.ylabel('Speedup')
    plt.title('Courbe de speedup en fonction du nombre de processeurs')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
