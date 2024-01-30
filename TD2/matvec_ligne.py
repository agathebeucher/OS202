# Produit matrice-vecteur v = A.u par LIGNE
import numpy as np
from time import time
from mpi4py import MPI

#PARALLELISATION
deb = time()
# Dimension du problème (peut-être changé)
dim = 120
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N_loc=dim//size #nb de colonnes de la matrice par processeur

# Initialisation de la matrice
local_A = np.array([[(i+j) % dim+1. for j in range(dim)]for i in range(rank * N_loc, (rank + 1) * N_loc)])
print(f"local_A = {local_A} pour le processeur {rank}")

# Initialisation du vecteur u_local
#u = np.array([i+1. for i in range(dim)])
local_u = np.array([i+1. for i in range(dim)])
print(f"u = {local_u}")

# Produit matrice-vecteur
local_v = local_A.dot(local_u)

# Collecte des résultats partiels de chaque processus dans le processus 0
v_parallel = np.empty(dim * size, dtype=np.float64)
comm.Gather(local_v, v_parallel, root=0)
#comm.ALLgather(local_v, v)

# Affichage du résultat sur chaque processus
#print(f"Process {rank}: v = {v}")
fin = time()
print(f"Temps du calcul du produit matriciel en parallèle : {fin-deb}")
