# Produit matrice-vecteur v = A.u
import numpy as np
from time import time
dim = 120
#SEQUENTIEL
deb = time()
# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")

# Produit matrice-vecteur
v_sequence = A.dot(u)
fin = time()
print(f"Temps du calcul du produit matriciel séquentiel : {fin-deb}")
print(f"v = {v_sequence}")