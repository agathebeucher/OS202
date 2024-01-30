# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
from mpi4py import MPI
import matplotlib.cm

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


def calculate_mandelbrot_row(start_row, end_row, width, height, scaleX, scaleY, mandelbrot_set):
    row_results = np.empty((width, end_row - start_row), dtype=np.double)
    for y in range(start_row, end_row):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            row_results[x,y - start_row] = mandelbrot_set.convergence(c, smooth=True)
    return row_results

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((width, height), dtype=np.double)

# Calcul de l'ensemble de mandelbrot :
deb = time()

#SEQUENTIEL
"""for y in range(height):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#PARALLELISATION
# Calcul de la répartition des lignes de l'image pour chaque processus
rows_per_process = height // size #nombre de ligne traitées par processus
start_row = rank * rows_per_process #ligne de début
end_row = start_row + rows_per_process #ligne de fin

# Calcul de l'ensemble de Mandelbrot sur les lignes allouées à ce processus
row_results = calculate_mandelbrot_row(start_row, end_row, width, height, scaleX, scaleY, mandelbrot_set)

# Rassemblement des résultats sur le processus 0
if rank==0:
    convergence = np.empty((width, height), dtype=np.double)
    convergence[:,start_row:end_row] = row_results
    for i in range(1, size):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process
        convergence[:,start_row:end_row] = comm.recv(source=i, tag=i)
else :
    comm.send(row_results, dest=0, tag=rank)

fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")

# Constitution de l'image résultante:
deb = time()
image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
fin = time()
print(f"Temps de constitution de l'image : {fin-deb}")
image.save("mandelbrot.png")