# Examen machine 2024

## Paramètres machines 
Ma machine dispose de :
- 4 coeurs physiques
- 8 processeurs logiques
- cache de niveau 1 : 258 ko
- cache de niveau 2 : 1.0 Mo
- cache de niveau 3 : 8.0 Mo

## Introduction projet
Le but de ce projet est de colorer une image en noir et blanc. La colorisation est un processus assisté par ordinateur qui consiste à rajouter de la couleur à une photo ou un film monochrome. Dans les algorithmes classiques de colorisation, l'algorithme segmente l'image en régions. En pratique, ce n'est pas une tâche robuste et l'intervention de l'utilisateur reste une tâche pénible et consommatrice de temps.

L'algorithme utilisé dans notre programme est une méthode de colorisation qui ne demande pas une segmentation précise de l'image mais se base sur la simple supposition que des pixels voisins ayant des intensités de gris similaires devraient avoir une couleur similaire. Cet algorithme a été proposé par A. Levin, D. Lischinsky et Y. Weiss dans le papier au Sig'graph 2004: "Colorization using Optimization".

## Stratégies
On commence par paralléliser l'image en nbp tranches d'images
### Partition verticale ou horizontale ?
On pourrait se demander quel oritentations dispose de zone plus homogènes, mais il ne me semble pas que ce soit le cas. Ne revanche, il me semble que les images sont souvent stockées en mémoire ligne par ligne. Une partition verticale exploiterait donc plus efficacement la localité spatiale des données, car le traitement de chaque tranche  bénéficierais d'un l'accès séquentiel à la mémoire.
Je choisit donc un partition **verticale**. 
### Parallélisation des calculs 
#### Matrice locale
Pour commencer, chaque processus essaye de coloriser sa portion d'image à partir des conditions de Dirichlet correspondant à sa portion d'image et en construisant une matrice uniquement locale à cette portion d'image.
Pour cela, la processeur 0 est le processeur maître et réaliser la répartition. 
- Il charge l'image, la transforme en tableau 
- Il divise ce tableau par tranche verticle équilibrée en fonction du nombre de processeur
- Il envoie à chaque processeur y compris lui-même une 3 matrices locale de ce tableau (fonction *comm.scatter*)
    - *slice_values_gray*
    - *slice_val_ycbcr* 
    - *slice_val_hsv*
- Chaque processeur traite alors ses matrices locales
- Le processeur maître récupère toutes les matrices finales et les concatène (fonction *comm.gather*)

## Analyse
### Memory Bound oud CPU bound ?
### Parallélisme embarrassant

## Speed-up
