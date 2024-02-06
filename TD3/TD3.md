### TD3
## Implémentation de la méthode bucket list : étapes principales

On implémente la méthode tel que : 
- le process 0 génère un tableau de nombres arbitraires (on reprend dans le code l'exemple du cours),
- il calcul la longueur des intervalles en fonction de la répartition des valeurs (val_min, val_max) et du nombre de processeurs utilisés
- il crée une liste de buckets vide et associe à chacun un iterval
- il répartie les éléments du tableau de départ dans les buckets en fonction de sa valeur et l'appartenance dans l'interval
- il les dispatch aux autres process (fonction *SCATTER*),
- **Chaque process participe au tri en parallèle** (fonction *SORT*),
- le tableau trié est rassemblé sur le process 0 (fonction *GATHER*),
- il modifie la liste de buckets en une seule liste triée !
