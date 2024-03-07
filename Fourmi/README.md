### Fourmi2024

# Problématisation : 

1. Problème 1 :
On veut que chaque processeur gère un ensemble de fourmi et les fasse évoluer à l'aide de la méthode '*Colony.advance*'. La méthode responsable de la mise à jour de l'interface est '*Colony.display*'. Il faut donc reconstruire une colonie globale mise à jour avec les autres processeurs dans P0 pour qu'il puisse gérer l'affichage.

Toutefois, on ne peut pas transmettre toutes les colonies locales d'un processeur à un autre (*Pas possible de faire : ants_glob_list=comm.gather(ants_local,root=0)*). 

On doit donc fournir à P0 les caractéristiques nécéssaires pour 'display' toute les colonies, c'est-à-dire :
- **ants.directions** : *fournit pour chaque fourmi la direction vers laquelle la fourmi fait face*
- **ants.historic_path** : *l'historique de passage de chaque fourmi*, (ants.historic_path[ants.age]) fournit la position actuelle
- **ants.age** : *les âges de chaque fourmi*
- **ants.is_loaded** : *Les états de chaque fourmi*

2. Problème 2 :
Colony.historic_path est un tableau numpy tridimensionnel de dimensions (nb_ants, max_life+1, 2), donc on récupère dans chaque processeur un tableau numpy tridimensionnel de dimension (my_ants_count, max_life+1,2). On veut ensuite les rassembler dans le processeur 0, mais pour ça il faut le concaténer seulement sur la première dimension (axis=0)
=> Problème résolu

3. Problème 3 :
Quand on prend nb_ants=3 fourmis, on ne voit à l'affichage qu'une seule fourmi. En réalité, c'est bien trois fourmis différentes superposées qui effectue les mêmes déplacement, parce qu'à chaque fois, elles sont initialisé par la même "seed". En effet, l'initialisation de la classe "Colony" initialise self.seeds avec un tableau d'entiers successifs servant de graines uniques pour chaque fourmi dans la simulation. Chaque fourmi, de l'indice 1 à nb_ants, reçoit une graine aléatoire unique qui peut être utilisée pour générer des séquences aléatoires indépendantes dans des simulations ou des calculs ultérieurs, assurant ainsi que le comportement aléatoire de chaque fourmi est distinct.
Donc lors de l'initialisation de ants_local, chaque colonie de fourmi reçoit une série de nombre identique.
=> idée de résolution : modifier self__init__.seeds pour générer une série de seeds entre [rank, rank+my_ants_counts]

4. Problème 4 : on ne recoit pas à chaque itération les données pour toutes les fourmis (par exemple pour historic_path : error : could not broadcast input array from shape (38,501,2) into shape (52,501,2)), il faut donc en tenir compte pour concaténer les historic_path reçu, que chaqun aura une taille différente.
