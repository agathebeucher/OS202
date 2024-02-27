### Fourmi2024

# Problématisation : 

On veut que chaque processeur gère un ensemble de fourmi et les faces évoluer à l'aide de la méthode '*Colony.advance*'. LA méthode qui gère la mise à jour du screen est '*Colony.display*'. Il faut donc reconstruire la colony globale mise à jour dans P0 pour qu'il puisse gérer l'affichage.

Toutefois, on ne peut pas transmettre toutes les colonies locales d'un processeur à un autre (*Pas possible de faire : ants_glob_list=comm.gather(ants_local,root=0)*). 

On doit donc fournir à P0 les caractéristiques nécéssaires pour 'display' toute les colonies, c'est-à-dire :
- **ants.directions** : *fournit pour chaque fourmi la direction vers laquelle la fourmi fait face*
- **ants.historic_path** : *l'historique de passage de chaque fourmi*, (ants.historic_path[ants.age]) fournit la position actuelle
- **ants.age** : *les âges de chaque fourmi*
- **ants.is_loaded** : *Les états de chaque fourmi*
