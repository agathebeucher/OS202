***Lifegame***

=> *Tracer le speedup en fonction du nombre de CPUs*

1. Allouer un processeur qui s'occupe de l'affichage et l'autre du calcul
2. P0 affiche P1/P2


Vis à Vis de l'alignement mémoire, il est plus intéréssant de couper horizontalement avec le rangement du tableau

Je n'ai pas eu le temps de gérer le problème des cellules fantômes : 
Il aurait fallu que je rajoute un 'gather' après que chaque processeur ait 'avance()' afin qu'il se communique une ligne de cellule fantôme (dans mon cas la ligne 50) qui servira au traitement de la dernière ligne de chaque surface et dont l'évolution sera fausse. 

3. P0 affiche P1, P2, P3, P4 : Je n'ai pas non plus eu le temps de traiter ce problème