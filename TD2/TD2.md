# Travaux dirigés n°2

## Questions de cours

### Interblocage
On reprend l'exemple avec 3 processeurs donné dans le cours : P2 peut recevoir des messages de n'importe quel processeur donc deux scénarios possibles se produisent
1. **Scénario où il n'y a pas d'interblocage** : Soit P2 reçoit de P0, donc renvoie un message à P0, qui peut le recevoir car il a déjà envoyé à P2, donc ça se passe bien --> pas de blocage
2. **Scénario où il y a un interblocage** : Soit P2 reçoit un message de P1, puis renvoie un message à P0 qui envoie également un message à P2, donc P0 et P2 attendant tout les deux le message de l'autre --> interblocage

### Execution en parallèle
En utilisant la *loi d'Amdhal*, l'accélération maximale que pourra obtenir Alice avec son code est de : S_n = 1/0.1=10.
Pour ce jeu de donné spécifique, il semble raisonnable de prendre à peu près 5 noeuds de calcul pour ne pas gaspiller de ressources CPU.
La *loi de Gustafon* s'applique lorsque la taille du problème augmente avec le nombre de processeurs : on a S=1−P+P×n avec P=0.9 ici.
Donc si on double la quantité de données à traiter et qu'on prend n=4, on a S=1.6.
