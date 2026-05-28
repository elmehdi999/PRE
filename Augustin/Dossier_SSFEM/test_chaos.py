import p_chaos

ordreKL = 2
ordrePC = 2

import math
nb_polynomes = math.factorial(ordreKL + ordrePC) // (math.factorial(ordreKL)*math.factorial(ordrePC))

print(f"Liste des polynomes (M={ordreKL}, q={ordrePC})")
for j in range(nb_polynomes):
    alpha = p_chaos.alpha_gras(ordreKL, j)
    print(f"Polynome j={j} : vecteur alpha = {alpha}")

print(f"\nMatrice c_ijk pour i=0")
matrice = p_chaos.matrice_cijk(0, ordreKL, nb_polynomes - 1)
print(matrice)