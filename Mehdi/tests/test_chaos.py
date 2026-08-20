import p_chaos

# 1. Testons la combinatoire
ordreKL = 2  # Dimension (M)
ordrePC = 2  # Degré max (q)

# Calcul du nombre total de polynômes (formule (M+q)! / (M! * q!))
import math
nb_polynomes = math.factorial(ordreKL + ordrePC) // (math.factorial(ordreKL) * math.factorial(ordrePC))

print(f"--- LISTE DES POLYNÔMES (M={ordreKL}, q={ordrePC}) ---")
for j in range(nb_polynomes):
    alpha = p_chaos.alpha_gras(ordreKL, j)
    print(f"Polynôme j={j} : vecteur alpha = {alpha}")

# 2. Testons la matrice d'intégration pour i = 0 (la moyenne)
print(f"\n--- MATRICE c_ijk pour i=0 ---")
matrice = p_chaos.matrice_cijk(0, ordreKL, nb_polynomes - 1)
print(matrice)