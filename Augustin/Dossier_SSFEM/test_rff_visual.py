import numpy as np
import matplotlib.pyplot as plt
from rff_kernel import RandomFourierFeatures

# --- Paramètres ---
DIMENSION = 2
PRECISION_D = 1000  # Nombre de fréquences (essaie avec 10, puis 1000 pour voir la différence)
LONGUEUR_L = 1    # Longueur de corrélation
TAILLE_GRILLE = 50  # Résolution de l'image

# Initialisation de notre générateur mathématique
rff = RandomFourierFeatures(DIMENSION, PRECISION_D, LONGUEUR_L)

# Création d'une grille spatiale 2D (de -1 à 1 sur X et Y)
x = np.linspace(-1, 1, TAILLE_GRILLE)
y = np.linspace(-1, 1, TAILLE_GRILLE)
X, Y = np.meshgrid(x, y)

# Matrices pour stocker les résultats
Z_exact = np.zeros((TAILLE_GRILLE, TAILLE_GRILLE))
Z_approx = np.zeros((TAILLE_GRILLE, TAILLE_GRILLE))

# On calcule la covariance entre le centre (0,0) et tous les autres points
point_central = np.array([0.0, 0.0])

print("Calcul en cours... (ça prend quelques secondes)")
for i in range(TAILLE_GRILLE):
    for j in range(TAILLE_GRILLE):
        point_courant = np.array([X[i, j], Y[i, j]])
        
        # Remplissage de la matrice exacte
        Z_exact[i, j] = rff.vraie_covariance(point_central, point_courant)
        
        # Remplissage de la matrice approchée par ta méthode SSFEM
        Z_approx[i, j] = rff.approx_covariance(point_central, point_courant)

# --- Affichage Visuel ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Graphe 1 : Le vrai noyau Gaussien
c1 = ax1.contourf(X, Y, Z_exact, levels=20, cmap='viridis')
ax1.set_title("Noyau Gaussien Exact")
fig.colorbar(c1, ax=ax1)

# Graphe 2 : L'approximation RFF
c2 = ax2.contourf(X, Y, Z_approx, levels=20, cmap='viridis')
ax2.set_title(f"Approximation RFF (D={PRECISION_D})")
fig.colorbar(c2, ax=ax2)

# Graphe 3 : L'erreur absolue
erreur = np.abs(Z_exact - Z_approx)
c3 = ax3.contourf(X, Y, erreur, levels=20, cmap='Reds')
ax3.set_title("Erreur Absolue")
fig.colorbar(c3, ax=ax3)

plt.tight_layout()
plt.show()

# PARTIE 2 : GÉNÉRATION DU CHAMP ALÉATOIRE

# 1. On tire nos 2D variables aléatoires standards (le vecteur xi)
# On a besoin d'un poids pour chaque élément de Z(x), donc 2 * PRECISION_D
xi = np.random.normal(0, 1, 2 * PRECISION_D)

champ_aleatoire = np.zeros((TAILLE_GRILLE, TAILLE_GRILLE))

print("Génération du champ aléatoire en cours...")
for i in range(TAILLE_GRILLE):
    for j in range(TAILLE_GRILLE):
        point_courant = np.array([X[i, j], Y[i, j]])
        
        # On calcule la signature spatiale du point
        Z_x = rff.eval_Z(point_courant)
        
        # Le champ est le simple produit scalaire entre la géométrie Z(x) et l'aléa xi
        champ_aleatoire[i, j] = np.dot(Z_x, xi)

# --- Affichage du Champ ---
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, champ_aleatoire, levels=30, cmap='coolwarm')
plt.title(f"Réalisation du Champ Aléatoire 2D (l={LONGUEUR_L})")
plt.colorbar()
plt.tight_layout()
plt.show()