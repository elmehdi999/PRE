import numpy as np
import ks_stat  # On importe le module d'Augustin

# Paramètres de base du noeud que l'on observe
TAILLE_ECHANTILLON = 100
VRAIE_MOYENNE = 50.0
VRAIE_VARIANCE = 5.0


print("\nSCÉNARIO 1 : Le code SSFEM est parfait")

# On simule un résultat SSFEM parfait (vraie loi normale)
echantillon_parfait = np.random.normal(VRAIE_MOYENNE, np.sqrt(VRAIE_VARIANCE), TAILLE_ECHANTILLON)

# On utilise la syntaxe exacte d'Augustin
test1 = ks_stat.ks(TAILLE_ECHANTILLON, VRAIE_MOYENNE, VRAIE_VARIANCE)
test1.setter_echantillon(echantillon_parfait.tolist())
test1.kstest()
test1.affichage()


print("\nSCÉNARIO 2 : Le code SSFEM a un bug de moyenne")

# On simule un bug où la SSFEM s'est trompée de +2.0 sur le déplacement moyen
echantillon_bug = np.random.normal(VRAIE_MOYENNE + 2.0, np.sqrt(VRAIE_VARIANCE), TAILLE_ECHANTILLON)

test2 = ks_stat.ks(TAILLE_ECHANTILLON, VRAIE_MOYENNE, VRAIE_VARIANCE)
test2.setter_echantillon(echantillon_bug.tolist())
test2.kstest()
test2.affichage()


print("\nSCÉNARIO 3 : Pas assez de tirages (Bruit statistique)")

# On ne fait que 10 tirages. Le hasard peut nous faire échouer le test !
echantillon_petit = np.random.normal(VRAIE_MOYENNE, np.sqrt(VRAIE_VARIANCE), 10)

test3 = ks_stat.ks(10, VRAIE_MOYENNE, VRAIE_VARIANCE)
test3.setter_echantillon(echantillon_petit.tolist())
test3.kstest()
test3.affichage()