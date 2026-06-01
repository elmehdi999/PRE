import numpy as np
from rff_kernel import RandomFourierFeatures

def trouver_D_optimal_robuste(longueur_l, epsilon_cible, taille_grille=30, max_D=5000):
    """
    Cherche D optimal avec arrêt prématuré (Early Stopping) si la convergence stagne.
    """
    D_test = 50
    pas_D = 100
    
    # Paramètres d'arrêt prématuré
    patience_max = 10  # Nombre d'itérations tolérées sans amélioration significative
    patience_actuelle = 0
    seuil_amelioration = 0.002 # On exige de gagner au moins 0.2% d'erreur à chaque palier
    
    best_erreur = float('inf')
    best_D = D_test
    
    # Échantillon figé pour comparer équitablement
    points = np.random.uniform(-1, 1, (taille_grille, 2))
    
    print(f"--- Calibration RFF (l={longueur_l}, cible < {epsilon_cible}) ---")
    
    while D_test <= max_D:
        rff = RandomFourierFeatures(2, D_test, longueur_l)
        erreurs_locales = []
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                K_vrai = rff.vraie_covariance(points[i], points[j])
                K_approx = rff.approx_covariance(points[i], points[j])
                erreurs_locales.append(np.abs(K_vrai - K_approx))
                
        erreur_max = np.max(erreurs_locales)
        print(f"Test D={D_test} -> Erreur max = {erreur_max:.4f}")
        
        # --- LOGIQUE D'ARRET PREMATURE ---
        
        # 1. Condition de succès absolue
        if erreur_max <= epsilon_cible:
            print(f">>> SUCCÈS : Cible atteinte. D optimal = {D_test}")
            return D_test
            
        # 2. Vérification du rendement marginal (stagnation)
        if (best_erreur - erreur_max) > seuil_amelioration:
            # L'algorithme a fait un vrai progrès
            best_erreur = erreur_max
            best_D = D_test
            patience_actuelle = 0 # On remet le compteur à zéro
        else:
            # L'algorithme stagne ou régresse (bruit stochastique)
            patience_actuelle += 1
            print(f"    Rendement faible. Stagnation ({patience_actuelle}/{patience_max})")
            
            if patience_actuelle >= patience_max:
                print(f">>> ABANDON : La convergence stagne. On s'arrête pour éviter le surcoût.")
                print(f">>> RETOUR DU MEILLEUR COMPROMIS : D = {best_D} (Erreur = {best_erreur:.4f})")
                return best_D
                
        D_test += pas_D
        
    print(f">>> LIMITE ATTEINTE : On a touché max_D={max_D}. Retour de D={best_D}")
    return best_D