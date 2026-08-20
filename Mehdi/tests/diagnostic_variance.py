import numpy as np
import os
import sys

def diagnostic_interne():
    fichier_cache = "Y_eval_full_D50_N2048.npy"
    
    if not os.path.exists(fichier_cache):
        print(f"Fichier {fichier_cache} introuvable.")
        sys.exit(1)
        
    data = np.load(fichier_cache, allow_pickle=True).item()
    Y_full = data['Y_eval_full']  # taille attendue : (2048, 942)
    
    N_total = Y_full.shape[0]
    demi_N = N_total // 2
    
    print(f"Analyse de cohérence interne sur {N_total} évaluations.")
    
    # separation en deux groupes independants
    var_groupe_A = np.var(Y_full[:demi_N, :], axis=0, ddof=1)
    var_groupe_B = np.var(Y_full[demi_N:, :], axis=0, ddof=1)
    
    # calcul de l'erreur relative L2 entre les deux groupes
    erreur_interne = np.linalg.norm(var_groupe_A - var_groupe_B) / np.linalg.norm(var_groupe_A)
    
    print(f"Erreur L2 entre Groupe A (N={demi_N}) et Groupe B (N={demi_N}) : {erreur_interne*100:.2f} %")
    
    print("\n Interprétation")
    if erreur_interne < 0.06: # 6%
        print(" L'erreur interne est faible (cohérente avec le bruit d'échantillonnage théorique).")
        print(" Conclusion : Le plateau de 19% observé face au Monte Carlo est bien un ÉCART DE MODÈLE.")
        print("   Le calibrage scalaire (0.0401) est insuffisant, car l'atténuation du filtre MEF++")
        print("   varie spatialement (près des bords vs centre).")
    else:
        print(" L'erreur interne est élevée.")
        print(" Conclusion : L'échantillon LHS n'est pas encore convergé sur le plan de la variance.")

if __name__ == "__main__":
    diagnostic_interne()