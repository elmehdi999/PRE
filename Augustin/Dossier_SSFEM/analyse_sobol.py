import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
from nisp_rff import NISP_RFF
from sklearn.linear_model import OrthogonalMatchingPursuitCV

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculer_sobol(D=30, P=3, N=2048):
    print("="*60)
    print(f" CALCUL DES INDICES DE SOBOL TOTAUX (D={D}, P={P})")
    print("="*60)
    
    # 1. Chargement instantané des données de la dernière simulation
    fichier_cache = f"Y_eval_full_D{D}_N{N}.npy"
    if not os.path.exists(fichier_cache):
        print(f"Erreur : Le fichier cache '{fichier_cache}' est introuvable.")
        print(f"Veuillez d'abord exécuter : python nisp_rff.py {D} {P} {N}")
        sys.exit(1)
        
    print(f" Chargement des données depuis '{fichier_cache}'...")
    data = np.load(fichier_cache, allow_pickle=True).item()
    Y_eval_full = data['Y_eval_full']
    Xi = data['Xi']
    
    # 2. Reconstitution ultra-rapide de l'espace mathématique
    nisp = NISP_RFF(D, P, N)
    nisp.Xi = Xi
    nisp.Y_eval_full = Y_eval_full
    nisp.noeud_cible = int(np.argmax(np.mean(Y_eval_full, axis=0)))
    nisp.Y_eval = Y_eval_full[:, nisp.noeud_cible]
    
    nisp.generer_multi_indices(q=0.75)
    nisp.evaluer_polynomes_hermite()
    
    # 3. Régression OMP-CV (Apprentissage)
    print(f" Régression OMP-CV sur le noeud cible ({nisp.noeud_cible})...")
    omp_cv = OrthogonalMatchingPursuitCV(fit_intercept=False, cv=5)
    omp_cv.fit(nisp.Psi, nisp.Y_eval)
    
    coeffs = omp_cv.coef_
    multi_indices = nisp.multi_indices
    
    # 4. Calcul mathématique de la Variance et de Sobol
    variance_totale = np.sum(coeffs[1:]**2)
    
    if variance_totale == 0:
        print("Erreur : La variance totale est nulle.")
        sys.exit(1)
        
    print(f" Variance analytique capturée : {variance_totale:.4f}")
        
    sobol_totaux_par_onde = np.zeros(D)
    
    for p_idx, alpha in enumerate(multi_indices):
        if p_idx == 0:
            continue # On ignore c_0 (Espérance)
            
        coeff_carre = coeffs[p_idx]**2
        
        # Si le polynôme a été conservé par l'OMP
        if coeff_carre > 1e-12:
            # On cherche quelles ondes composent ce polynôme
            for j in range(D):
                idx_cos = 2 * j       # Variable Xi associée au cosinus de l'onde j
                idx_sin = 2 * j + 1   # Variable Xi associée au sinus de l'onde j
                
                # Si l'onde j participe à la combinaison polynomiale 'alpha'
                if alpha[idx_cos] > 0 or alpha[idx_sin] > 0:
                    sobol_totaux_par_onde[j] += coeff_carre / variance_totale

    # 5. Tracé du spectre de Sobol
    os.makedirs("resultats", exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    ondes_indices = np.arange(1, D + 1)
    
    # Mise en évidence des ondes dominantes (> 5% d'influence)
    seuil = 0.05
    couleurs = ['crimson' if s > seuil else 'steelblue' for s in sobol_totaux_par_onde]
    
    plt.bar(ondes_indices, sobol_totaux_par_onde, color=couleurs, alpha=0.8, edgecolor='black')
    plt.axhline(y=seuil, color='black', linestyle='--', label=f"Seuil d'influence forte ({seuil*100}%)")
    
    plt.xlabel("Indice de l'Onde Spatiale RFF ($j$)", fontsize=12)
    plt.ylabel("Indice de Sobol Total $S_{T_j}$", fontsize=12)
    plt.title(f"Spectre de Sensibilité Thermique au Noeud Chaud (D={D}, P={P})", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(ondes_indices)
    plt.legend()
    
    nom_fig = f"resultats/Sobol_Indices_D{D}_P{P}.png"
    plt.tight_layout()
    plt.savefig(nom_fig, dpi=300)
    
    print(f"\n [SUCCÈS] Histogramme de Sobol généré : {nom_fig}")
    n_influentes = np.sum(sobol_totaux_par_onde > seuil)
    print(f" -> {n_influentes} ondes sur {D} contrôlent l'essentiel de la variance.")

    nisp.finalise()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        calculer_sobol(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    else:
        # Configuration par défaut selon ton dernier gros run validé
        calculer_sobol(30, 3, 2048)