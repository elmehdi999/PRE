import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

# On importe ta classe NISP
from nisp_rff import NISP_RFF

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculer_sobol():
    D = 50
    P = 3
    N = 500
    
    print("="*60)
    print(" CALCUL DES INDICES DE SOBOL TOTAUX (NISP-RFF)")
    print("="*60)
    
    # 1. Exécution de la chaîne NISP jusqu'à la régression
    nisp = NISP_RFF(D, P, N)
    nisp.initialiser_frequences()
    nisp.generer_plan_experience()
    nisp.evaluer_boite_noire()
    nisp.generer_multi_indices(q=0.75)
    nisp.evaluer_polynomes_hermite()
    nisp.regression_omp()
    
    coeffs = nisp.coefficients
    multi_indices = nisp.multi_indices
    
    # 2. Calcul de la variance totale
    variance_totale = np.sum(coeffs[1:]**2)
    
    if variance_totale == 0:
        print("Erreur : La variance totale est nulle. Impossible de calculer les indices de Sobol.")
        sys.exit(1)
        
    sobol_totaux_par_onde = np.zeros(D)
    
    # 3. Calcul analytique des indices de Sobol Totaux (S_T)
    # L'onde 'j' est contrôlée par les variables aléatoires 2*j (cos) et 2*j+1 (sin)
    for p_idx, alpha in enumerate(multi_indices):
        if p_idx == 0:
            continue # On ignore la constante (Moyenne)
            
        coeff_carre = coeffs[p_idx]**2
        
        # Si le coefficient est actif (choisi par OMP)
        if coeff_carre > 1e-12:
            # Pour chaque onde D, on vérifie si elle est impliquée dans ce polynôme
            for j in range(D):
                idx_cos = 2 * j
                idx_sin = 2 * j + 1
                
                # Si l'une des variables de l'onde est dans le polynôme, on ajoute la contribution
                if alpha[idx_cos] > 0 or alpha[idx_sin] > 0:
                    sobol_totaux_par_onde[j] += coeff_carre / variance_totale

    # 4. Tracé du spectre de sensibilité
    plt.figure(figsize=(12, 6))
    
    ondes_indices = np.arange(1, D + 1)
    
    # On met en évidence les ondes les plus influentes
    seuil = 0.05 # 5% de contribution
    couleurs = ['red' if s > seuil else 'blue' for s in sobol_totaux_par_onde]
    
    plt.bar(ondes_indices, sobol_totaux_par_onde, color=couleurs, alpha=0.7, edgecolor='black')
    
    # Ligne de seuil de sensibilité
    plt.axhline(y=seuil, color='black', linestyle='--', label=f'Seuil d\'influence forte ({seuil*100}%)')
    
    plt.xlabel("Indice de l'Onde RFF", fontsize=12)
    plt.ylabel("Indice de Sobol Total $S_{T_i}$", fontsize=12)
    plt.title(f"Spectre de Sensibilité Thermique des Fréquences RFF (D={D})", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("sobol_indices.png", dpi=300)
    print("\n[SUCCÈS] Histogramme de Sobol généré : sobol_indices.png")
    
    n_influentes = np.sum(sobol_totaux_par_onde > seuil)
    print(f"-> Nombre d'ondes dominantes (>5% d'influence) : {n_influentes} sur {D}")

if __name__ == "__main__":
    calculer_sobol()