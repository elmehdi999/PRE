import numpy as np
import sys
import os
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def gaussian_cov(r, l_corr):
    return np.exp(- (r / l_corr)**2)

def mesurer_longueur_correlation(n_tirages=1000):
    print("="*60)
    print(f" MESURE EMPIRIQUE DE LA LONGUEUR DE CORRÉLATION (N={n_tirages})")
    print("="*60)

    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    opts = PETSc.Options()
    if opts.hasName('options_slinksp_monitor'):
        opts.delValue('options_slinksp_monitor')
        
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    
    N_elements = vec_bruit.getSize()
    indices_elem = np.arange(N_elements, dtype=np.int32)
    
    # ---------------------------------------------------------
    # RÉCUPÉRATION DU MAILLAGE ET CIBLAGE DU CENTRE
    # ---------------------------------------------------------
    gfc.reqPP("pp_copie_X_elem").execute()
    gfc.reqPP("pp_copie_Y_elem").execute()
    X_elem = gfc.reqVecteurPETSc("Vec_X_elem").reqVec().getArray()
    Y_elem = gfc.reqVecteurPETSc("Vec_Y_elem").reqVec().getArray()
    
    # On prend le barycentre géométrique global pour éviter les limites
    xc, yc = np.mean(X_elem), np.mean(Y_elem)
    dist_au_centre = np.sqrt((X_elem - xc)**2 + (Y_elem - yc)**2)
    elem_ref = int(np.argmin(dist_au_centre))
    
    print(f"-> Élément de référence (Centre du domaine) : Index {elem_ref}")
    print(f"-> Coordonnées : X = {X_elem[elem_ref]:.4f}, Y = {Y_elem[elem_ref]:.4f}")
    
    # Distances euclidiennes de tous les éléments par rapport au centre
    distances = np.sqrt((X_elem - X_elem[elem_ref])**2 + (Y_elem - Y_elem[elem_ref])**2)

    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_gaussien")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")

    rng = np.random.default_rng(42)
    K_historique = np.zeros((n_tirages, N_elements))

    # ---------------------------------------------------------
    # BOUCLE MONTE CARLO (Génération de champs purs)
    # ---------------------------------------------------------
    print(f"\n1. Génération des {n_tirages} champs K (Monte Carlo)...")
    fd_stdout = sys.stdout.fileno()
    old_stdout = os.dup(fd_stdout)
    devnull = os.open(os.devnull, os.O_WRONLY)

    start = time.time()
    for i in range(n_tirages):
        bruit = rng.normal(0, 0.3, N_elements)
        bruit = np.clip(bruit, -0.9, 0.9)
        vec_bruit.setValues(indices_elem, bruit)
        vec_bruit.assemble()
        
        os.dup2(devnull, fd_stdout)
        pp_import_bruit.execute()
        pp_filtre.execute()
        pp_interpole_K.execute()
        gfc.reqPP("pp_copie_K_imp").execute()
        os.dup2(old_stdout, fd_stdout)
        
        K_historique[i, :] = vec_K.getValues(indices_elem).copy()
        
        if (i+1) % 500 == 0:
            print(f"   [{i+1}/{n_tirages}] itérations...")

    os.close(devnull)
    print(f"   Terminé en {time.time()-start:.2f}s.")

    # ---------------------------------------------------------
    # CALCUL MATRICIEL DE L'AUTOCORRÉLATION
    # ---------------------------------------------------------
    print("\n2. Analyse spatiale de l'autocorrélation...")
    # np.corrcoef génère une matrice (942, 942). On extrait la ligne de l'élément de référence.
    corr_matrix = np.corrcoef(K_historique, rowvar=False)
    corr_vector = corr_matrix[elem_ref, :]
    
    # On filtre les NaN possibles causés par des divisions par 0 si un point a une variance nulle
    valid = ~np.isnan(corr_vector)
    dist_valid = distances[valid]
    corr_valid = corr_vector[valid]
    
    # Fitting par les moindres carrés non-linéaires
    popt, pcov = curve_fit(gaussian_cov, dist_valid, corr_valid, p0=[0.12])
    l_corr_mesure = popt[0]
    
    print(f"\n---> RÉSULTAT DU FITTING <---")
    print(f"Longueur de corrélation devinée (Ancienne) : 0.1200")
    print(f"Longueur de corrélation MESURÉE (Nouvelle) : {l_corr_mesure:.4f}")
    
    # ---------------------------------------------------------
    # TRACÉ ET VALIDATION VISUELLE
    # ---------------------------------------------------------
    os.makedirs("resultats", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_valid, corr_valid, alpha=0.5, label='Données MC Empiriques', color='steelblue', s=10)
    
    r_plot = np.linspace(0, np.max(dist_valid), 200)
    plt.plot(r_plot, gaussian_cov(r_plot, l_corr_mesure), 'r-', lw=3, label=f'Fit Gaussien ($l_{{corr}} = {l_corr_mesure:.4f}$)')
    plt.plot(r_plot, gaussian_cov(r_plot, 0.12), 'k--', lw=2, label='Ancienne hypothèse ($l_{{corr}} = 0.12$)')
    
    plt.axhline(0, color='black', lw=1)
    plt.xlabel('Distance $r$ depuis le centre', fontsize=12)
    plt.ylabel('Coefficient de corrélation spatiale $\\rho(r)$', fontsize=12)
    plt.title('Mesure de la longueur de corrélation du filtre MEF++', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    nom_fig = "resultats/Autocorrelation_Filtre_MC.png"
    plt.tight_layout()
    plt.savefig(nom_fig, dpi=300)
    print(f"\nGraphique sauvegardé : {nom_fig}")

    mefpp.finalise()

if __name__ == "__main__":
    mesurer_longueur_correlation()