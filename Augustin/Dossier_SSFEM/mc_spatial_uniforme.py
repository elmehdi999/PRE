################################################################################
# Monte Carlo Spatial Indépendant (Vérité Absolue par Convolution)
# Utilisation de la résolution native MEF++ pour garantir la physique
################################################################################

import numpy as np
import sys
import os
import glob
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def monte_carlo_spatial(n_mc=1000):
    print(f"--- Démarrage du Vrai Monte Carlo Spatial ({n_mc} itérations) ---")
    print("Calcul de la physique en cours... (Le moteur C++ a été rendu totalement silencieux)")
    
    petsc4py.init(sys.argv)
    prefixe = "conduction_trou"
    mefpp.initialise(prefixe)
    
    # Résolution initiale déterministe pour initialiser la mémoire
    mefpp.litEtExecuteActionsDansCollection()
    
    # Suppression dynamiquement l'option de monitoring KSP
    opts = PETSc.Options()
    if opts.hasName('options_slinksp_monitor'):
        opts.delValue('options_slinksp_monitor')
        
    corps = mefpp.reqCollectionDeCorps().reqCorps(prefixe)
    gfc = corps.reqGFC()

    # Handles PETSc et MEF++
    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    T_imp = gfc.reqVecteurPETSc("T_imp").reqVec()
    
    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_uniforme")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")
    pp_assemblage = gfc.reqPP("ppAssMatEtRes")
    pp_resolution = gfc.reqPP("resolution")
    pp_copie_T = gfc.reqPP("pp_copie_T_imp")
    
    N_noeuds = T_imp.getSize()
    N_elements = vec_bruit.getSize()
    
    # --- OPTIMISATION ET REPRODUCTIBILITÉ (Retours de Claude) ---
    rng = np.random.default_rng(42) # Seed fixe pour reproductibilité
    indices = np.arange(N_elements, dtype=np.int32) # Sorti de la boucle
    
    # --- ALGORITHME DE WELFORD POUR LA VARIANCE ---
    T_mean = np.zeros(N_noeuds)
    M2 = np.zeros(N_noeuds) # Accumulateur pour la variance
    
    # --- PRÉPARATION DU SILENCIEUX C++ ---
    fd_stdout = sys.stdout.fileno()
    fd_stderr = sys.stderr.fileno()
    old_stdout = os.dup(fd_stdout)
    old_stderr = os.dup(fd_stderr)
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    for i in range(n_mc):
        bruit = rng.normal(0, 0.3, N_elements)
        # Si tu passes à K = exp(bruit_filtre) dans le .champs, tu peux supprimer cette ligne :
        bruit = np.clip(bruit, -0.9, 0.9)
        
        vec_bruit.setValues(indices, bruit)
        vec_bruit.assemble()
        
        # --- BLOC TRY/FINALLY (Sécurité absolue du terminal) ---
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, fd_stdout)
        os.dup2(devnull, fd_stderr)
        
        try:
            pp_import_bruit.execute()
            pp_filtre.execute()
            pp_interpole_K.execute()
            pp_assemblage.execute()
            pp_resolution.execute()
            pp_copie_T.execute()
        finally:
            os.dup2(old_stdout, fd_stdout)
            os.dup2(old_stderr, fd_stderr)
        
        # Extraction de la température
        T_courant = np.array(T_imp[:])
        
        # --- MISE À JOUR DE WELFORD (Moyenne et Variance en direct) ---
        delta = T_courant - T_mean
        T_mean += delta / (i + 1)
        delta2 = T_courant - T_mean
        M2 += delta * delta2
        
        # Affichage du diagnostic de convergence
        if (i+1) % 10 == 0:
            var_estime = M2 / (i + 1)
            # Erreur standard max sur tout le domaine (sigma / sqrt(N))
            std_err = np.sqrt(np.max(var_estime) / (i + 1))
            print(f"Itération {i+1}/{n_mc} | T_max: {np.max(T_courant):.2f}°C | Err. Standard MC: {std_err:.4e}")
            
        # Nettoyage automatique des fichiers générés par MEF++ tous les 50 pas
        if (i+1) % 50 == 0:
            for f in glob.glob("resultats/T_resultat_ssfem*"):
                try: os.remove(f)
                except Exception: pass
            
    # Finalisation
    T_var = M2 / n_mc
    np.save("vraie_solution_mc_spatial.npy", T_mean)
    np.save("variance_mc_spatial.npy", T_var) # Sauvegarde de la vraie variance spatiale !
    
    os.close(devnull)
    print("\nTerminé ! La Vérité Absolue est sauvegardée sans polluer le dossier.")
    mefpp.finalise()

if __name__ == "__main__":
    monte_carlo_spatial(1000)