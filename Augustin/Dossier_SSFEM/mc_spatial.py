################################################################################
# Monte Carlo par convolution
# Utilisation de la resolution native MEF++ pour garantir la physique
# Auteur : El Mehdi En-Nahas
################################################################################

import numpy as np
import sys
import os
import time
import glob
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def monte_carlo_spatial(n_mc=1000):
    print(f"--- Demarrage de Monte Carlo ({n_mc} iterations) ---")
    
    petsc4py.init(sys.argv)
    prefixe = "conduction_trou"
    mefpp.initialise(prefixe)
    
    # resolution initiale deterministe pour initialiser la memoire
    mefpp.litEtExecuteActionsDansCollection()
    
    # suppression dynamiquement l'option de monitoring KSP
    opts = PETSc.Options()
    if opts.hasName('options_slinksp_monitor'):
        opts.delValue('options_slinksp_monitor')
        
    corps = mefpp.reqCollectionDeCorps().reqCorps(prefixe)
    gfc = corps.reqGFC()

    # handles PETSc et MEF++
    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    T_imp = gfc.reqVecteurPETSc("T_imp").reqVec()
    
    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_gaussien")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")
    pp_assemblage = gfc.reqPP("ppAssMatEtRes")
    pp_resolution = gfc.reqPP("resolution")
    pp_copie_T = gfc.reqPP("pp_copie_T_imp")
    
    N_noeuds = T_imp.getSize()
    N_elements = vec_bruit.getSize()
    
    # optimisation et reproductibilite
    rng = np.random.default_rng(42) # seed fixe pour reproductibilite
    indices = np.arange(N_elements, dtype=np.int32) 
    
    # algorithme de welford pour la variance
    T_mean = np.zeros(N_noeuds)
    M2 = np.zeros(N_noeuds) # accumulateur pour la variance
    
    fd_stdout = sys.stdout.fileno()
    fd_stderr = sys.stderr.fileno()
    old_stdout = os.dup(fd_stdout)
    old_stderr = os.dup(fd_stderr)
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    for i in range(n_mc):
        bruit = rng.normal(0, 0.3, N_elements)
        bruit = np.clip(bruit, -0.9, 0.9)
        
        vec_bruit.setValues(indices, bruit)
        vec_bruit.assemble()
        
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
        
        # extraction de la temperature
        T_courant = np.array(T_imp[:])
        
        # mise a jour de welford (moyenne et variance)
        delta = T_courant - T_mean
        T_mean += delta / (i + 1)
        delta2 = T_courant - T_mean
        M2 += delta * delta2
        
        # affichage du diagnostic de convergence
        if (i+1) % 10 == 0:
            var_estime = M2 / (i + 1)
            # Erreur standard max sur tout le domaine (sigma / sqrt(N))
            std_err = np.sqrt(np.max(var_estime) / (i + 1))
            print(f"Iteration {i+1}/{n_mc} | T_max: {np.max(T_courant):.2f}°C | Err. Standard MC: {std_err:.4e}")
            
        # nettoyage automatique des fichiers generes par MEF++ tous les 50 pas
        if (i+1) % 50 == 0:
            for f in glob.glob("resultats/T_resultat_ssfem*"):
                try: os.remove(f)
                except Exception: pass
            
    T_var = M2 / n_mc
    np.save("vraie_solution_mc_spatial.npy", T_mean)
    np.save("variance_mc_spatial.npy", T_var) # sauvegarde de la variance
    
    os.close(devnull)
    print("\nTermine !")
    mefpp.finalise()

if __name__ == "__main__":
    start = time.perf_counter()
    monte_carlo_spatial(1000)
    end = time.perf_counter()
    print(f"Temps d'exécution : {end - start:.4f} secondes")