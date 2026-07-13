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

def monte_carlo_spatial(chunk_size=100):
    print(f"Demarrage de Monte Carlo (Bloc de {chunk_size} iterations)")
    
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
    
    indices = np.arange(N_elements, dtype=np.int32) 
    
    # checkpointing
    fichier_etat = "mc_state.npy"
    if os.path.exists(fichier_etat):
        etat = np.load(fichier_etat, allow_pickle=True).item()
        T_mean = etat['T_mean']
        M2 = etat['M2']
        iter_debut = etat['iterations_faites']
        print(f"Reprise du calcul : {iter_debut} itérations déjà calculées.")
    else:
        T_mean = np.zeros(N_noeuds)
        M2 = np.zeros(N_noeuds)
        iter_debut = 0
        print("Nouveau calcul : départ à l'itération 0.")

    # graine unique par bloc pour garantir des tirages differents a chaque redemarrage
    rng = np.random.default_rng(42 + iter_debut)
    
    fd_stdout = sys.stdout.fileno()
    fd_stderr = sys.stderr.fileno()
    old_stdout = os.dup(fd_stdout)
    old_stderr = os.dup(fd_stderr)
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    iter_actuelle = iter_debut

    try:
        for i in range(chunk_size):
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
            
            T_courant = np.array(T_imp[:])
            
            iter_actuelle = iter_debut + i + 1
            
            # mise a jour de welford
            delta = T_courant - T_mean
            T_mean += delta / iter_actuelle
            delta2 = T_courant - T_mean
            M2 += delta * delta2
            
            # affichage console (estimateur sans biais N-1)
            if iter_actuelle % 10 == 0 and iter_actuelle > 1:
                var_estime = M2 / (iter_actuelle - 1)
                std_err = np.sqrt(np.max(var_estime) / iter_actuelle)
                print(f"Iteration globale {iter_actuelle}, T_max: {np.max(T_courant):.2f}°C, Err. Standard MC: {std_err:.4e}")
                
            # nettoyage automatique des fichiers MEF++
            if iter_actuelle % 50 == 0:
                for f in glob.glob("resultats/T_resultat_ssfem*"):
                    try: os.remove(f)
                    except Exception: pass

    except KeyboardInterrupt:
        os.dup2(old_stdout, fd_stdout)
        os.dup2(old_stderr, fd_stderr)
        print(f"\nArrêt manuel détecté à l'itération {iter_actuelle}.")
        sys.exit(0)
                
    finally:
        # sauvegarde
        os.dup2(old_stdout, fd_stdout)
        os.dup2(old_stderr, fd_stderr)
        os.close(devnull)
        
        if iter_actuelle > iter_debut:
            etat = {'T_mean': T_mean, 'M2': M2, 'iterations_faites': iter_actuelle}
            np.save(fichier_etat, etat)
            
            if iter_actuelle > 1:
                T_var = M2 / (iter_actuelle - 1)
                np.save("vraie_solution_mc_spatial.npy", T_mean)
                np.save("variance_mc_spatial.npy", T_var) 
            print(f"État enregistré à l'itération {iter_actuelle}.")
        
    print("Fin du processus MEF++ local. Purge de la mémoire vive.")
    mefpp.finalise()

if __name__ == "__main__":
    chunk = 100
    if len(sys.argv) > 1:
        chunk = int(sys.argv[1])
        
    start = time.perf_counter()
    monte_carlo_spatial(chunk)
    end = time.perf_counter()
    print(f"Temps d'exécution du bloc : {end - start:.4f} secondes")