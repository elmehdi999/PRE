################################################################################
# Monte Carlo Spatial
# Utilisation de la resolution native MEF++ pour garantir la physique
# Auteur: El Mehdi EN-NAHAS
################################################################################

import numpy.random as rd
import numpy as np
import sys
import os
import glob
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def monte_carlo_spatial(n_mc=1000):
    print(f"- Demarrage de Monte Carlo ({n_mc} iterations)")
    
    petsc4py.init(sys.argv)
    prefixe = "conduction_trou"
    mefpp.initialise(prefixe)
    
    # resolution initiale deterministe pour initialiser la memoire
    mefpp.litEtExecuteActionsDansCollection()
        
    corps = mefpp.reqCollectionDeCorps().reqCorps(prefixe)
    gfc = corps.reqGFC()

    # on ne recupere que les vecteurs d'entree (bruit) et de sortie (temperature)
    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    T_imp = gfc.reqVecteurPETSc("T_imp").reqVec()
    
    # outils MEF++
    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_uniforme")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")
    
    # moteurs natifs de MEF++
    pp_assemblage = gfc.reqPP("ppAssMatEtRes")
    pp_resolution = gfc.reqPP("resolution")
    pp_copie_T = gfc.reqPP("pp_copie_T_imp")
    
    N_noeuds = T_imp.getSize()
    N_elements = vec_bruit.getSize()
    
    T_mean = np.zeros(N_noeuds)
    
    # On capture les sorties standard (stdout et stderr) de l'OS
    fd_stdout = sys.stdout.fileno()
    fd_stderr = sys.stderr.fileno()
    old_stdout = os.dup(fd_stdout)
    old_stderr = os.dup(fd_stderr)
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    for i in range(n_mc):
        # bruit blanc
        bruit = rd.normal(0, 0.3, N_elements)
        # securite : on bride le bruit pour garantir K > 0 (car K = 1 + bruit)
        bruit = np.clip(bruit, -0.9, 0.9)
        
        indices = np.arange(N_elements, dtype=np.int32)
        vec_bruit.setValues(indices, bruit)
        vec_bruit.assemble()
        
        # utile pour garder le terminal propre
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, fd_stdout)
        os.dup2(devnull, fd_stderr)
        
        # execution de MEF++ (totalement invisible dans le terminal)
        pp_import_bruit.execute()
        pp_filtre.execute()
        pp_interpole_K.execute()
        pp_assemblage.execute()
        pp_resolution.execute()
        pp_copie_T.execute()
        
        # on active le terminal
        os.dup2(old_stdout, fd_stdout)
        os.dup2(old_stderr, fd_stderr)
        
        # extraction de la temperature
        T_courant = np.array(T_imp[:])
        T_mean += T_courant
        
        # affichage d'avancement
        if (i+1) % 10 == 0:
            print(f"Iteration MC : {i+1}/{n_mc} (Temp max locale = {np.max(T_courant):.2f} °C)")
            
    fichiers_a_supprimer = glob.glob("resultats/T_resultat_ssfem*")
    for f in fichiers_a_supprimer:
        try: os.remove(f)
        except Exception: pass
            
    # finalisation
    T_mean /= n_mc
    np.save("vraie_solution_mc_spatial.npy", T_mean)
    
    # nettoyage des fichiers et liberation de l'OS
    fichiers_finaux = glob.glob("resultats/T_resultat_ssfem*")
    for f in fichiers_finaux:
        try: os.remove(f)
        except Exception: pass
    os.close(devnull)
    
    print("\nTermine !")
    mefpp.finalise()

if __name__ == "__main__":
    monte_carlo_spatial(1000)