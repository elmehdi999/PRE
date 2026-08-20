import numpy as np
import sys
import os
import time
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def comparer_variance_K(n_tirages=3000):
    print("="*60)
    print(f" MESURE HAUTE FIDÉLITÉ VAR(K) : MC vs RFF ({n_tirages} tirages)")
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
    # MAPPING SPATIAL STRICT : Nœud vs Élément
    # ---------------------------------------------------------
    gfc.reqPP("pp_copie_X_elem").execute()
    gfc.reqPP("pp_copie_Y_elem").execute()
    
    X_elem = gfc.reqVecteurPETSc("Vec_X_elem").reqVec().getArray()
    Y_elem = gfc.reqVecteurPETSc("Vec_Y_elem").reqVec().getArray()
    
    # Le trou de Neumann est centré en (0,0). On cherche l'élément le plus proche.
    Rayon = np.sqrt(X_elem**2 + Y_elem**2)
    elem_bord_trou = int(np.argmin(Rayon))
    
    print(f"-> Élément géométrique au bord du trou identifié : Index {elem_bord_trou}")
    print(f"-> Coordonnées du centroïde : X = {X_elem[elem_bord_trou]:.4f}, Y = {Y_elem[elem_bord_trou]:.4f}\n")
    
    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_gaussien")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")

    rng = np.random.default_rng(42)
    K_historique = np.zeros((n_tirages, N_elements))

    print(f"1. Calcul de la variance spatiale de K ({n_tirages} tirages)...")
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
            print(f"   [{i+1}/{n_tirages}] itérations MEF++...")

    os.close(devnull)
    print(f"   Terminé en {time.time()-start:.2f}s.")

    # ---------------------------------------------------------
    # CALCULS STATISTIQUES ET RATIO
    # ---------------------------------------------------------
    var_K_mc = np.var(K_historique, axis=0, ddof=1)
    var_K_rff_val = 0.0401**2
    
    var_chaud_mc = var_K_mc[elem_bord_trou]
    ratio_claude = var_chaud_mc / var_K_rff_val

    print("\n--- ANALYSE DE LA VARIANCE D'ENTRÉE (Var(K)) ---")
    print(f"Var(K) RFF (Analytique constante)     : {var_K_rff_val:.6f}")
    print(f"Var(K) MC  (Max global au centre)     : {np.max(var_K_mc):.6f}")
    print(f"Var(K) MC  (Min global sur les bords) : {np.min(var_K_mc):.6f}")
    print(f"Var(K) MC  (Élément exact du trou)    : {var_chaud_mc:.6f}")
    
    print("\n--- RATIO SPATIAL DE CLAUDE ---")
    print(f"Ratio Var(K)_MC / Var(K)_RFF au bord  : {ratio_claude:.3f}")

    # Injection propre via vec_bruit_blanc (non-verrouillé, 942 DDL)
    vec_bruit.setValues(indices_elem, var_K_mc)
    vec_bruit.assemble()
    
    gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_vark [vec_bruit_blanc, bruit_blanc, bruit_blanc]')
    gfc.reqPP("pp_push_vark").execute()

    nom_fichier = "resultats/Cartographie_Var_K_MC_HauteFidelite"
    gfc.lireLigne('exportation exp_var_k format_exportation VTK')
    gfc.lireLigne('exportation exp_var_k ajoute_champ bruit_blanc')
    gfc.lireLigne(f'pp_exportation run_exp_k [exp_var_k, "{nom_fichier}",0,true,false,false,false]')
    gfc.reqPP("run_exp_k").execute()

    print(f"\nExport VTU terminé : {nom_fichier}.vtu")

    mefpp.finalise()

if __name__ == "__main__":
    comparer_variance_K()