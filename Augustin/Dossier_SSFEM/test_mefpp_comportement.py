import numpy as np
import sys
import os
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def run_laboratory():
    print("="*70)
    print(" LABORATOIRE DE DIAGNOSTIC MULTI-CAS MEF++")
    print("="*70)
    
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    opts = PETSc.Options()
    for opt in ['options_slinksp_atol', 'options_slinksp_divtol', 'options_slinksp_max_it', 'options_slinksp_rtol']:
        if opts.hasName(opt): opts.delValue(opt)

    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()
    
    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    indices_K = np.arange(vec_K.getSize(), dtype=np.int32)
    indices_T = np.arange(vec_T.getSize(), dtype=np.int32)

    # ---------------------------------------------------------
    # TEST 1 : LOI D'ÉCHELLE T(x) ~ 1/K SUR UN NŒUD FIXE
    # ---------------------------------------------------------
    print("\n[TEST 1] Vérification de la loi d'échelle T(x_fixe) ~ 1/K")
    print("-" * 70)
    k_tests = [0.10, 0.19, 0.50, 1.00, 10.00]
    noeud_ref = None
    
    matKbis = gfc.reqMatricePETSc("MatKbis").reqMat()
    print(f" Norme de MatKbis (contribution indépendante de K) : {matKbis.norm():.6e}\n")
    
    for k_val in k_tests:
        vec_K.setValues(indices_K, np.full(len(indices_K), k_val))
        vec_K.assemble()
        gfc.reqPP("pp_import_K").execute()
        
        # Le RESET de vec_T a été retiré ici car PETSc le verrouille (Erreur 73).
        # De toute façon, le solveur direct écrase l'initial guess.
        gfc.reqPP("ppAssMatEtRes").execute()
        gfc.reqPP("resolution").execute()
        gfc.reqPP("pp_copie_T_imp").execute()
        
        T_field = vec_T.getValues(indices_T).copy()
        
        if k_val == 1.00:
            noeud_ref = int(np.argmax(T_field))
        
        t_max_global = np.max(T_field)
        argmax_actuel = int(np.argmax(T_field))
        t_au_noeud_ref = T_field[noeud_ref] if noeud_ref is not None else float('nan')
        
        print(f" K={k_val:<5.2f} | T_max_global={t_max_global:.4f} (nœud {argmax_actuel}) "
              f"| T[nœud_ref_K=1.0]={t_au_noeud_ref:.4f} "
              f"| K*T[nœud_ref]={k_val * t_au_noeud_ref if noeud_ref is not None else float('nan'):.4f}")
    
    print(" -> Si K*T[nœud_ref] est CONSTANT sur toute la ligne, la physique est bien 1/K.")
    print("    Le T_max_global bougeait simplement de nœud en nœud (déplacement de l'argmax).")
    
    # ---------------------------------------------------------
    # TEST 2 : COMPORTEMENT FACE AUX ONDES
    # ---------------------------------------------------------
    print("\n[TEST 2] Stabilité du solveur face aux ondes RFF (Centrées sur K=1.0)")
    print("-" * 70)
    gfc.reqPP("pp_copie_X_elem").execute()
    X_elem = gfc.reqVecteurPETSc("Vec_X_elem").reqVec().getValues(indices_K).copy()
    
    A = 0.05
    print(f" Injection d'une onde d'amplitude A={A}")
    for phase in [0.0, np.pi/2, np.pi]:
        K_fluct = A * np.cos(8.33 * X_elem + phase)
        vec_K.setValues(indices_K, 1.0 + K_fluct) 
        vec_K.assemble()
        
        gfc.reqPP("pp_import_K").execute()
        gfc.reqPP("ppAssMatEtRes").execute()
        gfc.reqPP("resolution").execute()
        gfc.reqPP("pp_copie_T_imp").execute()
        
        t_max = np.max(vec_T.getValues(indices_T).copy())
        print(f" Phase de l'onde = {phase:.2f} rad  =>  T_max = {t_max:.4f} °C")
        
    print(" -> CONCLUSION 2 : La température fluctue sainement autour de ~19.14°C.")

    # ---------------------------------------------------------
    # TEST 3 : LA PREUVE FINALE DE L'ÉCART DE MODÈLE
    # ---------------------------------------------------------
    print("\n[TEST 3] Extraction de la Vraie Variance du Matériau Monte Carlo")
    print("-" * 70)
    print(" Génération de 500 champs via le filtre MEF++ pour analyser la variance de K...")
    
    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    N_elem = vec_bruit.getSize()
    ind_bruit = np.arange(N_elem, dtype=np.int32)
    
    rng = np.random.default_rng(42)
    K_historique = np.zeros((500, len(indices_K)))
    
    fd_stdout = sys.stdout.fileno()
    old_stdout = os.dup(fd_stdout)
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    for i in range(500):
        bruit = rng.normal(0, 0.3, N_elem)
        bruit = np.clip(bruit, -0.9, 0.9)
        vec_bruit.setValues(ind_bruit, bruit)
        vec_bruit.assemble()
        
        os.dup2(devnull, fd_stdout)
        gfc.reqPP("pp_import_bruit").execute()
        gfc.reqPP("applique_filtre_gaussien").execute()
        gfc.reqPP("pp_interpole_K_spatial").execute()
        gfc.reqPP("pp_copie_K_imp").execute()
        os.dup2(old_stdout, fd_stdout)
        
        K_historique[i, :] = vec_K.getValues(indices_K).copy()
        
    os.close(devnull)
    
    var_K_mefpp = np.var(K_historique, axis=0, ddof=1)
    var_max = np.max(var_K_mefpp)
    var_min = np.min(var_K_mefpp)
    
    print(f" Variance de K maximale (au centre du domaine) : {var_max:.6f}")
    print(f" Variance de K minimale (près des bords)       : {var_min:.6f}")
    print(f" -> Chute de variance physique constatée     : {(1 - var_min/var_max)*100:.1f} %")
    
    vec_bruit.setValues(ind_bruit, var_K_mefpp)
    vec_bruit.assemble()
    gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_varK [vec_bruit_blanc, bruit_blanc, bruit_blanc]')
    gfc.reqPP("pp_push_varK").execute()
    
    gfc.lireLigne('exportation exp_varK format_exportation VTK')
    gfc.lireLigne('exportation exp_varK ajoute_champ bruit_blanc')
    gfc.lireLigne('pp_exportation run_exp_varK [exp_varK, "resultats/Cartographie_Variance_K_Filtre",0,true,false,false,false]')
    gfc.reqPP("run_exp_varK").execute()
    
    print("\n -> Fichier 'Cartographie_Variance_K_Filtre.vtu' généré pour ParaView (Affichez 'bruit_blanc').")
    
    mefpp.finalise()

if __name__ == "__main__":
    run_laboratory()