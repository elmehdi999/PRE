import numpy as np
import scipy.sparse as sp
import sys
import os
import petsc4py
from mefpp4py import mefpp

def petsc_to_scipy(petsc_mat):
    petsc_mat.assemble()
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())

def run_diagnostics():
    petsc4py.init(sys.argv)
    
    # FIX PROPRE CORRIGÉ : Sans underscore entre slin et ksp/pc
    opts = petsc4py.PETSc.Options()
    opts.setValue('options_slinksp_type', 'gmres')
    opts.setValue('options_slinpc_type', 'jacobi')
    for opt in ['options_slinksp_atol', 'options_slinksp_divtol', 'options_slinksp_max_it', 'options_slinksp_rtol']:
        if opts.hasName(opt): opts.delValue(opt)

    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    # ---------------------------------------------------------
    # 1. VERIFICATION T_EXACTE NUL
    # ---------------------------------------------------------
    vec_T_exacte = gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
    indices_T = np.arange(vec_T_exacte.getSize(), dtype=np.int32)
    valeurs_Texacte = vec_T_exacte.getValues(indices_T)
    print(f"--- 1. VERIFICATION T_exacte ---")
    print(f"Valeur max absolue dans T_exacte_vec : {np.max(np.abs(valeurs_Texacte)):.6e}")

    # ---------------------------------------------------------
    # 2. ASYMETRIE DE MATK
    # ---------------------------------------------------------
    gfc.reqPP("ppAssMatEtRes").execute()
    matK_petsc = gfc.reqMatricePETSc("MatK").reqMat()
    K_sp = petsc_to_scipy(matK_petsc)
    asymetrie = np.abs(K_sp - K_sp.T).max()
    print(f"\n--- 2. VERIFICATION ASYMETRIE MatK ---")
    print(f"Asymétrie max : {asymetrie:.6e}")
    if asymetrie > 1e-10:
        print("-> Matrice ASYMETRIQUE (Pénalisation de Dirichlet probable).")
    else:
        print("-> Matrice symétrique.")

    # ---------------------------------------------------------
    # 3. TEST DE VALIDATION DU SOLVEUR (Sans hack lireLigne ni reset)
    # ---------------------------------------------------------
    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    indices_K = np.arange(vec_K.getSize(), dtype=np.int32)
    k_tests = [0.10, 0.19, 0.50, 1.00, 10.00]
    
    print(f"\n--- 3. TEST DE VALIDATION DU SOLVEUR PROPRE (GMRES+Jacobi) ---")
    for k in k_tests:
        vec_K.setValues(indices_K, np.full(len(indices_K), k))
        vec_K.assemble()
        gfc.reqPP("pp_import_K").execute()
        gfc.reqPP("ppAssMatEtRes").execute()
        # Appel de la résolution standard de MEF++ SANS recréer d'objet
        gfc.reqPP("resolution").execute()
        gfc.reqPP("pp_copie_T_imp").execute()
        
        t = vec_T.getValues(indices_T).copy()[97]
        print(f"K={k:<5.2f} | T = {t:.6f} | K*T = {k*t:.6f}")

    mefpp.finalise()

if __name__ == "__main__":
    run_diagnostics()