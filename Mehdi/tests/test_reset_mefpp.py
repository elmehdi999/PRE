import numpy as np
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def test_reset():
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    indices_K = np.arange(vec_K.getSize(), dtype=np.int32)
    indices_T = np.arange(vec_T.getSize(), dtype=np.int32)

    # Création dynamique de l'opérateur de reset via MEF++
    # On interpole le scalaire 'T_exacte' (qui vaut 0) dans le champ 'T'
    gfc.lireLigne('pp_reinterpole pp_reset_T [T, T_exacte]')
    pp_reset = gfc.reqPP("pp_reset_T")

    k_tests = [0.10, 0.19, 0.50, 1.00, 10.00]
    noeud_cible = 97

    print("="*60)
    print(" TEST DU RESET VIA PP_REINTERPOLE NATIVE MEF++")
    print("="*60)

    for k in k_tests:
        vec_K.setValues(indices_K, np.full(len(indices_K), k))
        vec_K.assemble()
        gfc.reqPP("pp_import_K").execute()

        # LE VRAI RESET : Écrase la mémoire C++ du champ T avec des zéros
        pp_reset.execute()

        gfc.reqPP("ppAssMatEtRes").execute()
        gfc.reqPP("resolution").execute()
        gfc.reqPP("pp_copie_T_imp").execute()

        t = vec_T.getValues(indices_T).copy()[noeud_cible]
        print(f"K={k:<5.2f} | T = {t:.6f} | K*T = {k*t:.6f}")

    mefpp.finalise()

if __name__ == "__main__":
    test_reset()