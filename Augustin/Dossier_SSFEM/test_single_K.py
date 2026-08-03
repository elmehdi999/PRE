import sys
import numpy as np
import petsc4py
from mefpp4py import mefpp

def test_K(k_val, noeud_cible):
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()
    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    indices_K = np.arange(vec_K.getSize(), dtype=np.int32)
    indices_T = np.arange(vec_T.getSize(), dtype=np.int32)
    
    vec_K.setValues(indices_K, np.full(len(indices_K), k_val))
    vec_K.assemble()
    gfc.reqPP("pp_import_K").execute()
    gfc.reqPP("ppAssMatEtRes").execute()
    
    # LE FIX ULTIME : Création d'objets avec de nouveaux noms pour contourner ERR_OBJET_DEJA_EXISTANT.
    # Cela force MEF++ à créer une nouvelle instance KSP PETSc et à refaire la factorisation LU.
    gfc.lireLigne('solveur_lin Solveur_Frais(ProbDida) prefixe_options options_slin')
    gfc.lireLigne('pp_resolution_probleme reso_fraiche [ProbDida,Solveur_Frais(ProbDida)]')
    
    gfc.reqPP("reso_fraiche").execute()
    gfc.reqPP("pp_copie_T_imp").execute()
    
    T_field = vec_T.getValues(indices_T).copy()
    print(f"RESULTAT K={k_val} T_noeud97={T_field[noeud_cible]:.6f}")
    
    mefpp.finalise()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_K(float(sys.argv[1]), 97)