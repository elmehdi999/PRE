import numpy as np
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def scanner_matrice():
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()
    
    # Assemblage de la matrice K0
    gfc.reqPP("ppAssMatEtRes").execute()
    K0 = gfc.reqMatricePETSc("MatK").reqMat()
    
    diag = K0.getDiagonal().getArray()
    
    print("\n" + "="*50)
    print(" ANALYSE DE LA DIAGONALE DE K0 (MEF++)")
    print("="*50)
    print(f"Nombre total de DDL      : {len(diag)}")
    print(f"Valeur Minimum           : {np.min(diag):.4e}")
    print(f"Valeur Maximum           : {np.max(diag):.4e}")
    print(f"Valeur Moyenne           : {np.mean(diag):.4e}")
    print(f"Médiane                  : {np.median(diag):.4e}")
    
    # Recherche de la signature de Dirichlet
    if np.max(diag) > 1e4:
        dofs = np.where(diag > 1e4)[0]
        print(f"\n[DÉTECTION] Méthode de pénalisation détectée !")
        print(f"-> {len(dofs)} nœuds ont une pénalité > 10^4.")
        print(f"-> Valeur exacte de la pénalité : {np.max(diag):.15e}")
    else:
        dofs = np.where(diag == 1.0)[0]
        print(f"\n[DÉTECTION] Pas de pénalité massive. Recherche de substitution (Diag=1.0)...")
        print(f"-> {len(dofs)} nœuds ont une diagonale exactement égale à 1.0.")
        
    print("="*50)
    mefpp.finalise()

if __name__ == "__main__":
    scanner_matrice()