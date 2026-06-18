################################################################################
# Monte Carlo Spatial Indépendant (Vérité Absolue par Convolution)
# Génère la Vraie Solution parfaitement symétrique pour évaluer SSFEM
################################################################################

import numpy.random as rd
import numpy as np
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def monte_carlo_spatial(n_mc=1000):
    print(f"--- Démarrage du Monte Carlo Spatial de Référence ({n_mc} itérations) ---")
    
    petsc4py.init(sys.argv)
    prefixe = "conduction_trou"
    mefpp.initialise(prefixe)
    mefpp.litEtExecuteActionsDansCollection()
    corps = mefpp.reqCollectionDeCorps().reqCorps(prefixe)
    gfc = corps.reqGFC()

    matK = gfc.reqMatricePETSc("MatK").reqMat()
    residu = gfc.reqVecteurPETSc("Residu").reqVec()
    vec_bruit = gfc.reqVecteurPETSc("vec_bruit_blanc").reqVec()
    
    pp_import_bruit = gfc.reqPP("pp_import_bruit")
    pp_filtre = gfc.reqPP("applique_filtre_uniforme")
    pp_interpole_K = gfc.reqPP("pp_interpole_K_spatial")
    pp_assemblage = gfc.reqPP("ppAssMatK")
    
    # LA CORRECTION EST ICI : PP pour assembler la force (flux de Neumann)
    pp_assemblage_init = gfc.reqPP("ppAssMatEtRes")
    
    N_noeuds = matK.getSize()[0]
    N_elements = vec_bruit.getSize()
    
    T_mean = np.zeros(N_noeuds)
    vec_T = PETSc.Vec().createSeq(N_noeuds)
    
    ksp = PETSc.KSP().create()
    ksp.setType('preonly')
    ksp.getPC().setType('lu')
    
    # On calcule le vecteur de force (residu) UNE FOIS avant de commencer
    pp_assemblage_init.execute()
    residu.assemble()
    
    for i in range(n_mc):
        bruit = rd.normal(0, 0.01, N_elements)
        indices = np.arange(N_elements, dtype=np.int32)
        vec_bruit.setValues(indices, bruit)
        vec_bruit.assemble()
        
        pp_import_bruit.execute()
        pp_filtre.execute()
        pp_interpole_K.execute()
        
        pp_assemblage.execute()
        matK.assemble()
        
        # Le solveur voit maintenant le flux de chaleur dans 'residu' !
        ksp.setOperators(matK)
        ksp.solve(residu, vec_T)
        
        T_mean += np.array(vec_T.getArray())
        
        if (i+1) % 10 == 0:
            print(f"Itération MC Spatial : {i+1}/{n_mc}")
            
    T_mean /= n_mc
    np.save("vraie_solution_mc_spatial.npy", T_mean)
    print("Terminé ! La Vérité Absolue est sauvegardée dans 'vraie_solution_mc_spatial.npy'")
    mefpp.finalise()

if __name__ == "__main__":
    monte_carlo_spatial(1000)