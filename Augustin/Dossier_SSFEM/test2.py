#############
#Test comparaison déterministe et ssfem(0,0)
#############
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
#Bibliothèques nécessaires pour la méthode des Éléments finis
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc
# import ks_stat
# import kl_expansion 
# import p_chaos
# import test_init

def init(ordreKL,ordrePC):
    petsc4py.init(sys.argv)
    prefixe = "conduction"
    mefpp.initialise(prefixe,pModeTV=False)  #Si pModeTV=True , déclenchement de Warning et erreurs qui empêchent l'éxecution
    mefpp.litEtExecuteActionsDansCollection()
    collection=mefpp.reqCollectionDeCorps()
    collection.lisDonneesDeBase([prefixe])
    corps=collection.reqCorps(prefixe)
    gfc = corps.reqGFC()


    

    #lecture des lignes
        #Lecture manuelle des lignes mefpp
    requete_dict = {}
    variable_dict = {}
    for i in range(1):
        requete_dict[i] = "scalaire phi_" + str(i) + " " + "1"
        gfc.lireLigne(requete_dict[i])
        variable_dict["K_intermediaire" + str(i)] = "scalaire K_intermediaire2_"+str(i)+" f(K_intermediaire,sqrt_lambda_i, phi_"+str(i)+")=K_intermediaire*2*sqrt_lambda_i*phi_"+ str(i)
        gfc.lireLigne(variable_dict["K_intermediaire" + str(i)])
        variable_dict["pp_reinterpole"+str(i)] = "pp_reinterpole pp_interpoleK_"+str(i)+" [K, K_intermediaire2_"+str(i)+"]"
        gfc.lireLigne(variable_dict["pp_reinterpole" + str(i)])


    matK = gfc.reqMatricePETSc("MatK").reqMat()
    matK.assemble()
    residu = gfc.reqVecteurPETSc("Residu").reqVec()
    residu.assemble()
    print("\n\n\n\n")
    print("Residu avant assemblage matKetRes")
    residu.view()
    lambda_i = gfc.reqChamp("sqrt_lambda_i")
    print("Voici le lambda_i :")
    print(lambda_i.reqValeur())
    pp_resolution = gfc.reqPP("resolution")
    pp_assemblageMatEtRes = gfc.reqPP("ppAssMatEtRes")
    pp_reinterpole = gfc.reqPP("pp_interpoleK_0")

    #construction T
    N = matK.getSize()
    listeconstruction = []
    Tj = PETSc.Vec().createSeq(N)
    Tj.setUp()
    listeconstruction.append(Tj)
    T = PETSc.Vec().createNest(listeconstruction)
    T.setUp()
    T.assemble()

    #Assemblage et résolution initiale
    pp_assemblageMatEtRes.execute()
    residu.assemble()
#################################    
    compteur = 0

    #Affichages
    # print(f"Voici le residu pour {compteur} résolution :\n")
    # residu.view()

    # pp_resolution.execute()
    
    # compteur += 1
    print(f"Voici le résidu pour {compteur} résolution \n:")
    residu.view()
    matK.view()
##################################
    #Résolution
    ksp = PETSc.KSP().create()
    ksp.setOperators(matK)
    ksp.setTolerances(atol=1e-20, rtol=1e-12, divtol=1e30, max_it=4000)
    ksp.setType('cg')
    def monitor(ksp, its, rnorm):
        if its <=5 :
            print(f"iteration {its}: residual norm = {rnorm}")
    ksp.setMonitor(monitor)
    try:
        print(f"Voici taille F : {residu.getSize()} et taille T : {T.getSize()}")
        ksp.solve(residu, T)

        #Vérification de la convergence
        if ksp.is_converged:
            print(f"convergence en {ksp.getIterationNumber()} itérations")
            print(f"Résidu final : {ksp.getResidualNorm()}")
        else:
            print("La résolution n'a pas convergé")
            print(f"Résidu final : {ksp.getResidualNorm()}")

    except PETSc.Error as e:
        print(f"Erreur lors de la résolution : {e}")
            
    return T

T = init(0,0)
T.view()