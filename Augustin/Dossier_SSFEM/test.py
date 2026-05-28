###################
#Code test de petsc4py
###################
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
import petsc4py
import sys
from mefpp4py import mefpp
import ks_stat
import kl_expansion 
import p_chaos
import SSFEM.kl_varphi as kl_varphi

class deterministe:
    #résolution déterministe du problème ef
    
    def __init__(self):
        petsc4py.init(sys.argv)
        prefixe = "conduction" 
        mefpp.initialise(prefixe,pModeTV=False)
        mefpp.litEtExecuteActionsDansCollection()
        collection = mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        gfc=corps.reqGFC()
        self.gfc=gfc

        #variables du .champs
        self.matKbis = gfc.reqMatricePETSc("MatKbis").reqMat()
        self.residu2 = gfc.reqVecteurPETSc("Residu").reqVec()

        #pré-post traitements
        self.pp_resolution = gfc.reqPP("resolution")
        self.pp_assemblageMatEtRes = gfc.reqPP("ppAssMatEtRes")

    
    def resolution(self):
        #construction de T
        N = self.matKbis.getSize()
        print(f"Voici la taille de K:{N}")
        self.T_assemble = PETSc.Vec().createSeq(N)
        self.T_assemble.setUp()
        self.T_assemble.assemble()

        #résolution linéaire
        self.pp_assemblageMatEtRes.execute()
        print("Voici K:")
        self.matKbis.view()
        print("Voici F:")
        self.residu2.view()
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.matKbis)
        ksp.setTolerances(atol=1e-20, rtol=1e-12, divtol=1e30, max_it=4000)
        ksp.setType('cholesky')
        # pc = PETSc.PC()
        # pc.create(PETSc.COMM_SELF)
        # pc.setType("lu")
        # pc.setOperators(self.matKbis)
        # ksp.setPC(pc)
        def monitor(ksp, its, rnorm):
            print(f"iteration {its}: residual norm = {rnorm}")
        ksp.setMonitor(monitor)
        try:
            print(f"Voici taille F : {self.residu2.getSize()} et taille T : {self.T_assemble.getSize()}")
            ksp.solve(self.residu2, self.T_assemble)
            #Vérification de la convergence
            if ksp.is_converged:
                print(f"convergence en {ksp.getIterationNumber()} itérations")
                print(f"Résidu final : {ksp.getResidualNorm()}")
            else:
                print("La résolution n'a pas convergé")
                print(f"Résidu final : {ksp.getResidualNorm()}")

        except PETSc.Error as e:
            print(f"Erreur lors de la résolution : {e}")


        return self.T_assemble
    
    def visualisation(self):
        T_exporte = self.gfc.reqVecteurPETSc("T_exporte").reqVec()
        pp_exporte = self.gfc.reqPP("")
        return None
    
# a = deterministe()
# T = a.resolution()
# T.view()

#Code test de petsc4py
A11 = PETSc.Mat().createConstantDiagonal(5,2.0)
A22 = PETSc.Mat().createConstantDiagonal(5,2.0)
A21 = PETSc.Mat().createConstantDiagonal(5,1.0)
A12 = PETSc.Mat().createConstantDiagonal(5,1.5)
liste = [[A11,A12],[A21,A22]]
Mat_parbloc = PETSc.Mat().createNest(liste)
Mat_parbloc.setUp()
Mat_parbloc.assemble()
Mat_parbloc.view()

#Construction F
liste_construction = []
for i in range(2):
    new_vec = PETSc.Vec().create()
    new_vec.setSizes(5)
    new_vec.setFromOptions()
    if i == 0 :
        new_vec.set(1.0)
    else:
        new_vec.set(1.0)
    liste_construction.append(new_vec)
F = PETSc.Vec().createNest(liste_construction)
F.setUp()
F.assemble()
print("Voici F:")
F.view()

#Construction T
liste_construction = []
for j in range(2):
    Tj = PETSc.Vec().createSeq(5)
    Tj.setUp()
    liste_construction.append(Tj)
T = PETSc.Vec().createNest(liste_construction)
T.setUp()
T.assemble()

#Résolution linéaire

ksp = PETSc.KSP().create()
ksp.setOperators(Mat_parbloc)
ksp.setTolerances(atol=1e-20, rtol=1e-12, divtol=1e30, max_it=4000)

ksp.setType('cg')
# pc = ksp.getPC()
# pc.setType("fieldsplit")
# pc.setOperators(Mat_parbloc)
# ksp.setPC(pc)
def monitor(ksp, its, rnorm):
    print(f"iteration {its}: residual norm = {rnorm}")
ksp.setMonitor(monitor)
print("Voici K:")
Mat_parbloc.view()
print(f"Voici F:, taille {F.getSize()}")
F.view()
try:
    print(f"Voici taille F : {F.getSize()} et taille T : {T.getSize()}")
    
    ksp.solve(F, T)

    #Vérification de la convergence
    if ksp.is_converged:
        print(f"convergence en {ksp.getIterationNumber()} itérations")
        print(f"Résidu final : {ksp.getResidualNorm()}")
    else:
        print("La résolution n'a pas convergé")
        print(f"Résidu final : {ksp.getResidualNorm()}")

except PETSc.Error as e:
    print(f"Erreur lors de la résolution : {e}")
    
T.view()
print("KSP reason:", ksp.getConvergedReason())